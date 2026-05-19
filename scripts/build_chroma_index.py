#!/usr/bin/env python3
"""
Small helper to build a Chroma index using OpenAI embeddings.

Usage:
  python scripts/build_chroma_index.py --persist_dir chroma_db
"""
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Try imports with fallbacks. Prefer LangChain's embedding/vectorstore APIs
# but fall back to raw chromadb + an OpenAI embedding function if needed.
OpenAIEmbeddings = None
Chroma = None
chromadb = None
embedding_functions = None
import_errs = []
try:
    try:
        from langchain.embeddings.openai import OpenAIEmbeddings as _OpenAIEmbeddings
        OpenAIEmbeddings = _OpenAIEmbeddings
    except Exception:
        try:
            from langchain_openai import OpenAIEmbeddings as _OpenAIEmbeddings
            OpenAIEmbeddings = _OpenAIEmbeddings
        except Exception as e:
            import_errs.append(f"OpenAIEmbeddings import error: {e}")

    try:
        from langchain.vectorstores import Chroma as _Chroma
        Chroma = _Chroma
    except Exception as e:
        import_errs.append(f"langchain.vectorstores.Chroma import error: {e}")

    # Try raw chromadb as a fallback for vector storage
    try:
        import chromadb as _chromadb
        from chromadb.utils import embedding_functions as _embedding_functions
        chromadb = _chromadb
        embedding_functions = _embedding_functions
    except Exception as e:
        import_errs.append(f"chromadb import error: {e}")

    if OpenAIEmbeddings is None and chromadb is None:
        raise ImportError("Required embedding/vectorstore classes not available: " + "; ".join(import_errs))
except Exception as e:
    print("Required packages not available:", e)
    print("Install requirements (pip install -r requirements.txt) and try again.")
    sys.exit(1)


def build_index(persist_dir: str = "chroma_db"):
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    texts = [
        "LangGraph is a stateful orchestration library for LLM workflows.",
        "Sequentum MCP exposes tools to LangChain through an MCP API.",
        "Chroma is a lightweight vector database used for retrieval augmentation.",
    ]
    metadatas = [{"source": f"example_{i}"} for i in range(len(texts))]

    # Ensure the OpenAI API key is present before attempting embeddings
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it in your environment and try again.")
        sys.exit(1)

    # If LangChain's Chroma vectorstore is available, prefer that path
    if Chroma is not None and OpenAIEmbeddings is not None:
        embeddings = OpenAIEmbeddings()
        # Create and persist a Chroma collection via LangChain
        try:
            try:
                vectordb = Chroma.from_texts(
                    texts,
                    embedding=embeddings,
                    metadatas=metadatas,
                    persist_directory=str(persist_path),
                )
            except TypeError:
                vectordb = Chroma.from_texts(
                    texts,
                    embedding_function=embeddings,
                    metadatas=metadatas,
                    persist_directory=str(persist_path),
                )
            try:
                vectordb.persist()
            except Exception:
                pass
            print(f"Built Chroma index at: {persist_path} (via langchain.vectorstores.Chroma)")
            return
        except Exception as e:
            print(f"LangChain Chroma path failed: {e}; falling back to raw chromadb if available.")

    # Fallback: use raw chromadb client; prefer chromadb.embedding_functions if
    # it provides a compatible OpenAIEmbeddingFunction, otherwise compute
    # embeddings via the OpenAI SDK and add them directly.
    if chromadb is not None:
        ids = [f"doc_{i}" for i in range(len(texts))]

        # Try different client constructors to avoid deprecated Settings issues
        client = None
        try:
            # Preferred: allow chromadb to manage persistence via simple arg
            client = chromadb.Client(persist_directory=str(persist_path))
        except Exception:
            try:
                client = chromadb.Client()
            except Exception as e:
                print(f"Failed to construct chromadb client: {e}")
                client = None

        ef_created = False
        ef = None
        if embedding_functions is not None and hasattr(embedding_functions, "OpenAIEmbeddingFunction"):
            ef_cls = embedding_functions.OpenAIEmbeddingFunction
            key = os.environ.get("OPENAI_API_KEY")
            # Try several common parameter names for different chromadb versions
            tried = [
                {"api_key": key, "model": "text-embedding-3-small"},
                {"api_key": key, "model_name": "text-embedding-3-small"},
                {"api_key": key},
            ]
            for kw in tried:
                try:
                    ef = ef_cls(**kw)
                    ef_created = True
                    break
                except TypeError:
                    continue
                except Exception:
                    ef_created = False
                    break

        if ef_created and ef is not None:
            collection = client.create_collection(name="example_collection", embedding_function=ef)
            collection.add(documents=texts, metadatas=metadatas, ids=ids)
            try:
                client.persist()
            except Exception:
                pass
            print(f"Built Chroma index at: {persist_path} (via chromadb embedding_functions)")
            return

        # Fallback to computing embeddings via OpenAI SDK and adding them directly
        try:
            import openai

            openai.api_key = os.environ.get("OPENAI_API_KEY")
            resp = openai.Embeddings.create(model="text-embedding-3-small", input=texts)
            vectors = [d["embedding"] for d in resp["data"]]

            collection = client.create_collection(name="example_collection")
            collection.add(documents=texts, metadatas=metadatas, ids=ids, embeddings=vectors)
            try:
                client.persist()
            except Exception:
                pass

            print(f"Built Chroma index at: {persist_path} (via chromadb + OpenAI embeddings)")
            return
        except Exception as e:
            print(f"Failed to build via chromadb fallback: {e}")
            # fall through to error

    # If we reach here, nothing worked
    print("Failed to build index: no available vectorstore backend found.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="chroma_db")
    args = parser.parse_args()
    build_index(args.persist_dir)


if __name__ == "__main__":
    main()
