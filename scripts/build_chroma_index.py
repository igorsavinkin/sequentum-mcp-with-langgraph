#!/usr/bin/env python3
"""
Small helper to build a Chroma index using OpenAI embeddings.

Usage:
  python scripts/build_chroma_index.py --persist_dir chroma_db
"""
import argparse
import sys
from pathlib import Path

try:
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
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

    embeddings = OpenAIEmbeddings()

    # Create and persist a Chroma collection
    vectordb = Chroma.from_texts(
        texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(persist_path),
    )

    try:
        vectordb.persist()
    except Exception:
        # Some Chroma wrappers auto-persist; ignore if not supported
        pass

    print(f"Built Chroma index at: {persist_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persist_dir", default="chroma_db")
    args = parser.parse_args()
    build_index(args.persist_dir)


if __name__ == "__main__":
    main()
