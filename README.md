# AWIC — Automated Web Intelligence Coordinator

AWIC is a local agent scaffold that combines LangGraph (stateful workflow orchestration)
with Sequentum Cloud MCP (production scraper tooling). The project demonstrates how to
discover MCP tools, bind them to an LLM, and execute them inside a LangGraph workflow.

Features
- LangGraph-based supervisor + tool workflow
- Sequentum MCP integration via a small adapter (`mcp_client.py`)
- Async-first tool invocation to support async-only StructuredTool implementations

Quickstart

1. Create and activate a virtual environment (Windows):

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` → `.env` and set `OPENAI_API_KEY` and `SEQUENTUM_API_KEY`.

4. Run the console agent:

```powershell
python main.py
```

Tests

- A minimal pytest smoke test is included at `tests/test_smoke.py`.
- Run locally:

```powershell
python -m pip install pytest
pytest -q
```

CI

- A GitHub Actions workflow is added at `.github/workflows/ci.yml` that installs
  dependencies and runs the smoke tests on push and pull requests.

Notes

- The graph uses async APIs (`ainvoke`, `astream`) to avoid sync/async invocation
  mismatches with MCP-backed StructuredTool implementations.
- `mcp_client.py` launches a local `npx -y sequentum-mcp` process by default (stdio
  transport). For production, point `MultiServerMCPClient` to your managed MCP endpoint.

License

This repository is provided as-is for development and experimentation.
