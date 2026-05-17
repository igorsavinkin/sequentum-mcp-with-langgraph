import os
from contextlib import asynccontextmanager
from langchain_mcp_adapters.client import MultiServerMCPClient


class _SingleServerAdapter:
    """Small adapter to expose a single-server `get_tools()` API.

    The project previously used a single-server `MCPClient` context manager.
    Newer langchain-mcp-adapters expose `MultiServerMCPClient`, so this
    adapter keeps the original `get_tools()` callsite shape.
    """

    def __init__(self, client, server_name: str):
        self._client = client
        self._server_name = server_name

    async def get_tools(self):
        return await self._client.get_tools(server_name=self._server_name)


@asynccontextmanager
async def sequentum_mcp_client():
    """
    Context manager to connect to the Sequentum Cloud MCP server.
    Uses `MultiServerMCPClient` under the hood and yields an adapter that
    exposes `get_tools()` for compatibility with the rest of the codebase.
    """
    api_key = os.environ.get("SEQUENTUM_API_KEY")
    if not api_key:
        raise ValueError("SEQUENTUM_API_KEY environment variable is not set")

    # We use stdio transport calling npx to run sequentum-mcp
    server_params = {
        "command": "npx",
        "args": ["-y", "sequentum-mcp"],
        "env": {
            "SEQUENTUM_API_KEY": api_key,
            # Inherit path so npx can be found
            "PATH": os.environ.get("PATH", "")
        },
        "transport": "stdio",
    }

    # Use the MultiServerMCPClient and present a single-server façade.
    client = MultiServerMCPClient({"sequentum": server_params})
    try:
        yield _SingleServerAdapter(client, "sequentum")
    finally:
        # MultiServerMCPClient currently doesn't require explicit shutdown;
        # sessions are created per-call by the adapter methods.
        pass
