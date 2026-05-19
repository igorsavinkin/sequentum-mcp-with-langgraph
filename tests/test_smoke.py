import inspect


def test_basic_imports():
    # ensure core modules import without side effects
    import mcp_client
    import graph
    import main

    assert callable(mcp_client.sequentum_mcp_client)
    assert hasattr(graph, "create_coordinator_graph")
    assert hasattr(main, "main")


def test_create_graph_compiles():
    # creating the coordinator graph should return a compiled app object
    import graph

    class DummyLLM:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            from langchain_core.messages import AIMessage

            return AIMessage(content="ok")

    app = graph.create_coordinator_graph(DummyLLM(), [])
    assert hasattr(app, "invoke") or hasattr(app, "astream")
