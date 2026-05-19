from typing import Annotated, Sequence, TypedDict
import asyncio
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Define the State of the Graph
# add_messages is a reducer that appends new messages to the list
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_coordinator_graph(llm, tools, retriever=None, retriever_k: int = 3):
    """
    Creates and compiles the AWIC LangGraph.
    
    Args:
        llm: A LangChain Chat Model instance (e.g. ChatOpenAI, ChatAnthropic)
        tools: A list of LangChain tools obtained from the MCP server
    """
    
    # 1. Bind tools to the LLM so it knows what it can call
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Define the Retriever Node (optional) and Agent/Supervisor Node
    async def retriever_node(state: AgentState):
        """
        Optional node that queries a retriever for relevant documents and
        injects them into the message list as a SystemMessage.
        """
        if retriever is None:
            return {"messages": state["messages"]}

        print("--- [Retriever] Fetching context... ---")
        messages = state["messages"]
        # Use the latest user message as the retrieval query
        last = messages[-1]
        query = getattr(last, "content", str(last))

        # Support both sync and async retriever APIs
        docs = []
        if hasattr(retriever, "aget_relevant_documents"):
            docs = await retriever.aget_relevant_documents(query)
        elif hasattr(retriever, "get_relevant_documents"):
            loop = asyncio.get_running_loop()
            docs = await loop.run_in_executor(None, retriever.get_relevant_documents, query)

        # Build a compact context from the returned documents
        context_blocks = []
        for d in (docs or [])[:retriever_k]:
            # Documents are expected to have `page_content` attribute
            txt = getattr(d, "page_content", None) or getattr(d, "content", str(d))
            context_blocks.append(txt)

        if context_blocks:
            context = "\n\n".join(context_blocks)
            sys_msg = SystemMessage(content=f"Retrieved context:\n{context}")
            return {"messages": messages + [sys_msg]}

        return {"messages": messages}
    async def agent_node(state: AgentState):
        """
        The main supervisor node that looks at the conversation history
        and decides whether to answer directly, ask for clarification,
        or invoke an MCP tool.
        """
        print("--- [Agent] Thinking... ---")
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}
        
    # 3. Define the Tool Node
    # ToolNode automatically handles executing the tool calls requested by the LLM
    # and appends ToolMessages back to the state.
    async def tool_node(state: AgentState):
        """
        Executes tools and returns their results.
        """
        print("--- [Tool] Executing requested actions... ---")
        # We use the prebuilt ToolNode
        node_executor = ToolNode(tools=tools)
        return await node_executor.ainvoke(state)
        
    # 4. Conditional Edge Logic
    def should_continue(state: AgentState):
        """
        Checks if the LLM has requested a tool call.
        If yes, route to tools. If no, we are done (END).
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        # If the LLM made a tool call, route to tools
        if last_message.tool_calls:
            return "tools"
        # Otherwise, the LLM replied to the user
        return END

    # 5. Build the Graph
    workflow = StateGraph(AgentState)

    # Add nodes
    if retriever is not None:
        workflow.add_node("retriever", retriever_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    # Add edges
    if retriever is not None:
        workflow.add_edge(START, "retriever")
        workflow.add_edge("retriever", "agent")
    else:
        workflow.add_edge(START, "agent")

    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app
