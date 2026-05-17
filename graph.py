from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# Define the State of the Graph
# add_messages is a reducer that appends new messages to the list
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def create_coordinator_graph(llm, tools):
    """
    Creates and compiles the AWIC LangGraph.
    
    Args:
        llm: A LangChain Chat Model instance (e.g. ChatOpenAI, ChatAnthropic)
        tools: A list of LangChain tools obtained from the MCP server
    """
    
    # 1. Bind tools to the LLM so it knows what it can call
    llm_with_tools = llm.bind_tools(tools)
    
    # 2. Define the Agent/Supervisor Node
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
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    app = workflow.compile()
    
    return app
