import os
import asyncio
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI  # Example using OpenAI

from mcp_client import sequentum_mcp_client
from graph import create_coordinator_graph

async def main():
    # Load environment variables
    load_dotenv()
    
    # Needs OPENAI_API_KEY and SEQUENTUM_API_KEY
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment.")
    
    # 1. Initialize the LLM (e.g. GPT-4o or GPT-4o-mini)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    print("Connecting to Sequentum Cloud MCP...")
    try:
        # 2. Connect to MCP and extract tools
        async with sequentum_mcp_client() as mcp:
            # Reformat MCP tools into LangChain tools using the built-in adapter method
            # get_tools() returns a list of langchain_core.tools.BaseTool objects
            mcp_tools = await mcp.get_tools()
            
            print(f"Successfully loaded {len(mcp_tools)} tools from Sequentum MCP.")
            
            # 3. Compile the LangGraph
            app = create_coordinator_graph(llm, mcp_tools)
            print("\nGraph compiled and ready! Type 'exit' to quit.\n")
            
            # 4. Interactive loop
            while True:
                user_input = input("User: ")
                if user_input.lower() in ["exit", "quit"]:
                    break
                    
                if not user_input.strip():
                    continue
                
                # Starting state configuration
                initial_state = {"messages": [HumanMessage(content=user_input)]}
                
                # Stream the state updates through the graph asynchronously
                # This will print when nodes are executed
                try:
                    async for event in app.astream(initial_state, config={"recursion_limit": 50}):
                        for node_name, node_state in event.items():
                            # Print the latest message produced by the node
                            latest_msg = node_state["messages"][-1]
                            # Tool calls or final responses will show up here
                            msg_content = latest_msg.content if latest_msg.content else ""
                            if getattr(latest_msg, "tool_calls", None):
                                print(f"[{node_name}] Called tools: {[tc['name'] for tc in latest_msg.tool_calls]}")
                            elif msg_content:
                                print(f"[{node_name}] {msg_content}\n")
                except Exception as e:
                    print(f"Error during graph execution: {e}")
                    
    except Exception as e:
        print(f"Failed to start AWIC: {e}")

if __name__ == "__main__":
    asyncio.run(main())
