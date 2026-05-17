# AWIC (Automated Web Intelligence Coordinator)

Great, we've written the basic project code! Now we have a fully working skeleton of a LangGraph-based local agent with a connected Sequentum Cloud MCP.

## What has been implemented

All files are saved in the directory `c:/Users/igors/RnD/MachineLearning/LangChain/sequentum-mcp /`:

1. **`requirements.txt`** — Project dependencies, including the `langgraph` and `langchain-mcp-adapters` libraries — an official adapter that automatically turns MCP Tools into native LangChain tools. 
2. **`mcp_client.py `** — Utility `sequentum_mcp_client'. It uses 'MCPClient`, which runs `npx -y sequentum-mcp' locally ("stdio" connection) and forwards your API key there.
3. **`graph.py `** — Logic core:
- **`AgentState`**: Contains a list of messages (dialog history).
   - **Node `agent_node'**: Invokes the LLM itself, trained to invoke tools (via `bind_tools()`).
- **Node `tool_node`**: Executes the tools requested from the LLM, communicating with the running Sequentum MCP server.
4. **`main.py `** is the entry point. It reads the environment variables, initializes the LLM (configured for OpenAI `gpt-4o` by default, but can be changed), tightens the tools, compiles the graph and launches the console chat.
5. **`.env.example`** is a template for your keys. 

## How to test

To run, you need to:

1. Open the terminal in the folder `c:/Users/igors/RnD/MachineLearning/LangChain/sequentum-mcp `
2. Create a virtual environment: `python -m venv venv`
3. Activate it (for Windows): `venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt `
5. Create a `.env` file based on `.env.example' and add two keys there:
   - `OPENAI_API_KEY' (if you leave ChatGPT as the basis)
   - `SEQUENTUM_API_KEY' (issued to you in Sequentum Cloud)
6. Run the script: `python main.py `

## How it works in practice

When the application starts, the agent will first read the server schema and all 30+ tools (tool definition).
Then you can write in the console, for example:

> **User:** "What parsers do I have?"
>
> *LLM understands that it needs a list, returns the command to call the `list_agents` function.*
> *LangGraph will go to `ToolNode', call `list_agents` via MCP.*
> *The received JSON will be returned to LLM.*
> *LLM will generate a response:* "You have 2 agents: ScraperX and AmazonTracker."

> **User:** "Where can I find billing information?"
>
> *LLM will call `get_credits_balance'.*

> **User:** "Schedule AmazonTracker for every Saturday"
>
> *LLM will find the schedule through `create_agent_schedule'.*

## What's next?

Now the agent has direct "super-rights" — he can call any toolbox functions if we ask him.
In a real project ("production") in `graph.py ` add **conditions and checks**:
For example, if LLM decided to call `start_agent` (to start the parser), the graph should pause and redirect the request to the node `human_approval` so that you confirm the start (since it costs Sequentum credits)