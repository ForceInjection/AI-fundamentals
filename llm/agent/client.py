from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio

model = ChatOpenAI(model="deepseek-chat")

server_params = StdioServerParameters(
    command="python",
    args=["math_server.py"],  # 替换为绝对路径
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)
            result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            return result

if __name__ == "__main__":
    print(asyncio.run(run_agent()))