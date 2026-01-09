from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
import sys

model = ChatOpenAI(model="deepseek-chat", verbose=True)

async def run_agent():
    # 从命令行参数获取SSE服务器URL
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/sse"
    
    # 使用SSE客户端连接远程服务器
    async with sse_client(url=server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools, debug=True) 
            result = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
            return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <SSE_SERVER_URL>")
        sys.exit(1)
    print(asyncio.run(run_agent()))