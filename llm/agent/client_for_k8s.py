from mcp import ClientSession
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage

import asyncio
import sys

model = ChatOpenAI(model="deepseek-chat")

# 你可以在这里自定义多轮提问
QUESTIONS = [
    "我的 K8s 集群有几个节点？",
    "我的集群有哪些 Pods？"
]

# 输出 Agent 和模型交互信息
def print_step_debug(step):
    messages = step.get("messages", [])
    print("\n--- 🧠 Agent 思考过程 ---")
    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"👤 User: {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"🤖 Thought: {msg.content.strip() if msg.content else '(工具调用阶段省略内容)'}")
                for call in msg.tool_calls:
                    print(f"🔧 Tool Call: {call['name']}")
                    print(f"   Input: {call['args']}")
            else:
                print(f"💬 Final Answer: {msg.content.strip()}")
        elif isinstance(msg, ToolMessage):
            print(f"📦 Tool Output (id={msg.tool_call_id}): {msg.content.strip()}")


async def run_agent():
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/sse"
    async with sse_client(url=server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            for question in QUESTIONS:
                print(f"\n====================")
                print(f"📝 Question: {question}")
                result = await agent.ainvoke({"messages": question})
                print_step_debug(result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <SSE_SERVER_URL>")
        sys.exit(1)
    asyncio.run(run_agent())