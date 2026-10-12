"""通过 Streamable HTTP 匿名调用 Parallel Search MCP，不加载模型或本地密钥。"""

import argparse
import asyncio
from datetime import timedelta
import sys
import uuid

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


SERVER_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "AI-fundamentals/parallel-search-demo"


async def run_tool(name, arguments):
    """先发现工具，再调用；超时或取消时由上下文管理器关闭连接。"""
    async with asyncio.timeout(90):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True
        ) as client:
            async with streamable_http_client(
                SERVER_URL, http_client=client
            ) as (read, write, _):
                async with ClientSession(
                    read, write, read_timeout_seconds=timedelta(seconds=60)
                ) as session:
                    await session.initialize()
                    tools = []
                    cursor = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        tools.extend(page.tools)
                        cursor = page.nextCursor
                        if not cursor:
                            break
                    if name is None:
                        return tools
                    if name not in {tool.name for tool in tools}:
                        raise RuntimeError(f"服务端未提供工具：{name}")
                    result = await session.call_tool(name, arguments)
                    if result.isError:
                        message = "\n".join(
                            item.text for item in result.content if item.type == "text"
                        )
                        raise RuntimeError(f"工具调用失败：{message}")
                    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-id", default=None, help="同一研究会话的稳定 ID；默认生成 UUID"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tools", help="查看服务端工具及当前输入 schema")
    search = commands.add_parser("search", help="搜索网页，输出来源和相关摘录")
    search.add_argument("--objective", required=True, help="要查找的信息")
    search.add_argument("--query", action="append", required=True, help="关键词，可重复")
    fetch = commands.add_parser("fetch", help="提取指定网页的内容")
    fetch.add_argument("urls", nargs="+", help="HTTP 或 HTTPS URL，最多 20 个")
    fetch.add_argument("--objective", help="关注的信息，最多 200 个字符")
    args = parser.parse_args(argv)
    if args.session_id is not None and not 1 <= len(args.session_id) <= 100:
        parser.error("session-id 长度须为 1 至 100 个字符")
    if args.command == "fetch":
        if len(args.urls) > 20:
            parser.error("每次 fetch 最多 20 个 URL")
        if any(httpx.URL(url).scheme not in ("http", "https") for url in args.urls):
            parser.error("fetch 仅接受 HTTP 或 HTTPS URL")
        if args.objective is not None and len(args.objective) > 200:
            parser.error("fetch objective 最多 200 个字符")
    return args


def main(argv=None):
    args = parse_args(argv)
    session_id = args.session_id or str(uuid.uuid4())
    print(f"会话 ID：{session_id}", file=sys.stderr)
    if args.command == "tools":
        name, arguments = None, {}
    elif args.command == "search":
        name = "web_search"
        arguments = {
            "objective": args.objective,
            "search_queries": args.query,
            "session_id": session_id,
        }
    else:
        name = "web_fetch"
        arguments = {"urls": args.urls, "session_id": session_id}
        if args.objective is not None:
            arguments["objective"] = args.objective
    result = asyncio.run(run_tool(name, arguments))
    if name is None:
        for tool in result:
            print(tool.model_dump_json(indent=2, exclude_none=True))
    else:
        print(result.model_dump_json(indent=2, exclude_none=True))


if __name__ == "__main__":
    main()
