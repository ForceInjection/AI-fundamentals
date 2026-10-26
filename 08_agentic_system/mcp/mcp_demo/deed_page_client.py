"""通过 Streamable HTTP 匿名调用 deed.page MCP，把本地 HTML 发布成公开网页，不加载模型。"""

import argparse
import asyncio
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


SERVER_URL = "https://deed.page/mcp"
USER_AGENT = "AI-fundamentals/deed-page-demo"
TOKEN_ENV = "DEED_PAGE_TOKEN"
MAX_HTML_BYTES = 1_000_000


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
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tools", help="查看服务端工具名称")
    publish = commands.add_parser("publish", help="把一个 HTML 文件发布为 index.html")
    publish.add_argument("file", type=Path, help="本地 HTML 文件")
    publish.add_argument("--slug", help="可选的子域名，例如 mcp-demo-report")
    publish.add_argument(
        "--ttl", type=int, default=3600, help="多少秒后自动过期，默认 3600；0 表示不过期"
    )
    delete = commands.add_parser("delete", help=f"用 {TOKEN_ENV} 中的令牌删除页面")
    delete.add_argument("slug", help="要删除的页面 slug")
    args = parser.parse_args(argv)
    if args.command == "publish":
        if args.ttl < 0:
            parser.error("ttl 不能为负数")
        if not args.file.is_file():
            parser.error(f"找不到文件：{args.file}")
        if args.file.stat().st_size > MAX_HTML_BYTES:
            parser.error("示例只发布 1 MB 以内的单个 HTML 文件")
    if args.command == "delete" and not os.environ.get(TOKEN_ENV):
        parser.error(f"删除页面需要先设置环境变量 {TOKEN_ENV}")
    return args


def build_call(args):
    if args.command == "tools":
        return None, {}
    if args.command == "publish":
        arguments = {"files": {"index.html": args.file.read_text(encoding="utf-8")}}
        if args.slug:
            arguments["slug"] = args.slug
        if args.ttl:
            arguments["ttl_seconds"] = args.ttl
        return "publish_site", arguments
    return "delete_site", {"slug": args.slug, "token": os.environ[TOKEN_ENV]}


def main(argv=None):
    args = parse_args(argv)
    name, arguments = build_call(args)
    result = asyncio.run(run_tool(name, arguments))
    if name is None:
        for tool in result:
            print(tool.name)
        return
    text = "\n".join(item.text for item in result.content if item.type == "text")
    print(text)
    if name == "publish_site":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return
        if data.get("token"):
            print(
                f"首次发布返回了页面令牌；更新或删除这个页面时需要它，请设置为 {TOKEN_ENV} 并妥善保存。",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
