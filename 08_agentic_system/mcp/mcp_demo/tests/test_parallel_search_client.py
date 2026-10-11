"""通过模拟 HTTP 服务验证 MCP 协议、匿名请求和错误传播，不访问外网。"""

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel_search_client as demo


class ClientTests(unittest.IsolatedAsyncioTestCase):
    async def call(self, name, arguments, *, error=False, missing=False):
        requests = []
        messages = []

        def respond(request):
            requests.append(request)
            self.assertEqual(str(request.url), demo.SERVER_URL)
            self.assertEqual(request.headers["User-Agent"], demo.USER_AGENT)
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("Cookie", request.headers)
            message = json.loads(request.content)
            messages.append(message)
            if "id" not in message:
                return httpx.Response(202)
            if message["method"] == "initialize":
                result = {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "测试服务", "version": "1"},
                }
            elif message["method"] == "tools/list":
                cursor = message.get("params", {}).get("cursor")
                tool_name = "web_fetch" if cursor else "web_search"
                result = {"tools": [] if missing else [{
                    "name": tool_name,
                    "inputSchema": {"type": "object", "properties": {}},
                }]}
                if not cursor:
                    result["nextCursor"] = "第二页"
            else:
                self.assertEqual(message["method"], "tools/call")
                self.assertEqual(message["params"]["name"], name)
                self.assertEqual(message["params"]["arguments"], arguments)
                result = {
                    "content": [{"type": "text", "text": "来源 https://example.org 摘录"}],
                    "isError": error,
                }
            return httpx.Response(200, json={
                "jsonrpc": "2.0", "id": message["id"], "result": result,
            })

        original_client = httpx.AsyncClient
        clients = []

        def make_client(**kwargs):
            client = original_client(transport=httpx.MockTransport(respond), **kwargs)
            clients.append(client)
            return client

        try:
            with patch.object(demo.httpx, "AsyncClient", side_effect=make_client):
                result = await demo.run_tool(name, arguments)
            return result, messages
        finally:
            self.assertTrue(requests)
            self.assertTrue(all(client.is_closed for client in clients))

    async def test_search_uses_discovery_and_anonymous_http(self):
        arguments = {"objective": "CUDA 同步语义", "search_queries": ["CUDA stream sync"],
                     "session_id": "研究会话"}
        result, messages = await self.call("web_search", arguments)
        self.assertIn("https://example.org", result.content[0].text)
        self.assertEqual([m["method"] for m in messages], [
            "initialize", "notifications/initialized", "tools/list", "tools/list", "tools/call",
        ])

    async def test_fetch_uses_second_discovery_page(self):
        result, _ = await self.call("web_fetch", {
            "urls": ["https://example.org"], "session_id": "研究会话",
        })
        self.assertFalse(result.isError)

    async def test_tools_discovery_does_not_dispatch(self):
        tools, messages = await self.call(None, {})
        self.assertEqual([tool.name for tool in tools], ["web_search", "web_fetch"])
        self.assertNotIn("tools/call", [m["method"] for m in messages])

    async def test_tool_failure_is_not_success(self):
        with self.assertRaisesRegex(ExceptionGroup, "TaskGroup") as raised:
            await self.call("web_search", {}, error=True)
        # SDK 上下文会将工具错误包装为异常组，叶子错误仍须保留诊断。
        def leaves(error):
            if isinstance(error, BaseExceptionGroup):
                return [leaf for child in error.exceptions for leaf in leaves(child)]
            return [error]
        self.assertTrue(any("工具调用失败" in str(e) for e in leaves(raised.exception)))

    async def test_missing_tool_is_not_dispatched(self):
        with self.assertRaises(ExceptionGroup):
            await self.call("web_search", {}, missing=True)


class ArgumentTests(unittest.TestCase):
    def test_repeat_queries_and_shared_session(self):
        args = demo.parse_args(["--session-id", "研究会话", "search", "--objective", "同步语义",
                                "--query", "CUDA streams", "--query", "CUDA events"])
        self.assertEqual(args.query, ["CUDA streams", "CUDA events"])
        self.assertEqual(args.session_id, "研究会话")

    def test_fetch_limits(self):
        for arguments in (["fetch", *["https://example.org"] * 21],
                          ["fetch", "file:///tmp/example"],
                          ["fetch", "https://example.org", "--objective", "字" * 201],
                          ["--session-id", "", "tools"]):
            with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as raised:
                demo.parse_args(arguments)
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
