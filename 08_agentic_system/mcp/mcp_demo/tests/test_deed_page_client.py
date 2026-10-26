"""通过模拟 HTTP 服务验证 deed.page 示例的 MCP 调用、匿名请求和参数检查，不访问外网。"""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deed_page_client as demo


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
                tool_name = "delete_site" if cursor else "publish_site"
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
                text = json.dumps({"url": "https://demo.deed.page/", "slug": "demo"})
                result = {"content": [{"type": "text", "text": text}], "isError": error}
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

    async def test_publish_uses_discovery_and_anonymous_http(self):
        arguments = {"files": {"index.html": "<h1>报告</h1>"}, "ttl_seconds": 3600}
        result, messages = await self.call("publish_site", arguments)
        self.assertIn("https://demo.deed.page/", result.content[0].text)
        self.assertEqual([m["method"] for m in messages], [
            "initialize", "notifications/initialized", "tools/list", "tools/list", "tools/call",
        ])

    async def test_delete_uses_second_discovery_page(self):
        result, _ = await self.call("delete_site", {"slug": "demo", "token": "deed_tok_test"})
        self.assertFalse(result.isError)

    async def test_tools_discovery_does_not_dispatch(self):
        tools, messages = await self.call(None, {})
        self.assertEqual([tool.name for tool in tools], ["publish_site", "delete_site"])
        self.assertNotIn("tools/call", [m["method"] for m in messages])

    async def test_tool_failure_is_not_success(self):
        with self.assertRaises(ExceptionGroup) as raised:
            await self.call("publish_site", {}, error=True)

        def leaves(error):
            if isinstance(error, BaseExceptionGroup):
                return [leaf for child in error.exceptions for leaf in leaves(child)]
            return [error]
        self.assertTrue(any("工具调用失败" in str(e) for e in leaves(raised.exception)))

    async def test_missing_tool_is_not_dispatched(self):
        with self.assertRaises(ExceptionGroup):
            await self.call("publish_site", {}, missing=True)


class ArgumentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.page = Path(self.tmp.name) / "report.html"
        self.page.write_text("<h1>GPU 笔记</h1>", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_publish_arguments(self):
        args = demo.parse_args(["publish", str(self.page), "--slug", "gpu-notes", "--ttl", "600"])
        self.assertEqual(demo.build_call(args), ("publish_site", {
            "files": {"index.html": "<h1>GPU 笔记</h1>"}, "slug": "gpu-notes", "ttl_seconds": 600,
        }))

    def test_ttl_zero_omits_expiry(self):
        args = demo.parse_args(["publish", str(self.page), "--ttl", "0"])
        self.assertNotIn("ttl_seconds", demo.build_call(args)[1])

    def test_delete_reads_token_from_environment(self):
        with patch.dict(os.environ, {demo.TOKEN_ENV: "deed_tok_test"}):
            args = demo.parse_args(["delete", "gpu-notes"])
            self.assertEqual(demo.build_call(args), (
                "delete_site", {"slug": "gpu-notes", "token": "deed_tok_test"},
            ))

    def test_invalid_arguments(self):
        with patch.dict(os.environ, {}, clear=True):
            for arguments in (["publish", str(self.page), "--ttl", "-1"],
                              ["publish", str(Path(self.tmp.name) / "missing.html")],
                              ["delete", "gpu-notes"]):
                with self.subTest(arguments=arguments), self.assertRaises(SystemExit) as raised:
                    demo.parse_args(arguments)
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
