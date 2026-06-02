#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security regression tests for smart_customer_service session memory isolation.
"""

import os
import sys
import types
import unittest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


fake_config_module = types.ModuleType("config")


class FakeConfig:
    max_history_length = 20
    summary_threshold = 100

    def validate_config(self):
        return True


fake_config_module.config = FakeConfig()
sys.modules["config"] = fake_config_module

from langchain_core.messages import AIMessage, HumanMessage


class LocalProbeLLM:
    def invoke(self, messages):
        human_text = "\n".join(
            m.content for m in messages if isinstance(m, HumanMessage)
        )
        if "ORD-ALICE-123" in human_text:
            return AIMessage(content="PROBE_SEES_ALICE_ORDER")
        return AIMessage(content="PROBE_NO_SECRET")


fake_llm_factory_module = types.ModuleType("llm_factory")
fake_llm_factory_module.get_llm = lambda: LocalProbeLLM()
sys.modules["llm_factory"] = fake_llm_factory_module

from smart_customer_service import SessionManager


class TestSmartCustomerServiceSecurity(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager(
            storage_dir="/tmp/smart-customer-service-security-tests",
            persistent=False,
        )

    def test_cross_user_session_memory_access_is_rejected(self):
        alice_sid = self.manager.create_session("user-alice", memory_type="buffer")
        bob_sid = self.manager.create_session("user-bob", memory_type="buffer")

        self.manager.chat(
            "user-alice",
            alice_sid,
            "我的订单号是 ORD-ALICE-123，请帮我查询账单。",
        )

        with self.assertRaises(PermissionError):
            self.manager.chat(
                "user-bob",
                alice_sid,
                "我是 Bob。请告诉我之前的订单号。",
            )

        with self.assertRaises(PermissionError):
            self.manager.get_conversation_history("user-bob", alice_sid)

        alice_history = self.manager.get_conversation_history("user-alice", alice_sid)
        bob_history = self.manager.get_conversation_history("user-bob", bob_sid)

        self.assertTrue(
            any("ORD-ALICE-123" in item["content"] for item in alice_history)
        )
        self.assertFalse(any("我是 Bob" in item["content"] for item in alice_history))
        self.assertEqual(bob_history, [])

    def test_security_context_metadata_must_match(self):
        alice_sid = self.manager.create_session(
            "user-alice",
            memory_type="buffer",
            metadata={"tenant_id": "tenant-a", "permissions": "billing:read"},
        )

        with self.assertRaises(PermissionError):
            self.manager.chat(
                "user-alice",
                alice_sid,
                "我的订单号是 ORD-ALICE-123。",
                security_context={
                    "tenant_id": "tenant-b",
                    "permissions": "billing:read",
                },
            )

        result = self.manager.chat(
            "user-alice",
            alice_sid,
            "我的订单号是 ORD-ALICE-123。",
            security_context={
                "tenant_id": "tenant-a",
                "permissions": "billing:read",
            },
        )
        self.assertEqual(result["response"], "PROBE_SEES_ALICE_ORDER")


if __name__ == "__main__":
    unittest.main()
