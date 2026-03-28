#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax LLM Provider - Unit Tests

Tests for MiniMax integration in LLMFactory and LLMConfig.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LLMConfig, check_config
from llm_factory import LLMFactory, get_minimax_llm, get_llm


class TestLLMConfigMiniMax(unittest.TestCase):
    """Tests for MiniMax configuration in LLMConfig."""

    def test_minimax_config_defaults(self):
        """Test that MiniMax config has correct defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig()
            self.assertEqual(cfg.minimax_api_key, "")
            self.assertEqual(cfg.minimax_base_url, "https://api.minimax.io/v1")
            self.assertEqual(cfg.minimax_model, "MiniMax-M2.7")

    def test_minimax_config_from_env(self):
        """Test that MiniMax config reads from environment variables."""
        env = {
            "MINIMAX_API_KEY": "test-key-123",
            "MINIMAX_BASE_URL": "https://custom.minimax.io/v1",
            "MINIMAX_MODEL": "MiniMax-M2.7-highspeed",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            self.assertEqual(cfg.minimax_api_key, "test-key-123")
            self.assertEqual(cfg.minimax_base_url, "https://custom.minimax.io/v1")
            self.assertEqual(cfg.minimax_model, "MiniMax-M2.7-highspeed")

    def test_get_minimax_config(self):
        """Test get_minimax_config returns correct dict structure."""
        env = {"MINIMAX_API_KEY": "key-abc"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            result = cfg.get_minimax_config()
            self.assertIn("api_key", result)
            self.assertIn("base_url", result)
            self.assertIn("model", result)
            self.assertIn("temperature", result)
            self.assertIn("max_tokens", result)
            self.assertIn("timeout", result)
            self.assertEqual(result["api_key"], "key-abc")

    def test_validate_config_minimax_with_key(self):
        """Test validate_config returns True when MINIMAX_API_KEY is set."""
        env = {"MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            self.assertTrue(cfg.validate_config("minimax"))

    def test_validate_config_minimax_without_key(self):
        """Test validate_config returns False when MINIMAX_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig()
            self.assertFalse(cfg.validate_config("minimax"))

    def test_validate_config_any_includes_minimax(self):
        """Test that validate_config(None) considers MiniMax API key."""
        env = {"MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            self.assertTrue(cfg.validate_config(None))

    def test_check_config_minimax_raises_without_key(self):
        """Test check_config raises ValueError for minimax without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload config to pick up cleared env
            import config as cfg_mod
            cfg_mod.config = LLMConfig()
            with self.assertRaises(ValueError) as ctx:
                check_config("minimax")
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))


class TestLLMFactoryMiniMax(unittest.TestCase):
    """Tests for MiniMax LLM creation in LLMFactory."""

    @patch("llm_factory.ChatOpenAI")
    def test_create_minimax_llm_basic(self, mock_chat):
        """Test creating MiniMax LLM with explicit parameters."""
        mock_chat.return_value = MagicMock()
        result = LLMFactory.create_minimax_llm(
            api_key="test-key",
            model="MiniMax-M2.7",
            temperature=0.5,
            max_tokens=1024,
            timeout=30,
        )
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "test-key")
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")
        self.assertEqual(call_kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs["max_tokens"], 1024)

    @patch("llm_factory.ChatOpenAI")
    def test_create_minimax_llm_highspeed_model(self, mock_chat):
        """Test creating MiniMax LLM with highspeed model."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_minimax_llm(
            api_key="test-key",
            model="MiniMax-M2.7-highspeed",
        )
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7-highspeed")

    @patch("llm_factory.ChatOpenAI")
    def test_create_minimax_temperature_clamping_zero(self, mock_chat):
        """Test that temperature=0.0 is clamped to 0.01 for MiniMax."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_minimax_llm(api_key="test-key", temperature=0.0)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.01)

    @patch("llm_factory.ChatOpenAI")
    def test_create_minimax_temperature_clamping_high(self, mock_chat):
        """Test that temperature>1.0 is clamped to 1.0 for MiniMax."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_minimax_llm(api_key="test-key", temperature=1.5)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 1.0)

    @patch("llm_factory.ChatOpenAI")
    def test_create_minimax_temperature_normal(self, mock_chat):
        """Test that valid temperature passes through unchanged."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_minimax_llm(api_key="test-key", temperature=0.7)
        call_kwargs = mock_chat.call_args[1]
        self.assertAlmostEqual(call_kwargs["temperature"], 0.7)

    def test_create_minimax_llm_raises_without_key(self):
        """Test that create_minimax_llm raises ValueError without API key."""
        with patch.dict(os.environ, {}, clear=True):
            import config as cfg_mod
            cfg_mod.config = LLMConfig()
            import llm_factory as fac_mod
            fac_mod.config = cfg_mod.config
            with self.assertRaises(ValueError) as ctx:
                LLMFactory.create_minimax_llm()
            self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch("llm_factory.ChatOpenAI")
    def test_create_llm_dispatcher_minimax(self, mock_chat):
        """Test that create_llm dispatches to MiniMax correctly."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_llm("minimax", api_key="test-key")
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")

    @patch("llm_factory.ChatOpenAI")
    def test_create_llm_dispatcher_minimax_case_insensitive(self, mock_chat):
        """Test that create_llm handles 'MiniMax' case-insensitively."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_llm("MiniMax", api_key="test-key")
        mock_chat.assert_called_once()

    def test_create_llm_unsupported_provider(self):
        """Test that create_llm raises for unknown providers."""
        with self.assertRaises(ValueError) as ctx:
            LLMFactory.create_llm("unknown_provider")
        self.assertIn("minimax", str(ctx.exception))

    @patch("llm_factory.ChatOpenAI")
    def test_get_minimax_llm_convenience(self, mock_chat):
        """Test the get_minimax_llm convenience function."""
        mock_chat.return_value = MagicMock()
        result = get_minimax_llm(api_key="test-key")
        self.assertIsNotNone(result)
        mock_chat.assert_called_once()

    @patch("llm_factory.ChatOpenAI")
    def test_get_llm_with_minimax_provider(self, mock_chat):
        """Test get_llm with provider='minimax'."""
        mock_chat.return_value = MagicMock()
        result = get_llm(provider="minimax", api_key="test-key")
        self.assertIsNotNone(result)

    @patch("llm_factory.ChatOpenAI")
    def test_auto_create_llm_selects_minimax(self, mock_chat):
        """Test auto_create_llm picks MiniMax when only MINIMAX_API_KEY is set."""
        mock_chat.return_value = MagicMock()
        env = {"MINIMAX_API_KEY": "test-key"}
        with patch.dict(os.environ, env, clear=True):
            import config as cfg_mod
            cfg_mod.config = LLMConfig()
            import llm_factory as fac_mod
            fac_mod.config = cfg_mod.config
            result = LLMFactory.auto_create_llm()
            self.assertIsNotNone(result)
            call_kwargs = mock_chat.call_args[1]
            self.assertEqual(call_kwargs["base_url"], "https://api.minimax.io/v1")

    @patch("llm_factory.ChatOpenAI")
    def test_auto_create_prefers_openai_over_minimax(self, mock_chat):
        """Test auto_create_llm prefers OpenAI when both keys are set."""
        mock_chat.return_value = MagicMock()
        env = {
            "OPENAI_API_KEY": "openai-key",
            "MINIMAX_API_KEY": "minimax-key",
        }
        with patch.dict(os.environ, env, clear=True):
            import config as cfg_mod
            cfg_mod.config = LLMConfig()
            import llm_factory as fac_mod
            fac_mod.config = cfg_mod.config
            LLMFactory.auto_create_llm()
            call_kwargs = mock_chat.call_args[1]
            # Should use OpenAI (higher priority)
            self.assertEqual(call_kwargs["api_key"], "openai-key")

    @patch("llm_factory.ChatOpenAI")
    def test_minimax_default_model(self, mock_chat):
        """Test that MiniMax defaults to MiniMax-M2.7 model."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_minimax_llm(api_key="test-key")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7")


class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests for MiniMax provider (require MINIMAX_API_KEY)."""

    @unittest.skipUnless(
        os.getenv("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration tests",
    )
    def test_minimax_llm_invoke(self):
        """Integration test: invoke MiniMax LLM with a simple prompt."""
        from langchain_core.messages import HumanMessage

        llm = LLMFactory.create_minimax_llm(
            api_key=os.getenv("MINIMAX_API_KEY"),
        )
        response = llm.invoke([HumanMessage(content="Say hello in one word.")])
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)

    @unittest.skipUnless(
        os.getenv("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration tests",
    )
    def test_minimax_llm_highspeed(self):
        """Integration test: invoke MiniMax highspeed model."""
        from langchain_core.messages import HumanMessage

        llm = LLMFactory.create_minimax_llm(
            api_key=os.getenv("MINIMAX_API_KEY"),
            model="MiniMax-M2.7-highspeed",
        )
        response = llm.invoke([HumanMessage(content="What is 2+2?")])
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)

    @unittest.skipUnless(
        os.getenv("MINIMAX_API_KEY"),
        "MINIMAX_API_KEY not set, skipping integration tests",
    )
    def test_minimax_via_get_llm(self):
        """Integration test: use get_llm('minimax') convenience function."""
        from langchain_core.messages import HumanMessage

        llm = get_llm(
            provider="minimax",
            api_key=os.getenv("MINIMAX_API_KEY"),
        )
        response = llm.invoke([HumanMessage(content="Reply with OK.")])
        self.assertIsNotNone(response)
        self.assertIn("OK", response.content.upper())


if __name__ == "__main__":
    unittest.main()
