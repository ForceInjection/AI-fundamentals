#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OrcaRouter LLM Provider - Unit Tests

Tests for OrcaRouter integration in LLMFactory and LLMConfig.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import LLMConfig, check_config
from llm_factory import LLMFactory, get_orcarouter_llm, get_llm


class TestLLMConfigOrcaRouter(unittest.TestCase):
    """Tests for OrcaRouter configuration in LLMConfig."""

    def test_orcarouter_config_defaults(self):
        """Test that OrcaRouter config has correct defaults when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig()
            self.assertEqual(cfg.orcarouter_api_key, "")
            self.assertEqual(cfg.orcarouter_base_url, "https://api.orcarouter.ai/v1")
            self.assertEqual(cfg.orcarouter_model, "orcarouter/auto")

    def test_orcarouter_config_from_env(self):
        """Test that OrcaRouter config reads from environment variables."""
        env = {
            "ORCAROUTER_API_KEY": "sk-orca-test-123",
            "ORCAROUTER_BASE_URL": "https://custom.orcarouter.ai/v1",
            "ORCAROUTER_MODEL": "orcarouter/fusion",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            self.assertEqual(cfg.orcarouter_api_key, "sk-orca-test-123")
            self.assertEqual(cfg.orcarouter_base_url, "https://custom.orcarouter.ai/v1")
            self.assertEqual(cfg.orcarouter_model, "orcarouter/fusion")

    def test_get_orcarouter_config(self):
        """Test get_orcarouter_config returns correct dict structure."""
        env = {"ORCAROUTER_API_KEY": "sk-orca-abc"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            result = cfg.get_orcarouter_config()
            self.assertIn("api_key", result)
            self.assertIn("base_url", result)
            self.assertIn("model", result)
            self.assertIn("temperature", result)
            self.assertIn("max_tokens", result)
            self.assertIn("timeout", result)
            self.assertEqual(result["api_key"], "sk-orca-abc")

    def test_validate_config_orcarouter_with_key(self):
        """Test validate_config returns True when ORCAROUTER_API_KEY is set."""
        env = {"ORCAROUTER_API_KEY": "sk-orca-test"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            self.assertTrue(cfg.validate_config("orcarouter"))

    def test_validate_config_orcarouter_without_key(self):
        """Test validate_config returns False when ORCAROUTER_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            cfg = LLMConfig()
            self.assertFalse(cfg.validate_config("orcarouter"))

    def test_validate_config_any_includes_orcarouter(self):
        """Test that validate_config(None) considers OrcaRouter API key."""
        env = {"ORCAROUTER_API_KEY": "sk-orca-test"}
        with patch.dict(os.environ, env, clear=True):
            cfg = LLMConfig()
            self.assertTrue(cfg.validate_config(None))

    def test_check_config_orcarouter_raises_without_key(self):
        """Test check_config raises ValueError for orcarouter without API key."""
        with patch.dict(os.environ, {}, clear=True):
            # Reload config to pick up cleared env
            import config as cfg_mod
            cfg_mod.config = LLMConfig()
            with self.assertRaises(ValueError) as ctx:
                check_config("orcarouter")
            self.assertIn("ORCAROUTER_API_KEY", str(ctx.exception))


class TestLLMFactoryOrcaRouter(unittest.TestCase):
    """Tests for OrcaRouter LLM creation in LLMFactory."""

    @patch("llm_factory.ChatOpenAI")
    def test_create_orcarouter_llm_basic(self, mock_chat):
        """Test creating OrcaRouter LLM with explicit parameters."""
        mock_chat.return_value = MagicMock()
        result = LLMFactory.create_orcarouter_llm(
            api_key="sk-orca-test",
            model="orcarouter/fusion",
            temperature=0.5,
            max_tokens=1024,
            timeout=30,
        )
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["api_key"], "sk-orca-test")
        self.assertEqual(call_kwargs["base_url"], "https://api.orcarouter.ai/v1")
        self.assertEqual(call_kwargs["model"], "orcarouter/fusion")
        self.assertEqual(call_kwargs["temperature"], 0.5)
        self.assertEqual(call_kwargs["max_tokens"], 1024)

    @patch("llm_factory.ChatOpenAI")
    def test_create_orcarouter_llm_fusion_model(self, mock_chat):
        """Test creating OrcaRouter LLM with fusion model."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_orcarouter_llm(
            api_key="sk-orca-test",
            model="orcarouter/fusion-flash",
        )
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "orcarouter/fusion-flash")

    def test_create_orcarouter_llm_raises_without_key(self):
        """Test that create_orcarouter_llm raises ValueError without API key."""
        with patch.dict(os.environ, {}, clear=True):
            import config as cfg_mod
            cfg_mod.config = LLMConfig()
            import llm_factory as fac_mod
            fac_mod.config = cfg_mod.config
            with self.assertRaises(ValueError) as ctx:
                LLMFactory.create_orcarouter_llm()
            self.assertIn("ORCAROUTER_API_KEY", str(ctx.exception))

    @patch("llm_factory.ChatOpenAI")
    def test_create_llm_dispatcher_orcarouter(self, mock_chat):
        """Test that create_llm dispatches to OrcaRouter correctly."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_llm("orcarouter", api_key="sk-orca-test")
        mock_chat.assert_called_once()
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.orcarouter.ai/v1")

    @patch("llm_factory.ChatOpenAI")
    def test_create_llm_dispatcher_orcarouter_case_insensitive(self, mock_chat):
        """Test that create_llm handles 'OrcaRouter' case-insensitively."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_llm("OrcaRouter", api_key="sk-orca-test")
        mock_chat.assert_called_once()

    def test_create_llm_unsupported_provider(self):
        """Test that create_llm raises for unknown providers."""
        with self.assertRaises(ValueError) as ctx:
            LLMFactory.create_llm("unknown_provider")
        self.assertIn("orcarouter", str(ctx.exception))

    @patch("llm_factory.ChatOpenAI")
    def test_get_orcarouter_llm_convenience(self, mock_chat):
        """Test the get_orcarouter_llm convenience function."""
        mock_chat.return_value = MagicMock()
        result = get_orcarouter_llm(api_key="sk-orca-test")
        self.assertIsNotNone(result)
        mock_chat.assert_called_once()

    @patch("llm_factory.ChatOpenAI")
    def test_get_llm_with_orcarouter_provider(self, mock_chat):
        """Test get_llm with provider='orcarouter'."""
        mock_chat.return_value = MagicMock()
        result = get_llm(provider="orcarouter", api_key="sk-orca-test")
        self.assertIsNotNone(result)

    @patch("llm_factory.ChatOpenAI")
    def test_auto_create_llm_selects_orcarouter(self, mock_chat):
        """Test auto_create_llm picks OrcaRouter when only ORCAROUTER_API_KEY is set."""
        mock_chat.return_value = MagicMock()
        env = {"ORCAROUTER_API_KEY": "sk-orca-test"}
        with patch.dict(os.environ, env, clear=True):
            import config as cfg_mod
            cfg_mod.config = LLMConfig()
            import llm_factory as fac_mod
            fac_mod.config = cfg_mod.config
            result = LLMFactory.auto_create_llm()
            self.assertIsNotNone(result)
            call_kwargs = mock_chat.call_args[1]
            self.assertEqual(call_kwargs["base_url"], "https://api.orcarouter.ai/v1")

    @patch("llm_factory.ChatOpenAI")
    def test_auto_create_prefers_openai_over_orcarouter(self, mock_chat):
        """Test auto_create_llm prefers OpenAI when both keys are set."""
        mock_chat.return_value = MagicMock()
        env = {
            "OPENAI_API_KEY": "openai-key",
            "ORCAROUTER_API_KEY": "sk-orca-test",
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
    def test_orcarouter_default_model(self, mock_chat):
        """Test that OrcaRouter defaults to orcarouter/auto model."""
        mock_chat.return_value = MagicMock()
        LLMFactory.create_orcarouter_llm(api_key="sk-orca-test")
        call_kwargs = mock_chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "orcarouter/auto")


class TestOrcaRouterIntegration(unittest.TestCase):
    """Integration tests for OrcaRouter provider (require ORCAROUTER_API_KEY)."""

    @unittest.skipUnless(
        os.getenv("ORCAROUTER_API_KEY"),
        "ORCAROUTER_API_KEY not set, skipping integration tests",
    )
    def test_orcarouter_llm_invoke(self):
        """Integration test: invoke OrcaRouter LLM with a simple prompt."""
        from langchain_core.messages import HumanMessage

        llm = LLMFactory.create_orcarouter_llm(
            api_key=os.getenv("ORCAROUTER_API_KEY"),
        )
        response = llm.invoke([HumanMessage(content="Say hello in one word.")])
        self.assertIsNotNone(response)
        self.assertTrue(len(response.content) > 0)

    @unittest.skipUnless(
        os.getenv("ORCAROUTER_API_KEY"),
        "ORCAROUTER_API_KEY not set, skipping integration tests",
    )
    def test_orcarouter_via_get_llm(self):
        """Integration test: use get_llm('orcarouter') convenience function."""
        from langchain_core.messages import HumanMessage

        llm = get_llm(
            provider="orcarouter",
            api_key=os.getenv("ORCAROUTER_API_KEY"),
        )
        response = llm.invoke([HumanMessage(content="Reply with OK.")])
        self.assertIsNotNone(response)
        self.assertIn("OK", response.content.upper())


if __name__ == "__main__":
    unittest.main()
