from __future__ import annotations

import unittest
from unittest.mock import patch

from app.llm.client import LLMClientError
from app.llm.client import LLMConfig
from app.llm.client import generate_grounded_answer
from app.llm.client import generate_grounded_answer_result
from app.llm.client import load_llm_config


class LoadLlmConfigTests(unittest.TestCase):
    def test_load_llm_config_reads_environment_values(self) -> None:
        config = load_llm_config(
            {
                "OPENAI_API_KEY": "test-key",
                "OPENAI_BASE_URL": "https://example.test/v1/chat/completions",
                "OPENAI_MODEL": "test-model",
                "OPENAI_TIMEOUT_SECONDS": "12.5",
            }
        )

        self.assertEqual(config.api_key, "test-key")
        self.assertEqual(config.base_url, "https://example.test/v1/chat/completions")
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.timeout_seconds, 12.5)

    def test_load_llm_config_rejects_invalid_timeout_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPENAI_TIMEOUT_SECONDS must be a positive number"):
            load_llm_config({"OPENAI_TIMEOUT_SECONDS": "not-a-number"})

    def test_load_llm_config_rejects_non_positive_timeout_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "OPENAI_TIMEOUT_SECONDS must be a positive number"):
            load_llm_config({"OPENAI_TIMEOUT_SECONDS": "0"})


class GenerateGroundedAnswerTests(unittest.TestCase):
    def test_generate_grounded_answer_returns_fallback_without_api_key(self) -> None:
        summary = generate_grounded_answer(
            question="What signals matter?",
            supporting_signals=["Latest simple return: 1.00%."],
            retrieved_context_texts=["ETF momentum improved."],
            fallback_summary="Deterministic fallback summary.",
            config=LLMConfig(api_key=None),
        )

        self.assertEqual(summary, "Deterministic fallback summary.")

    def test_generate_grounded_answer_result_marks_fallback_source_without_api_key(self) -> None:
        result = generate_grounded_answer_result(
            question="What signals matter?",
            supporting_signals=["Latest simple return: 1.00%."],
            retrieved_context_texts=["ETF momentum improved."],
            fallback_summary="Deterministic fallback summary.",
            config=LLMConfig(api_key=None),
        )

        self.assertEqual(result.summary, "Deterministic fallback summary.")
        self.assertEqual(result.source, "deterministic_fallback")

    @patch("app.llm.client._request_chat_completion")
    def test_generate_grounded_answer_uses_provider_when_configured(self, mock_request: object) -> None:
        mock_request.return_value = "Provider-backed grounded answer."

        summary = generate_grounded_answer(
            question="What signals matter?",
            supporting_signals=["Latest simple return: 1.00%."],
            retrieved_context_texts=["ETF momentum improved."],
            fallback_summary="Deterministic fallback summary.",
            config=LLMConfig(api_key="test-key"),
        )

        self.assertEqual(summary, "Provider-backed grounded answer.")
        mock_request.assert_called_once()

    @patch("app.llm.client._request_chat_completion")
    def test_generate_grounded_answer_result_marks_provider_source(self, mock_request: object) -> None:
        mock_request.return_value = "Provider-backed grounded answer."

        result = generate_grounded_answer_result(
            question="What signals matter?",
            supporting_signals=["Latest simple return: 1.00%."],
            retrieved_context_texts=["ETF momentum improved."],
            fallback_summary="Deterministic fallback summary.",
            config=LLMConfig(api_key="test-key"),
        )

        self.assertEqual(result.summary, "Provider-backed grounded answer.")
        self.assertEqual(result.source, "provider_backed")

    @patch("app.llm.client._request_chat_completion")
    def test_generate_grounded_answer_falls_back_on_provider_error(self, mock_request: object) -> None:
        mock_request.side_effect = LLMClientError("Provider-backed answer generation failed.")

        summary = generate_grounded_answer(
            question="What signals matter?",
            supporting_signals=["Latest simple return: 1.00%."],
            retrieved_context_texts=["ETF momentum improved."],
            fallback_summary="Deterministic fallback summary.",
            config=LLMConfig(api_key="test-key"),
        )

        self.assertEqual(summary, "Deterministic fallback summary.")


if __name__ == "__main__":
    unittest.main()
