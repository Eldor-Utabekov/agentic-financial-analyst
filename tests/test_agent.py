from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from app.llm.agent import AgentResponse
from app.llm.agent import answer_financial_question
from app.llm.client import SummaryGenerationResult
from app.rag.ingest import Chunk


def _sample_price_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "close": [100.0, 102.0, 101.0, 104.0, 106.0],
            "symbol": ["SPY"] * 5,
        }
    )


def _sample_chunks() -> list[Chunk]:
    return [
        {
            "chunk_id": "chunk-1",
            "source": "doc1.txt",
            "text": "ETF momentum improved as inflows increased this week.",
            "metadata": {"file_name": "doc1.txt"},
        },
        {
            "chunk_id": "chunk-2",
            "source": "doc2.txt",
            "text": "Drawdown risk remains manageable for diversified ETFs.",
            "metadata": {"file_name": "doc2.txt"},
        },
        {
            "chunk_id": "chunk-3",
            "source": "doc3.txt",
            "text": "Commodity prices moved independently of equity ETFs.",
            "metadata": {"file_name": "doc3.txt"},
        },
    ]


class AnswerFinancialQuestionTests(unittest.TestCase):
    def test_answer_financial_question_returns_structured_response(self) -> None:
        response = answer_financial_question(
            question="What signals matter for this ETF?",
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
            top_k=2,
        )

        self.assertIsInstance(response, AgentResponse)
        self.assertEqual(response.question, "What signals matter for this ETF?")
        self.assertTrue(response.summary)
        self.assertGreaterEqual(len(response.supporting_signals), 1)
        self.assertGreaterEqual(len(response.retrieved_context), 1)
        self.assertEqual(len(response.tool_trace), 5)

    @patch("app.llm.agent.generate_grounded_answer_result")
    def test_answer_financial_question_uses_provider_backed_summary_when_available(self, mock_generate: object) -> None:
        mock_generate.return_value = SummaryGenerationResult(
            summary="Provider-backed grounded summary.",
            source="provider_backed",
        )

        response = answer_financial_question(
            question="What signals matter for this ETF?",
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
            top_k=2,
        )

        self.assertEqual(response.summary, "Provider-backed grounded summary.")
        mock_generate.assert_called_once()
        self.assertIn("provider_backed", response.tool_trace[-1].details)

    def test_answer_financial_question_rejects_empty_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "question must not be empty"):
            answer_financial_question("   ", _sample_price_data(), _sample_chunks())

    def test_answer_financial_question_rejects_missing_required_price_columns(self) -> None:
        price_data = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})

        with self.assertRaisesRegex(ValueError, "price_data is missing required columns: close"):
            answer_financial_question("What signals matter?", price_data, _sample_chunks())

    def test_answer_financial_question_handles_no_relevant_chunks(self) -> None:
        chunks: list[Chunk] = [
            {
                "chunk_id": "chunk-1",
                "source": "doc1.txt",
                "text": "Natural gas supply changed materially.",
                "metadata": {"file_name": "doc1.txt"},
            }
        ]

        response = answer_financial_question(
            question="ETF momentum outlook",
            price_data=_sample_price_data(),
            chunks=chunks,
            top_k=3,
        )

        self.assertEqual(response.retrieved_context, [])
        self.assertIn("No relevant local context was retrieved for this question.", response.risks)
        self.assertIn("No directly relevant local context was retrieved.", response.summary)

    def test_answer_financial_question_tool_trace_records_steps(self) -> None:
        response = answer_financial_question(
            question="ETF drawdown risk",
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
        )

        self.assertEqual(
            [step.step for step in response.tool_trace],
            [
                "validate_question",
                "extract_price_series",
                "retrieve_context",
                "summarize_financial_signals",
                "build_response",
            ],
        )
        self.assertTrue(all(step.status == "completed" for step in response.tool_trace))

    def test_answer_financial_question_response_context_includes_scores(self) -> None:
        response = answer_financial_question(
            question="ETF momentum improved",
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
            top_k=1,
        )

        self.assertEqual(len(response.retrieved_context), 1)
        self.assertGreater(response.retrieved_context[0].score, 0.0)
        self.assertTrue(response.retrieved_context[0].text)

    def test_answer_financial_question_preserves_deterministic_fallback_without_api_key(self) -> None:
        response = answer_financial_question(
            question="ETF momentum improved",
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
            top_k=1,
        )

        self.assertIn("Latest close is 106.00.", response.summary)
        self.assertIn("deterministic_fallback", response.tool_trace[-1].details)


if __name__ == "__main__":
    unittest.main()
