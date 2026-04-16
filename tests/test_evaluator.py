from __future__ import annotations

import unittest

import pandas as pd

from app.llm.evaluator import EvaluationCase
from app.llm.evaluator import evaluate_batch
from app.llm.evaluator import evaluate_single_case
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
    ]


class EvaluateSingleCaseTests(unittest.TestCase):
    def test_evaluate_single_case_returns_successful_result(self) -> None:
        result = evaluate_single_case(
            question="What signals matter for this ETF?",
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
            expected_tool_steps=[
                "validate_question",
                "extract_price_series",
                "retrieve_context",
                "summarize_financial_signals",
                "build_response",
            ],
        )

        self.assertTrue(result.success)
        self.assertTrue(result.tool_trace_present)
        self.assertTrue(result.summary_present)
        self.assertGreater(result.supporting_signal_count, 0)
        self.assertGreater(result.risk_count, 0)
        self.assertTrue(result.expected_tools_matched)
        self.assertEqual(result.notes, [])

    def test_evaluate_single_case_returns_failure_for_bad_input(self) -> None:
        bad_price_data = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})

        result = evaluate_single_case(
            question="What signals matter for this ETF?",
            price_data=bad_price_data,
            chunks=_sample_chunks(),
        )

        self.assertFalse(result.success)
        self.assertFalse(result.tool_trace_present)
        self.assertFalse(result.summary_present)
        self.assertGreaterEqual(len(result.notes), 1)
        self.assertIn("missing required columns: close", result.notes[0])

    def test_evaluate_single_case_records_question_as_string_on_failure(self) -> None:
        result = evaluate_single_case(
            question=123,  # type: ignore[arg-type]
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
        )

        self.assertFalse(result.success)
        self.assertIsInstance(result.question, str)
        self.assertEqual(result.question, "123")

    def test_evaluate_single_case_populates_note_for_expected_step_mismatch(self) -> None:
        result = evaluate_single_case(
            question="What signals matter for this ETF?",
            price_data=_sample_price_data(),
            chunks=_sample_chunks(),
            expected_tool_steps=["validate_question", "retrieve_context"],
        )

        self.assertFalse(result.success)
        self.assertFalse(result.expected_tools_matched)
        self.assertIn("Tool trace did not match expected steps.", result.notes)


class EvaluateBatchTests(unittest.TestCase):
    def test_evaluate_batch_returns_results_for_multiple_cases(self) -> None:
        cases = [
            EvaluationCase(
                question="What signals matter for this ETF?",
                price_data=_sample_price_data(),
                chunks=_sample_chunks(),
            ),
            EvaluationCase(
                question="ETF momentum outlook",
                price_data=_sample_price_data(),
                chunks=[
                    {
                        "chunk_id": "chunk-3",
                        "source": "doc3.txt",
                        "text": "Natural gas storage rose this month.",
                        "metadata": {"file_name": "doc3.txt"},
                    }
                ],
            ),
        ]

        results = evaluate_batch(cases)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].question, "What signals matter for this ETF?")
        self.assertEqual(results[1].question, "ETF momentum outlook")

    def test_evaluate_batch_rejects_invalid_case_type(self) -> None:
        with self.assertRaisesRegex(TypeError, "Each case must be an EvaluationCase"):
            evaluate_batch(["not-a-case"])  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
