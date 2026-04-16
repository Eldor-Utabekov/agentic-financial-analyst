from __future__ import annotations

import unittest

from app.rag.ingest import Chunk
from app.rag.retriever import retrieve_top_k
from app.rag.retriever import score_chunk


class ScoreChunkTests(unittest.TestCase):
    def test_score_chunk_returns_higher_score_for_more_overlap(self) -> None:
        high_score = score_chunk("etf momentum outlook", "ETF momentum outlook improved this week")
        low_score = score_chunk("etf momentum outlook", "Bond duration risk rose today")

        self.assertGreater(high_score, low_score)

    def test_score_chunk_is_case_insensitive(self) -> None:
        result = score_chunk("ETF Momentum", "etf momentum improved")

        self.assertEqual(result, 1.0)

    def test_score_chunk_rejects_empty_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "query must not be empty"):
            score_chunk("   ", "sample chunk")


class RetrieveTopKTests(unittest.TestCase):
    def test_retrieve_top_k_returns_ranked_results_with_scores(self) -> None:
        chunks: list[Chunk] = [
            {"chunk_id": "c1", "source": "a.txt", "text": "ETF momentum improved strongly", "metadata": {}},
            {"chunk_id": "c2", "source": "b.txt", "text": "ETF momentum and flows improved", "metadata": {}},
            {"chunk_id": "c3", "source": "c.txt", "text": "Bond yields rose sharply", "metadata": {}},
        ]

        results = retrieve_top_k("etf momentum improved", chunks, k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertIn("score", results[0])
        self.assertGreaterEqual(float(results[0]["score"]), float(results[1]["score"]))

    def test_retrieve_top_k_filters_out_zero_score_chunks(self) -> None:
        chunks: list[Chunk] = [
            {"chunk_id": "c1", "source": "a.txt", "text": "ETF momentum improved strongly", "metadata": {}},
            {"chunk_id": "c2", "source": "b.txt", "text": "Bond yields rose sharply", "metadata": {}},
        ]

        results = retrieve_top_k("etf momentum", chunks, k=3)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_retrieve_top_k_uses_chunk_id_as_tie_breaker(self) -> None:
        chunks: list[Chunk] = [
            {"chunk_id": "b", "source": "b.txt", "text": "ETF outlook", "metadata": {}},
            {"chunk_id": "a", "source": "a.txt", "text": "ETF outlook", "metadata": {}},
        ]

        results = retrieve_top_k("ETF outlook", chunks, k=2)

        self.assertEqual([chunk["chunk_id"] for chunk in results], ["a", "b"])

    def test_retrieve_top_k_rejects_invalid_chunk_shape(self) -> None:
        chunks = [{"chunk_id": "c1", "source": "a.txt", "text": "ETF text"}]

        with self.assertRaisesRegex(ValueError, "Each chunk must include keys"):
            retrieve_top_k("etf", chunks)

    def test_retrieve_top_k_rejects_empty_chunk_text(self) -> None:
        chunks = [{"chunk_id": "c1", "source": "a.txt", "text": "   ", "metadata": {}}]

        with self.assertRaisesRegex(ValueError, "text must be a non-empty string"):
            retrieve_top_k("etf", chunks)

    def test_retrieve_top_k_rejects_non_dict_metadata(self) -> None:
        chunks = [{"chunk_id": "c1", "source": "a.txt", "text": "ETF text", "metadata": "bad"}]

        with self.assertRaisesRegex(TypeError, "metadata must be a dictionary"):
            retrieve_top_k("etf", chunks)

    def test_retrieve_top_k_rejects_invalid_k(self) -> None:
        chunks: list[Chunk] = [{"chunk_id": "c1", "source": "a.txt", "text": "ETF text", "metadata": {}}]

        with self.assertRaisesRegex(ValueError, "k must be greater than 0"):
            retrieve_top_k("etf", chunks, k=0)
