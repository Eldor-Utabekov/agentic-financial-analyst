from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.main import app


def _sample_payload() -> dict[str, object]:
    return {
        "question": "What signals matter for this ETF?",
        "price_data": [
            {"date": "2024-01-01", "close": 100.0, "symbol": "SPY"},
            {"date": "2024-01-02", "close": 102.0, "symbol": "SPY"},
            {"date": "2024-01-03", "close": 101.0, "symbol": "SPY"},
            {"date": "2024-01-04", "close": 104.0, "symbol": "SPY"},
            {"date": "2024-01-05", "close": 106.0, "symbol": "SPY"},
        ],
        "chunks": [
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
        ],
        "top_k": 2,
    }


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_returns_ok(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_ask_returns_structured_response(self) -> None:
        response = self.client.post("/ask", json=_sample_payload())

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["question"], "What signals matter for this ETF?")
        self.assertIn("summary", payload)
        self.assertIn("supporting_signals", payload)
        self.assertIn("risks", payload)
        self.assertIn("retrieved_context", payload)
        self.assertIn("tool_trace", payload)
        self.assertEqual(len(payload["tool_trace"]), 5)

    def test_ask_returns_422_for_invalid_payload(self) -> None:
        payload = _sample_payload()
        payload["question"] = 123

        response = self.client.post("/ask", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_ask_returns_422_for_missing_required_fields(self) -> None:
        response = self.client.post("/ask", json={"question": "What signals matter?"})

        self.assertEqual(response.status_code, 422)

    def test_ask_returns_422_for_invalid_top_k(self) -> None:
        payload = _sample_payload()
        payload["top_k"] = 0

        response = self.client.post("/ask", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_ask_returns_400_for_invalid_price_data_content(self) -> None:
        payload = _sample_payload()
        payload["price_data"] = [{"date": "2024-01-01", "symbol": "SPY"}]

        response = self.client.post("/ask", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing required columns: close", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
