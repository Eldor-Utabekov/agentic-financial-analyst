from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import patch

import pandas as pd

from app.tools.market_data import MarketDataError, PriceHistoryRequest, fetch_price_history


class PriceHistoryRequestTests(unittest.TestCase):
    def test_normalized_symbol_trims_and_uppercases(self) -> None:
        request = PriceHistoryRequest(symbol=" spy ", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

        self.assertEqual(request.normalized_symbol(), "SPY")

    def test_validate_rejects_empty_symbol(self) -> None:
        request = PriceHistoryRequest(symbol="   ", start_date=date(2024, 1, 1), end_date=date(2024, 1, 31))

        with self.assertRaisesRegex(ValueError, "Symbol must not be empty"):
            request.validate()

    def test_validate_rejects_reversed_date_range(self) -> None:
        request = PriceHistoryRequest(symbol="SPY", start_date=date(2024, 2, 1), end_date=date(2024, 1, 31))

        with self.assertRaisesRegex(ValueError, "start_date must be on or before end_date"):
            request.validate()


class FetchPriceHistoryTests(unittest.TestCase):
    @patch("app.tools.market_data._download_price_history")
    def test_fetch_price_history_returns_normalized_dataframe(self, mock_download: object) -> None:
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Adj Close": [100.4, 101.4],
                "Volume": [1_000_000, 1_100_000],
            },
            index=pd.to_datetime(["2024-01-03", "2024-01-02"]),
        )

        result = fetch_price_history("spy", date(2024, 1, 2), date(2024, 1, 3))

        self.assertEqual(
            list(result.columns),
            ["date", "open", "high", "low", "close", "adj_close", "volume", "symbol"],
        )
        self.assertEqual(result["symbol"].tolist(), ["SPY", "SPY"])
        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d").tolist(), ["2024-01-02", "2024-01-03"])
        called_request = mock_download.call_args.args[0]
        self.assertEqual(called_request.normalized_symbol(), "SPY")
        self.assertEqual(called_request.start_date, date(2024, 1, 2))
        self.assertEqual(called_request.end_date, date(2024, 1, 3))
        self.assertEqual(called_request.interval, "1d")

    def test_fetch_price_history_rejects_empty_symbol(self) -> None:
        with self.assertRaisesRegex(ValueError, "Symbol must not be empty"):
            fetch_price_history("   ", date(2024, 1, 2), date(2024, 1, 3))

    def test_fetch_price_history_rejects_reversed_date_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "start_date must be on or before end_date"):
            fetch_price_history("SPY", date(2024, 1, 3), date(2024, 1, 2))

    @patch("app.tools.market_data._download_price_history")
    def test_fetch_price_history_raises_for_empty_results(self, mock_download: object) -> None:
        mock_download.return_value = pd.DataFrame()

        with self.assertRaisesRegex(MarketDataError, "No market data returned"):
            fetch_price_history("SPY", date(2024, 1, 1), date(2024, 1, 2))

    @patch("app.tools.market_data._download_price_history")
    def test_fetch_price_history_raises_for_missing_columns(self, mock_download: object) -> None:
        mock_download.return_value = pd.DataFrame(
            {"Open": [100.0], "Close": [100.5]},
            index=pd.to_datetime(["2024-01-02"]),
        )

        with self.assertRaisesRegex(MarketDataError, "Missing required price columns"):
            fetch_price_history("SPY", date(2024, 1, 2), date(2024, 1, 2))

    @patch("app.tools.market_data._download_price_history")
    def test_fetch_price_history_normalizes_timezone_aware_index_values(self, mock_download: object) -> None:
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [101.0, 100.0],
                "High": [102.0, 101.0],
                "Low": [100.0, 99.0],
                "Close": [101.5, 100.5],
                "Adj Close": [101.4, 100.4],
                "Volume": [1_100_000, 1_000_000],
            },
            index=pd.DatetimeIndex(
                [
                    "2024-01-03 00:00:00+00:00",
                    "2024-01-02 00:00:00+00:00",
                ]
            ),
        )

        result = fetch_price_history("SPY", date(2024, 1, 2), date(2024, 1, 3))

        self.assertEqual(result["date"].dt.tz, None)
        self.assertEqual(result["date"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist(), ["2024-01-02 00:00:00", "2024-01-03 00:00:00"])

    def test_fetch_price_history_raises_when_yfinance_is_unavailable(self) -> None:
        with self.assertRaisesRegex(MarketDataError, "yfinance is required to fetch market data"):
            with patch("builtins.__import__", side_effect=_mock_import_without_yfinance):
                from app.tools.market_data import _download_price_history

                _download_price_history(
                    PriceHistoryRequest(
                        symbol="SPY",
                        start_date=date(2024, 1, 1),
                        end_date=date(2024, 1, 2),
                    )
                )


def _mock_import_without_yfinance(name: str, globals: object = None, locals: object = None, fromlist: object = (), level: int = 0) -> object:
    if name == "yfinance":
        raise ModuleNotFoundError("No module named 'yfinance'")
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


_ORIGINAL_IMPORT = __import__


if __name__ == "__main__":
    unittest.main()
