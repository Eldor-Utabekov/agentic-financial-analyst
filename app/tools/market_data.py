"""Utilities for retrieving and normalizing ETF market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime
from datetime import timedelta
from typing import Final
from typing import cast

import pandas as pd

REQUIRED_PRICE_COLUMNS: Final[tuple[str, ...]] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
)


class MarketDataError(Exception):
    """Raised when market data retrieval fails or returns invalid results."""


@dataclass(frozen=True, slots=True)
class PriceHistoryRequest:
    """Input parameters for fetching historical ETF prices."""

    symbol: str
    start_date: date
    end_date: date
    interval: str = "1d"

    def normalized_symbol(self) -> str:
        """Return the cleaned symbol used for market data requests."""
        cleaned_symbol = self.symbol.strip().upper()
        if not cleaned_symbol:
            raise ValueError("Symbol must not be empty.")
        return cleaned_symbol

    def validate(self) -> None:
        """Validate the request fields before fetching market data."""
        self.normalized_symbol()
        normalized_start_date = _normalize_request_date(self.start_date, field_name="start_date")
        normalized_end_date = _normalize_request_date(self.end_date, field_name="end_date")
        if normalized_start_date > normalized_end_date:
            raise ValueError("start_date must be on or before end_date.")


def fetch_price_history(
    symbol: str,
    start_date: date,
    end_date: date,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch and normalize historical ETF OHLCV data.

    Returns a DataFrame with the columns:
    `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `symbol`.
    """

    request = PriceHistoryRequest(
        symbol=symbol,
        start_date=_normalize_request_date(start_date, field_name="start_date"),
        end_date=_normalize_request_date(end_date, field_name="end_date"),
        interval=interval,
    )
    request.validate()

    raw_prices = _download_price_history(request)

    return _normalize_price_history(raw_prices, request.normalized_symbol())


def _exclusive_end_date(inclusive_end_date: date) -> date:
    """Convert an inclusive end date into the exclusive boundary used by yfinance."""
    return inclusive_end_date + timedelta(days=1)


def _download_price_history(request: PriceHistoryRequest) -> pd.DataFrame:
    """Download raw historical price data for a validated request."""
    try:
        import yfinance as yf
    except ModuleNotFoundError as exc:
        raise MarketDataError(
            "yfinance is required to fetch market data. Install it in the active environment."
        ) from exc

    return yf.download(
        tickers=request.normalized_symbol(),
        start=request.start_date.isoformat(),
        end=_exclusive_end_date(request.end_date).isoformat(),
        interval=request.interval,
        progress=False,
        auto_adjust=False,
        group_by="column",
    )


def _normalize_price_history(raw_prices: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Validate and normalize raw yfinance output into a stable schema."""
    if raw_prices.empty:
        raise MarketDataError(f"No market data returned for symbol '{symbol}'.")

    missing_columns = [column for column in REQUIRED_PRICE_COLUMNS if column not in raw_prices.columns]
    if missing_columns:
        missing_str = ", ".join(missing_columns)
        raise MarketDataError(f"Missing required price columns: {missing_str}.")

    normalized = raw_prices.loc[:, list(REQUIRED_PRICE_COLUMNS)].rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    normalized = normalized.reset_index()
    normalized = normalized.rename(columns={normalized.columns[0]: "date"})

    normalized["date"] = pd.Series(
        [_normalize_timestamp(value) for value in normalized["date"]],
        index=normalized.index,
    )
    normalized["symbol"] = symbol

    ordered_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "symbol",
    ]

    return normalized.loc[:, ordered_columns].sort_values("date").reset_index(drop=True)


def _normalize_request_date(value: date, field_name: str) -> date:
    """Normalize supported date-like inputs to a plain date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"{field_name} must be a date-like value.")


def _normalize_timestamp(value: object) -> pd.Timestamp:
    """Normalize index values to timezone-naive pandas timestamps."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return cast(pd.Timestamp, timestamp.tz_localize(None))
    return timestamp
