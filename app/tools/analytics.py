"""Deterministic analytics helpers for ETF price series."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_simple_returns(prices: pd.Series) -> pd.Series:
    """Calculate period-over-period simple returns from a price series."""
    validated_prices = _validate_price_series(prices, series_name="prices")
    return validated_prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """Calculate period-over-period log returns from a price series."""
    validated_prices = _validate_price_series(prices, series_name="prices")
    return pd.Series(
        np.log(validated_prices / validated_prices.shift(1)),
        index=validated_prices.index,
        name=validated_prices.name,
    )


def calculate_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    periods_per_year: int = 252,
) -> pd.Series:
    """Calculate rolling standard deviation for a return series."""
    validated_returns = _validate_numeric_series(returns, series_name="returns")
    if validated_returns.empty:
        raise ValueError("returns must not be empty.")
    validated_window = _validate_window(window)

    rolling_volatility = validated_returns.rolling(window=validated_window).std()
    if annualize:
        validated_periods_per_year = _validate_periods_per_year(periods_per_year)
        rolling_volatility = rolling_volatility * math.sqrt(validated_periods_per_year)

    return rolling_volatility


def calculate_momentum(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate trailing price momentum as a simple return over a window."""
    validated_prices = _validate_price_series(prices, series_name="prices")
    validated_window = _validate_window(window)
    return validated_prices / validated_prices.shift(validated_window) - 1.0


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """Calculate drawdown relative to the running peak of a price series."""
    validated_prices = _validate_price_series(prices, series_name="prices")
    running_peak = validated_prices.cummax()
    return validated_prices / running_peak - 1.0


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Return the worst drawdown observed in a price series."""
    drawdown = calculate_drawdown(prices)
    return float(drawdown.min())


def _validate_price_series(prices: pd.Series, series_name: str) -> pd.Series:
    """Validate a price series used by analytics functions."""
    validated_prices = _validate_numeric_series(prices, series_name=series_name)
    if validated_prices.empty:
        raise ValueError(f"{series_name} must not be empty.")
    if validated_prices.isna().any():
        raise ValueError(f"{series_name} must not contain missing values.")
    if (validated_prices <= 0).any():
        raise ValueError(f"{series_name} must contain only positive values.")
    return validated_prices.astype(float)


def _validate_numeric_series(values: pd.Series, series_name: str) -> pd.Series:
    """Validate that a value is a pandas Series with numeric dtype."""
    if not isinstance(values, pd.Series):
        raise TypeError(f"{series_name} must be a pandas Series.")
    if not pd.api.types.is_numeric_dtype(values):
        raise TypeError(f"{series_name} must contain numeric values.")
    return values


def _validate_window(window: int) -> int:
    """Validate rolling window parameters."""
    if not isinstance(window, int):
        raise TypeError("window must be an integer.")
    if window <= 0:
        raise ValueError("window must be greater than 0.")
    return window


def _validate_periods_per_year(periods_per_year: int) -> int:
    """Validate annualization period settings."""
    if not isinstance(periods_per_year, int):
        raise TypeError("periods_per_year must be an integer.")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be greater than 0.")
    return periods_per_year
