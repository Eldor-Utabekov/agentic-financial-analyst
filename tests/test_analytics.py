from __future__ import annotations

import math
import unittest

import pandas as pd

from app.tools.analytics import calculate_drawdown
from app.tools.analytics import calculate_log_returns
from app.tools.analytics import calculate_max_drawdown
from app.tools.analytics import calculate_momentum
from app.tools.analytics import calculate_rolling_volatility
from app.tools.analytics import calculate_simple_returns


class ReturnsTests(unittest.TestCase):
    def test_calculate_simple_returns_returns_expected_values(self) -> None:
        prices = pd.Series([100.0, 110.0, 121.0], index=pd.RangeIndex(3))

        result = calculate_simple_returns(prices)

        expected = pd.Series([float("nan"), 0.10, 0.10], index=prices.index)
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_log_returns_returns_expected_values(self) -> None:
        prices = pd.Series([100.0, 110.0, 121.0], index=pd.RangeIndex(3))

        result = calculate_log_returns(prices)

        expected_value = math.log(1.1)
        self.assertTrue(pd.isna(result.iloc[0]))
        self.assertAlmostEqual(result.iloc[1], expected_value)
        self.assertAlmostEqual(result.iloc[2], expected_value)

    def test_calculate_log_returns_rejects_non_positive_prices(self) -> None:
        prices = pd.Series([100.0, 0.0, 121.0], index=pd.RangeIndex(3))

        with self.assertRaisesRegex(ValueError, "prices must contain only positive values"):
            calculate_log_returns(prices)


class RollingVolatilityTests(unittest.TestCase):
    def test_calculate_rolling_volatility_returns_expected_values(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=pd.RangeIndex(4))

        result = calculate_rolling_volatility(returns, window=2, annualize=False)

        expected = returns.rolling(window=2).std()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_rolling_volatility_annualizes_when_requested(self) -> None:
        returns = pd.Series([0.01, 0.02, 0.03], index=pd.RangeIndex(3))

        result = calculate_rolling_volatility(returns, window=2, annualize=True, periods_per_year=12)

        expected = returns.rolling(window=2).std() * math.sqrt(12)
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_rolling_volatility_rejects_invalid_window(self) -> None:
        returns = pd.Series([0.01, 0.02], index=pd.RangeIndex(2))

        with self.assertRaisesRegex(ValueError, "window must be greater than 0"):
            calculate_rolling_volatility(returns, window=0)

    def test_calculate_rolling_volatility_rejects_invalid_periods_per_year(self) -> None:
        returns = pd.Series([0.01, 0.02], index=pd.RangeIndex(2))

        with self.assertRaisesRegex(ValueError, "periods_per_year must be greater than 0"):
            calculate_rolling_volatility(returns, periods_per_year=0)


class MomentumAndDrawdownTests(unittest.TestCase):
    def test_calculate_momentum_returns_expected_values(self) -> None:
        prices = pd.Series([100.0, 105.0, 110.0, 121.0], index=pd.RangeIndex(4))

        result = calculate_momentum(prices, window=2)

        expected = pd.Series([float("nan"), float("nan"), 0.10, 0.15238095238095228], index=prices.index)
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_drawdown_returns_expected_values(self) -> None:
        prices = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=pd.RangeIndex(5))

        result = calculate_drawdown(prices)

        expected = pd.Series([0.0, 0.0, -0.25, -0.20833333333333337, 0.0], index=prices.index)
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_max_drawdown_returns_worst_drawdown(self) -> None:
        prices = pd.Series([100.0, 120.0, 90.0, 95.0, 130.0], index=pd.RangeIndex(5))

        result = calculate_max_drawdown(prices)

        self.assertEqual(result, -0.25)


class ValidationTests(unittest.TestCase):
    def test_price_functions_reject_empty_series(self) -> None:
        prices = pd.Series(dtype=float)

        for function in (
            calculate_simple_returns,
            calculate_log_returns,
            calculate_momentum,
            calculate_drawdown,
            calculate_max_drawdown,
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "prices must not be empty"):
                    if function is calculate_momentum:
                        function(prices, window=2)
                    else:
                        function(prices)

    def test_price_functions_reject_missing_values(self) -> None:
        prices = pd.Series([100.0, None, 120.0], dtype=float)

        for function in (
            calculate_simple_returns,
            calculate_log_returns,
            calculate_momentum,
            calculate_drawdown,
            calculate_max_drawdown,
        ):
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "prices must not contain missing values"):
                    if function is calculate_momentum:
                        function(prices, window=2)
                    else:
                        function(prices)

    def test_returns_functions_reject_non_series_input(self) -> None:
        with self.assertRaisesRegex(TypeError, "prices must be a pandas Series"):
            calculate_simple_returns([100.0, 110.0])  # type: ignore[arg-type]

        with self.assertRaisesRegex(TypeError, "returns must be a pandas Series"):
            calculate_rolling_volatility([0.01, 0.02])  # type: ignore[arg-type]
