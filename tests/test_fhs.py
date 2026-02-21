"""Tests for the Filtered Historical Simulation module."""

import numpy as np
import pandas as pd

from investment_portfolio.fhs import (
    calculate_fhs_percentiles,
    calculate_fhs_risk_metrics,
    calculate_filtered_historical_returns,
    calculate_return_statistics,
    forecast_prices,
)


class TestCalculateFilteredHistoricalReturns:
    def test_output_length(self, sample_fx_rates: pd.Series) -> None:
        result = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
        )
        expected_len = len(sample_fx_rates) - 50 - 1
        assert len(result) == expected_len

    def test_filtered_differs_from_unfiltered(self, sample_fx_rates: pd.Series) -> None:
        filtered = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            apply_filter=True,
        )
        unfiltered = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            apply_filter=False,
        )
        assert not np.allclose(filtered, unfiltered)

    def test_unfiltered_returns_raw(self, sample_fx_rates: pd.Series) -> None:
        result = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            apply_filter=False,
        )
        assert len(result) > 0


class TestCalculateReturnStatistics:
    def test_expected_keys(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        result = calculate_return_statistics(returns=returns)
        expected = {"mean", "median", "std", "skewness", "kurtosis", "min", "max"}
        assert set(result.keys()) == expected

    def test_min_max_ordering(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        stats = calculate_return_statistics(returns=returns)
        assert stats["min"] <= stats["mean"] <= stats["max"]


class TestForecastPrices:
    def test_all_positive(self, sample_fx_rates: pd.Series) -> None:
        returns = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
        )
        current = float(sample_fx_rates.iloc[-1])
        prices = forecast_prices(current_price=current, filtered_returns=returns)
        assert np.all(prices > 0)

    def test_shape_matches_returns(self) -> None:
        returns = np.array([0.01, -0.02, 0.03])
        prices = forecast_prices(current_price=1.10, filtered_returns=returns)
        assert prices.shape == returns.shape


class TestFHSPercentiles:
    def test_ordering(self) -> None:
        rng = np.random.default_rng(42)
        prices = 1.10 + rng.normal(0, 0.05, 1000)
        p = calculate_fhs_percentiles(forecasted_prices=prices, current_price=1.10)
        assert (
            p["p5_price"]
            <= p["p25_price"]
            <= p["p50_price"]
            <= p["p75_price"]
            <= p["p95_price"]
        )


class TestFHSRiskMetrics:
    def test_expected_keys(self, sample_fx_rates: pd.Series) -> None:
        returns = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
        )
        current = float(sample_fx_rates.iloc[-1])
        prices = forecast_prices(current_price=current, filtered_returns=returns)
        result = calculate_fhs_risk_metrics(
            filtered_returns=returns,
            forecasted_prices=prices,
            current_price=current,
        )
        expected = {
            "var_95",
            "cvar_95",
            "expected_price",
            "expected_return",
            "return_std",
            "sharpe",
            "prob_appreciation",
        }
        assert set(result.keys()) == expected
