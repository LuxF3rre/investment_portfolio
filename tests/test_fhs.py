"""Tests for the Filtered Historical Simulation module."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from investment_portfolio.fhs import (
    FHSMethod,
    FHSPercentiles,
    FHSRiskMetrics,
    ReturnStatistics,
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

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("forecast_horizon", 0, "forecast_horizon must be positive"),
            ("forecast_horizon", -1, "forecast_horizon must be positive"),
            ("volatility_window", 0, "volatility_window must be positive"),
            ("volatility_window", -5, "volatility_window must be positive"),
            ("num_simulations", 0, "num_simulations must be positive"),
        ],
    )
    def test_invalid_inputs_raise(
        self,
        sample_fx_rates: pd.Series,
        field: str,
        value: int,
        match: str,
    ) -> None:
        defaults: dict[str, object] = {
            "prices": sample_fx_rates,
            "forecast_horizon": 50,
            "volatility_window": 20,
            "ewma_decay": 0.99,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            calculate_filtered_historical_returns(**defaults)  # type: ignore[invalid-argument-type]

    def test_insufficient_prices_raises(self) -> None:
        prices = pd.Series(
            [1.0, 1.01, 1.02],
            index=pd.bdate_range("2024-01-02", periods=3),
        )
        with pytest.raises(ValueError, match="more observations than forecast_horizon"):
            calculate_filtered_historical_returns(
                prices=prices,
                forecast_horizon=10,
                volatility_window=2,
                ewma_decay=0.99,
            )


class TestCalculateReturnStatistics:
    def test_returns_dataclass(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.005])
        result = calculate_return_statistics(returns=returns)
        assert isinstance(result, ReturnStatistics)
        fields = [
            "mean",
            "median",
            "std",
            "skewness",
            "kurtosis",
            "minimum",
            "maximum",
        ]
        assert all(hasattr(result, f) for f in fields)

    def test_min_max_ordering(self) -> None:
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 500)
        stats = calculate_return_statistics(returns=returns)
        assert stats.minimum <= stats.mean <= stats.maximum

    def test_round_trip_via_asdict(self) -> None:
        returns = np.array([0.01, -0.02, 0.03])
        stats = calculate_return_statistics(returns=returns)
        d = dataclasses.asdict(stats)
        assert set(d.keys()) == {
            "mean",
            "median",
            "std",
            "skewness",
            "kurtosis",
            "minimum",
            "maximum",
        }


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

    def test_zero_current_price_raises(self) -> None:
        with pytest.raises(ValueError, match="current_price must be positive"):
            forecast_prices(current_price=0.0, filtered_returns=np.array([0.01]))

    def test_negative_current_price_raises(self) -> None:
        with pytest.raises(ValueError, match="current_price must be positive"):
            forecast_prices(current_price=-10.0, filtered_returns=np.array([0.01]))


class TestFHSPercentiles:
    def test_returns_dataclass(self) -> None:
        rng = np.random.default_rng(42)
        prices = 1.10 + rng.normal(0, 0.05, 1000)
        p = calculate_fhs_percentiles(forecasted_prices=prices, current_price=1.10)
        assert isinstance(p, FHSPercentiles)

    def test_ordering(self) -> None:
        rng = np.random.default_rng(42)
        prices = 1.10 + rng.normal(0, 0.05, 1000)
        p = calculate_fhs_percentiles(forecasted_prices=prices, current_price=1.10)
        assert p.p5_price <= p.p25_price <= p.p50_price <= p.p75_price <= p.p95_price


class TestStandardizedResidualsMethod:
    def test_output_length(self, sample_fx_rates: pd.Series) -> None:
        result = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            method=FHSMethod.RESIDUALS,
            rng=np.random.default_rng(42),
            num_simulations=500,
        )
        assert len(result) == 500

    def test_deterministic_with_seed(self, sample_fx_rates: pd.Series) -> None:
        a = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            method=FHSMethod.RESIDUALS,
            num_simulations=200,
            rng=np.random.default_rng(7),
        )
        b = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            method=FHSMethod.RESIDUALS,
            num_simulations=200,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(a, b)

    def test_differs_from_ratio(self, sample_fx_rates: pd.Series) -> None:
        ratio = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            method=FHSMethod.RATIO,
        )
        residuals = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            method=FHSMethod.RESIDUALS,
            rng=np.random.default_rng(42),
            num_simulations=len(ratio),
        )
        assert not np.allclose(ratio, residuals)

    def test_differs_from_unfiltered(self, sample_fx_rates: pd.Series) -> None:
        unfiltered = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            apply_filter=False,
        )
        residuals = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
            method=FHSMethod.RESIDUALS,
            rng=np.random.default_rng(42),
            num_simulations=len(unfiltered),
        )
        assert not np.allclose(unfiltered, residuals)


class TestFHSRiskMetrics:
    def test_returns_dataclass(self, sample_fx_rates: pd.Series) -> None:
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
        assert isinstance(result, FHSRiskMetrics)

    def test_has_expected_fields(self, sample_fx_rates: pd.Series) -> None:
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
        assert set(dataclasses.asdict(result).keys()) == expected

    def test_risk_free_rate_affects_sharpe(self, sample_fx_rates: pd.Series) -> None:
        returns = calculate_filtered_historical_returns(
            prices=sample_fx_rates,
            forecast_horizon=50,
            volatility_window=20,
            ewma_decay=0.99,
        )
        current = float(sample_fx_rates.iloc[-1])
        prices = forecast_prices(current_price=current, filtered_returns=returns)

        r0 = calculate_fhs_risk_metrics(
            filtered_returns=returns,
            forecasted_prices=prices,
            current_price=current,
            risk_free_rate=0.0,
        )
        r1 = calculate_fhs_risk_metrics(
            filtered_returns=returns,
            forecasted_prices=prices,
            current_price=current,
            risk_free_rate=0.05,
        )
        assert r0.sharpe > r1.sharpe
