"""Tests for the Monte Carlo simulation module."""

import numpy as np
import pandas as pd

from investment_portfolio.monte_carlo import (
    calculate_annualized_volatility,
    calculate_simulation_percentiles,
    calculate_simulation_risk_metrics,
    geometric_brownian_motion,
    geometric_brownian_motion_paths,
)


class TestGeometricBrownianMotion:
    def test_terminal_prices_shape(self, rng: np.random.Generator) -> None:
        result = geometric_brownian_motion(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=500,
            rng=rng,
        )
        assert result.shape == (500,)

    def test_all_prices_positive(self, rng: np.random.Generator) -> None:
        result = geometric_brownian_motion(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=1000,
            rng=rng,
        )
        assert np.all(result > 0)

    def test_deterministic_with_seed(self) -> None:
        a = geometric_brownian_motion(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=50,
            rng=np.random.default_rng(7),
        )
        b = geometric_brownian_motion(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=50,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(a, b)


class TestGeometricBrownianMotionPaths:
    def test_paths_shape(self, rng: np.random.Generator) -> None:
        paths = geometric_brownian_motion_paths(
            current_price=50.0,
            annual_return=0.08,
            annual_volatility=0.25,
            time_horizon=1.0,
            num_simulations=200,
            num_steps=126,
            rng=rng,
        )
        assert paths.shape == (127, 200)

    def test_first_row_is_current_price(self, rng: np.random.Generator) -> None:
        price = 123.45
        paths = geometric_brownian_motion_paths(
            current_price=price,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=10,
            rng=rng,
        )
        np.testing.assert_allclose(paths[0], price)

    def test_deterministic_with_seed(self) -> None:
        a = geometric_brownian_motion_paths(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=20,
            num_steps=50,
            rng=np.random.default_rng(7),
        )
        b = geometric_brownian_motion_paths(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=20,
            num_steps=50,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(a, b)


class TestCalculateAnnualizedVolatility:
    def test_positive(self, sample_prices: pd.Series) -> None:
        vol = calculate_annualized_volatility(prices=sample_prices, lookback_years=5)
        assert vol > 0

    def test_returns_float(self, sample_prices: pd.Series) -> None:
        vol = calculate_annualized_volatility(prices=sample_prices, lookback_years=5)
        assert isinstance(vol, float)


class TestCalculateSimulationPercentiles:
    def test_keys(self, rng: np.random.Generator) -> None:
        values = rng.normal(100, 20, 1000)
        result = calculate_simulation_percentiles(values=values)
        assert set(result.keys()) == {"p5", "p25", "p50", "p75", "p95"}

    def test_ordering(self, rng: np.random.Generator) -> None:
        values = rng.normal(100, 20, 1000)
        p = calculate_simulation_percentiles(values=values)
        assert p["p5"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["p95"]


class TestCalculateSimulationRiskMetrics:
    def test_keys(self, rng: np.random.Generator) -> None:
        terminals = geometric_brownian_motion(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=500,
            rng=rng,
        )
        result = calculate_simulation_risk_metrics(
            terminal_prices=terminals, current_price=100.0
        )
        expected_keys = {
            "var_95",
            "cvar_95",
            "sharpe",
            "prob_profit",
            "mean_return",
            "return_std",
        }
        assert set(result.keys()) == expected_keys

    def test_cvar_worse_than_var(self, rng: np.random.Generator) -> None:
        terminals = geometric_brownian_motion(
            current_price=100.0,
            annual_return=0.10,
            annual_volatility=0.20,
            time_horizon=1.0,
            num_simulations=5000,
            rng=rng,
        )
        m = calculate_simulation_risk_metrics(
            terminal_prices=terminals, current_price=100.0
        )
        assert m["cvar_95"] <= m["var_95"]
