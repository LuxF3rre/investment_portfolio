"""Tests for the Monte Carlo simulation module."""

import numpy as np
import pandas as pd
import pytest

from investment_portfolio.monte_carlo import (
    SimulationPercentiles,
    SimulationRiskMetrics,
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

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("current_price", -1.0, "current_price must be positive"),
            ("current_price", 0.0, "current_price must be positive"),
            ("time_horizon", 0.0, "time_horizon must be positive"),
            ("time_horizon", -1.0, "time_horizon must be positive"),
            ("num_simulations", 0, "num_simulations must be positive"),
            ("num_simulations", -5, "num_simulations must be positive"),
        ],
    )
    def test_invalid_inputs_raise(
        self, field: str, value: float | int, match: str
    ) -> None:
        defaults = {
            "current_price": 100.0,
            "annual_return": 0.10,
            "annual_volatility": 0.20,
            "time_horizon": 1.0,
            "num_simulations": 10,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            geometric_brownian_motion(**defaults)  # type: ignore[invalid-argument-type]


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

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("current_price", 0.0, "current_price must be positive"),
            ("time_horizon", 0.0, "time_horizon must be positive"),
            ("num_simulations", 0, "num_simulations must be positive"),
            ("num_steps", 0, "num_steps must be positive"),
            ("num_steps", -1, "num_steps must be positive"),
        ],
    )
    def test_invalid_inputs_raise(
        self, field: str, value: float | int, match: str
    ) -> None:
        defaults: dict[str, float | int] = {
            "current_price": 100.0,
            "annual_return": 0.10,
            "annual_volatility": 0.20,
            "time_horizon": 1.0,
            "num_simulations": 10,
            "num_steps": 50,
        }
        defaults[field] = value
        with pytest.raises(ValueError, match=match):
            geometric_brownian_motion_paths(**defaults)  # type: ignore[invalid-argument-type]


class TestCalculateAnnualizedVolatility:
    def test_positive(self, sample_prices: pd.Series) -> None:
        vol = calculate_annualized_volatility(prices=sample_prices)
        assert vol > 0

    def test_returns_float(self, sample_prices: pd.Series) -> None:
        vol = calculate_annualized_volatility(prices=sample_prices)
        assert isinstance(vol, float)

    def test_fewer_than_two_observations_raises(self) -> None:
        prices = pd.Series([100.0], index=pd.bdate_range("2024-01-02", periods=1))
        with pytest.raises(ValueError, match="at least 2 observations"):
            calculate_annualized_volatility(prices=prices)

    def test_empty_series_raises(self) -> None:
        prices = pd.Series([], dtype=float, index=pd.DatetimeIndex([]))
        with pytest.raises(ValueError, match="at least 2 observations"):
            calculate_annualized_volatility(prices=prices)


class TestCalculateSimulationPercentiles:
    def test_returns_dataclass(self, rng: np.random.Generator) -> None:
        values = rng.normal(100, 20, 1000)
        result = calculate_simulation_percentiles(values=values)
        assert isinstance(result, SimulationPercentiles)

    def test_ordering(self, rng: np.random.Generator) -> None:
        values = rng.normal(100, 20, 1000)
        p = calculate_simulation_percentiles(values=values)
        assert p.p5 <= p.p25 <= p.p50 <= p.p75 <= p.p95


class TestCalculateSimulationRiskMetrics:
    def test_returns_dataclass(self, rng: np.random.Generator) -> None:
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
        assert isinstance(result, SimulationRiskMetrics)

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
        assert m.cvar_95 <= m.var_95

    def test_zero_std_returns_zero_sharpe(self) -> None:
        terminals = np.full(100, 110.0)
        m = calculate_simulation_risk_metrics(
            terminal_prices=terminals, current_price=100.0
        )
        assert m.sharpe == 0.0
