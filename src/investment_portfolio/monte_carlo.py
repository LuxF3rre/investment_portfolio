"""Geometric Brownian Motion simulation and risk metrics."""

__all__ = [
    "SimulationPercentiles",
    "SimulationRiskMetrics",
    "calculate_annualized_volatility",
    "calculate_simulation_percentiles",
    "calculate_simulation_risk_metrics",
    "geometric_brownian_motion",
    "geometric_brownian_motion_paths",
]

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from investment_portfolio._validation import validate_finite

_MIN_OBSERVATIONS: Final = 20
_ANN_FACTOR_MIN: Final = 200.0
_ANN_FACTOR_MAX: Final = 260.0


@dataclass(frozen=True, slots=True)
class SimulationPercentiles:
    """Percentile values from a simulation distribution.

    Attributes:
        p5: 5th percentile.
        p25: 25th percentile.
        p50: 50th percentile (median).
        p75: 75th percentile.
        p95: 95th percentile.
    """

    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


@dataclass(frozen=True, slots=True)
class SimulationRiskMetrics:
    """Risk metrics derived from a terminal price distribution.

    Attributes:
        var_95: Value at Risk at 95% confidence (5th percentile of returns).
        cvar_95: Conditional VaR (expected shortfall beyond VaR).
        sharpe: Sharpe ratio.
        prob_profit: Probability of positive return.
        mean_return: Mean simulated return.
        return_std: Standard deviation of simulated returns.
    """

    var_95: float
    cvar_95: float
    sharpe: float
    prob_profit: float
    mean_return: float
    return_std: float


def geometric_brownian_motion(
    *,
    current_price: float,
    annual_return: float,
    annual_volatility: float,
    time_horizon: float,
    num_simulations: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate terminal asset prices using vectorized GBM.

    Args:
        current_price: Current asset price (S_t). Must be positive.
        annual_return: Expected annual rate of return (mu).
        annual_volatility: Annualized volatility (sigma).
        time_horizon: Forecast period in years (delta-t). Must be positive.
        num_simulations: Number of terminal prices to generate. Must be positive.
        rng: NumPy random generator for reproducibility.

    Returns:
        1-D array of simulated terminal prices.

    Raises:
        ValueError: If inputs are out of valid range.
    """
    if current_price <= 0:
        msg = "current_price must be positive"
        raise ValueError(msg)
    if time_horizon <= 0:
        msg = "time_horizon must be positive"
        raise ValueError(msg)
    if num_simulations <= 0:
        msg = "num_simulations must be positive"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    shocks = rng.standard_normal(num_simulations)
    drift = (annual_return - 0.5 * annual_volatility**2) * time_horizon
    diffusion = annual_volatility * shocks * math.sqrt(time_horizon)
    return current_price * np.exp(drift + diffusion)


def geometric_brownian_motion_paths(
    *,
    current_price: float,
    annual_return: float,
    annual_volatility: float,
    time_horizon: float,
    num_simulations: int,
    num_steps: int = 252,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate complete price paths using GBM.

    Args:
        current_price: Current asset price. Must be positive.
        annual_return: Expected annual rate of return.
        annual_volatility: Annualized volatility.
        time_horizon: Forecast period in years. Must be positive.
        num_simulations: Number of paths. Must be positive.
        num_steps: Time steps per path (default 252 trading days). Must be positive.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape ``(num_steps + 1, num_simulations)``.

    Raises:
        ValueError: If inputs are out of valid range.
    """
    if current_price <= 0:
        msg = "current_price must be positive"
        raise ValueError(msg)
    if time_horizon <= 0:
        msg = "time_horizon must be positive"
        raise ValueError(msg)
    if num_simulations <= 0:
        msg = "num_simulations must be positive"
        raise ValueError(msg)
    if num_steps <= 0:
        msg = "num_steps must be positive"
        raise ValueError(msg)

    if rng is None:
        rng = np.random.default_rng()

    dt = time_horizon / num_steps
    shocks = rng.standard_normal((num_steps, num_simulations))
    drift = (annual_return - 0.5 * annual_volatility**2) * dt
    diffusion = annual_volatility * math.sqrt(dt)

    log_increments = drift + diffusion * shocks
    prices = np.empty((num_steps + 1, num_simulations))
    prices[0] = current_price
    prices[1:] = current_price * np.exp(np.cumsum(log_increments, axis=0))
    return prices


def calculate_annualized_volatility(*, prices: pd.Series) -> float:
    """Calculate annualized volatility from a price series.

    Infers the annualization factor from the datetime index by computing
    the average number of observations per calendar year.

    Args:
        prices: Historical closing prices with a DatetimeIndex.

    Returns:
        Annualized volatility estimate.

    Raises:
        ValueError: If the price series has fewer than 2 observations.
    """
    if len(prices) < _MIN_OBSERVATIONS:
        msg = f"prices must have at least {_MIN_OBSERVATIONS} observations"
        raise ValueError(msg)
    validate_finite(arr=prices.values, name="prices")

    daily_returns = prices.pct_change().dropna()
    daily_std = daily_returns.std()

    total_calendar_days = (prices.index[-1] - prices.index[0]).days
    if total_calendar_days <= 0:
        ann_factor = 252.0
    else:
        ann_factor = (len(prices) - 1) * 365.25 / total_calendar_days
    ann_factor = max(_ANN_FACTOR_MIN, min(ann_factor, _ANN_FACTOR_MAX))

    return float(daily_std * math.sqrt(ann_factor))


def calculate_simulation_percentiles(*, values: np.ndarray) -> SimulationPercentiles:
    """Compute 5/25/50/75/95 percentiles.

    Args:
        values: 1-D array of simulated values.

    Returns:
        Percentile values.
    """
    p5, p25, p50, p75, p95 = np.percentile(values, [5, 25, 50, 75, 95])
    return SimulationPercentiles(
        p5=float(p5),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p95=float(p95),
    )


def calculate_simulation_risk_metrics(
    *,
    terminal_prices: np.ndarray,
    current_price: float,
    risk_free_rate: float = 0.04,
    time_horizon: float = 1.0,
) -> SimulationRiskMetrics:
    """Derive risk metrics from terminal price distribution.

    VaR is the 5th percentile of returns (95% confidence).  CVaR (Expected
    Shortfall) is the mean of all returns at or below the VaR threshold;
    when ties exist at the threshold this may include slightly more than 5%
    of observations.

    Args:
        terminal_prices: 1-D array of simulated terminal prices.
        current_price: Starting price.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.
        time_horizon: Forecast period in years for compounding the risk-free rate.

    Returns:
        Risk metrics.
    """
    validate_finite(arr=terminal_prices, name="terminal_prices")

    returns = (terminal_prices - current_price) / current_price

    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean())

    mean_return = float(returns.mean())
    return_std = float(returns.std(ddof=1))
    eps: float = 1e-12
    rf_period = (1 + risk_free_rate) ** time_horizon - 1
    sharpe = (mean_return - rf_period) / return_std if return_std > eps else 0.0
    prob_profit = float((terminal_prices > current_price).mean())

    return SimulationRiskMetrics(
        var_95=var_95,
        cvar_95=cvar_95,
        sharpe=sharpe,
        prob_profit=prob_profit,
        mean_return=mean_return,
        return_std=return_std,
    )
