"""Geometric Brownian Motion simulation and risk metrics."""

__all__ = [
    "calculate_annualized_volatility",
    "calculate_simulation_percentiles",
    "calculate_simulation_risk_metrics",
    "geometric_brownian_motion",
    "geometric_brownian_motion_paths",
]

import math

import numpy as np
import pandas as pd


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
        current_price: Current asset price (S_t).
        annual_return: Expected annual rate of return (mu).
        annual_volatility: Annualized volatility (sigma).
        time_horizon: Forecast period in years (delta-t).
        num_simulations: Number of terminal prices to generate.
        rng: NumPy random generator for reproducibility.

    Returns:
        1-D array of simulated terminal prices.
    """
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
        current_price: Current asset price.
        annual_return: Expected annual rate of return.
        annual_volatility: Annualized volatility.
        time_horizon: Forecast period in years.
        num_simulations: Number of paths.
        num_steps: Time steps per path (default 252 trading days).
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape ``(num_steps + 1, num_simulations)``.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = time_horizon / num_steps
    shocks = rng.standard_normal((num_steps, num_simulations))
    drift = (annual_return - 0.5 * annual_volatility**2) * dt
    diffusion = annual_volatility * math.sqrt(dt)

    prices = np.empty((num_steps + 1, num_simulations))
    prices[0] = current_price
    for t in range(1, num_steps + 1):
        prices[t] = prices[t - 1] * np.exp(drift + diffusion * shocks[t - 1])
    return prices


def calculate_annualized_volatility(*, prices: pd.Series, lookback_years: int) -> float:
    """Calculate annualized volatility from a price series.

    Args:
        prices: Historical closing prices.
        lookback_years: Length of lookback period in years.

    Returns:
        Annualized volatility estimate.
    """
    daily_returns = prices.pct_change().dropna()
    daily_std = daily_returns.std()
    time_interval = lookback_years / len(prices)
    return float(daily_std / math.sqrt(time_interval))


def calculate_simulation_percentiles(*, values: np.ndarray) -> dict[str, float]:
    """Compute 5/25/50/75/95 percentiles.

    Args:
        values: 1-D array of simulated values.

    Returns:
        Mapping from percentile label to value.
    """
    labels = [5, 25, 50, 75, 95]
    return {f"p{p}": float(np.percentile(values, p)) for p in labels}


def calculate_simulation_risk_metrics(
    *,
    terminal_prices: np.ndarray,
    current_price: float,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Derive risk metrics from terminal price distribution.

    Args:
        terminal_prices: 1-D array of simulated terminal prices.
        current_price: Starting price.
        risk_free_rate: Annual risk-free rate for Sharpe calculation.

    Returns:
        Dictionary with VaR, CVaR, Sharpe, probability of profit,
        expected return, and return volatility.
    """
    returns = (terminal_prices - current_price) / current_price

    var_95 = float(np.percentile(returns, 5))
    cvar_95 = float(returns[returns <= var_95].mean())

    mean_return = float(returns.mean())
    return_std = float(returns.std())
    sharpe = (mean_return - risk_free_rate) / return_std if return_std > 0 else 0.0
    prob_profit = float((terminal_prices > current_price).mean())

    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "sharpe": sharpe,
        "prob_profit": prob_profit,
        "mean_return": mean_return,
        "return_std": return_std,
    }
