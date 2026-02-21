"""Filtered Historical Simulation for asset price forecasting."""

__all__ = [
    "FHSMethod",
    "FHSPercentiles",
    "FHSRiskMetrics",
    "ReturnStatistics",
    "calculate_fhs_percentiles",
    "calculate_fhs_risk_metrics",
    "calculate_filtered_historical_returns",
    "calculate_return_statistics",
    "forecast_prices",
]

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import assert_never

import numpy as np
import pandas as pd
import scipy.stats


class FHSMethod(StrEnum):
    """FHS volatility-filtering method."""

    RATIO = auto()  # Practitioner: vol ratio on multi-period returns
    RESIDUALS = auto()  # Academic: bootstrap standardized daily residuals


@dataclass(frozen=True, slots=True)
class ReturnStatistics:
    """Descriptive statistics for a returns array.

    Attributes:
        mean: Mean return.
        median: Median return.
        std: Standard deviation.
        skewness: Skewness.
        kurtosis: Excess kurtosis.
        minimum: Minimum return.
        maximum: Maximum return.
    """

    mean: float
    median: float
    std: float
    skewness: float
    kurtosis: float
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class FHSPercentiles:
    """Percentile prices and percentage changes from FHS forecast.

    Attributes:
        p5_price: 5th percentile price.
        p5_pct: 5th percentile percentage change.
        p25_price: 25th percentile price.
        p25_pct: 25th percentile percentage change.
        p50_price: 50th percentile price.
        p50_pct: 50th percentile percentage change.
        p75_price: 75th percentile price.
        p75_pct: 75th percentile percentage change.
        p95_price: 95th percentile price.
        p95_pct: 95th percentile percentage change.
    """

    p5_price: float
    p5_pct: float
    p25_price: float
    p25_pct: float
    p50_price: float
    p50_pct: float
    p75_price: float
    p75_pct: float
    p95_price: float
    p95_pct: float


@dataclass(frozen=True, slots=True)
class FHSRiskMetrics:
    """Risk metrics from Filtered Historical Simulation.

    Attributes:
        var_95: Value at Risk at 95% confidence (percentage).
        cvar_95: Conditional VaR (percentage).
        expected_price: Expected forecasted price.
        expected_return: Mean filtered return.
        return_std: Standard deviation of filtered returns.
        sharpe: Sharpe ratio.
        prob_appreciation: Probability of price increase.
    """

    var_95: float
    cvar_95: float
    expected_price: float
    expected_return: float
    return_std: float
    sharpe: float
    prob_appreciation: float


def _ratio_scaling_method(
    *,
    prices: pd.Series,
    forecast_horizon: int,
    volatility_window: int,
    ewma_decay: float,
) -> np.ndarray:
    """Practitioner FHS — scale multi-period returns by volatility ratio."""
    pct_returns = prices.pct_change().dropna()

    current_volatility = float(
        pct_returns.iloc[-volatility_window:].ewm(alpha=1 - ewma_decay).std().iloc[-1]
    )

    max_periods = len(prices) - forecast_horizon
    returns: list[float] = []

    for i in range(1, max_periods):
        end_idx = -i
        start_idx = -i - forecast_horizon
        historical_return = (
            prices.iloc[end_idx] - prices.iloc[start_idx]
        ) / prices.iloc[start_idx]

        end_pos = len(prices) - i
        start_pos = max(0, end_pos - volatility_window)
        hist_returns = pct_returns.iloc[start_pos:end_pos]

        if len(hist_returns) >= volatility_window:
            historical_volatility = float(
                hist_returns.ewm(alpha=1 - ewma_decay).std().iloc[-1]
            )
            if historical_volatility > 0:
                scaling_factor = current_volatility / historical_volatility
                historical_return *= scaling_factor

        returns.append(float(historical_return))

    return np.array(returns)


def _standardized_residuals_method(
    *,
    prices: pd.Series,
    forecast_horizon: int,
    ewma_decay: float,
    rng: np.random.Generator,
    num_simulations: int,
) -> np.ndarray:
    """Academic FHS — bootstrap standardized residuals (Barone-Adesi et al. 1998).

    Steps:
        1. Compute daily log-returns.
        2. Compute conditional volatility via EWMA std of log-returns.
        3. Standardize: z_t = r_t / vol_t.
        4. Bootstrap ``num_simulations`` paths of ``forecast_horizon`` days.
        5. Re-scale each sampled residual by current conditional volatility,
           sum daily log-returns, and convert to simple returns.
    """
    log_returns = np.diff(np.log(prices.values))

    ewma_vol = pd.Series(log_returns).ewm(alpha=1 - ewma_decay).std()

    vol_values = ewma_vol.values
    eps: float = 1e-12
    valid = vol_values > eps
    z = log_returns[valid] / vol_values[valid]

    current_vol = float(vol_values[~np.isnan(vol_values)][-1])

    sampled_z = rng.choice(z, size=(num_simulations, forecast_horizon), replace=True)
    daily_log_returns = sampled_z * current_vol
    cumulative = daily_log_returns.sum(axis=1)

    return np.expm1(cumulative)


def _unfiltered_returns(
    *,
    prices: pd.Series,
    forecast_horizon: int,
) -> np.ndarray:
    """Raw multi-period simple returns without any volatility filtering."""
    max_periods = len(prices) - forecast_horizon
    returns: list[float] = []

    for i in range(1, max_periods):
        end_idx = -i
        start_idx = -i - forecast_horizon
        historical_return = (
            prices.iloc[end_idx] - prices.iloc[start_idx]
        ) / prices.iloc[start_idx]
        returns.append(float(historical_return))

    return np.array(returns)


def calculate_filtered_historical_returns(
    *,
    prices: pd.Series,
    forecast_horizon: int,
    volatility_window: int,
    ewma_decay: float,
    apply_filter: bool = True,
    method: FHSMethod = FHSMethod.RATIO,
    rng: np.random.Generator | None = None,
    num_simulations: int = 10_000,
) -> np.ndarray:
    """Calculate historical returns adjusted for current volatility.

    Implements two Filtered Historical Simulation methods:

    * **RATIO** (practitioner): scales multi-period returns by the ratio of
      current to historical EWMA volatility.
    * **RESIDUALS** (academic, Barone-Adesi et al. 1998): extracts standardized
      daily residuals, bootstraps them, and re-scales by the current
      conditional volatility to produce simulated multi-period returns.

    Args:
        prices: Time series of prices.
        forecast_horizon: Number of periods ahead to forecast. Must be positive.
        volatility_window: Window size for volatility estimation. Must be positive.
        ewma_decay: EWMA decay factor (lambda). 0.94 is the RiskMetrics standard.
        apply_filter: If ``True``, apply volatility scaling.
        method: Which FHS method to use.
        rng: Random generator for the residuals method.
        num_simulations: Number of bootstrap paths (residuals method only).
            Must be positive.

    Returns:
        Array of (optionally filtered) historical returns.

    Raises:
        ValueError: If inputs are out of valid range.
    """
    if forecast_horizon <= 0:
        msg = "forecast_horizon must be positive"
        raise ValueError(msg)
    if volatility_window <= 0:
        msg = "volatility_window must be positive"
        raise ValueError(msg)
    if num_simulations <= 0:
        msg = "num_simulations must be positive"
        raise ValueError(msg)
    if len(prices) <= forecast_horizon:
        msg = "prices must have more observations than forecast_horizon"
        raise ValueError(msg)

    if not apply_filter:
        return _unfiltered_returns(
            prices=prices,
            forecast_horizon=forecast_horizon,
        )

    match method:
        case FHSMethod.RATIO:
            return _ratio_scaling_method(
                prices=prices,
                forecast_horizon=forecast_horizon,
                volatility_window=volatility_window,
                ewma_decay=ewma_decay,
            )
        case FHSMethod.RESIDUALS:
            if rng is None:
                rng = np.random.default_rng()
            return _standardized_residuals_method(
                prices=prices,
                forecast_horizon=forecast_horizon,
                ewma_decay=ewma_decay,
                rng=rng,
                num_simulations=num_simulations,
            )
        case _:
            assert_never(method)


def calculate_return_statistics(*, returns: np.ndarray) -> ReturnStatistics:
    """Compute descriptive statistics for a returns array.

    Args:
        returns: 1-D array of returns.

    Returns:
        Descriptive statistics.
    """
    return ReturnStatistics(
        mean=float(np.mean(returns)),
        median=float(np.median(returns)),
        std=float(np.std(returns, ddof=1)),
        skewness=float(scipy.stats.skew(returns)),
        kurtosis=float(scipy.stats.kurtosis(returns)),
        minimum=float(np.min(returns)),
        maximum=float(np.max(returns)),
    )


def forecast_prices(
    *, current_price: float, filtered_returns: np.ndarray
) -> np.ndarray:
    """Project future prices from filtered returns.

    Args:
        current_price: Current asset price. Must be positive.
        filtered_returns: Array of filtered historical returns.

    Returns:
        Array of forecasted prices.

    Raises:
        ValueError: If current_price is not positive.
    """
    if current_price <= 0:
        msg = "current_price must be positive"
        raise ValueError(msg)
    return (filtered_returns + 1) * current_price


def calculate_fhs_percentiles(
    *, forecasted_prices: np.ndarray, current_price: float
) -> FHSPercentiles:
    """Compute percentiles and percentage changes for forecasted prices.

    Args:
        forecasted_prices: 1-D array of forecasted prices.
        current_price: Starting price.

    Returns:
        Percentile prices and percentage changes.
    """
    labels = [5, 25, 50, 75, 95]
    values: dict[str, float] = {}
    for p in labels:
        price = float(np.percentile(forecasted_prices, p))
        pct = (price / current_price - 1) * 100
        values[f"p{p}_price"] = price
        values[f"p{p}_pct"] = pct

    return FHSPercentiles(**values)


def calculate_fhs_risk_metrics(
    *,
    filtered_returns: np.ndarray,
    forecasted_prices: np.ndarray,
    current_price: float,
    risk_free_rate: float = 0.0,
    time_horizon: float = 1.0,
) -> FHSRiskMetrics:
    """Derive risk metrics from FHS results.

    Args:
        filtered_returns: Array of filtered historical returns.
        forecasted_prices: Corresponding forecasted prices.
        current_price: Starting price.
        risk_free_rate: Annual risk-free rate for Sharpe ratio calculation.
        time_horizon: Forecast period in years for compounding the risk-free rate.

    Returns:
        Risk metrics.
    """
    returns_pct = filtered_returns * 100

    var_95 = float(np.percentile(returns_pct, 5))
    cvar_95 = float(returns_pct[returns_pct <= var_95].mean())

    mean_return = float(np.mean(filtered_returns))
    return_std = float(np.std(filtered_returns, ddof=1))
    eps: float = 1e-12
    rf_period = (1 + risk_free_rate) ** time_horizon - 1
    sharpe = (mean_return - rf_period) / return_std if return_std > eps else 0.0

    expected_price = float(np.mean(forecasted_prices))
    prob_appreciation = float((forecasted_prices > current_price).mean())

    return FHSRiskMetrics(
        var_95=var_95,
        cvar_95=cvar_95,
        expected_price=expected_price,
        expected_return=mean_return,
        return_std=return_std,
        sharpe=sharpe,
        prob_appreciation=prob_appreciation,
    )
