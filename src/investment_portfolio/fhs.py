"""Filtered Historical Simulation for asset price forecasting."""

__all__ = [
    "calculate_fhs_percentiles",
    "calculate_fhs_risk_metrics",
    "calculate_filtered_historical_returns",
    "calculate_return_statistics",
    "forecast_prices",
]

import numpy as np
import pandas as pd
import scipy.stats


def calculate_filtered_historical_returns(
    *,
    prices: pd.Series,
    forecast_horizon: int,
    volatility_window: int,
    ewma_decay: float,
    apply_filter: bool = True,
) -> np.ndarray:
    """Calculate historical returns adjusted for current volatility.

    Implements the Filtered Historical Simulation (FHS) method which scales
    historical returns by the ratio of current to historical volatility.

    Args:
        prices: Time series of prices.
        forecast_horizon: Number of periods ahead to forecast.
        volatility_window: Window size for volatility estimation.
        ewma_decay: Decay factor (``com`` parameter) for EWMA.
        apply_filter: If ``True``, apply volatility scaling.

    Returns:
        Array of (optionally filtered) historical returns.
    """
    recent_rates = prices.iloc[-volatility_window:]
    current_volatility = float(np.std(recent_rates.ewm(com=ewma_decay).mean()))

    max_periods = len(prices) - forecast_horizon
    returns: list[float] = []

    for i in range(1, max_periods):
        end_idx = -i
        start_idx = -i - forecast_horizon
        historical_return = (
            prices.iloc[end_idx] - prices.iloc[start_idx]
        ) / prices.iloc[start_idx]

        if apply_filter:
            hist_window_start = max(0, -i - volatility_window)
            hist_window_end = -i if i > 0 else None
            historical_rates = prices.iloc[hist_window_start:hist_window_end]

            if len(historical_rates) >= volatility_window:
                historical_volatility = float(
                    np.std(historical_rates.ewm(com=ewma_decay).mean())
                )
                if historical_volatility > 0:
                    scaling_factor = current_volatility / historical_volatility
                    historical_return *= scaling_factor

        returns.append(float(historical_return))

    return np.array(returns)


def calculate_return_statistics(*, returns: np.ndarray) -> dict[str, float]:
    """Compute descriptive statistics for a returns array.

    Args:
        returns: 1-D array of returns.

    Returns:
        Dictionary with mean, median, std, skewness, kurtosis, min, max.
    """
    return {
        "mean": float(np.mean(returns)),
        "median": float(np.median(returns)),
        "std": float(np.std(returns)),
        "skewness": float(scipy.stats.skew(returns)),
        "kurtosis": float(scipy.stats.kurtosis(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
    }


def forecast_prices(
    *, current_price: float, filtered_returns: np.ndarray
) -> np.ndarray:
    """Project future prices from filtered returns.

    Args:
        current_price: Current asset price.
        filtered_returns: Array of filtered historical returns.

    Returns:
        Array of forecasted prices.
    """
    return (filtered_returns + 1) * current_price


def calculate_fhs_percentiles(
    *, forecasted_prices: np.ndarray, current_price: float
) -> dict[str, float]:
    """Compute percentiles and percentage changes for forecasted prices.

    Args:
        forecasted_prices: 1-D array of forecasted prices.
        current_price: Starting price.

    Returns:
        Dictionary keyed by percentile label with price and pct change.
    """
    labels = [5, 25, 50, 75, 95]
    result: dict[str, float] = {}
    for p in labels:
        price = float(np.percentile(forecasted_prices, p))
        pct = (price / current_price - 1) * 100
        result[f"p{p}_price"] = price
        result[f"p{p}_pct"] = pct
    return result


def calculate_fhs_risk_metrics(
    *,
    filtered_returns: np.ndarray,
    forecasted_prices: np.ndarray,
    current_price: float,
) -> dict[str, float]:
    """Derive risk metrics from FHS results.

    Args:
        filtered_returns: Array of filtered historical returns.
        forecasted_prices: Corresponding forecasted prices.
        current_price: Starting price.

    Returns:
        Dictionary with VaR, CVaR, expected price, expected return,
        probability of appreciation, and Sharpe ratio.
    """
    returns_pct = filtered_returns * 100

    var_95 = float(np.percentile(returns_pct, 5))
    cvar_95 = float(returns_pct[returns_pct <= var_95].mean())

    mean_return = float(np.mean(filtered_returns))
    return_std = float(np.std(filtered_returns))
    sharpe = mean_return / return_std if return_std > 0 else 0.0

    expected_price = float(np.mean(forecasted_prices))
    prob_appreciation = float((forecasted_prices > current_price).mean())

    return {
        "var_95": var_95,
        "cvar_95": cvar_95,
        "expected_price": expected_price,
        "expected_return": mean_return,
        "return_std": return_std,
        "sharpe": sharpe,
        "prob_appreciation": prob_appreciation,
    }
