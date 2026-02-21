"""Modern Portfolio Theory — portfolio optimization via riskfolio-lib."""

__all__ = [
    "OBJECTIVES",
    "RISK_MEASURES",
    "PortfolioResult",
    "build_efficient_frontier",
    "build_portfolio",
    "calculate_asset_statistics",
    "fetch_multi_history",
    "optimize_portfolio",
    "preferred_solvers",
]

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import riskfolio as rp
import yfinance as yf
from riskfolio.src import ParamsEstimation

TRADING_DAYS: int = 252
_MIN_TICKERS: int = 2

# Risk measures requiring EXP or POW cone solvers.
# MOSEK is strongly recommended; CLARABEL may fail, SCS may be slow.
_EXP_POW_MEASURES: frozenset[str] = frozenset(
    {
        "GMD",
        "TG",
        "RLVaR",
        "TGRG",
        "RVRG",
        "RLDaR",  # POW cone
        "EVaR",
        "EVRG",
        "EDaR",  # EXP cone
    }
)


def preferred_solvers(*, rm: str) -> list[str]:
    """Return solver preference order for the given risk measure.

    EXP / POW cone problems (marked ** in the riskfolio-lib docs)
    strongly benefit from MOSEK.  For LP / QP / SOCP / SDP problems
    CLARABEL is a good open-source default.
    """
    if rm in _EXP_POW_MEASURES:
        return ["MOSEK", "CLARABEL", "SCS"]
    return ["CLARABEL", "MOSEK", "SCS"]


RISK_MEASURES: dict[str, str] = {
    # Dispersion
    "Standard Deviation": "MV",
    "Square Root Kurtosis": "KT",
    "Mean Absolute Deviation": "MAD",
    "Gini Mean Difference": "GMD",
    "CVaR Range": "CVRG",
    "Tail Gini Range": "TGRG",
    "EVaR Range": "EVRG",
    "RLVaR Range": "RVRG",
    "Range": "RG",
    # Downside
    "Semi Standard Deviation": "MSV",
    "Square Root Semi Kurtosis": "SKT",
    "First Lower Partial Moment": "FLPM",
    "Second Lower Partial Moment": "SLPM",
    "Conditional Value at Risk": "CVaR",
    "Tail Gini": "TG",
    "Entropic Value at Risk": "EVaR",
    "Relativistic Value at Risk": "RLVaR",
    "Worst Realization": "WR",
    # Drawdown
    "Average Drawdown": "ADD",
    "Ulcer Index": "UCI",
    "Conditional Drawdown at Risk": "CDaR",
    "Entropic Drawdown at Risk": "EDaR",
    "Relativistic Drawdown at Risk": "RLDaR",
    "Maximum Drawdown": "MDD",
}

OBJECTIVES: dict[str, str] = {
    "Minimum Risk": "MinRisk",
    "Maximum Return": "MaxRet",
    "Maximum Utility Function": "Utility",
    "Maximum Risk Adjusted Return Ratio": "Sharpe",
}


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    """Optimal portfolio allocation with risk/return metrics.

    Attributes:
        weights: Mapping of ticker to allocation weight.
        expected_return: Annualized expected return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Annualized Sharpe ratio.
    """

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


# -- Helpers ------------------------------------------------------------------


def _infer_ann_factor(*, index: pd.Index) -> float:
    """Infer the annualization factor from a DatetimeIndex.

    Computes the average number of observations per calendar year
    based on the date range spanned by the index.
    """
    total_calendar_days = (index[-1] - index[0]).days
    if total_calendar_days <= 0:
        return float(TRADING_DAYS)
    return len(index) * 365.25 / total_calendar_days


def build_portfolio(
    *,
    prices: pd.DataFrame,
    allow_short_selling: bool,
    alpha: float = 0.05,
    kappa: float = 0.30,
) -> tuple[rp.Portfolio, np.ndarray, np.ndarray, float]:
    """Create a riskfolio Portfolio and compute asset statistics.

    Args:
        prices: Multi-asset price DataFrame (one column per ticker).
        allow_short_selling: Allow negative weights if ``True``.
        alpha: Significance level for tail-risk measures (e.g. 0.05
            for 95 % confidence).
        kappa: Deformation parameter for relativistic VaR / DaR
            measures.  Must be between 0 and 1.

    Returns:
        Tuple of ``(portfolio, mu_daily, cov_daily, ann_factor)`` where
        *mu_daily* and *cov_daily* are in the same frequency as the
        input returns and *ann_factor* is the inferred annualization
        multiplier.

    Raises:
        RuntimeError: If asset statistics could not be computed.
    """
    returns = prices.pct_change().dropna()
    ann_factor = _infer_ann_factor(index=returns.index)

    port = rp.Portfolio(returns=returns)
    port.assets_stats(method_mu="hist", method_cov="hist", method_kurt="hist")
    port.skurt = ParamsEstimation.cokurt_matrix(returns, method="semi")
    port.sht = allow_short_selling
    port.alpha = alpha
    port.kappa = kappa

    if port.mu is None or port.cov is None:
        msg = "asset statistics could not be computed"
        raise RuntimeError(msg)

    mu: np.ndarray = port.mu.values.flatten()
    cov_mat: np.ndarray = port.cov.values
    return port, mu, cov_mat, ann_factor


# -- Public API ---------------------------------------------------------------


def fetch_multi_history(*, tickers: list[str], period: str = "5y") -> pd.DataFrame:
    """Download and align closing prices for multiple tickers.

    Args:
        tickers: List of yfinance ticker symbols (minimum 2).
        period: Look-back period accepted by yfinance.

    Returns:
        DataFrame with a ``DatetimeIndex`` and one column per ticker
        (inner-joined on dates).

    Raises:
        ValueError: If fewer than 2 tickers are supplied.
    """
    if len(tickers) < _MIN_TICKERS:
        msg = "at least 2 tickers are required for portfolio optimization"
        raise ValueError(msg)

    raw: pd.DataFrame = yf.download(
        tickers,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        msg = f"no data returned for tickers={tickers!r}, period={period!r}"
        raise ValueError(msg)

    prices = raw["Close"].dropna()
    prices.index.name = "Date"
    return prices


def optimize_portfolio(
    *,
    prices: pd.DataFrame,
    rm: str = "MV",
    obj: str = "Sharpe",
    kelly: str | None = None,
    rf: float = 0.0,
    allow_short_selling: bool = False,
    risk_aversion: float = 1.0,
    alpha: float = 0.05,
    kappa: float = 0.30,
) -> PortfolioResult:
    """Optimize a portfolio using riskfolio-lib.

    Args:
        prices: Multi-asset price DataFrame (one column per ticker).
        rm: Risk measure code (see ``RISK_MEASURES``).
        obj: Objective code (see ``OBJECTIVES``).
        kelly: Kelly criterion — ``None``, ``"approx"``, or ``"exact"``.
        rf: Annual risk-free rate.
        allow_short_selling: Allow negative weights if ``True``.
        risk_aversion: Risk aversion parameter (used when *obj* is
            ``"Utility"``).
        alpha: Significance level for tail-risk measures.
        kappa: Deformation parameter for relativistic measures.

    Returns:
        ``PortfolioResult`` with optimal weights and metrics.

    Raises:
        RuntimeError: If the optimization fails to converge.
    """
    port, mu, cov_mat, ann_factor = build_portfolio(
        prices=prices,
        allow_short_selling=allow_short_selling,
        alpha=alpha,
        kappa=kappa,
    )
    port.solvers = preferred_solvers(rm=rm)
    rf_daily = rf / ann_factor
    w = port.optimization(
        model="Classic",
        rm=rm,
        obj=obj,
        kelly=kelly,
        rf=rf_daily,
        l=risk_aversion,
    )
    if w is None:
        msg = "optimization failed to converge"
        raise RuntimeError(msg)

    tickers = list(w.index)
    weights_dict = dict(zip(tickers, w["weights"].tolist(), strict=True))
    w_arr = w.values.flatten()

    port_return = float(mu @ w_arr) * ann_factor
    port_vol = float(np.sqrt(w_arr @ cov_mat @ w_arr)) * math.sqrt(ann_factor)
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0

    return PortfolioResult(
        weights=weights_dict,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
    )


def build_efficient_frontier(
    *,
    prices: pd.DataFrame,
    rm: str = "MV",
    kelly: str | None = None,
    num_points: int = 50,
    rf: float = 0.0,
    allow_short_selling: bool = False,
    alpha: float = 0.05,
    kappa: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build the efficient frontier using riskfolio-lib.

    Args:
        prices: Multi-asset price DataFrame.
        rm: Risk measure code (see ``RISK_MEASURES``).
        kelly: Kelly criterion — ``None``, ``"approx"``, or ``"exact"``.
        num_points: Number of frontier points.
        rf: Annual risk-free rate.
        allow_short_selling: Allow negative weights if ``True``.
        alpha: Significance level for tail-risk measures.
        kappa: Deformation parameter for relativistic measures.

    Returns:
        Tuple of ``(risks, returns, weights_df)`` where *weights_df* has
        assets as rows and frontier points as columns.

    Raises:
        RuntimeError: If the frontier computation fails.
    """
    port, mu, cov_mat, ann_factor = build_portfolio(
        prices=prices,
        allow_short_selling=allow_short_selling,
        alpha=alpha,
        kappa=kappa,
    )
    port.solvers = preferred_solvers(rm=rm)
    rf_daily = rf / ann_factor
    frontier = port.efficient_frontier(
        model="Classic",
        rm=rm,
        kelly=kelly,
        points=num_points,
        rf=rf_daily,
    )
    if frontier is None:
        msg = "efficient frontier computation failed"
        raise RuntimeError(msg)

    n_points = frontier.shape[1]
    returns_arr = np.empty(n_points)
    risks_arr = np.empty(n_points)

    sqrt_ann = math.sqrt(ann_factor)
    for i, col in enumerate(frontier.columns):
        w_arr = frontier[col].values
        returns_arr[i] = float(mu @ w_arr) * ann_factor
        risks_arr[i] = float(np.sqrt(w_arr @ cov_mat @ w_arr)) * sqrt_ann

    # Sort by risk so the frontier plots as a smooth curve.
    order = np.argsort(risks_arr)
    return risks_arr[order], returns_arr[order], frontier.iloc[:, order]


def calculate_asset_statistics(*, prices: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute annualized return and volatility per asset.

    Infers the annualization factor from the DatetimeIndex.

    Args:
        prices: Multi-asset price DataFrame with a DatetimeIndex.

    Returns:
        Mapping of ticker to ``{"annualized_return": …, "annualized_volatility": …}``.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    ann_factor = _infer_ann_factor(index=log_returns.index)
    result: dict[str, dict[str, float]] = {}
    for col in prices.columns:
        ann_ret = float(log_returns[col].mean() * ann_factor)
        ann_vol = float(log_returns[col].std() * math.sqrt(ann_factor))
        result[col] = {
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
        }
    return result
