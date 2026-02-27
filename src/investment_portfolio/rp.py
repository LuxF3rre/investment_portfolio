"""Risk Parity — equal risk contribution portfolio optimization via riskfolio-lib."""

__all__ = [
    "RP_RISK_MEASURES",
    "RiskParityResult",
    "compare_risk_budgets",
    "optimize_risk_parity",
]

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from riskfolio.src.RiskFunctions import Risk_Contribution

from investment_portfolio.mpt import build_portfolio, preferred_solvers

RP_RISK_MEASURES: dict[str, str] = {
    # Dispersion
    "Standard Deviation": "MV",
    "Square Root Kurtosis": "KT",
    "Mean Absolute Deviation": "MAD",
    "Gini Mean Difference": "GMD",
    "CVaR Range": "CVRG",
    "Tail Gini Range": "TGRG",
    "EVaR Range": "EVRG",
    "RLVaR Range": "RVRG",
    # Downside
    "Semi Standard Deviation": "MSV",
    "Square Root Semi Kurtosis": "SKT",
    "First Lower Partial Moment (Omega Ratio)": "FLPM",
    "Second Lower Partial Moment (Sortino Ratio)": "SLPM",
    "Conditional Value at Risk": "CVaR",
    "Tail Gini": "TG",
    "Entropic Value at Risk": "EVaR",
    "Relativistic Value at Risk": "RLVaR",
    # Drawdown
    "Ulcer Index": "UCI",
    "Conditional Drawdown at Risk": "CDaR",
    "Entropic Drawdown at Risk": "EDaR",
    "Relativistic Drawdown at Risk": "RLDaR",
}


@dataclass(frozen=True, slots=True)
class RiskParityResult:
    """Risk parity portfolio allocation with risk contribution metrics.

    Attributes:
        weights: Mapping of ticker to allocation weight.
        expected_return: Annualized expected return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        risk_contributions: Absolute risk contribution per asset.
        risk_contribution_pct: Percentage risk contribution per asset.
        budget: Target risk budget per asset.
    """

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_contributions: dict[str, float]
    risk_contribution_pct: dict[str, float]
    budget: dict[str, float]


def optimize_risk_parity(
    *,
    prices: pd.DataFrame,
    rm: str = "MV",
    rf: float = 0.0,
    budget: np.ndarray | None = None,
    allow_short_selling: bool = False,
    alpha: float = 0.05,
    kappa: float = 0.30,
) -> RiskParityResult:
    """Optimize a portfolio using risk parity (equal risk contribution).

    Args:
        prices: Multi-asset price DataFrame (one column per ticker).
        rm: Risk measure code (see ``RP_RISK_MEASURES``).
        rf: Annual risk-free rate.
        budget: Target risk budget vector of shape ``(n_assets, 1)``.
            Must be positive and will be internally normalized.
            If ``None``, defaults to equal ``1/N`` budget.
        allow_short_selling: Allow negative weights if ``True``.
        alpha: Significance level for tail-risk measures.
        kappa: Deformation parameter for relativistic measures.

    Returns:
        ``RiskParityResult`` with optimal weights and risk contributions.

    Raises:
        ValueError: If *budget* has wrong shape or non-positive entries.
        RuntimeError: If the optimization fails to converge.
    """
    n_assets = prices.shape[1]

    if budget is not None:
        if budget.shape != (n_assets, 1):
            msg = f"budget must have shape ({n_assets}, 1), got {budget.shape}"
            raise ValueError(msg)
        if np.any(budget <= 0):
            msg = "all budget entries must be positive"
            raise ValueError(msg)
        budget = budget / budget.sum()

    port, mu, cov_mat, ann_factor = build_portfolio(
        prices=prices,
        allow_short_selling=allow_short_selling,
        alpha=alpha,
        kappa=kappa,
    )
    port.solvers = preferred_solvers(rm=rm)
    rf_daily = rf / ann_factor

    w = port.rp_optimization(model="Classic", rm=rm, rf=rf_daily, b=budget)
    if w is None:
        msg = "risk parity optimization failed to converge"
        raise RuntimeError(msg)

    tickers = list(w.index)
    weights_dict = dict(zip(tickers, w["weights"].tolist(), strict=True))
    w_arr = w.values.flatten()

    # Risk contributions
    returns = prices.pct_change().dropna()
    rc = Risk_Contribution(
        w=w,
        returns=returns.values,
        cov=port.cov.values,  # type: ignore[union-attr]
        rm=rm,
        rf=rf_daily,
        alpha=alpha,
        kappa=kappa,
        solver=port.solvers[0],
    )
    rc_flat = rc.flatten()
    rc_total = rc_flat.sum()
    rc_pct = rc_flat / rc_total if rc_total > 0 else rc_flat

    risk_contributions = dict(zip(tickers, rc_flat.tolist(), strict=True))
    risk_contribution_pct = dict(zip(tickers, rc_pct.tolist(), strict=True))

    # Annualized metrics
    port_return = float(mu @ w_arr) * ann_factor
    port_vol = float(np.sqrt(w_arr @ cov_mat @ w_arr)) * math.sqrt(ann_factor)
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0

    # Budget dict
    if budget is not None:
        budget_dict = dict(zip(tickers, budget.flatten().tolist(), strict=True))
    else:
        budget_dict = dict(zip(tickers, [1.0 / n_assets] * n_assets, strict=True))

    return RiskParityResult(
        weights=weights_dict,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        risk_contributions=risk_contributions,
        risk_contribution_pct=risk_contribution_pct,
        budget=budget_dict,
    )


def compare_risk_budgets(
    *,
    prices: pd.DataFrame,
    rm: str = "MV",
    rf: float = 0.0,
    budgets: dict[str, np.ndarray | None],
    allow_short_selling: bool = False,
    alpha: float = 0.05,
    kappa: float = 0.30,
) -> dict[str, RiskParityResult]:
    """Compare multiple risk budget allocations.

    Args:
        prices: Multi-asset price DataFrame (one column per ticker).
        rm: Risk measure code (see ``RP_RISK_MEASURES``).
        rf: Annual risk-free rate.
        budgets: Mapping of label to budget vector (or ``None`` for
            equal budget).
        allow_short_selling: Allow negative weights if ``True``.
        alpha: Significance level for tail-risk measures.
        kappa: Deformation parameter for relativistic measures.

    Returns:
        Mapping of label to ``RiskParityResult``.
    """
    results: dict[str, RiskParityResult] = {}
    for label, budget in budgets.items():
        results[label] = optimize_risk_parity(
            prices=prices,
            rm=rm,
            rf=rf,
            budget=budget,
            allow_short_selling=allow_short_selling,
            alpha=alpha,
            kappa=kappa,
        )
    return results
