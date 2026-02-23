"""Nested Clustered Optimization — NCO via riskfolio-lib."""

__all__ = [
    "NCO_OBJECTIVES",
    "NCO_RISK_MEASURES",
    "NCOResult",
    "optimize_nco",
]

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import riskfolio as rp

from investment_portfolio.mpt import infer_ann_factor

NCO_OBJECTIVES: dict[str, str] = {
    "Minimum Risk": "MinRisk",
    "Maximum Risk Adjusted Return Ratio": "Sharpe",
    "Maximum Utility Function": "Utility",
    "Equal Risk Contribution": "ERC",
}

# NCO supports 24 risk measures — excludes vol, VaR, DaR, VRG,
# and all _Rel (compounded drawdown) variants from HC_RISK_MEASURES.
NCO_RISK_MEASURES: dict[str, str] = {
    # Dispersion (8)
    "Variance": "MV",
    "Square Root Kurtosis": "KT",
    "Mean Absolute Deviation": "MAD",
    "Gini Mean Difference": "GMD",
    "CVaR Range": "CVRG",
    "Tail Gini Range": "TGRG",
    "EVaR Range": "EVRG",
    "RLVaR Range": "RVRG",
    "Range": "RG",
    # Downside (8)
    "Semi Standard Deviation": "MSV",
    "Square Root Semi Kurtosis": "SKT",
    "First Lower Partial Moment": "FLPM",
    "Second Lower Partial Moment": "SLPM",
    "Conditional Value at Risk": "CVaR",
    "Entropic Value at Risk": "EVaR",
    "Relativistic Value at Risk": "RLVaR",
    "Tail Gini": "TG",
    "Worst Realization": "WR",
    # Drawdown — uncompounded (7)
    "Average Drawdown": "ADD",
    "Ulcer Index": "UCI",
    "Conditional Drawdown at Risk": "CDaR",
    "Entropic Drawdown at Risk": "EDaR",
    "Relativistic Drawdown at Risk": "RLDaR",
    "Maximum Drawdown": "MDD",
}

# Risk measures requiring EXP or POW cone solvers (no _Rel variants for NCO).
_EXP_POW_MEASURES: frozenset[str] = frozenset(
    {
        "GMD",
        "TG",
        "RLVaR",
        "TGRG",
        "RVRG",
        "RLDaR",
        "EVaR",
        "EVRG",
        "EDaR",
    }
)


@dataclass(frozen=True, slots=True)
class NCOResult:
    """Nested Clustered Optimization portfolio with clustering metadata.

    Attributes:
        weights: Mapping of ticker to allocation weight.
        expected_return: Annualized expected return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        clustering: Scipy linkage matrix of shape ``(N-1, 4)``.
        asset_order: Leaf ordering from the dendrogram.
        k: Number of clusters used by NCO.
        objective: Objective code used for optimization.
    """

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    clustering: np.ndarray
    asset_order: list[str]
    k: int | None
    objective: str


def optimize_nco(
    *,
    prices: pd.DataFrame,
    obj: str = "MinRisk",
    codependence: str = "pearson",
    linkage: str = "single",
    rm: str = "MV",
    rf: float = 0.0,
    risk_aversion: float = 2.0,
    alpha: float = 0.05,
    kappa: float = 0.30,
    max_k: int = 10,
    leaf_order: bool = True,
) -> NCOResult:
    """Optimize a portfolio using Nested Clustered Optimization (NCO).

    NCO combines hierarchical clustering with objective-based intra-cluster
    and inter-cluster optimization, supporting 4 objectives and 24 risk measures.

    Args:
        prices: Multi-asset price DataFrame (one column per ticker).
        obj: Objective code — ``"MinRisk"``, ``"Sharpe"``, ``"Utility"``,
            or ``"ERC"``.
        codependence: Codependence measure (see ``CODEPENDENCE_MEASURES``).
        linkage: Linkage method (see ``LINKAGE_METHODS``).
        rm: Risk measure code (see ``NCO_RISK_MEASURES``).
        rf: Annual risk-free rate.
        risk_aversion: Risk aversion parameter (used when *obj* is
            ``"Utility"``).
        alpha: Significance level for tail-risk measures.
        kappa: Deformation parameter for relativistic measures.
        max_k: Maximum number of clusters.
        leaf_order: Whether to optimize leaf ordering in the dendrogram.

    Returns:
        ``NCOResult`` with optimal weights, metrics, and clustering data.

    Raises:
        RuntimeError: If the optimization fails to converge.
    """
    returns = prices.pct_change().dropna()
    ann_factor = infer_ann_factor(index=returns.index)

    if rm in _EXP_POW_MEASURES:
        solver_rl = "MOSEK"
        solvers = ["MOSEK", "CLARABEL", "SCS"]
    else:
        solver_rl = "CLARABEL"
        solvers = ["CLARABEL", "SCS", "ECOS"]

    hc_port = rp.HCPortfolio(
        returns=returns,
        alpha=alpha,
        kappa=kappa,
        solver_rl=solver_rl,
        solvers=solvers,
    )

    w = hc_port.optimization(
        model="NCO",
        obj=obj,
        codependence=codependence,
        method_mu="hist",
        method_cov="hist",
        rm=rm,
        rf=rf / ann_factor,
        l=risk_aversion,
        linkage=linkage,
        max_k=max_k,
        leaf_order=leaf_order,
    )

    if w is None:
        msg = "NCO optimization failed to converge"
        raise RuntimeError(msg)

    tickers = list(w.index)
    weights_dict = dict(zip(tickers, w["weights"].tolist(), strict=True))
    w_arr = w.values.flatten()

    # Annualized metrics from the fitted model
    if hc_port.mu is None or hc_port.cov is None:
        msg = "asset statistics could not be computed"
        raise RuntimeError(msg)
    mu: np.ndarray = hc_port.mu.values.flatten()
    cov_mat: np.ndarray = hc_port.cov.values

    port_return = float(mu @ w_arr) * ann_factor
    port_vol = float(np.sqrt(w_arr @ cov_mat @ w_arr)) * math.sqrt(ann_factor)
    sharpe = (port_return - rf) / port_vol if port_vol > 0 else 0.0

    if hc_port.clustering is None or hc_port.asset_order is None:
        msg = "clustering data not available after optimization"
        raise RuntimeError(msg)
    clustering: np.ndarray = hc_port.clustering
    asset_order: list[str] = list(hc_port.asset_order)
    k: int | None = getattr(hc_port, "k", None)

    return NCOResult(
        weights=weights_dict,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        clustering=clustering,
        asset_order=asset_order,
        k=k,
        objective=obj,
    )
