"""Hierarchical Clustering Portfolio — HRP & HERC optimization via riskfolio-lib."""

__all__ = [
    "CODEPENDENCE_MEASURES",
    "HC_MODELS",
    "HC_RISK_MEASURES",
    "LINKAGE_METHODS",
    "HCResult",
    "optimize_hierarchical_clustering",
]

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import riskfolio as rp

from investment_portfolio.mpt import _infer_ann_factor

# Risk measures requiring EXP or POW cone solvers.
_EXP_POW_MEASURES: frozenset[str] = frozenset(
    {
        "GMD",
        "TG",
        "RLVaR",
        "TGRG",
        "RVRG",
        "RLDaR",
        "RLDaR_Rel",
        "EVaR",
        "EVRG",
        "EDaR",
        "EDaR_Rel",
    }
)

HC_RISK_MEASURES: dict[str, str] = {
    # Dispersion (11)
    "Standard Deviation": "vol",
    "Variance": "MV",
    "Square Root Kurtosis": "KT",
    "Mean Absolute Deviation": "MAD",
    "Gini Mean Difference": "GMD",
    "Value at Risk Range": "VRG",
    "CVaR Range": "CVRG",
    "Tail Gini Range": "TGRG",
    "EVaR Range": "EVRG",
    "RLVaR Range": "RVRG",
    "Range": "RG",
    # Downside (11)
    "Semi Standard Deviation": "MSV",
    "Square Root Semi Kurtosis": "SKT",
    "First Lower Partial Moment": "FLPM",
    "Second Lower Partial Moment": "SLPM",
    "Value at Risk": "VaR",
    "Conditional Value at Risk": "CVaR",
    "Entropic Value at Risk": "EVaR",
    "Relativistic Value at Risk": "RLVaR",
    "Tail Gini": "TG",
    "Worst Realization": "WR",
    # Drawdown — uncompounded (7)
    "Average Drawdown": "ADD",
    "Ulcer Index": "UCI",
    "Drawdown at Risk": "DaR",
    "Conditional Drawdown at Risk": "CDaR",
    "Entropic Drawdown at Risk": "EDaR",
    "Relativistic Drawdown at Risk": "RLDaR",
    "Maximum Drawdown": "MDD",
    # Drawdown — compounded (7)
    "Average Drawdown (Relative)": "ADD_Rel",
    "Ulcer Index (Relative)": "UCI_Rel",
    "Drawdown at Risk (Relative)": "DaR_Rel",
    "Conditional Drawdown at Risk (Relative)": "CDaR_Rel",
    "Entropic Drawdown at Risk (Relative)": "EDaR_Rel",
    "Relativistic Drawdown at Risk (Relative)": "RLDaR_Rel",
    "Maximum Drawdown (Relative)": "MDD_Rel",
}

HC_MODELS: dict[str, str] = {
    "HRP (Hierarchical Risk Parity)": "HRP",
    "HERC (Hierarchical Equal Risk Contribution)": "HERC",
}

CODEPENDENCE_MEASURES: dict[str, str] = {
    "Pearson": "pearson",
    "Spearman": "spearman",
    "Kendall": "kendall",
    "Gerber Statistic 1": "gerber1",
    "Gerber Statistic 2": "gerber2",
    "Absolute Value": "abs_pearson",
    "Absolute Value (Spearman)": "abs_spearman",
    "Absolute Value (Kendall)": "abs_kendall",
    "Distance Correlation": "distance",
    "Mutual Information": "mutual_info",
    "Tail Dependence": "tail",
}

LINKAGE_METHODS: dict[str, str] = {
    "Single": "single",
    "Complete": "complete",
    "Average": "average",
    "Weighted": "weighted",
    "Centroid": "centroid",
    "Median": "median",
    "Ward": "ward",
    "DBHT": "DBHT",
}


@dataclass(frozen=True, slots=True)
class HCResult:
    """Hierarchical clustering portfolio allocation with clustering metadata.

    Attributes:
        weights: Mapping of ticker to allocation weight.
        expected_return: Annualized expected return.
        volatility: Annualized portfolio volatility.
        sharpe_ratio: Annualized Sharpe ratio.
        clustering: Scipy linkage matrix of shape ``(N-1, 4)``.
        asset_order: Leaf ordering from the dendrogram.
        k: Number of clusters (``None`` for HRP, auto-detected for HERC).
        model: Model used — ``"HRP"`` or ``"HERC"``.
    """

    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    clustering: np.ndarray
    asset_order: list[str]
    k: int | None
    model: str


def optimize_hierarchical_clustering(
    *,
    prices: pd.DataFrame,
    model: str = "HRP",
    codependence: str = "pearson",
    linkage: str = "single",
    rm: str = "vol",
    rf: float = 0.0,
    alpha: float = 0.05,
    kappa: float = 0.30,
    max_k: int = 10,
    leaf_order: bool = True,
) -> HCResult:
    """Optimize a portfolio using hierarchical clustering (HRP or HERC).

    Args:
        prices: Multi-asset price DataFrame (one column per ticker).
        model: Clustering model — ``"HRP"`` or ``"HERC"``.
        codependence: Codependence measure (see ``CODEPENDENCE_MEASURES``).
        linkage: Linkage method (see ``LINKAGE_METHODS``).
        rm: Risk measure code (see ``HC_RISK_MEASURES``).
        rf: Annual risk-free rate.
        alpha: Significance level for tail-risk measures.
        kappa: Deformation parameter for relativistic measures.
        max_k: Maximum number of clusters for HERC.
        leaf_order: Whether to optimize leaf ordering in the dendrogram.

    Returns:
        ``HCResult`` with optimal weights, metrics, and clustering data.

    Raises:
        RuntimeError: If the optimization fails to converge.
    """
    returns = prices.pct_change().dropna()
    ann_factor = _infer_ann_factor(index=returns.index)

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
        model=model,
        codependence=codependence,
        method_mu="hist",
        method_cov="hist",
        rm=rm,
        rf=rf / ann_factor,
        linkage=linkage,
        max_k=max_k,
        leaf_order=leaf_order,
    )

    if w is None:
        msg = "hierarchical clustering optimization failed to converge"
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

    return HCResult(
        weights=weights_dict,
        expected_return=port_return,
        volatility=port_vol,
        sharpe_ratio=sharpe,
        clustering=clustering,
        asset_order=asset_order,
        k=k,
        model=model,
    )
