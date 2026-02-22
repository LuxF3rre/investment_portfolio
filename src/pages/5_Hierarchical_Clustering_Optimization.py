"""Hierarchical Clustering Optimization page (Streamlit UI)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from investment_portfolio.hcp import (
    CODEPENDENCE_MEASURES,
    HC_MODELS,
    HC_RISK_MEASURES,
    LINKAGE_METHODS,
    optimize_hierarchical_clustering,
)
from investment_portfolio.mpt import calculate_asset_statistics, fetch_multi_history

_MIN_ACTIVE_WEIGHT: float = 0.005

# Risk measure codes that use the alpha (confidence) parameter.
_ALPHA_MEASURES: frozenset[str] = frozenset(
    {
        "CVaR",
        "CVRG",
        "TG",
        "TGRG",
        "EVaR",
        "EVRG",
        "RLVaR",
        "RVRG",
        "VaR",
        "VRG",
        "CDaR",
        "EDaR",
        "RLDaR",
        "DaR",
        "CDaR_Rel",
        "EDaR_Rel",
        "RLDaR_Rel",
        "DaR_Rel",
    }
)

# Risk measure codes that use the kappa (deformation) parameter.
_KAPPA_MEASURES: frozenset[str] = frozenset({"RLVaR", "RVRG", "RLDaR", "RLDaR_Rel"})

st.set_page_config(
    page_title="Hierarchical Clustering Optimization",
    page_icon=":deciduous_tree:",
    layout="wide",
)

st.title("Hierarchical Clustering Optimization")

# -- Sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")
    tickers_input = st.text_input(
        "Tickers (comma-separated)",
        value="AAPL, MSFT, GOOGL, AMZN",
        help="Enter 2 or more yfinance ticker symbols separated by commas.",
    )
    period = st.selectbox(
        "History period",
        ["1y", "2y", "5y", "10y", "max"],
        index=2,
        help=(
            "How far back to look for historical data. Longer periods capture "
            "more market regimes but may include outdated patterns."
        ),
    )
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0,
        max_value=0.20,
        value=0.04,
        step=0.005,
        format="%.3f",
        help=(
            "Annual return on a risk-free asset (e.g. Treasury bill yield). "
            "Used in Sharpe ratio calculations."
        ),
    )

    model_label = st.selectbox(
        "Model",
        list(HC_MODELS.keys()),
        help=(
            "HRP uses top-down recursive bisection — works without clusters. "
            "HERC uses flat clusters for more balanced risk allocation."
        ),
    )
    model_code = HC_MODELS[model_label]

    rm_label = st.selectbox(
        "Risk measure",
        list(HC_RISK_MEASURES.keys()),
        help=(
            "How risk is measured when allocating weights. "
            "HCPortfolio supports 35 risk measures across dispersion, "
            "downside, and drawdown categories."
        ),
    )
    rm_code = HC_RISK_MEASURES[rm_label]

    alpha = 0.05
    if rm_code in _ALPHA_MEASURES:
        confidence = st.slider(
            "Confidence level (%)",
            min_value=90,
            max_value=99,
            value=95,
            step=1,
            help=(
                "Confidence level for tail-risk measures. 95 % means "
                "we focus on the worst 5 % of scenarios. Higher values "
                "are more conservative."
            ),
        )
        alpha = (100 - confidence) / 100

    kappa = 0.30
    if rm_code in _KAPPA_MEASURES:
        kappa = st.slider(
            "Deformation (kappa)",
            min_value=0.01,
            max_value=0.99,
            value=0.30,
            step=0.01,
            help=(
                "Controls conservatism of relativistic measures. "
                "Lower values behave like EVaR/EDaR, higher values "
                "approach worst-case scenarios."
            ),
        )

    codep_label = st.selectbox(
        "Codependence measure",
        list(CODEPENDENCE_MEASURES.keys()),
        index=0,
        help=(
            "How similarity between assets is measured for clustering. "
            "Pearson captures linear correlation; Spearman and Kendall "
            "handle nonlinear relationships."
        ),
    )
    codep_code = CODEPENDENCE_MEASURES[codep_label]

    linkage_label = st.selectbox(
        "Linkage method",
        list(LINKAGE_METHODS.keys()),
        index=6,  # Ward
        help=(
            "How clusters are merged in the hierarchy. Ward minimizes "
            "within-cluster variance and tends to produce balanced trees."
        ),
    )
    linkage_code = LINKAGE_METHODS[linkage_label]

    max_k = 10
    if model_code == "HERC":
        max_k = st.slider(
            "Max clusters",
            min_value=2,
            max_value=20,
            value=10,
            step=1,
            help=(
                "Maximum number of clusters for HERC. The optimal number "
                "is auto-detected up to this limit."
            ),
        )

    run = st.button("Run", type="primary")


# -- Cached data fetch --------------------------------------------------------
@st.cache_data(show_spinner="Fetching data…")
def _fetch(*, tickers: list[str], period: str) -> pd.DataFrame:
    return fetch_multi_history(tickers=tickers, period=period)


# -- Computation (only on Run click) ------------------------------------------
_sidebar_tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

if run:
    tickers = _sidebar_tickers
    if len(tickers) < 2:  # noqa: PLR2004
        st.error("Please enter at least 2 tickers.")
        st.stop()

    try:
        prices = _fetch(tickers=tickers, period=period)
    except Exception as exc:
        st.error(f"Could not fetch data: {exc}")
        st.stop()

    with st.spinner("Optimizing hierarchical clustering portfolio…"):
        result = optimize_hierarchical_clustering(
            prices=prices,
            model=model_code,
            codependence=codep_code,
            linkage=linkage_code,
            rm=rm_code,
            rf=risk_free_rate,
            alpha=alpha,
            kappa=kappa,
            max_k=max_k,
        )
        asset_stats = calculate_asset_statistics(prices=prices)

    st.session_state["hc_has_run"] = True
    st.session_state["hc_data"] = {
        "tickers": tickers,
        "prices": prices,
        "result": result,
        "asset_stats": asset_stats,
    }

_has_run: bool = (
    st.session_state.get("hc_has_run", False) and "hc_data" in st.session_state
)

# -- Guide --------------------------------------------------------------------
with st.expander(
    "What is Hierarchical Clustering Optimization?",
    expanded=(not _has_run),
):
    st.markdown(
        """
Hierarchical Clustering Optimization uses **tree-based allocation**
instead of traditional covariance matrix inversion. This makes it
more robust when dealing with many assets or noisy correlations.

**Two models are available:**

- **HRP (Hierarchical Risk Parity)** — Builds a hierarchical tree
  of assets, then allocates top-down via recursive bisection.
  No cluster count needed.
- **HERC (Hierarchical Equal Risk Contribution)** — Extends HRP
  by identifying flat clusters and equalizing risk contribution
  within and between clusters.

**Key advantages over traditional optimization:**
- Does **not** invert the covariance matrix — avoids numerical
  instability with many correlated assets.
- Naturally respects the hierarchical structure of asset
  relationships.
- More stable allocations — small changes in input data don't
  cause dramatic weight swings.
- Works well even with more assets than observations.
"""
    )

with st.expander("Models explained"):
    st.markdown(
        """\
**HRP (Hierarchical Risk Parity)** by Marcos Lopez de Prado (2016):
1. Compute a distance matrix from asset codependence.
2. Build a hierarchical tree using the chosen linkage method.
3. Reorder assets by dendrogram leaf ordering.
4. Allocate weights top-down via recursive bisection — at each
   split, inversely proportional to cluster variance.

*Best for:* Simple, stable allocations without cluster tuning.

---

**HERC (Hierarchical Equal Risk Contribution)** by Raffinot (2018):
1. Same tree construction as HRP.
2. Cut the tree into *k* flat clusters (auto-detected).
3. Allocate risk equally **between** clusters and **within**
   each cluster.

*Best for:* When you want risk-balanced clusters, especially
with assets that form natural groups (e.g. sectors, geographies).
"""
    )

with st.expander("Risk measures explained"):
    st.markdown(
        """\
There are **35 risk measures** grouped into four categories.

---

**Dispersion measures (11)** — how spread out are returns?

- **Standard Deviation** (`vol`) — The classic volatility measure.
- **Variance** (`MV`) — Squared standard deviation.
- **Square Root Kurtosis** — Captures extreme outlier risk.
- **Mean Absolute Deviation** — Less influenced by outliers.
- **Gini Mean Difference** — Average pairwise return difference.
- **Value at Risk Range** — Spread of the VaR distribution.
- **CVaR Range** — Spread between best and worst tails.
- **Tail Gini Range** — Gini focused on extreme returns.
- **EVaR Range** — Information-theoretic tail spread.
- **RLVaR Range** — Flexible tail spread measure.
- **Range** — Simple max minus min return.

---

**Downside measures (11)** — how bad can losses get?

- **Semi Standard Deviation** — Volatility of negative returns only.
- **Square Root Semi Kurtosis** — Extreme downside outlier risk.
- **First Lower Partial Moment** — Frequency of below-target returns.
- **Second Lower Partial Moment** — Magnitude of below-target returns.
- **Value at Risk** — Maximum expected loss at a confidence level.
- **Conditional Value at Risk** — Average loss beyond VaR.
- **Entropic Value at Risk** — Upper bound on CVaR.
- **Relativistic Value at Risk** — Between EVaR and worst case.
- **Tail Gini** — Alternative to CVaR with different tail weighting.
- **Worst Realization** — The single worst return observed.

---

**Drawdown measures — uncompounded (7)**

- **Average Drawdown** — Mean peak-to-trough decline.
- **Ulcer Index** — Depth and duration of drawdowns.
- **Drawdown at Risk** — Maximum drawdown at a confidence level.
- **Conditional Drawdown at Risk** — Average of worst drawdowns.
- **Entropic Drawdown at Risk** — Upper bound on CDaR.
- **Relativistic Drawdown at Risk** — Between EDaR and worst case.
- **Maximum Drawdown** — The deepest single drawdown.

---

**Drawdown measures — compounded (7)**

Same as above but computed on compounded (geometric) returns.
Suffixed with `_Rel` internally.

---

**Quick recommendations:**
- **New to this?** *Standard Deviation* — most intuitive.
- **Worried about crashes?** *Conditional Value at Risk.*
- **Long-term investor?** *Conditional Drawdown at Risk.*
"""
    )

with st.expander("Codependence and linkage explained"):
    st.markdown(
        """\
**Codependence measures** determine how similarity between
assets is computed before clustering.

- **Pearson** — Linear correlation. Fast and widely understood.
  *Recommended default.*
- **Spearman** — Rank correlation. Robust to outliers.
- **Kendall** — Concordance-based rank correlation.
- **Gerber 1 & 2** — Focus on co-movements beyond a threshold.
- **Absolute Value** variants — Use absolute correlations
  (groups anti-correlated assets together).
- **Distance Correlation** — Captures nonlinear dependencies.
- **Mutual Information** — Information-theoretic dependence.
- **Tail Dependence** — Correlation in extreme events.

---

**Linkage methods** determine how clusters merge.

- **Single** — Nearest-neighbor. Tends to produce chain-like
  clusters.
- **Complete** — Farthest-neighbor. Compact, spherical clusters.
- **Average** — Mean pairwise distance. Balanced approach.
- **Weighted** — Like average but weights by cluster size.
- **Centroid** — Distance between cluster centroids.
- **Median** — Like centroid but robust to outliers.
- **Ward** — Minimizes within-cluster variance.
  *Recommended default.* Produces balanced, even-sized clusters.
- **DBHT** — Direct Bubble Hierarchical Tree. Data-driven
  approach that doesn't require a distance metric choice.

**Recommendation:** Start with **Pearson + Ward** for most
portfolios. Switch to **Spearman** if you suspect nonlinear
relationships, and try **DBHT** for a fully data-driven approach.
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()

# -- Display from session state -----------------------------------------------
_d = st.session_state["hc_data"]
_result = _d["result"]
_tickers = _d["tickers"]

top_ticker = max(_result.weights, key=_result.weights.get)
top_weight = _result.weights[top_ticker]
active_count = sum(1 for w in _result.weights.values() if abs(w) > _MIN_ACTIVE_WEIGHT)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Expected Return", f"{_result.expected_return:.2%}")
c2.metric("Volatility", f"{_result.volatility:.2%}")
c3.metric("Sharpe Ratio", f"{_result.sharpe_ratio:.3f}")
c4.metric("Top Holding", top_ticker, f"{top_weight:.1%}")
c5.metric("Active Positions", f"{active_count} / {len(_tickers)}")

tab_dendro, tab_weights, tab_corr, tab_stats = st.tabs(
    [
        "Dendrogram",
        "Portfolio Weights",
        "Correlation Matrix",
        "Asset Statistics",
    ]
)

with tab_dendro:
    # Build dendrogram data without plotting
    dendro_data = dendrogram(
        _result.clustering,
        labels=_result.asset_order,
        no_plot=True,
        leaf_rotation=90,
    )

    fig_dendro = go.Figure()

    # Draw U-shaped links from icoord/dcoord
    for xs, ys in zip(dendro_data["icoord"], dendro_data["dcoord"], strict=True):
        fig_dendro.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={"color": "teal", "width": 1.5},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Add leaf labels
    leaf_labels = dendro_data["ivl"]
    n_leaves = len(leaf_labels)
    tick_positions = [5 + 10 * i for i in range(n_leaves)]

    fig_dendro.update_layout(
        title="Hierarchical Clustering Dendrogram",
        xaxis={
            "tickvals": tick_positions,
            "ticktext": leaf_labels,
            "tickangle": 45,
        },
        yaxis_title="Distance",
        height=500,
        showlegend=False,
    )
    st.plotly_chart(fig_dendro, width="stretch")

    if _result.k is not None:
        st.caption(f"HERC detected **{_result.k}** clusters.")

with tab_weights:
    names = list(_result.weights.keys())
    vals = list(_result.weights.values())
    fig_w = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker_color="teal",
            text=[f"{v:.1%}" for v in vals],
            textposition="outside",
        )
    )
    fig_w.update_layout(
        title="Portfolio Allocation",
        yaxis_title="Weight",
        yaxis_tickformat=".0%",
        height=400,
    )
    st.plotly_chart(fig_w, width="stretch")

with tab_corr:
    _prices = _d["prices"]
    log_returns = np.log(_prices / _prices.shift(1)).dropna()
    corr_matrix = log_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdYlGn",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig_corr.update_layout(
        title="Return Correlation Matrix",
        height=500,
    )
    st.plotly_chart(fig_corr, width="stretch")

with tab_stats:
    _stats = _d["asset_stats"]
    rows = []
    for t in _tickers:
        s = _stats[t]
        rows.append(
            {
                "Ticker": t,
                "Annualized Return": f"{s['annualized_return']:.2%}",
                "Annualized Volatility": f"{s['annualized_volatility']:.2%}",
            }
        )
    st.table(pd.DataFrame(rows))

st.divider()
st.markdown(
    "<div style='text-align: center'>Made with ❤ by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
