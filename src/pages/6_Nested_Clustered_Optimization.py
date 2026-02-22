"""Nested Clustered Optimization page (Streamlit UI)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import dendrogram

from investment_portfolio.hcp import CODEPENDENCE_MEASURES, LINKAGE_METHODS
from investment_portfolio.mpt import calculate_asset_statistics, fetch_multi_history
from investment_portfolio.nco import NCO_OBJECTIVES, NCO_RISK_MEASURES, optimize_nco

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
        "CDaR",
        "EDaR",
        "RLDaR",
    }
)

# Risk measure codes that use the kappa (deformation) parameter.
_KAPPA_MEASURES: frozenset[str] = frozenset({"RLVaR", "RVRG", "RLDaR"})

st.set_page_config(
    page_title="Nested Clustered Optimization",
    page_icon=":link:",
    layout="wide",
)

st.title("Nested Clustered Optimization")

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

    obj_label = st.selectbox(
        "Objective function",
        list(NCO_OBJECTIVES.keys()),
        help=(
            "The optimization goal applied within and across clusters. "
            "MinRisk minimizes portfolio risk; Sharpe maximizes risk-adjusted "
            "return; Utility maximizes expected utility; ERC equalizes risk "
            "contribution."
        ),
    )
    obj_code = NCO_OBJECTIVES[obj_label]

    rm_label = st.selectbox(
        "Risk measure",
        list(NCO_RISK_MEASURES.keys()),
        help=(
            "How risk is measured when optimizing weights. "
            "NCO supports 24 risk measures across dispersion, "
            "downside, and drawdown categories."
        ),
    )
    rm_code = NCO_RISK_MEASURES[rm_label]

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

    risk_aversion = 2.0
    if obj_code == "Utility":
        risk_aversion = st.slider(
            "Risk aversion (\u03bb)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help=(
                "Controls the tradeoff between return and risk in the "
                "utility function. Higher values penalize risk more heavily."
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

    max_k = st.slider(
        "Max clusters",
        min_value=2,
        max_value=20,
        value=10,
        step=1,
        help=(
            "Maximum number of clusters. NCO always uses clusters — "
            "the optimal number is auto-detected up to this limit."
        ),
    )

    run = st.button("Run", type="primary")


# -- Cached data fetch --------------------------------------------------------
@st.cache_data(show_spinner="Fetching data\u2026")
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

    with st.spinner("Optimizing nested clustered portfolio\u2026"):
        result = optimize_nco(
            prices=prices,
            obj=obj_code,
            codependence=codep_code,
            linkage=linkage_code,
            rm=rm_code,
            rf=risk_free_rate,
            risk_aversion=risk_aversion,
            alpha=alpha,
            kappa=kappa,
            max_k=max_k,
        )
        asset_stats = calculate_asset_statistics(prices=prices)

    st.session_state["nco_has_run"] = True
    st.session_state["nco_data"] = {
        "tickers": tickers,
        "prices": prices,
        "result": result,
        "asset_stats": asset_stats,
    }

_has_run: bool = (
    st.session_state.get("nco_has_run", False) and "nco_data" in st.session_state
)

# -- Guide --------------------------------------------------------------------
with st.expander(
    "What is Nested Clustered Optimization?",
    expanded=(not _has_run),
):
    st.markdown(
        """
Nested Clustered Optimization (NCO) is a hybrid method that combines
**hierarchical clustering** with **objective-based optimization**.
Unlike HRP/HERC which use heuristic allocation rules, NCO applies a
full optimization step **within** each cluster and then **across**
clusters.

**How NCO works:**

1. Build a hierarchical tree of assets using codependence and linkage.
2. Cut the tree into *k* clusters (auto-detected).
3. **Intra-cluster optimization** — solve the chosen objective
   (e.g. minimize risk, maximize Sharpe) within each cluster
   independently.
4. **Inter-cluster optimization** — treat each cluster as a single
   "meta-asset" and optimize across clusters using the same objective.
5. Combine the two levels of weights into final asset allocations.

**Key advantages:**
- Reduces dimensionality via clustering — the optimizer only sees
  a small number of assets at each level.
- More robust than full-universe optimization when assets are
  correlated or data is noisy.
- Supports objective functions (unlike HRP) — you can target
  minimum risk, maximum Sharpe, utility maximization, or equal
  risk contribution.
"""
    )

with st.expander("Objective functions explained"):
    st.markdown(
        """\
NCO supports **4 objective functions** applied within and across clusters.

---

**Minimum Risk** (`MinRisk`) — Minimize total portfolio risk as
measured by the chosen risk measure. No return forecast needed.
*Best for:* Conservative investors seeking the least risky allocation.

---

**Maximum Risk Adjusted Return Ratio** (`Sharpe`) — Maximize the
ratio of expected return to risk (generalized Sharpe ratio). Works
with any risk measure, not just standard deviation.
*Best for:* Balancing return and risk in a single metric.

---

**Maximum Utility Function** (`Utility`) — Maximize
*E[R] - \u03bb \u00d7 Risk*, where \u03bb is the risk aversion parameter.
Higher \u03bb penalizes risk more, producing more conservative allocations.
*Best for:* Investors with a specific risk tolerance who want to
control the return-risk tradeoff directly.

---

**Equal Risk Contribution** (`ERC`) — Allocate so that every asset
contributes equally to the total portfolio risk. Similar to Risk Parity
but applied within the NCO framework.
*Best for:* Diversification-focused investors who want balanced risk.
"""
    )

with st.expander("Risk measures explained"):
    st.markdown(
        """\
NCO supports **24 risk measures** grouped into three categories.

---

**Dispersion measures (9)** \u2014 how spread out are returns?

- **Variance** (`MV`) \u2014 Classic squared-deviation risk.
- **Square Root Kurtosis** \u2014 Captures extreme outlier risk.
- **Mean Absolute Deviation** \u2014 Less influenced by outliers.
- **Gini Mean Difference** \u2014 Average pairwise return difference.
- **CVaR Range** \u2014 Spread between best and worst tails.
- **Tail Gini Range** \u2014 Gini focused on extreme returns.
- **EVaR Range** \u2014 Information-theoretic tail spread.
- **RLVaR Range** \u2014 Flexible tail spread measure.
- **Range** \u2014 Simple max minus min return.

---

**Downside measures (9)** \u2014 how bad can losses get?

- **Semi Standard Deviation** \u2014 Volatility of negative returns only.
- **Square Root Semi Kurtosis** \u2014 Extreme downside outlier risk.
- **First Lower Partial Moment** \u2014 Frequency of below-target returns.
- **Second Lower Partial Moment** \u2014 Magnitude of below-target returns.
- **Conditional Value at Risk** \u2014 Average loss beyond VaR.
- **Entropic Value at Risk** \u2014 Upper bound on CVaR.
- **Relativistic Value at Risk** \u2014 Between EVaR and worst case.
- **Tail Gini** \u2014 Alternative to CVaR with different tail weighting.
- **Worst Realization** \u2014 The single worst return observed.

---

**Drawdown measures \u2014 uncompounded (6)**

- **Average Drawdown** \u2014 Mean peak-to-trough decline.
- **Ulcer Index** \u2014 Depth and duration of drawdowns.
- **Conditional Drawdown at Risk** \u2014 Average of worst drawdowns.
- **Entropic Drawdown at Risk** \u2014 Upper bound on CDaR.
- **Relativistic Drawdown at Risk** \u2014 Between EDaR and worst case.
- **Maximum Drawdown** \u2014 The deepest single drawdown.

---

**Quick recommendations:**
- **New to this?** *Variance* \u2014 most intuitive.
- **Worried about crashes?** *Conditional Value at Risk.*
- **Long-term investor?** *Conditional Drawdown at Risk.*
"""
    )

with st.expander("Codependence and linkage explained"):
    st.markdown(
        """\
**Codependence measures** determine how similarity between
assets is computed before clustering.

- **Pearson** \u2014 Linear correlation. Fast and widely understood.
  *Recommended default.*
- **Spearman** \u2014 Rank correlation. Robust to outliers.
- **Kendall** \u2014 Concordance-based rank correlation.
- **Gerber 1 & 2** \u2014 Focus on co-movements beyond a threshold.
- **Absolute Value** variants \u2014 Use absolute correlations
  (groups anti-correlated assets together).
- **Distance Correlation** \u2014 Captures nonlinear dependencies.
- **Mutual Information** \u2014 Information-theoretic dependence.
- **Tail Dependence** \u2014 Correlation in extreme events.

---

**Linkage methods** determine how clusters merge.

- **Single** \u2014 Nearest-neighbor. Tends to produce chain-like
  clusters.
- **Complete** \u2014 Farthest-neighbor. Compact, spherical clusters.
- **Average** \u2014 Mean pairwise distance. Balanced approach.
- **Weighted** \u2014 Like average but weights by cluster size.
- **Centroid** \u2014 Distance between cluster centroids.
- **Median** \u2014 Like centroid but robust to outliers.
- **Ward** \u2014 Minimizes within-cluster variance.
  *Recommended default.* Produces balanced, even-sized clusters.
- **DBHT** \u2014 Direct Bubble Hierarchical Tree. Data-driven
  approach that doesn\u2019t require a distance metric choice.

**Recommendation:** Start with **Pearson + Ward** for most
portfolios. Switch to **Spearman** if you suspect nonlinear
relationships, and try **DBHT** for a fully data-driven approach.
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()

# -- Display from session state -----------------------------------------------
_d = st.session_state["nco_data"]
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
        st.caption(f"NCO detected **{_result.k}** clusters.")

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
    "<div style='text-align: center'>Made with \u2764 by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
