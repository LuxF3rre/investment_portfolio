"""Portfolio Optimization page — riskfolio-lib powered (Streamlit UI)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from investment_portfolio.mpt import (
    OBJECTIVES,
    RISK_MEASURES,
    build_efficient_frontier,
    calculate_asset_statistics,
    fetch_multi_history,
    optimize_portfolio,
)

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
    page_title="Portfolio Optimization",
    page_icon=":balance_scale:",
    layout="wide",
)

st.title("Portfolio Optimization")

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
            "Used in Sharpe ratio and utility calculations."
        ),
    )
    allow_short = st.checkbox(
        "Allow short selling",
        value=False,
        help=(
            "If enabled, the optimizer can assign negative weights (betting "
            "against an asset). Most retail portfolios keep this off."
        ),
    )

    model_label = st.selectbox(
        "Model",
        ["Mean Risk", "Logarithmic Mean Risk (Kelly Criterion)"],
        help=(
            "Mean Risk is the standard approach. Kelly Criterion maximizes "
            "long-term growth rate — more aggressive, best for long horizons."
        ),
    )
    kelly: str | None = None if model_label == "Mean Risk" else "exact"

    obj_label = st.selectbox(
        "Objective function",
        list(OBJECTIVES.keys()),
        help=(
            "What the optimizer tries to achieve. See the guide on the main "
            "panel for details on each objective."
        ),
    )
    obj_code = OBJECTIVES[obj_label]

    rm_label = st.selectbox(
        "Risk measure",
        list(RISK_MEASURES.keys()),
        help=(
            "How risk is measured. Different measures emphasize different "
            "aspects of risk. See the guide on the main panel for details."
        ),
    )
    rm_code = RISK_MEASURES[rm_label]

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
                "Lower values behave like CVaR/CDaR, higher values "
                "approach worst-case scenarios."
            ),
        )

    risk_aversion = 1.0
    if obj_label == "Maximum Utility Function":
        risk_aversion = st.slider(
            "Risk aversion",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help=(
                "Higher values penalize risk more heavily, producing more "
                "conservative portfolios."
            ),
        )

    frontier_points = st.slider(
        "Frontier points",
        10,
        200,
        50,
        help=(
            "Number of points to compute on the efficient frontier curve. "
            "More points give a smoother curve but take longer."
        ),
    )
    run = st.button("Run", type="primary")


# -- Cached data fetch --------------------------------------------------------
@st.cache_data(show_spinner="Fetching data…")
def _fetch(*, tickers: list[str], period: str) -> pd.DataFrame:
    return fetch_multi_history(tickers=tickers, period=period)


# -- Computation (only on Run click) ------------------------------------------
if run:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:  # noqa: PLR2004
        st.error("Please enter at least 2 tickers.")
        st.stop()

    try:
        prices = _fetch(tickers=tickers, period=period)
    except Exception as exc:
        st.error(f"Could not fetch data: {exc}")
        st.stop()

    with st.spinner("Optimizing portfolio…"):
        result = optimize_portfolio(
            prices=prices,
            rm=rm_code,
            obj=obj_code,
            kelly=kelly,
            rf=risk_free_rate,
            allow_short_selling=allow_short,
            risk_aversion=risk_aversion,
            alpha=alpha,
            kappa=kappa,
        )
        frontier_risks, frontier_rets, _ = build_efficient_frontier(
            prices=prices,
            rm=rm_code,
            kelly=kelly,
            num_points=frontier_points,
            rf=risk_free_rate,
            allow_short_selling=allow_short,
            alpha=alpha,
            kappa=kappa,
        )
        asset_stats = calculate_asset_statistics(prices=prices)

    st.session_state["po_has_run"] = True
    st.session_state["po_data"] = {
        "tickers": tickers,
        "prices": prices,
        "result": result,
        "frontier_risks": frontier_risks,
        "frontier_rets": frontier_rets,
        "asset_stats": asset_stats,
    }

_has_run: bool = (
    st.session_state.get("po_has_run", False) and "po_data" in st.session_state
)

# -- Guide ---------------------------------------------------------------------
with st.expander(
    "What is Portfolio Optimization?",
    expanded=(not _has_run),
):
    st.markdown(
        """
Portfolio optimization answers the question: **"Given these
assets, how should I split my money to get the best
risk-return trade-off?"**

The core idea (Modern Portfolio Theory) is that
diversification reduces risk. By combining assets that don't
move in lockstep, you can build a portfolio with less risk
than any individual asset — or more return for the same risk.

The **efficient frontier** is the set of portfolios that
offer the highest possible return for each level of risk.
Any portfolio below the frontier is sub-optimal.

**How to read the results:**
- **Efficient Frontier tab** — the frontier curve, individual
  assets, and your optimal portfolio (star marker).
- **Portfolio Weights tab** — how much to allocate to each
  asset.
- **Correlation Matrix tab** — how assets move relative to
  each other (lower correlation = better diversification).
- **Asset Statistics tab** — annualized return and volatility
  of each asset.
"""
    )

with st.expander("Models explained"):
    st.markdown(
        """\
**Mean Risk** — The standard approach from Modern Portfolio
Theory. The optimizer directly balances expected return
against risk. The most common and well-understood model.
Good for most investors.

**Logarithmic Mean Risk (Kelly Criterion)** — Optimizes
the growth rate of your portfolio (logarithm of returns).
This naturally avoids "betting too big" because the log
function heavily penalizes large losses. Tends to produce
more concentrated portfolios than Mean Risk.
- Best for long-term buy-and-hold investors focused on
  compounding.
- Can be more aggressive — consider pairing with a
  conservative risk measure.

**Recommendation:** Start with *Mean Risk* for a balanced
result. Try *Kelly Criterion* if you are investing for the
long term and comfortable with more concentration.
"""
    )

with st.expander("Objective functions explained"):
    st.markdown(
        """\
**Minimum Risk** — Finds the portfolio with the lowest
possible risk. Best for conservative investors who want to
minimize losses.

**Maximum Return** — Maximizes expected return regardless of
risk. Best for aggressive investors with high risk tolerance.

**Maximum Utility Function** — Balances return and risk using
a risk-aversion parameter. Best for investors who want to
tune the return/risk trade-off.

**Maximum Risk Adjusted Return Ratio** — Maximizes return per
unit of risk (Sharpe ratio). Best for most investors — the
"best bang for your buck" approach.

**Recommendation:** Start with *Maximum Risk Adjusted Return
Ratio* (Sharpe) — it's the most widely used and gives a
balanced result.
"""
    )

with st.expander("Risk measures explained"):
    st.markdown(
        """\
There are 24 risk measures grouped into three categories.

---

**Dispersion measures** — how spread out are returns?

- **Standard Deviation** — The classic volatility measure.
  How much returns bounce around their average. Higher means
  more unpredictable. *Good default for most users.*
- **Mean Absolute Deviation** — Similar to standard
  deviation but less influenced by extreme outliers.
- **Gini Mean Difference** — Average difference between
  all pairs of returns. More robust than standard deviation
  to unusual data points.
- **Square Root Kurtosis** — Captures extreme outlier
  risk. Sensitive to unusually large gains or losses.
- **CVaR Range** — The spread between the best and worst
  tail outcomes.
- **Tail Gini Range** — Like Gini Mean Difference but
  focused on extreme returns only.
- **EVaR Range** — Uses information theory to measure
  the spread of extreme outcomes. More conservative than
  CVaR Range.
- **RLVaR Range** — A flexible middle ground between
  CVaR Range and worst-case range.
- **Range** — Gap between the best and worst observed
  returns. Most intuitive but sensitive to single outliers.

---

**Downside measures** — how bad can losses get?

- **Conditional Value at Risk (CVaR)** — Average loss in
  the worst scenarios (e.g. worst 5 %). The industry
  standard for tail risk. Also called Expected Shortfall.
  *Best choice if you worry about crashes.*
- **Semi Standard Deviation** — Like standard deviation
  but only counts downside moves. Ignores upside
  volatility entirely.
- **Square Root Semi Kurtosis** — Captures extreme
  downside outlier risk specifically.
- **First Lower Partial Moment** — How often returns fall
  below a target (usually zero). Think of it as the
  "frequency of losses".
- **Second Lower Partial Moment** — How far returns fall
  below a target. Combines frequency and severity of
  losses.
- **Tail Gini** — Similar to CVaR but weights extreme
  losses differently. More robust when the worst outcomes
  are very spread out.
- **Entropic Value at Risk (EVaR)** — An upper bound
  on CVaR using information theory. More conservative —
  assumes worse scenarios are possible.
- **Relativistic Value at Risk (RLVaR)** — Flexible
  measure between CVaR and worst case. Adjustable
  conservatism level.
- **Worst Realization** — The single worst return ever
  observed. The most conservative measure — optimizes for
  the absolute worst case.

---

**Drawdown measures** — how deep can peak-to-trough drops
be?

- **Maximum Drawdown** — The single deepest peak-to-trough
  drop. The most intuitive drawdown measure — "how much
  could I lose from a peak?" *Best choice for long-term
  investors.*
- **Average Drawdown** — The average of all peak-to-trough
  drops. Gives a typical sense of how deep pullbacks are.
- **Ulcer Index** — Measures both depth and duration of
  drawdowns. Deep prolonged drops score highest.
- **Conditional Drawdown at Risk (CDaR)** — Average
  drawdown in the worst episodes. Like CVaR but for
  drawdowns instead of returns.
- **Entropic Drawdown at Risk (EDaR)** — An upper bound
  on CDaR. More conservative — assumes deeper drawdowns
  are possible.
- **Relativistic Drawdown at Risk (RLDaR)** — Flexible
  measure between CDaR and worst-case drawdown.

---

**Quick recommendations:**
- **New to this?** *Standard Deviation* — most intuitive.
- **Worried about crashes?** *Conditional Value at Risk.*
- **Long-term investor?** *Maximum Drawdown* or
  *Conditional Drawdown at Risk.*
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()

# -- Display from session state ------------------------------------------------
_d = st.session_state["po_data"]
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

tab_frontier, tab_weights, tab_corr, tab_stats = st.tabs(
    [
        "Efficient Frontier",
        "Portfolio Weights",
        "Correlation Matrix",
        "Asset Statistics",
    ]
)

with tab_frontier:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=_d["frontier_risks"],
            y=_d["frontier_rets"],
            mode="lines",
            line={"width": 2, "color": "rgb(99,110,250)"},
            name="Efficient Frontier",
        )
    )

    _stats = _d["asset_stats"]
    for t in _tickers:
        s = _stats[t]
        fig.add_trace(
            go.Scatter(
                x=[s["annualized_volatility"]],
                y=[s["annualized_return"]],
                mode="markers+text",
                marker={"size": 10},
                text=[t],
                textposition="top right",
                name=t,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[_result.volatility],
            y=[_result.expected_return],
            mode="markers",
            marker={"size": 18, "symbol": "star", "color": "#EF553B"},
            name="Optimal Portfolio",
        )
    )

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=550,
    )
    st.plotly_chart(fig, width="stretch")

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
                "Annualized Volatility": (f"{s['annualized_volatility']:.2%}"),
            }
        )
    st.table(pd.DataFrame(rows))
