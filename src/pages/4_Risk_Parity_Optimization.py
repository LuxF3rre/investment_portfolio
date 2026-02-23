"""Risk Parity Optimization page (Streamlit UI)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from investment_portfolio.mpt import calculate_asset_statistics, fetch_multi_history
from investment_portfolio.rp import RP_RISK_MEASURES, optimize_risk_parity
from investment_portfolio.theme import COLORSCALE_DIVERGING, TEAL

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
    page_title="Risk Parity Optimization",
    page_icon=":scales:",
    layout="wide",
)

st.title("Risk Parity Optimization")

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
    allow_short = st.checkbox(
        "Allow short selling",
        value=False,
        help=(
            "If enabled, the optimizer can assign negative weights (betting "
            "against an asset). Most retail portfolios keep this off."
        ),
    )

    rm_label = st.selectbox(
        "Risk measure",
        list(RP_RISK_MEASURES.keys()),
        help=(
            "How risk is measured. Risk parity equalizes each asset's "
            "contribution to total portfolio risk under this measure."
        ),
    )
    rm_code = RP_RISK_MEASURES[rm_label]

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

    budget_mode = st.radio(
        "Budget mode",
        ["Equal (1/N)", "Custom"],
        help=(
            "Equal budget targets the same risk contribution from every "
            "asset. Custom lets you assign specific risk budgets."
        ),
    )

    # Parse tickers early so custom budget inputs can be shown.
    _sidebar_tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    budget: np.ndarray | None = None
    if budget_mode == "Custom" and len(_sidebar_tickers) >= 2:  # noqa: PLR2004
        st.subheader("Custom Budget")
        raw_budgets: list[float] = []
        for t in _sidebar_tickers:
            val = st.number_input(
                f"Budget: {t}",
                min_value=0.01,
                max_value=10.0,
                value=1.0,
                step=0.1,
                key=f"budget_{t}",
            )
            raw_budgets.append(val)
        budget = np.array(raw_budgets).reshape(-1, 1)

    run = st.button("Run", type="primary")


# -- Cached data fetch --------------------------------------------------------
@st.cache_data(show_spinner="Fetching data…")
def _fetch(*, tickers: list[str], period: str) -> pd.DataFrame:
    return fetch_multi_history(tickers=tickers, period=period)


# -- Computation (only on Run click) ------------------------------------------
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

    with st.spinner("Optimizing risk parity portfolio…"):
        result = optimize_risk_parity(
            prices=prices,
            rm=rm_code,
            rf=risk_free_rate,
            budget=budget,
            allow_short_selling=allow_short,
            alpha=alpha,
            kappa=kappa,
        )
        asset_stats = calculate_asset_statistics(prices=prices)

    st.session_state["rp_has_run"] = True
    st.session_state["rp_data"] = {
        "tickers": tickers,
        "prices": prices,
        "result": result,
        "asset_stats": asset_stats,
    }

_has_run: bool = (
    st.session_state.get("rp_has_run", False) and "rp_data" in st.session_state
)

# -- Guide --------------------------------------------------------------------
with st.expander(
    "What is Risk Parity?",
    expanded=(not _has_run),
):
    st.markdown(
        """
Risk Parity answers the question: **"How do I allocate capital
so that every asset contributes equally to my portfolio's risk?"**

Unlike traditional portfolio optimization which balances return
against risk, Risk Parity focuses solely on **risk budgeting**.
The idea is simple: if one asset dominates your portfolio's risk,
you're effectively undiversified — even if you hold many assets.

**Key advantages:**
- Does **not** require return forecasts — only risk estimates.
- Naturally diversified: every asset pulls its weight.
- More robust in practice since return estimates are notoriously
  noisy.

**How it works:**
1. Choose a risk measure (e.g. Standard Deviation, CVaR).
2. Set a target risk budget (equal or custom).
3. The optimizer finds weights so each asset's risk contribution
   matches the budget.

**How to read the results:**
- **Risk Contributions tab** — compare target budget vs actual
  risk contribution per asset.
- **Portfolio Weights tab** — how much to allocate to each
  asset.
- **Correlation Matrix tab** — how assets move relative to
  each other (lower correlation = better diversification).
- **Asset Statistics tab** — annualized return and volatility
  of each asset.
"""
    )

with st.expander("Risk measures explained"):
    st.markdown(
        """\
There are 20 convex risk measures grouped into three categories.

---

**Dispersion measures** — how spread out are returns?

- **Standard Deviation** — The classic volatility measure.
  *Good default for most users.*
- **Mean Absolute Deviation** — Similar to standard
  deviation but less influenced by extreme outliers.
- **Gini Mean Difference** — Average difference between
  all pairs of returns.
- **Square Root Kurtosis** — Captures extreme outlier risk.
- **CVaR Range** — The spread between the best and worst
  tail outcomes.
- **Tail Gini Range** — Like Gini Mean Difference but
  focused on extreme returns only.
- **EVaR Range** — Uses information theory to measure
  the spread of extreme outcomes.
- **RLVaR Range** — A flexible middle ground between
  EVaR Range and worst-case range.

---

**Downside measures** — how bad can losses get?

- **Semi Standard Deviation** — Like standard deviation
  but only counts downside moves.
- **Square Root Semi Kurtosis** — Captures extreme
  downside outlier risk.
- **First Lower Partial Moment (Omega Ratio)** — How
  often returns fall below a target.
- **Second Lower Partial Moment (Sortino Ratio)** — How
  far returns fall below a target.
- **Conditional Value at Risk (CVaR)** — Average loss in
  the worst scenarios. *Best choice if you worry about
  crashes.*
- **Tail Gini** — Similar to CVaR but weights extreme
  losses differently.
- **Entropic Value at Risk (EVaR)** — An upper bound
  on CVaR using information theory.
- **Relativistic Value at Risk (RLVaR)** — Flexible
  measure between EVaR and worst case.

---

**Drawdown measures** — how deep can peak-to-trough drops be?

- **Ulcer Index** — Measures both depth and duration of
  drawdowns.
- **Conditional Drawdown at Risk (CDaR)** — Average
  drawdown in the worst episodes.
- **Entropic Drawdown at Risk (EDaR)** — An upper bound
  on CDaR.
- **Relativistic Drawdown at Risk (RLDaR)** — Flexible
  measure between EDaR and worst-case drawdown.

---

**Quick recommendations:**
- **New to this?** *Standard Deviation* — most intuitive.
- **Worried about crashes?** *Conditional Value at Risk.*
- **Long-term investor?** *Conditional Drawdown at Risk.*
"""
    )

with st.expander("Understanding the budget vector"):
    st.markdown(
        """\
The **risk budget** controls how much each asset should contribute
to total portfolio risk.

**Equal (1/N)** — Each asset gets the same risk budget.
For 4 assets, each contributes 25 % of total risk. This is the
classic Risk Parity approach and the most common starting point.

**Custom** — You assign different risk budgets to each asset.
For example, you might want:
- 40 % of risk from a high-conviction equity position
- 20 % each from bonds and commodities
- 20 % from an alternative asset

The values you enter are **relative weights** — they are
normalized internally so they sum to 100 %. Entering
`[2, 1, 1, 1]` means the first asset gets 40 % of the risk
budget and the rest get 20 % each.

**Important:** The budget controls *risk contribution*, not
*capital allocation*. An asset with a 25 % risk budget might
receive more or less than 25 % of capital, depending on its
volatility and correlations.
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()

# -- Display from session state -----------------------------------------------
_d = st.session_state["rp_data"]
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

tab_rc, tab_weights, tab_corr, tab_stats = st.tabs(
    [
        "Risk Contributions",
        "Portfolio Weights",
        "Correlation Matrix",
        "Asset Statistics",
    ]
)

with tab_rc:
    rc_tickers = list(_result.risk_contribution_pct.keys())
    rc_actual = [_result.risk_contribution_pct[t] for t in rc_tickers]

    fig_pie = px.pie(
        names=rc_tickers,
        values=rc_actual,
        title="Risk Contribution Breakdown",
        hole=0.3,
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, width="stretch")

with tab_weights:
    names = list(_result.weights.keys())
    vals = list(_result.weights.values())
    fig_w = go.Figure(
        go.Bar(
            x=names,
            y=vals,
            marker_color=TEAL,
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
    df_w = pd.DataFrame({"Ticker": names, "Weight": [f"{v:.2%}" for v in vals]})
    st.dataframe(df_w, hide_index=True, width="stretch")

with tab_corr:
    _prices = _d["prices"]
    log_returns = np.log(_prices / _prices.shift(1)).dropna()
    corr_matrix = log_returns.corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale=COLORSCALE_DIVERGING,
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
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

st.divider()
st.markdown(
    "<div style='text-align: center'>Made with ❤ by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
