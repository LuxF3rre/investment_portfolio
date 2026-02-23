"""Filtered Historical Simulation page — Streamlit UI only."""

import dataclasses

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from investment_portfolio.data import fetch_history
from investment_portfolio.fhs import (
    FHSMethod,
    calculate_fhs_percentiles,
    calculate_fhs_risk_metrics,
    calculate_filtered_historical_returns,
    calculate_return_statistics,
    forecast_prices,
)
from investment_portfolio.theme import OVERLAY1, RED, SAPPHIRE, TEAL

st.set_page_config(
    page_title="Filtered Historical Simulation",
    page_icon=":chart_with_downwards_trend:",
    layout="wide",
)

st.title("Filtered Historical Simulation (FHS)")


_PRICE_PRECISION_THRESHOLD: int = 10


def _fmt_price(price: float) -> str:
    """Format price adaptively — more decimals for small values (FX)."""
    if abs(price) < _PRICE_PRECISION_THRESHOLD:
        return f"{price:.4f}"
    return f"{price:,.2f}"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input(
        "Ticker (yfinance)",
        value="EURUSD=X",
        help=(
            "Any yfinance-compatible symbol: stocks (AAPL), FX (EURUSD=X), "
            "commodities (GC=F), ETFs (SPY)."
        ),
    )
    method_label = st.selectbox(
        "Method",
        ["Ratio Scaling", "Standardized Residuals"],
        help=(
            "Ratio Scaling is simpler and faster. Standardized Residuals is "
            "more rigorous when volatility is changing rapidly. See the guide "
            "on the main panel for details."
        ),
    )
    fhs_method = (
        FHSMethod.RESIDUALS
        if method_label == "Standardized Residuals"
        else FHSMethod.RATIO
    )
    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        10,
        500,
        250,
        help="Number of trading days to forecast. 250 days is roughly 1 year.",
    )
    vol_window = st.slider(
        "Volatility window",
        10,
        60,
        20,
        help=(
            "Number of days used to estimate rolling volatility. Shorter "
            "windows react faster to recent market changes."
        ),
    )
    ewma_decay = st.slider(
        "EWMA decay",
        0.90,
        0.99,
        0.94,
        step=0.01,
        help=(
            "Controls how quickly the volatility estimate adapts. Lower "
            "values react faster to recent changes. 0.94 is the RiskMetrics "
            "standard."
        ),
    )
    if fhs_method == FHSMethod.RESIDUALS:
        num_sims = st.slider(
            "Simulations",
            1_000,
            50_000,
            10_000,
            step=1_000,
            help="Number of bootstrap samples to draw from the standardized residuals.",
        )
    else:
        num_sims = 10_000
    risk_free_rate = st.number_input(
        "Risk-free rate",
        0.0,
        0.20,
        0.0,
        step=0.01,
        format="%.2f",
        help=(
            "Annual return on a risk-free asset (e.g. Treasury bill yield). "
            "Used to calculate the Sharpe ratio."
        ),
    )
    run = st.button("Run", type="primary")


# ── Cached data fetch ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data…")
def _fetch(*, ticker: str) -> pd.DataFrame:
    return fetch_history(ticker=ticker)


# ── Computation (only on Run click) ──────────────────────────────────────────
if run:
    try:
        df = _fetch(ticker=ticker)
    except Exception as exc:
        st.error(f"Could not fetch data for **{ticker}**: {exc}")
        st.stop()

    with st.spinner("Running simulation…"):
        prices: pd.Series = df["Close"]
        current_price = float(prices.iloc[-1])

        filtered = calculate_filtered_historical_returns(
            prices=prices,
            forecast_horizon=forecast_horizon,
            volatility_window=vol_window,
            ewma_decay=ewma_decay,
            apply_filter=True,
            method=fhs_method,
            num_simulations=num_sims,
        )
        unfiltered = calculate_filtered_historical_returns(
            prices=prices,
            forecast_horizon=forecast_horizon,
            volatility_window=vol_window,
            ewma_decay=ewma_decay,
            apply_filter=False,
        )

        forecasted = forecast_prices(
            current_price=current_price, filtered_returns=filtered
        )
        total_cal_days = (prices.index[-1] - prices.index[0]).days
        if total_cal_days > 0:
            trading_days_per_year = (len(prices) - 1) * 365.25 / total_cal_days
        else:
            trading_days_per_year = 252.0
        time_horizon = forecast_horizon / trading_days_per_year

        risk = calculate_fhs_risk_metrics(
            filtered_returns=filtered,
            forecasted_prices=forecasted,
            current_price=current_price,
            risk_free_rate=risk_free_rate,
            time_horizon=time_horizon,
        )
        pctiles = calculate_fhs_percentiles(
            forecasted_prices=forecasted, current_price=current_price
        )
        stats = calculate_return_statistics(returns=filtered)

    st.session_state["fhs_has_run"] = True
    st.session_state["fhs_data"] = {
        "ticker": ticker,
        "forecast_horizon": forecast_horizon,
        "current_price": current_price,
        "prices": prices,
        "filtered": filtered,
        "unfiltered": unfiltered,
        "forecasted": forecasted,
        "risk": risk,
        "pctiles": pctiles,
        "stats": stats,
    }

_has_run: bool = (
    st.session_state.get("fhs_has_run", False) and "fhs_data" in st.session_state
)

# ── Guide ────────────────────────────────────────────────────────────────────
with st.expander(
    "What is Filtered Historical Simulation?",
    expanded=(not _has_run),
):
    st.markdown(
        """
Filtered Historical Simulation (FHS) forecasts prices by replaying **actual
historical returns** — but first adjusting them for how volatile the market is
*right now*.

Traditional historical simulation assumes the future will look exactly like the
past. FHS improves on this by recognizing that volatility changes over time: if
the market is currently turbulent, the simulation reflects that.

**How to read the results:**
- **Historical Rate tab** — the asset's raw price history.
- **Filtered vs Unfiltered tab** — comparison showing how the volatility filter
  reshapes the return distribution.
- **Forecast Distribution tab** — where the price might land at the end of the
  forecast horizon.
- **Percentiles & Stats tab** — key price levels and statistical properties of
  the filtered returns.

**Key metrics:**
- **VaR 95 %** — the worst-case loss you'd expect 95 % of the time.
- **CVaR 95 %** — the average loss in the worst 5 % of scenarios.
- **Sharpe ratio** — return per unit of risk (higher is better).
"""
    )

with st.expander("Which method should I choose?"):
    st.markdown(
        """\
**Ratio Scaling**
- Scales multi-period historical returns by the ratio of
  current to historical volatility.
- Fast and simple.
- Best for: quick analysis, stable volatility regimes.

**Standardized Residuals**
- Extracts standardized daily residuals, bootstraps them
  with replacement, re-scales by current volatility.
- Slower (generates many bootstrap samples).
- Best for: rapidly changing volatility, rigorous tail
  estimates.

**Recommendation:** Start with *Ratio Scaling* for a quick
first look. Switch to *Standardized Residuals* when you need
higher precision or the market is in a stress period.
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()

# ── Display from session state ───────────────────────────────────────────────
_d = st.session_state["fhs_data"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current", _fmt_price(_d["current_price"]))
c2.metric("Expected", _fmt_price(_d["risk"].expected_price))
c3.metric("VaR 95%", f"{_d['risk'].var_95:.2%}")
c4.metric("CVaR 95%", f"{_d['risk'].cvar_95:.2%}")
c5.metric("Sharpe", f"{_d['risk'].sharpe:.3f}")

tab_hist, tab_ret, tab_dist, tab_pct = st.tabs(
    [
        "Historical Rate",
        "Filtered vs Unfiltered",
        "Forecast Distribution",
        "Percentiles & Stats",
    ]
)

with tab_hist:
    _prices = _d["prices"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_prices.index,
            y=_prices.values,
            mode="lines",
            line={"width": 1},
            name=_d["ticker"],
        )
    )
    fig.update_layout(
        title=f"{_d['ticker']} Historical Price",
        xaxis_title="Date",
        yaxis_title="Price",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

with tab_ret:
    _filt = _d["filtered"]
    _unfilt = _d["unfiltered"]
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Filtered Returns", "Unfiltered Returns"),
    )
    fig2.add_trace(
        go.Histogram(
            x=_filt,
            nbinsx=50,
            opacity=0.7,
            marker_color=TEAL,
            name="Filtered",
        ),
        row=1,
        col=1,
    )
    fig2.add_vline(
        x=float(_filt.mean()),
        line_dash="dash",
        line_color=RED,
        annotation_text=f"Mean: {_filt.mean():.4f}",
        col=1,
    )
    fig2.add_trace(
        go.Histogram(
            x=_unfilt,
            nbinsx=50,
            opacity=0.7,
            marker_color=SAPPHIRE,
            name="Unfiltered",
        ),
        row=1,
        col=2,
    )
    fig2.add_vline(
        x=float(_unfilt.mean()),
        line_dash="dash",
        line_color=RED,
        annotation_text=f"Mean: {_unfilt.mean():.4f}",
        col=2,
    )
    fig2.update_layout(height=500)
    st.plotly_chart(fig2, width="stretch")

with tab_dist:
    _fc = _d["forecasted"]
    _cp = _d["current_price"]
    fig3 = go.Figure()
    fig3.add_trace(
        go.Histogram(
            x=_fc,
            nbinsx=60,
            histnorm="probability density",
            opacity=0.7,
            name="Distribution",
            marker_color=TEAL,
        )
    )
    kde = gaussian_kde(_fc)
    x = np.linspace(_fc.min(), _fc.max(), 300)
    fig3.add_trace(
        go.Scatter(
            x=x,
            y=kde(x),
            mode="lines",
            line={"width": 2, "color": RED},
            name="KDE",
        )
    )
    fig3.add_vline(
        x=_cp,
        line_dash="dash",
        line_color=OVERLAY1,
        annotation_text=f"Current: {_fmt_price(_cp)}",
    )
    fig3.update_layout(
        title=(
            f"{_d['ticker']} Forecast Distribution "
            f"({_d['forecast_horizon']}-Day Horizon)"
        ),
        xaxis_title="Forecasted Price",
        yaxis_title="Density",
        height=500,
    )
    st.plotly_chart(fig3, width="stretch")

with tab_pct:
    _pctiles = _d["pctiles"]
    _cp = _d["current_price"]
    pct_col, stat_col = st.columns(2)

    with pct_col:
        st.subheader("Percentiles")
        rows = []
        for p in [5, 25, 50, 75, 95]:
            rows.append(
                {
                    "Percentile": f"{p}th",
                    "Price": _fmt_price(getattr(_pctiles, f"p{p}_price")),
                    "Change": f"{getattr(_pctiles, f'p{p}_pct'):+.2f}%",
                }
            )
        st.table(pd.DataFrame(rows))
        st.metric(
            "Probability of appreciation",
            f"{_d['risk'].prob_appreciation:.1%}",
        )

    with stat_col:
        st.subheader("Return Statistics")
        stat_rows = [
            {
                "Metric": k.replace("_", " ").title(),
                "Value": f"{v:.6f}",
            }
            for k, v in dataclasses.asdict(_d["stats"]).items()
        ]
        st.table(pd.DataFrame(stat_rows))

st.divider()
st.markdown(
    "<div style='text-align: center'>Made with ❤ by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
