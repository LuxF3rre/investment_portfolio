"""Monte Carlo Simulation page — Streamlit UI only."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from investment_portfolio.data import fetch_history
from investment_portfolio.monte_carlo import (
    calculate_annualized_volatility,
    calculate_simulation_percentiles,
    calculate_simulation_risk_metrics,
    geometric_brownian_motion,
    geometric_brownian_motion_paths,
)

st.set_page_config(
    page_title="Monte Carlo Simulation",
    page_icon=":game_die:",
    layout="wide",
)

st.title("Monte Carlo Simulation (GBM)")


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
        value="AAPL",
        help=(
            "Any yfinance-compatible symbol: stocks (AAPL), FX (EURUSD=X), "
            "commodities (GC=F), ETFs (SPY)."
        ),
    )
    forecast_years = st.slider(
        "Forecast horizon (years)",
        1,
        5,
        1,
        help=(
            "How far into the future to simulate. Longer horizons produce "
            "wider outcome ranges."
        ),
    )
    num_simulations = st.slider(
        "Simulations",
        100,
        10_000,
        1_000,
        step=100,
        help=(
            "Number of random price paths to generate. More simulations "
            "give smoother distributions but take longer."
        ),
    )
    lookback_years = st.slider(
        "Lookback (years)",
        1,
        10,
        5,
        help=(
            "How many years of historical data to use for estimating return "
            "and volatility."
        ),
    )
    risk_free_rate = st.number_input(
        "Risk-free rate",
        0.0,
        0.20,
        0.04,
        step=0.01,
        format="%.2f",
        help=(
            "Annual return on a risk-free asset (e.g. Treasury bill yield). "
            "Used to calculate the Sharpe ratio."
        ),
    )
    seed = st.number_input(
        "Random seed",
        value=42,
        min_value=0,
        step=1,
        help=(
            "Fixes the random number generator so results are reproducible. "
            "Change to explore different outcomes."
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

        cutoff = prices.index.max() - pd.DateOffset(years=lookback_years)
        prices = prices.loc[prices.index >= cutoff]

        current_price = float(prices.iloc[-1])
        initial_price = float(prices.iloc[0])

        expected_annual_return = (current_price / initial_price) ** (
            1 / lookback_years
        ) - 1
        annual_vol = calculate_annualized_volatility(prices=prices)

        rng = np.random.default_rng(seed)

        terminal_prices = geometric_brownian_motion(
            current_price=current_price,
            annual_return=expected_annual_return,
            annual_volatility=annual_vol,
            time_horizon=forecast_years,
            num_simulations=num_simulations,
            rng=rng,
        )

        paths = geometric_brownian_motion_paths(
            current_price=current_price,
            annual_return=expected_annual_return,
            annual_volatility=annual_vol,
            time_horizon=forecast_years,
            num_simulations=min(num_simulations, 200),
            num_steps=252 * forecast_years,
            rng=np.random.default_rng(seed),
        )

        risk = calculate_simulation_risk_metrics(
            terminal_prices=terminal_prices,
            current_price=current_price,
            risk_free_rate=risk_free_rate,
        )
        pctiles = calculate_simulation_percentiles(values=terminal_prices)

    st.session_state["mc_has_run"] = True
    st.session_state["mc_data"] = {
        "ticker": ticker,
        "forecast_years": forecast_years,
        "current_price": current_price,
        "terminal_prices": terminal_prices,
        "paths": paths,
        "risk": risk,
        "pctiles": pctiles,
    }

_has_run: bool = (
    st.session_state.get("mc_has_run", False) and "mc_data" in st.session_state
)

# ── Guide ────────────────────────────────────────────────────────────────────
with st.expander("What is Monte Carlo Simulation?", expanded=(not _has_run)):
    st.markdown(
        """
Monte Carlo simulation generates thousands of possible future price paths to
answer the question: **"Given what we know about this asset's past behavior,
what range of prices is plausible in the future?"**

Each simulated path uses **Geometric Brownian Motion (GBM)** — a model that
combines the asset's average growth rate with random day-to-day fluctuations
(volatility). The result is a probability distribution of future prices, not a
single point forecast.

**How to read the results:**
- **Price Paths tab** — a fan of possible futures; the red line is the average.
- **Terminal Distribution tab** — histogram of where the price lands at the end
  of the forecast horizon.
- **Percentiles tab** — key price levels (5th, 25th, 50th, 75th, 95th
  percentile) and the probability of a positive return.

**Key metrics:**
- **VaR 95 %** — the worst-case loss you'd expect 95 % of the time.
- **CVaR 95 %** — the average loss in the worst 5 % of scenarios.
- **Sharpe ratio** — return per unit of risk (higher is better).
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()

# ── Display from session state ───────────────────────────────────────────────
_d = st.session_state["mc_data"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current", _fmt_price(_d["current_price"]))
c2.metric("Median", _fmt_price(_d["pctiles"].p50))
c3.metric("VaR 95%", f"{_d['risk'].var_95:.2%}")
c4.metric("CVaR 95%", f"{_d['risk'].cvar_95:.2%}")
c5.metric("Sharpe", f"{_d['risk'].sharpe:.3f}")

tab_paths, tab_dist, tab_pct = st.tabs(
    ["Price Paths", "Terminal Distribution", "Percentiles"]
)

with tab_paths:
    _paths = _d["paths"]
    _fy = _d["forecast_years"]
    time_steps = np.linspace(0, _fy, _paths.shape[0])
    fig = go.Figure()
    for i in range(_paths.shape[1]):
        fig.add_trace(
            go.Scatter(
                x=time_steps,
                y=_paths[:, i],
                mode="lines",
                line={"width": 0.5, "color": "rgba(99,110,250,0.15)"},
                showlegend=False,
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=time_steps,
            y=_paths.mean(axis=1),
            mode="lines",
            line={"width": 2, "color": "#EF553B"},
            name="Mean",
        )
    )
    fig.add_hline(
        y=_d["current_price"],
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Current {_fmt_price(_d['current_price'])}",
    )
    fig.update_layout(
        title=f"{_d['ticker']} Simulated Price Paths",
        xaxis_title="Years",
        yaxis_title="Price",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

with tab_dist:
    _tp = _d["terminal_prices"]
    fig2 = make_subplots()
    fig2.add_trace(
        go.Histogram(
            x=_tp,
            nbinsx=60,
            histnorm="probability density",
            opacity=0.7,
            name="Distribution",
            marker_color="rgb(99,110,250)",
        )
    )
    kde = gaussian_kde(_tp)
    x = np.linspace(_tp.min(), _tp.max(), 300)
    fig2.add_trace(
        go.Scatter(
            x=x,
            y=kde(x),
            mode="lines",
            line={"width": 2, "color": "#EF553B"},
            name="KDE",
        )
    )
    fig2.add_vline(
        x=_d["current_price"],
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Current {_fmt_price(_d['current_price'])}",
    )
    fig2.update_layout(
        title="Terminal Price Distribution",
        xaxis_title="Terminal Price",
        yaxis_title="Density",
        height=500,
    )
    st.plotly_chart(fig2, width="stretch")

with tab_pct:
    _pctiles = _d["pctiles"]
    _cp = _d["current_price"]
    rows = []
    for label in ["p5", "p25", "p50", "p75", "p95"]:
        price = getattr(_pctiles, label)
        ret = (price / _cp - 1) * 100
        rows.append(
            {
                "Percentile": label.upper(),
                "Price": _fmt_price(price),
                "Return": f"{ret:+.2f}%",
            }
        )
    st.table(pd.DataFrame(rows))
    st.metric(
        "Probability of positive return",
        f"{_d['risk'].prob_profit:.1%}",
    )
