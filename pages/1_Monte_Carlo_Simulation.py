"""Monte Carlo Simulation page — Streamlit UI only."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Ticker (yfinance)", value="AAPL")
    forecast_years = st.slider("Forecast horizon (years)", 1, 5, 1)
    num_simulations = st.slider("Simulations", 100, 10_000, 1_000, step=100)
    lookback_years = st.slider("Lookback (years)", 1, 10, 5)
    risk_free_rate = st.number_input(
        "Risk-free rate", 0.0, 0.20, 0.04, step=0.01, format="%.2f"
    )
    run = st.button("Run", type="primary")


# ── Cached data fetch ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data…")
def _fetch(*, ticker: str) -> pd.DataFrame:
    return fetch_history(ticker=ticker)


# ── Main ─────────────────────────────────────────────────────────────────────
if run:
    df = _fetch(ticker=ticker)

    prices: pd.Series = df["Close"]

    # Filter to lookback window
    cutoff = prices.index.max() - pd.DateOffset(years=lookback_years)
    prices = prices.loc[prices.index >= cutoff]

    current_price = float(prices.iloc[-1])
    initial_price = float(prices.iloc[0])

    expected_annual_return = (current_price / initial_price) ** (1 / lookback_years) - 1
    annual_vol = calculate_annualized_volatility(
        prices=prices, lookback_years=lookback_years
    )

    rng = np.random.default_rng(42)

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
        rng=np.random.default_rng(42),
    )

    risk = calculate_simulation_risk_metrics(
        terminal_prices=terminal_prices,
        current_price=current_price,
        risk_free_rate=risk_free_rate,
    )
    pctiles = calculate_simulation_percentiles(values=terminal_prices)

    # ── Metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current", f"${current_price:,.2f}")
    c2.metric("Median", f"${pctiles['p50']:,.2f}")
    c3.metric("VaR 95%", f"{risk['var_95']:.2%}")
    c4.metric("CVaR 95%", f"{risk['cvar_95']:.2%}")
    c5.metric("Sharpe", f"{risk['sharpe']:.3f}")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_paths, tab_dist, tab_pct = st.tabs(
        ["Price Paths", "Terminal Distribution", "Percentiles"]
    )

    with tab_paths:
        fig, ax = plt.subplots(figsize=(10, 5))
        time_steps = np.linspace(0, forecast_years, paths.shape[0])
        for i in range(paths.shape[1]):
            ax.plot(time_steps, paths[:, i], alpha=0.08, linewidth=0.5)
        ax.plot(
            time_steps,
            paths.mean(axis=1),
            color="red",
            linewidth=2,
            label="Mean",
        )
        ax.axhline(current_price, color="black", linestyle="--", label="Current")
        ax.set_xlabel("Years")
        ax.set_ylabel("Price ($)")
        ax.legend()
        ax.set_title(f"{ticker} Simulated Price Paths")
        st.pyplot(fig)

    with tab_dist:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.hist(terminal_prices, bins=60, density=True, alpha=0.7)
        kde = gaussian_kde(terminal_prices)
        x = np.linspace(terminal_prices.min(), terminal_prices.max(), 300)
        ax2.plot(x, kde(x), linewidth=2, color="red", label="KDE")
        ax2.axvline(
            current_price,
            color="black",
            linestyle="--",
            label=f"Current ${current_price:,.2f}",
        )
        ax2.set_xlabel("Terminal Price ($)")
        ax2.set_ylabel("Density")
        ax2.set_title("Terminal Price Distribution")
        ax2.legend()
        st.pyplot(fig2)

    with tab_pct:
        rows = []
        for label in ["p5", "p25", "p50", "p75", "p95"]:
            price = pctiles[label]
            ret = (price / current_price - 1) * 100
            rows.append(
                {
                    "Percentile": label.upper(),
                    "Price": f"${price:,.2f}",
                    "Return": f"{ret:+.2f}%",
                }
            )
        st.table(pd.DataFrame(rows))
        st.metric("Probability of positive return", f"{risk['prob_profit']:.1%}")

    st.session_state["mc_results"] = {
        "risk": risk,
        "percentiles": pctiles,
    }
