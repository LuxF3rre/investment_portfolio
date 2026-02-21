"""Filtered Historical Simulation page — Streamlit UI only."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

from investment_portfolio.data import fetch_history
from investment_portfolio.fhs import (
    calculate_fhs_percentiles,
    calculate_fhs_risk_metrics,
    calculate_filtered_historical_returns,
    calculate_return_statistics,
    forecast_prices,
)

st.set_page_config(
    page_title="Filtered Historical Simulation",
    page_icon=":chart_with_downwards_trend:",
    layout="wide",
)

st.title("Filtered Historical Simulation (FHS)")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    ticker = st.text_input("Ticker (yfinance)", value="EURUSD=X")
    forecast_horizon = st.slider("Forecast horizon (days)", 10, 500, 250)
    vol_window = st.slider("Volatility window", 10, 60, 20)
    ewma_decay = st.slider("EWMA decay", 0.90, 0.99, 0.99, step=0.01)
    run = st.button("Run", type="primary")


# ── Cached data fetch ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data…")
def _fetch(*, ticker: str) -> pd.DataFrame:
    return fetch_history(ticker=ticker)


# ── Main ─────────────────────────────────────────────────────────────────────
if run:
    df = _fetch(ticker=ticker)

    prices: pd.Series = df["Close"]
    current_price = float(prices.iloc[-1])

    # Filtered + unfiltered for comparison
    filtered = calculate_filtered_historical_returns(
        prices=prices,
        forecast_horizon=forecast_horizon,
        volatility_window=vol_window,
        ewma_decay=ewma_decay,
        apply_filter=True,
    )
    unfiltered = calculate_filtered_historical_returns(
        prices=prices,
        forecast_horizon=forecast_horizon,
        volatility_window=vol_window,
        ewma_decay=ewma_decay,
        apply_filter=False,
    )

    forecasted = forecast_prices(current_price=current_price, filtered_returns=filtered)
    risk = calculate_fhs_risk_metrics(
        filtered_returns=filtered,
        forecasted_prices=forecasted,
        current_price=current_price,
    )
    pctiles = calculate_fhs_percentiles(
        forecasted_prices=forecasted, current_price=current_price
    )
    stats = calculate_return_statistics(returns=filtered)

    # ── Metrics row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Current", f"{current_price:.5f}")
    c2.metric("Expected", f"{risk['expected_price']:.5f}")
    c3.metric("VaR 95%", f"{risk['var_95']:.2f}%")
    c4.metric("CVaR 95%", f"{risk['cvar_95']:.2f}%")
    c5.metric("Sharpe", f"{risk['sharpe']:.3f}")

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_hist, tab_ret, tab_dist, tab_pct = st.tabs(
        [
            "Historical Rate",
            "Filtered vs Unfiltered",
            "Forecast Distribution",
            "Percentiles & Stats",
        ]
    )

    with tab_hist:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(prices.index, prices.values, linewidth=0.8)
        ax.set_title(f"{ticker} Historical Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        st.pyplot(fig)

    with tab_ret:
        fig2, (ax_f, ax_u) = plt.subplots(1, 2, figsize=(12, 5))

        ax_f.hist(filtered, bins=50, alpha=0.7, color="teal", edgecolor="black")
        ax_f.axvline(
            filtered.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {filtered.mean():.4f}",
        )
        ax_f.set_title("Filtered Returns")
        ax_f.set_xlabel("Return")
        ax_f.legend()

        ax_u.hist(
            unfiltered,
            bins=50,
            alpha=0.7,
            color="steelblue",
            edgecolor="black",
        )
        ax_u.axvline(
            unfiltered.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {unfiltered.mean():.4f}",
        )
        ax_u.set_title("Unfiltered Returns")
        ax_u.set_xlabel("Return")
        ax_u.legend()

        plt.tight_layout()
        st.pyplot(fig2)

    with tab_dist:
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.hist(forecasted, bins=60, density=True, alpha=0.7, color="teal")
        kde = gaussian_kde(forecasted)
        x = np.linspace(forecasted.min(), forecasted.max(), 300)
        ax3.plot(x, kde(x), linewidth=2, color="red", label="KDE")
        ax3.axvline(
            current_price,
            color="black",
            linestyle="--",
            label=f"Current: {current_price:.5f}",
        )
        ax3.set_xlabel("Forecasted Price")
        ax3.set_ylabel("Density")
        ax3.set_title(
            f"{ticker} Forecast Distribution ({forecast_horizon}-Day Horizon)"
        )
        ax3.legend()
        st.pyplot(fig3)

    with tab_pct:
        pct_col, stat_col = st.columns(2)

        with pct_col:
            st.subheader("Percentiles")
            rows = []
            for p in [5, 25, 50, 75, 95]:
                rows.append(
                    {
                        "Percentile": f"{p}th",
                        "Price": f"{pctiles[f'p{p}_price']:.5f}",
                        "Change": f"{pctiles[f'p{p}_pct']:+.2f}%",
                    }
                )
            st.table(pd.DataFrame(rows))
            st.metric(
                "Probability of appreciation",
                f"{risk['prob_appreciation']:.1%}",
            )

        with stat_col:
            st.subheader("Return Statistics")
            stat_rows = [
                {"Metric": k.replace("_", " ").title(), "Value": f"{v:.6f}"}
                for k, v in stats.items()
            ]
            st.table(pd.DataFrame(stat_rows))

    st.session_state["fhs_results"] = {
        "risk": risk,
        "percentiles": pctiles,
        "stats": stats,
    }
