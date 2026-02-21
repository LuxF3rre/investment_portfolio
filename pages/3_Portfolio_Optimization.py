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
    )
    period = st.selectbox("History period", ["1y", "2y", "5y", "10y", "max"], index=2)
    risk_free_rate = st.number_input(
        "Risk-free rate (annual)",
        min_value=0.0,
        max_value=0.20,
        value=0.04,
        step=0.005,
        format="%.3f",
    )
    allow_short = st.checkbox("Allow short selling", value=False)

    model_label = st.selectbox(
        "Model",
        ["Mean Risk", "Logarithmic Mean Risk (Kelly Criterion)"],
    )
    kelly: str | None = None if model_label == "Mean Risk" else "exact"

    obj_label = st.selectbox("Objective function", list(OBJECTIVES.keys()))
    obj_code = OBJECTIVES[obj_label]

    rm_label = st.selectbox("Risk measure", list(RISK_MEASURES.keys()))
    rm_code = RISK_MEASURES[rm_label]

    risk_aversion = 1.0
    if obj_label == "Maximum Utility Function":
        risk_aversion = st.slider(
            "Risk aversion",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
        )

    frontier_points = st.slider("Frontier points", 10, 200, 50)
    run = st.button("Run", type="primary")


# -- Cached data fetch --------------------------------------------------------
@st.cache_data(show_spinner="Fetching data…")
def _fetch(*, tickers: list[str], period: str) -> pd.DataFrame:
    return fetch_multi_history(tickers=tickers, period=period)


# -- Main ---------------------------------------------------------------------
if run:
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
    if len(tickers) < 2:  # noqa: PLR2004
        st.error("Please enter at least 2 tickers.")
        st.stop()

    prices = _fetch(tickers=tickers, period=period)

    result = optimize_portfolio(
        prices=prices,
        rm=rm_code,
        obj=obj_code,
        kelly=kelly,
        rf=risk_free_rate,
        allow_short_selling=allow_short,
        risk_aversion=risk_aversion,
    )
    frontier_risks, frontier_rets, _ = build_efficient_frontier(
        prices=prices,
        rm=rm_code,
        kelly=kelly,
        num_points=frontier_points,
        rf=risk_free_rate,
        allow_short_selling=allow_short,
    )
    asset_stats = calculate_asset_statistics(prices=prices)

    # -- Metrics row ----------------------------------------------------------
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected Return", f"{result.expected_return:.2%}")
    c2.metric("Volatility", f"{result.volatility:.2%}")
    c3.metric("Sharpe Ratio", f"{result.sharpe_ratio:.3f}")
    c4.metric("Risk Measure", rm_label)
    c5.metric("Objective", obj_label)

    # -- Tabs -----------------------------------------------------------------
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

        # Frontier curve
        fig.add_trace(
            go.Scatter(
                x=frontier_risks,
                y=frontier_rets,
                mode="lines",
                line={"width": 2, "color": "navy"},
                name="Efficient Frontier",
            )
        )

        # Individual assets
        for t in tickers:
            s = asset_stats[t]
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

        # Optimal portfolio star
        fig.add_trace(
            go.Scatter(
                x=[result.volatility],
                y=[result.expected_return],
                mode="markers",
                marker={"size": 18, "symbol": "star", "color": "red"},
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
        names = list(result.weights.keys())
        vals = list(result.weights.values())
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
        log_returns = np.log(prices / prices.shift(1)).dropna()
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
        rows = []
        for t in tickers:
            s = asset_stats[t]
            rows.append(
                {
                    "Ticker": t,
                    "Annualized Return": f"{s['annualized_return']:.2%}",
                    "Annualized Volatility": f"{s['annualized_volatility']:.2%}",
                }
            )
        st.table(pd.DataFrame(rows))
