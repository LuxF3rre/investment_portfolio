"""Streamlit entry point — landing page."""

import streamlit as st

st.set_page_config(
    page_title="Investment Portfolio",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Investment Portfolio Analysis")

st.markdown(
    """
Welcome to the **Investment Portfolio Analysis** toolkit.
Use the sidebar to navigate between the analysis pages described below.
"""
)

with st.expander("Which tool should I use?", expanded=True):
    st.markdown(
        """\
**Monte Carlo Simulation** — Single-asset price forecasting
over months to years.
*Example: "What might AAPL be worth in 1 year?"*

**Filtered Historical Simulation** — Short-to-medium term
risk assessment using real market history.
*Example: "What's the 250-day risk of my EUR/USD position?"*

**Mean-Risk Optimization** — Deciding how to allocate capital
across multiple assets using return/risk tradeoffs.
*Example: "How should I split my money across 5 stocks?"*

**Risk Parity Optimization** — Allocating capital so every asset
contributes equally to portfolio risk — no return forecasts needed.
*Example: "How do I balance risk across 4 ETFs?"*

**Hierarchical Clustering Optimization** — Tree-based allocation
(HRP/HERC) that avoids covariance matrix inversion — more robust
with many assets or noisy correlations.
*Example: "How do I allocate across 10 sector ETFs without unstable weights?"*

**Nested Clustered Optimization** — Combines clustering with
objective-based optimization within and across clusters. Supports
minimum risk, maximum Sharpe, utility, and equal risk contribution.
*Example: "How do I maximize Sharpe while using clustering
for correlated assets?"*

**Black-Scholes Option Pricing** — Price European call/put options,
compute Greeks, visualize price sensitivity surfaces, and back out
implied volatility.
*Example: "What's a 1-year AAPL call worth at strike $150?"*
"""
    )

with st.expander("About Monte Carlo Simulation"):
    st.markdown(
        """
Simulates thousands of possible future price paths using **Geometric Brownian
Motion (GBM)**. Each path represents one scenario based on the asset's historical
return and volatility. The spread of outcomes gives you a probability distribution
of where the price might end up.

Works with any yfinance ticker — stocks (`AAPL`), FX pairs (`EURUSD=X`),
commodities (`GC=F`), ETFs (`SPY`).

**Use when:** You want a probabilistic view of a single asset's price over
a 1-5 year horizon.
"""
    )

with st.expander("About Filtered Historical Simulation"):
    st.markdown(
        """
Forecasts prices using actual historical returns, adjusted for current market
volatility. Unlike Monte Carlo (which assumes returns follow a bell curve), FHS
preserves the real-world shape of returns — including fat tails and skew.

Two methods are available:
- **Ratio Scaling** — scales historical multi-period returns by the ratio of
  current to historical volatility. Simpler and faster.
- **Standardized Residuals** — extracts and bootstraps standardized daily
  residuals, then re-scales by current volatility. More rigorous for
  non-stationary volatility.

**Use when:** You want a realistic short-to-medium term risk assessment that
reflects how volatile the market is *right now*, not just on average.
"""
    )

with st.expander("About Mean-Risk Optimization"):
    st.markdown(
        """
Finds the optimal way to allocate capital across multiple assets using Modern
Portfolio Theory. Choose from **24 risk measures** and **4 optimization
objectives**, powered by
[riskfolio-lib](https://riskfolio-lib.readthedocs.io/).

**Use when:** You have 2 or more assets and want to find the allocation that
maximizes return for a given risk level, or minimizes risk for a target return.
"""
    )

with st.expander("About Risk Parity Optimization"):
    st.markdown(
        """
Allocates capital so that every asset contributes **equally** (or by a custom
budget) to total portfolio risk. Unlike traditional optimization, Risk Parity
does **not** require return forecasts — only risk estimates.

Choose from **20 convex risk measures** including Standard Deviation, CVaR,
and drawdown-based measures. Set an equal risk budget (1/N) or define custom
risk budgets per asset.

**Use when:** You want a diversified portfolio where no single asset dominates
your risk — especially useful when you distrust return forecasts or want
a robust, low-maintenance allocation.
"""
    )

with st.expander("About Hierarchical Clustering Optimization"):
    st.markdown(
        """
Uses hierarchical clustering (HRP or HERC) to build a tree of asset
relationships, then allocates weights without inverting the covariance matrix.
This avoids the numerical instability that plagues traditional optimizers when
assets are highly correlated or the number of assets exceeds the number of
observations.

Choose from **35 risk measures**, **11 codependence measures**, and
**8 linkage methods**. HRP allocates via recursive bisection; HERC
identifies flat clusters and equalizes risk contribution within and between
them.

**Use when:** You have many correlated assets, want more stable allocations
that don't swing wildly with small data changes, or want a method that is
robust to estimation error in covariance matrices.
"""
    )

with st.expander("About Nested Clustered Optimization"):
    st.markdown(
        """
Nested Clustered Optimization (NCO) combines hierarchical clustering with
objective-based portfolio optimization. Unlike HRP/HERC which use heuristic
allocation rules, NCO applies a full optimization step **within** each cluster
and then **across** clusters — giving you the robustness of clustering with
the flexibility of objective-driven optimization.

Choose from **4 objectives** (Minimum Risk, Maximum Sharpe, Utility,
Equal Risk Contribution) and **24 risk measures** including variance,
CVaR, drawdown-based, and tail-risk measures.

**Use when:** You want the dimensionality reduction benefits of clustering
but need a specific optimization objective — e.g. maximizing Sharpe ratio
or targeting a utility function — rather than a fixed heuristic allocation.
"""
    )

with st.expander("About Black-Scholes Option Pricing"):
    st.markdown(
        """
Prices European call and put options using the **Black-Scholes-Merton** closed-form
formula. Computes all five Greeks (Delta, Gamma, Theta, Vega, Rho), generates a
price sensitivity heatmap across spot price and volatility, and solves for implied
volatility from an observed market price.

**Use when:** You want to quickly price a European option, understand its risk
sensitivities, or back out the market's implied volatility from a quoted premium.
"""
    )

st.divider()
st.markdown(
    "<div style='text-align: center'>Made with ❤ by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
