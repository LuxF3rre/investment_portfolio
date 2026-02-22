"""Black-Scholes Option Pricing page — Streamlit UI only."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from investment_portfolio.black_scholes import (
    calculate_greeks,
    implied_volatility,
    price_european,
    price_surface,
)

st.set_page_config(
    page_title="Black-Scholes Option Pricing",
    page_icon=":chart_with_downwards_trend:",
    layout="wide",
)

st.title("Black-Scholes Option Pricing")


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    spot = st.number_input(
        "Spot price (S)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="Current market price of the underlying asset.",
    )
    strike = st.number_input(
        "Strike price (K)",
        min_value=0.01,
        value=100.0,
        step=1.0,
        format="%.2f",
        help="Exercise price of the option.",
    )
    time_to_expiry = st.slider(
        "Time to expiry (years)",
        min_value=0.01,
        max_value=5.0,
        value=1.0,
        step=0.01,
        help="Time until the option expires, in years.",
    )
    risk_free_rate = st.number_input(
        "Risk-free rate (r)",
        min_value=0.0,
        max_value=0.20,
        value=0.04,
        step=0.01,
        format="%.2f",
        help="Annualized risk-free interest rate (e.g. Treasury yield).",
    )
    volatility = st.number_input(
        "Volatility (\u03c3)",
        min_value=0.01,
        max_value=2.0,
        value=0.20,
        step=0.01,
        format="%.2f",
        help="Annualized volatility of the underlying asset.",
    )
    run = st.button("Run", type="primary")


# ── Computation ──────────────────────────────────────────────────────────────
if run:
    with st.spinner("Pricing options\u2026"):
        option_price = price_european(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
        )
        call_greeks = calculate_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            is_call=True,
        )
        put_greeks = calculate_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            is_call=False,
        )

        # Price surface data
        spot_range = np.linspace(spot * 0.5, spot * 1.5, 50)
        vol_range = np.linspace(max(0.01, volatility * 0.25), volatility * 2.0, 50)
        call_surface = price_surface(
            spots=spot_range,
            volatilities=vol_range,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            is_call=True,
        )

        # Greeks vs spot data
        greeks_spot_range = np.linspace(spot * 0.5, spot * 1.5, 200)

    st.session_state["bs_has_run"] = True
    st.session_state["bs_data"] = {
        "spot": spot,
        "strike": strike,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "volatility": volatility,
        "option_price": option_price,
        "call_greeks": call_greeks,
        "put_greeks": put_greeks,
        "spot_range": spot_range,
        "vol_range": vol_range,
        "call_surface": call_surface,
        "greeks_spot_range": greeks_spot_range,
    }


_has_run: bool = (
    st.session_state.get("bs_has_run", False) and "bs_data" in st.session_state
)


# ── Guide ────────────────────────────────────────────────────────────────────
with st.expander("What is the Black-Scholes Model?", expanded=(not _has_run)):
    st.markdown(
        """\
The **Black-Scholes-Merton (BSM)** model is the foundational framework for
pricing European-style options. It answers the question: **"What is the fair
value of the right to buy (call) or sell (put) an asset at a fixed price on a
future date?"**

**Key assumptions:**
- The underlying follows geometric Brownian motion (log-normal prices).
- Volatility and the risk-free rate are constant over the option's life.
- No dividends, transaction costs, or taxes.
- Continuous trading is possible.

**How to read the results:**
- **Option Price tab** \u2014 call and put values with put-call parity verification
  and breakeven prices.
- **Greeks tab** \u2014 sensitivity measures showing how the option price responds
  to changes in spot price, volatility, time, and interest rates.
- **Price Surface tab** \u2014 a heatmap of option value across different spot
  prices and volatility levels.
- **Implied Volatility tab** \u2014 back out the market's volatility expectation
  from an observed option price.

**The Greeks:**
- **Delta (\u0394)** \u2014 price change per $1 move in the underlying.
- **Gamma (\u0393)** \u2014 rate of change of Delta; measures convexity.
- **Theta (\u0398)** \u2014 daily time decay; how much value the option loses per day.
- **Vega (\u03bd)** \u2014 price change per 1 % move in implied volatility.
- **Rho (\u03c1)** \u2014 price change per 1 % move in the risk-free rate.
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()


# ── Display from session state ───────────────────────────────────────────────
_d = st.session_state["bs_data"]
_op = _d["option_price"]
_cg = _d["call_greeks"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Call Price", f"{_op.call:.4f}")
c2.metric("Put Price", f"{_op.put:.4f}")
c3.metric("Delta", f"{_cg.delta:.4f}")
c4.metric("Gamma", f"{_cg.gamma:.4f}")
c5.metric("Vega", f"{_cg.vega:.4f}")


tab_price, tab_greeks, tab_surface, tab_iv = st.tabs(
    ["Option Price", "Greeks", "Price Surface", "Implied Volatility"]
)

# ── Tab 1: Option Price ──────────────────────────────────────────────────────
with tab_price:
    _spot = _d["spot"]
    _strike = _d["strike"]
    _r = _d["risk_free_rate"]
    _t = _d["time_to_expiry"]
    _pg = _d["put_greeks"]

    import math

    discount = math.exp(-_r * _t)

    col_call, col_put = st.columns(2)
    with col_call:
        st.subheader("Call Option")
        st.metric("Price", f"${_op.call:.4f}")
        breakeven_call = _strike + _op.call
        st.metric("Breakeven at expiry", f"${breakeven_call:.2f}")

    with col_put:
        st.subheader("Put Option")
        st.metric("Price", f"${_op.put:.4f}")
        breakeven_put = _strike - _op.put
        st.metric("Breakeven at expiry", f"${breakeven_put:.2f}")

    st.divider()
    st.subheader("Put-Call Parity Check")
    lhs = _op.call - _op.put
    rhs = _spot - _strike * discount
    st.markdown(
        f"C \u2212 P = **{lhs:.6f}** &emsp;|\u00a0 "
        f"S \u2212 K\u00b7e^(-rT) = **{rhs:.6f}** &emsp;|\u00a0 "
        f"Difference = **{abs(lhs - rhs):.2e}**"
    )


# ── Tab 2: Greeks ────────────────────────────────────────────────────────────
with tab_greeks:
    _pg = _d["put_greeks"]
    rows = []
    for name in ["delta", "gamma", "theta", "vega", "rho"]:
        rows.append(
            {
                "Greek": name.capitalize(),
                "Call": f"{getattr(_cg, name):.6f}",
                "Put": f"{getattr(_pg, name):.6f}",
            }
        )
    st.table(pd.DataFrame(rows))

    # Greeks vs. spot charts
    st.subheader("Greeks vs. Spot Price")
    _gs = _d["greeks_spot_range"]
    _strike = _d["strike"]
    _t = _d["time_to_expiry"]
    _r = _d["risk_free_rate"]
    _vol = _d["volatility"]

    greek_names = ["delta", "gamma", "theta", "vega", "rho"]
    for gname in greek_names:
        call_vals = []
        put_vals = []
        for s in _gs:
            cg = calculate_greeks(
                spot=float(s),
                strike=_strike,
                time_to_expiry=_t,
                risk_free_rate=_r,
                volatility=_vol,
                is_call=True,
            )
            pg = calculate_greeks(
                spot=float(s),
                strike=_strike,
                time_to_expiry=_t,
                risk_free_rate=_r,
                volatility=_vol,
                is_call=False,
            )
            call_vals.append(getattr(cg, gname))
            put_vals.append(getattr(pg, gname))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=_gs,
                y=call_vals,
                mode="lines",
                name="Call",
                line={"width": 2, "color": "#636EFA"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=_gs,
                y=put_vals,
                mode="lines",
                name="Put",
                line={"width": 2, "color": "#EF553B"},
            )
        )
        fig.add_vline(
            x=_strike,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Strike {_strike:.2f}",
        )
        fig.update_layout(
            title=f"{gname.capitalize()} vs. Spot Price",
            xaxis_title="Spot Price",
            yaxis_title=gname.capitalize(),
            height=350,
        )
        st.plotly_chart(fig, width="stretch")


# ── Tab 3: Price Surface ────────────────────────────────────────────────────
with tab_surface:
    _sr = _d["spot_range"]
    _vr = _d["vol_range"]
    _cs = _d["call_surface"]

    fig = go.Figure(
        data=go.Heatmap(
            z=_cs,
            x=np.round(_sr, 2),
            y=np.round(_vr * 100, 1),
            colorscale="Viridis",
            colorbar={"title": "Price"},
        )
    )
    fig.update_layout(
        title="Call Price Surface (Spot \u00d7 Volatility)",
        xaxis_title="Spot Price",
        yaxis_title="Volatility (%)",
        height=550,
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 4: Implied Volatility ───────────────────────────────────────────────
with tab_iv:
    st.subheader("Implied Volatility Calculator")
    st.markdown("Enter an observed market price to solve for the implied volatility.")

    iv_col1, iv_col2 = st.columns(2)
    with iv_col1:
        iv_market_price = st.number_input(
            "Market price",
            min_value=0.01,
            value=round(_op.call, 2),
            step=0.01,
            format="%.2f",
            key="iv_market_price",
        )
    with iv_col2:
        iv_is_call = st.selectbox(
            "Option type",
            options=["Call", "Put"],
            key="iv_option_type",
        )

    if st.button("Solve", key="iv_solve"):
        try:
            iv = implied_volatility(
                market_price=iv_market_price,
                spot=_d["spot"],
                strike=_d["strike"],
                time_to_expiry=_d["time_to_expiry"],
                risk_free_rate=_d["risk_free_rate"],
                is_call=(iv_is_call == "Call"),
            )
            st.success(f"Implied Volatility: **{iv:.4f}** ({iv * 100:.2f} %)")
        except ValueError as exc:
            st.error(f"Could not solve: {exc}")


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center'>Made with \u2764 by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
