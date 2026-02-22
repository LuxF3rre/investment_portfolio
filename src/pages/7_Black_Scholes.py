"""Black-Scholes Option Pricing page — Streamlit UI only."""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from investment_portfolio.black_scholes import (
    calculate_greeks,
    implied_volatility,
    price_binary,
    price_discrete_dividend,
    price_european,
    price_fx_option,
    price_perpetual_put,
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
    dividend_yield = st.number_input(
        "Dividend yield (q)",
        min_value=0.0,
        max_value=0.20,
        value=0.0,
        step=0.01,
        format="%.2f",
        help="Continuous dividend yield (e.g. 0.02 for 2%).",
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
            dividend_yield=dividend_yield,
        )
        call_greeks = calculate_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            is_call=True,
            dividend_yield=dividend_yield,
        )
        put_greeks = calculate_greeks(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=volatility,
            is_call=False,
            dividend_yield=dividend_yield,
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
            dividend_yield=dividend_yield,
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
        "dividend_yield": dividend_yield,
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
- No transaction costs or taxes.
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

**Extensions (new tabs):**
- **Binary Options** \u2014 cash-or-nothing and asset-or-nothing digital options
  that pay a fixed amount or the asset value at expiry.
- **Perpetual Put** \u2014 closed-form price for an American put with infinite
  expiration (no time decay).
- **Discrete Dividends** \u2014 European option pricing with discrete
  proportional dividend adjustments.
- **FX Options** \u2014 Garman-Kohlhagen (1983) model for currency options, using
  domestic and foreign interest rates.

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


(
    tab_price,
    tab_greeks,
    tab_surface,
    tab_iv,
    tab_binary,
    tab_perpetual,
    tab_discrete,
    tab_fx,
) = st.tabs(
    [
        "Option Price",
        "Greeks",
        "Price Surface",
        "Implied Volatility",
        "Binary Options",
        "Perpetual Put",
        "Discrete Dividends",
        "FX Options",
    ]
)

# ── Tab 1: Option Price ──────────────────────────────────────────────────────
with tab_price:
    _spot = _d["spot"]
    _strike = _d["strike"]
    _r = _d["risk_free_rate"]
    _t = _d["time_to_expiry"]
    _q = _d["dividend_yield"]
    _pg = _d["put_greeks"]

    discount = math.exp(-_r * _t)
    div_discount = math.exp(-_q * _t)

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
    rhs = _spot * div_discount - _strike * discount
    st.markdown(
        f"C \u2212 P = **{lhs:.6f}** &emsp;|\u00a0 "
        f"S\u00b7e^(-qT) \u2212 K\u00b7e^(-rT) = **{rhs:.6f}** &emsp;|\u00a0 "
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
    _q = _d["dividend_yield"]

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
                dividend_yield=_q,
            )
            pg = calculate_greeks(
                spot=float(s),
                strike=_strike,
                time_to_expiry=_t,
                risk_free_rate=_r,
                volatility=_vol,
                is_call=False,
                dividend_yield=_q,
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
                dividend_yield=_d["dividend_yield"],
            )
            st.success(f"Implied Volatility: **{iv:.4f}** ({iv * 100:.2f} %)")
        except ValueError as exc:
            st.error(f"Could not solve: {exc}")


# ── Tab 5: Binary Options ───────────────────────────────────────────────────
with tab_binary:
    st.subheader("Binary (Digital) Options")
    st.markdown(
        "Binary options pay a fixed amount (cash-or-nothing) or the asset "
        "value (asset-or-nothing) if the option expires in-the-money."
    )

    bp = price_binary(
        spot=_d["spot"],
        strike=_d["strike"],
        time_to_expiry=_d["time_to_expiry"],
        risk_free_rate=_d["risk_free_rate"],
        volatility=_d["volatility"],
        dividend_yield=_d["dividend_yield"],
    )

    bin_rows = [
        {"Type": "Cash-or-Nothing Call", "Price": f"${bp.cash_or_nothing_call:.6f}"},
        {"Type": "Cash-or-Nothing Put", "Price": f"${bp.cash_or_nothing_put:.6f}"},
        {"Type": "Asset-or-Nothing Call", "Price": f"${bp.asset_or_nothing_call:.6f}"},
        {"Type": "Asset-or-Nothing Put", "Price": f"${bp.asset_or_nothing_put:.6f}"},
    ]
    st.table(pd.DataFrame(bin_rows))

    _con_sum = bp.cash_or_nothing_call + bp.cash_or_nothing_put
    _exp_disc = math.exp(-_d["risk_free_rate"] * _d["time_to_expiry"])
    st.markdown(
        f"**Verification:** Cash-or-Nothing Call + Put = "
        f"**{_con_sum:.6f}** "
        f"(expected e^{{-rT}} = {_exp_disc:.6f})"
    )


# ── Tab 6: Perpetual Put ────────────────────────────────────────────────────
with tab_perpetual:
    st.subheader("Perpetual American Put")
    st.markdown(
        "Closed-form price for an American put with infinite expiration. "
        "Requires r > 0 (the formula diverges at r = 0)."
    )

    if _d["risk_free_rate"] <= 0:
        st.error(
            "The perpetual put requires a strictly positive risk-free rate. "
            "Please set r > 0 in the sidebar."
        )
    else:
        pp = price_perpetual_put(
            spot=_d["spot"],
            strike=_d["strike"],
            risk_free_rate=_d["risk_free_rate"],
            volatility=_d["volatility"],
            dividend_yield=_d["dividend_yield"],
        )

        pp_c1, pp_c2, pp_c3 = st.columns(3)
        pp_c1.metric("Perpetual Put Price", f"${pp.price:.4f}")
        pp_c2.metric("Exercise Boundary (S*)", f"${pp.exercise_boundary:.4f}")
        pp_c3.metric("\u03bb\u2082", f"{pp.lambda_2:.4f}")

        # Chart: perpetual put value vs spot
        st.subheader("Perpetual Put Value vs. Spot Price")
        pp_spot_range = np.linspace(_d["strike"] * 0.2, _d["strike"] * 2.0, 200)
        pp_values = []
        for s in pp_spot_range:
            res = price_perpetual_put(
                spot=float(s),
                strike=_d["strike"],
                risk_free_rate=_d["risk_free_rate"],
                volatility=_d["volatility"],
                dividend_yield=_d["dividend_yield"],
            )
            pp_values.append(res.price)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pp_spot_range,
                y=pp_values,
                mode="lines",
                name="Perpetual Put",
                line={"width": 2, "color": "#636EFA"},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pp_spot_range,
                y=[max(_d["strike"] - s, 0) for s in pp_spot_range],
                mode="lines",
                name="Intrinsic (K - S)",
                line={"width": 1, "dash": "dash", "color": "#EF553B"},
            )
        )
        fig.add_vline(
            x=pp.exercise_boundary,
            line_dash="dash",
            line_color="green",
            annotation_text=f"S* = {pp.exercise_boundary:.2f}",
        )
        fig.update_layout(
            xaxis_title="Spot Price",
            yaxis_title="Put Value",
            height=400,
        )
        st.plotly_chart(fig, width="stretch")


# ── Tab 7: Discrete Dividends ───────────────────────────────────────────────
with tab_discrete:
    st.subheader("Discrete Proportional Dividends")
    st.markdown(
        "Adjusts the spot price by a proportional dividend factor: "
        "S_adj = S \u00d7 (1 \u2212 d)^n, then applies standard Black-Scholes."
    )

    dd_col1, dd_col2 = st.columns(2)
    with dd_col1:
        dd_proportion = st.number_input(
            "Dividend proportion (d)",
            min_value=0.0,
            max_value=0.10,
            value=0.02,
            step=0.01,
            format="%.2f",
            key="dd_proportion",
            help="Proportional dividend per payment (e.g. 0.02 for 2%).",
        )
    with dd_col2:
        dd_num = st.number_input(
            "Number of dividends (n)",
            min_value=0,
            max_value=20,
            value=4,
            step=1,
            key="dd_num",
            help="Number of dividend payments before expiry.",
        )

    if st.button("Compute", key="dd_compute"):
        dd_result = price_discrete_dividend(
            spot=_d["spot"],
            strike=_d["strike"],
            time_to_expiry=_d["time_to_expiry"],
            risk_free_rate=_d["risk_free_rate"],
            volatility=_d["volatility"],
            dividend_proportion=dd_proportion,
            num_dividends=dd_num,
        )
        adjusted_spot = _d["spot"] * (1 - dd_proportion) ** dd_num

        st.markdown(f"**Adjusted spot:** ${adjusted_spot:.4f}")

        dd_rows = [
            {
                "": "Discrete Dividend",
                "Call": f"${dd_result.call:.4f}",
                "Put": f"${dd_result.put:.4f}",
            },
            {
                "": "Continuous Dividend (sidebar q)",
                "Call": f"${_op.call:.4f}",
                "Put": f"${_op.put:.4f}",
            },
        ]
        st.table(pd.DataFrame(dd_rows))


# ── Tab 8: FX Options ───────────────────────────────────────────────────────
with tab_fx:
    st.subheader("FX Options (Garman-Kohlhagen)")
    st.markdown(
        "Prices currency options using domestic and foreign risk-free rates. "
        "The foreign rate acts as the continuous dividend yield."
    )

    fx_col1, fx_col2 = st.columns(2)
    with fx_col1:
        fx_domestic = st.number_input(
            "Domestic rate (r_d)",
            min_value=0.0,
            max_value=0.20,
            value=_d["risk_free_rate"],
            step=0.01,
            format="%.2f",
            key="fx_domestic",
            help="Domestic risk-free interest rate.",
        )
    with fx_col2:
        fx_foreign = st.number_input(
            "Foreign rate (r_f)",
            min_value=0.0,
            max_value=0.20,
            value=0.02,
            step=0.01,
            format="%.2f",
            key="fx_foreign",
            help="Foreign risk-free interest rate.",
        )

    if st.button("Compute", key="fx_compute"):
        fx_result = price_fx_option(
            spot=_d["spot"],
            strike=_d["strike"],
            time_to_expiry=_d["time_to_expiry"],
            domestic_rate=fx_domestic,
            foreign_rate=fx_foreign,
            volatility=_d["volatility"],
        )

        fx_cc, fx_cp = st.columns(2)
        with fx_cc:
            st.metric("FX Call Price", f"${fx_result.call:.4f}")
        with fx_cp:
            st.metric("FX Put Price", f"${fx_result.put:.4f}")

        # Parity check
        fx_lhs = fx_result.call - fx_result.put
        fx_rhs = _d["spot"] * math.exp(-fx_foreign * _d["time_to_expiry"]) - _d[
            "strike"
        ] * math.exp(-fx_domestic * _d["time_to_expiry"])
        _fx_diff = abs(fx_lhs - fx_rhs)
        st.markdown(
            "**Put-Call Parity:** "
            f"C \u2212 P = **{fx_lhs:.6f}** | "
            f"S\u00b7e^(-r_f\u00b7T) \u2212 K\u00b7e^(-r_d\u00b7T) "
            f"= **{fx_rhs:.6f}** | "
            f"Diff = **{_fx_diff:.2e}**"
        )


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center'>Made with \u2764 by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
