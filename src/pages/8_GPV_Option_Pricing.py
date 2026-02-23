"""GPV Option Pricing page — Streamlit UI only."""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from investment_portfolio.black_scholes import implied_volatility, price_european
from investment_portfolio.gpv import (
    KERNEL_TYPES,
    GPVModelConfig,
    KernelType,
    compute_vix_squared,
    forward_variance,
    kernel_function,
    price_european_gpv,
    simulate_paths,
)

st.set_page_config(
    page_title="GPV Option Pricing",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("GPV Option Pricing")


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")

    kernel_label = st.selectbox(
        "Kernel type",
        options=list(KERNEL_TYPES.keys()),
        index=0,
        help="Volterra kernel for the Gaussian process.",
    )
    kernel_type = KernelType(KERNEL_TYPES[kernel_label])

    hurst = st.slider(
        "Hurst parameter (H)",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        help="Controls roughness of the volatility process.",
    )

    st.subheader("Polynomial Volatility")
    poly_degree = st.number_input(
        "Polynomial degree",
        min_value=0,
        max_value=4,
        value=0,
        step=1,
        help="Degree M of p(x) = a0 + a1*x + ... + aM*x^M.",
    )
    poly_coeffs: list[float] = []
    for i in range(poly_degree + 1):
        default_val = 1.0 if i == 0 else 0.0
        label = f"a{i} (constant)" if i == 0 else f"a{i} (x^{i} term)"
        coeff = st.number_input(
            label,
            value=default_val,
            step=0.1,
            format="%.4f",
            key=f"poly_coeff_{i}",
            help=f"Coefficient of x^{i} in p(x) = a0 + a1*x + ... + aM*x^M.",
        )
        poly_coeffs.append(coeff)

    st.subheader("Market")
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
        help="Annualized risk-free interest rate.",
    )

    st.subheader("Model")
    rho = st.slider(
        "Correlation (rho)",
        min_value=-1.0,
        max_value=1.0,
        value=-0.7,
        step=0.05,
        help="Correlation between asset and volatility Brownian motions.",
    )
    epsilon = st.number_input(
        "Epsilon",
        min_value=0.001,
        value=0.10,
        step=0.01,
        format="%.3f",
        help="Scale parameter for the kernel.",
    )
    flat_xi = st.number_input(
        "Flat forward variance (xi_0)",
        min_value=0.001,
        value=0.04,
        step=0.01,
        format="%.4f",
        help="Initial flat forward variance level (e.g. 0.04 = 20% vol).",
    )

    num_simulations = st.selectbox(
        "Number of simulations",
        options=[1_000, 5_000, 10_000, 50_000, 100_000],
        index=2,
        help="More simulations = more accuracy but slower.",
    )

    run = st.button("Run", type="primary")


# ── Computation ──────────────────────────────────────────────────────────
if run:
    with st.spinner("Running GPV simulation\u2026"):
        maturities = np.linspace(0.01, max(time_to_expiry + 0.5, 2.0), 100)
        xi_0 = np.full(len(maturities), flat_xi)

        config = GPVModelConfig(
            kernel_type=kernel_type,
            hurst=hurst,
            poly_coeffs=tuple(poly_coeffs),
            xi_0=xi_0,
            maturities=maturities,
            rho=rho,
            epsilon=epsilon,
        )

        gpv_price = price_european_gpv(
            config=config,
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            num_steps=100,
            num_simulations=num_simulations,
            rng=np.random.default_rng(42),
        )

        sim_result = simulate_paths(
            config=config,
            spot=spot,
            time_horizon=time_to_expiry,
            num_steps=100,
            num_simulations=min(num_simulations, 1000),
            rng=np.random.default_rng(42),
        )

        fwd_var = forward_variance(config=config, conditioning_time=0.0)

        vix_result = compute_vix_squared(config=config, observation_time=0.0)

        # Black-Scholes reference
        bs_price = price_european(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            volatility=math.sqrt(flat_xi),
        )

    st.session_state["gpv_has_run"] = True
    st.session_state["gpv_data"] = {
        "config": config,
        "gpv_price": gpv_price,
        "bs_price": bs_price,
        "sim_result": sim_result,
        "fwd_var": fwd_var,
        "vix_result": vix_result,
        "spot": spot,
        "strike": strike,
        "time_to_expiry": time_to_expiry,
        "risk_free_rate": risk_free_rate,
        "flat_xi": flat_xi,
        "num_simulations": num_simulations,
    }


_has_run: bool = (
    st.session_state.get("gpv_has_run", False) and "gpv_data" in st.session_state
)


# ── Guide ────────────────────────────────────────────────────────────────
with st.expander("What is the GPV Model?", expanded=(not _has_run)):
    st.markdown(
        """\
The **Gaussian Polynomial Volatility (GPV)** model is a stochastic volatility
framework for option pricing (Bonesini, Callegaro & Grasselli, 2022). It models
stock price volatility as a **polynomial function of a Gaussian Volterra
process**, enabling analytical VIX formulas and realistic implied volatility
smiles.

**Key features:**
- The volatility process is `sigma_t = sqrt(xi_0(t)) * p(X_t) / sqrt(g(t))`
  where `p` is a polynomial and `X_t` is a Gaussian Volterra process.
- Supports multiple kernels: **Exponential** (Markovian, best performer),
  Fractional, Log-modulated, and Shifted Fractional.
- The exponential kernel gives a Markovian model with ~6 parameters that
  outperforms rough (non-Markovian) alternatives.

**How to read the results:**
- **Option Price tab** \u2014 GPV Monte Carlo vs. Black-Scholes comparison.
- **Volatility Paths tab** \u2014 sample stochastic volatility paths.
- **Price Paths tab** \u2014 sample asset price paths under the GPV model.
- **Forward Variance tab** \u2014 the initial forward variance curve xi_0(u).
- **VIX tab** \u2014 model-implied VIX level.
- **Implied Vol Smile tab** \u2014 implied volatility across strikes from
  GPV prices, showing the characteristic skew/smile.

**Reference:** Bonesini, Callegaro & Grasselli (2022),
*Functional quantization of rough volatility and applications to the VIX*,
[arXiv:2212.08297v2](https://arxiv.org/abs/2212.08297v2).
"""
    )

if not _has_run:
    st.info("Configure the parameters in the sidebar and click **Run** to start.")
    st.stop()


# ── Display from session state ───────────────────────────────────────────
_d = st.session_state["gpv_data"]
_gp = _d["gpv_price"]
_bp = _d["bs_price"]
_vix = _d["vix_result"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("GPV Call", f"{_gp.call:.4f}")
c2.metric("GPV Put", f"{_gp.put:.4f}")
c3.metric("Call SE", f"{_gp.call_std_error:.4f}")
c4.metric("Put SE", f"{_gp.put_std_error:.4f}")
c5.metric("VIX", f"{float(_vix.vix_values[0]):.2f}%")


(
    tab_price,
    tab_vol_paths,
    tab_price_paths,
    tab_fwd_var,
    tab_vix,
    tab_smile,
) = st.tabs(
    [
        "Option Price",
        "Volatility Paths",
        "Price Paths",
        "Forward Variance",
        "VIX",
        "Implied Vol Smile",
    ]
)


# ── Tab 1: Option Price ─────────────────────────────────────────────────
with tab_price:
    col_gpv, col_bs = st.columns(2)
    with col_gpv:
        st.subheader("GPV Monte Carlo")
        st.metric("Call Price", f"${_gp.call:.4f}")
        st.metric("Put Price", f"${_gp.put:.4f}")
        st.caption(
            f"SE: call \u00b1{_gp.call_std_error:.4f}, "
            f"put \u00b1{_gp.put_std_error:.4f} "
            f"({_gp.num_simulations:,} sims)"
        )
    with col_bs:
        st.subheader("Black-Scholes Reference")
        st.metric("Call Price", f"${_bp.call:.4f}")
        st.metric("Put Price", f"${_bp.put:.4f}")
        st.caption(
            f"Using flat vol = {math.sqrt(_d['flat_xi']):.4f} "
            f"({math.sqrt(_d['flat_xi']) * 100:.1f}%)"
        )

    st.divider()
    st.subheader("Comparison")
    st.markdown(
        f"**Call difference:** GPV \u2212 BS = "
        f"**{_gp.call - _bp.call:+.4f}**\n\n"
        f"**Put difference:** GPV \u2212 BS = "
        f"**{_gp.put - _bp.put:+.4f}**"
    )


# ── Tab 2: Volatility Paths ─────────────────────────────────────────────
with tab_vol_paths:
    st.subheader("Sample Volatility Paths")
    _sim = _d["sim_result"]
    n_show = min(20, _sim.volatility_paths.shape[1])

    fig = go.Figure()
    for i in range(n_show):
        fig.add_trace(
            go.Scatter(
                x=_sim.time_grid,
                y=_sim.volatility_paths[:, i],
                mode="lines",
                line={"width": 0.8},
                showlegend=False,
                opacity=0.6,
            )
        )
    fig.update_layout(
        title=f"Volatility Paths ({n_show} samples)",
        xaxis_title="Time (years)",
        yaxis_title="Volatility",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 3: Price Paths ──────────────────────────────────────────────────
with tab_price_paths:
    st.subheader("Sample Price Paths")
    n_show = min(20, _sim.price_paths.shape[1])

    fig = go.Figure()
    for i in range(n_show):
        fig.add_trace(
            go.Scatter(
                x=_sim.time_grid,
                y=_sim.price_paths[:, i],
                mode="lines",
                line={"width": 0.8},
                showlegend=False,
                opacity=0.6,
            )
        )
    fig.add_hline(
        y=_d["strike"],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Strike {_d['strike']:.2f}",
    )
    fig.update_layout(
        title=f"Price Paths ({n_show} samples)",
        xaxis_title="Time (years)",
        yaxis_title="Price",
        height=450,
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 4: Forward Variance ─────────────────────────────────────────────
with tab_fwd_var:
    st.subheader("Forward Variance Curve at t=0")
    _fv = _d["fwd_var"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_fv.maturities,
            y=_fv.values,
            mode="lines",
            name="xi_0(u)",
            line={"width": 2, "color": "#636EFA"},
        )
    )
    fig.update_layout(
        title="Forward Variance xi_0(u)",
        xaxis_title="Maturity (years)",
        yaxis_title="Forward Variance",
        height=400,
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 5: VIX ──────────────────────────────────────────────────────────
with tab_vix:
    st.subheader("Model-Implied VIX")
    vix_val = float(_vix.vix_values[0])
    vix_sq = float(_vix.vix_squared[0])

    vc1, vc2 = st.columns(2)
    vc1.metric("VIX", f"{vix_val:.2f}%")
    vc2.metric("VIX\u00b2", f"{vix_sq:.4f}")

    st.markdown(
        "The VIX is computed as "
        "`VIX = 100 * sqrt((1/Delta) * int_T^{T+Delta} xi_T(u) du)` "
        "where Delta = 30/365 days."
    )

    # Show kernel shape
    st.subheader("Kernel Shape")
    _cfg = _d["config"]
    t_grid = np.linspace(0.001, 2.0, 200)
    k_vals = kernel_function(
        t=t_grid,
        kernel_type=_cfg.kernel_type,
        hurst=_cfg.hurst,
        epsilon=_cfg.epsilon,
        theta=_cfg.theta,
        beta=_cfg.beta,
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t_grid,
            y=k_vals,
            mode="lines",
            name="K(t)",
            line={"width": 2, "color": "#EF553B"},
        )
    )
    fig.update_layout(
        title="Volterra Kernel K(t)",
        xaxis_title="t",
        yaxis_title="K(t)",
        height=350,
    )
    st.plotly_chart(fig, width="stretch")


# ── Tab 6: Implied Vol Smile ────────────────────────────────────────────
with tab_smile:
    st.subheader("Implied Volatility Smile")
    st.markdown(
        "GPV prices at multiple strikes are inverted via the Black-Scholes "
        "formula to extract implied volatilities."
    )

    _spot = _d["spot"]
    _tte = _d["time_to_expiry"]
    _r = _d["risk_free_rate"]
    _cfg = _d["config"]
    _n_sims = _d["num_simulations"]

    strikes_smile = np.linspace(_spot * 0.7, _spot * 1.3, 15)
    ivs: list[float | None] = []

    with st.spinner("Computing smile\u2026"):
        for k in strikes_smile:
            try:
                gp = price_european_gpv(
                    config=_cfg,
                    spot=_spot,
                    strike=float(k),
                    time_to_expiry=_tte,
                    risk_free_rate=_r,
                    num_steps=50,
                    num_simulations=min(_n_sims, 10_000),
                    rng=np.random.default_rng(42),
                )
                iv = implied_volatility(
                    market_price=max(gp.call, 1e-8),
                    spot=_spot,
                    strike=float(k),
                    time_to_expiry=_tte,
                    risk_free_rate=_r,
                    is_call=True,
                )
                ivs.append(iv * 100)
            except ValueError:
                ivs.append(None)

    valid_k = [k for k, iv in zip(strikes_smile, ivs, strict=True) if iv is not None]
    valid_iv = [iv for iv in ivs if iv is not None]

    if valid_iv:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=valid_k,
                y=valid_iv,
                mode="lines+markers",
                name="GPV Implied Vol",
                line={"width": 2, "color": "#636EFA"},
            )
        )
        fig.add_vline(
            x=_spot,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"Spot {_spot:.2f}",
        )
        fig.update_layout(
            title="Implied Volatility Smile",
            xaxis_title="Strike",
            yaxis_title="Implied Volatility (%)",
            height=450,
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Could not compute implied volatilities for the given parameters.")


# ── Footer ───────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center'>Made with \u2764 by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
