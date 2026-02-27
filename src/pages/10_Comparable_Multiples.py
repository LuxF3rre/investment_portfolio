"""Comparable Multiples — peer-group relative valuation."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from investment_portfolio.fundamentals import Fundamentals, fetch_fundamentals
from investment_portfolio.theme import (
    BLUE,
    GREEN,
    OVERLAY1,
    RED,
    SAPPHIRE,
    TEAL,
)
from investment_portfolio.valuation import (
    ComparablesResult,
    MultipleValuation,
    calculate_comparables,
)

st.set_page_config(
    page_title="Comparable Multiples",
    page_icon=":bar_chart:",
    layout="wide",
)

st.title("Comparable Multiples")

# ── Constants ────────────────────────────────────────────────────────────────

_MODES: dict[str, str] = {
    "P/E": "Price-to-Earnings",
    "EV/EBITDA": "Enterprise Value to EBITDA",
    "P/B": "Price-to-Book",
    "P/S": "Price-to-Sales",
}

_MODE_HELP: dict[str, str] = {
    "P/E": (
        "Values the target by applying the **peer-median P/E ratio** to the "
        "target's earnings per share. The most widely used equity multiple.\n\n"
        "$$P_{\\text{implied}} = \\widetilde{\\text{P/E}}_{\\text{peers}}"
        " \\times \\text{EPS}_{\\text{target}}$$\n\n"
        "where $\\widetilde{\\text{P/E}}_{\\text{peers}}$ is the median "
        "trailing (or forward) P/E of the peer group.\n\n"
        "**Strengths:** Intuitive, universally reported, directly links "
        "price to profitability.\n\n"
        "**Limitations:** Meaningless for loss-making companies; sensitive "
        "to one-off earnings items and accounting differences."
    ),
    "EV/EBITDA": (
        "Values the target by applying the **peer-median EV/EBITDA** to the "
        "target's EBITDA, then converting from enterprise value to equity "
        "value.\n\n"
        "$$\\text{EV}_{\\text{implied}} = \\widetilde{\\text{EV/EBITDA}}"
        "_{\\text{peers}} \\times \\text{EBITDA}_{\\text{target}}$$\n\n"
        "$$P_{\\text{implied}} = \\frac{\\text{EV}_{\\text{implied}}"
        " - D + C}{N}$$\n\n"
        "where $D$ is total debt, $C$ is total cash, and $N$ is shares "
        "outstanding.\n\n"
        "**Strengths:** Capital-structure neutral; ignores depreciation "
        "differences; works across leveraged and unleveraged peers.\n\n"
        "**Limitations:** Ignores capex intensity; meaningless for negative "
        "EBITDA companies."
    ),
    "P/B": (
        "Values the target by applying the **peer-median P/B ratio** to the "
        "target's book value per share.\n\n"
        "$$P_{\\text{implied}} = \\widetilde{\\text{P/B}}_{\\text{peers}}"
        " \\times \\text{BVPS}_{\\text{target}}$$\n\n"
        "**Strengths:** Useful for asset-heavy sectors (banks, REITs, "
        "insurance); less sensitive to earnings volatility.\n\n"
        "**Limitations:** Book value may not reflect fair value of "
        "intangibles; accounting standards affect comparability."
    ),
    "P/S": (
        "Values the target by applying the **peer-median P/S ratio** to the "
        "target's revenue per share.\n\n"
        "$$P_{\\text{implied}} = \\widetilde{\\text{P/S}}_{\\text{peers}}"
        " \\times \\text{RPS}_{\\text{target}}$$\n\n"
        "**Strengths:** Always positive (revenue is never negative); useful "
        "for early-stage or loss-making companies.\n\n"
        "**Limitations:** Ignores profitability entirely; a company with "
        "high revenue but no margin may appear cheap."
    ),
}

_PEER_COLORS = [TEAL, SAPPHIRE, GREEN, "#c6a0f6", "#f5a97f", "#f4dbd6", "#ee99a0"]


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Parameters")
    target_ticker = (
        st.text_input(
            "Ticker (yfinance)",
            value="AAPL",
            help="Single target ticker symbol.",
        )
        .strip()
        .upper()
    )

    peer_str = st.text_input(
        "Peer tickers (comma-separated)",
        value="MSFT, GOOGL, AMZN, META",
        help="Comparable companies for the peer group.",
    )

    mode = st.selectbox(
        "Valuation multiple",
        list(_MODES.keys()),
        format_func=lambda k: f"{k} — {_MODES[k]}",
    )

    use_forward_pe = False
    if mode == "P/E":
        use_forward_pe = st.toggle(
            "Use forward P/E",
            value=False,
            help="Use forward (analyst consensus) P/E and EPS instead of trailing.",
        )

    run = st.button("Run", type="primary")


# ── Computation ──────────────────────────────────────────────────────────────
if run:
    peer_tickers = [t.strip().upper() for t in peer_str.split(",") if t.strip()]

    if not target_ticker:
        st.error("Enter a target ticker.")
        st.stop()
        msg = "unreachable"
        raise RuntimeError(msg)

    if not peer_tickers:
        st.error("Enter at least one peer ticker.")
        st.stop()
        msg = "unreachable"
        raise RuntimeError(msg)

    with st.spinner("Fetching fundamentals\u2026"):
        try:
            target_fund = fetch_fundamentals(ticker=target_ticker)
        except ValueError:
            st.error(f"Could not fetch data for target **{target_ticker}**.")
            st.stop()
            msg = "unreachable"
            raise

        peer_funds: list[Fundamentals] = []
        for pt in peer_tickers:
            try:
                peer_funds.append(fetch_fundamentals(ticker=pt))
            except ValueError:
                st.warning(f"Skipped **{pt}** — no data available.")

        if not peer_funds:
            st.error("No valid peer data. Cannot run valuation.")
            st.stop()
            msg = "unreachable"
            raise RuntimeError(msg)

    with st.spinner("Calculating valuations\u2026"):
        result = calculate_comparables(
            target=target_fund,
            peers=tuple(peer_funds),
            use_forward_pe=use_forward_pe,
        )

    st.session_state["val_has_run"] = True
    st.session_state["val_data"] = {
        "result": result,
        "mode": mode,
        "target": target_fund,
        "peers": peer_funds,
        "use_forward_pe": use_forward_pe,
    }


# ── State check ──────────────────────────────────────────────────────────────
_has_run: bool = (
    st.session_state.get("val_has_run", False) and "val_data" in st.session_state
)


# ── Educational expanders ────────────────────────────────────────────────────
with st.expander(
    "Understanding Comparable Multiples — overview", expanded=(not _has_run)
):
    st.markdown("""\
### What is Comparable Multiples Valuation?

**Comparable multiples** (or "comps") is a relative valuation method that
estimates a stock's fair value by comparing its financial ratios to those
of similar companies. The core idea: if comparable firms trade at similar
multiples, a deviation suggests the target may be over- or under-valued.

### How it works

1. **Select peers** — companies in the same sector/industry with similar
   size, growth, and risk profiles.
2. **Compute multiples** — for each peer, calculate valuation ratios
   (P/E, EV/EBITDA, P/B, P/S).
3. **Take the median** — the peer-group median multiple serves as the
   "fair" benchmark (median is more robust than mean to outliers).
4. **Apply to target** — multiply the median multiple by the target's
   corresponding financial metric to get an implied share price.
5. **Compare** — if the implied price exceeds the current price, the
   stock may be undervalued relative to peers (and vice versa).

### Which multiple to use?

| Multiple | Best for | Metric |
|---|---|---|
| **P/E** | Profitable companies, quick comparison | Earnings per share |
| **EV/EBITDA** | Capital-structure differences, leveraged firms | EBITDA |
| **P/B** | Asset-heavy sectors (banks, REITs) | Book value per share |
| **P/S** | Early-stage or unprofitable companies | Revenue per share |

### Limitations

- Assumes peers are truly comparable (same risk, growth, margins)
- "Cheap relative to peers" does not mean "cheap in absolute terms" — the
  entire sector could be overvalued
- Backward-looking metrics may not reflect future prospects
- Sensitive to peer selection and one-off items

In practice, using **multiple methods** and triangulating the implied prices
gives the most robust estimate.
""")

with st.expander(
    f"What is {_MODES.get(mode, mode)}?",
    expanded=False,
):
    st.markdown(_MODE_HELP.get(mode, ""))

if not _has_run:
    st.info("Enter a target ticker, peer tickers, and click **Run**.")
    st.stop()


# ── Retrieve data ────────────────────────────────────────────────────────────
_d = st.session_state["val_data"]
result: ComparablesResult = _d["result"]
_mode: str = _d["mode"]
_target: Fundamentals = _d["target"]
_peers: list[Fundamentals] = _d["peers"]
_use_forward_pe: bool = _d["use_forward_pe"]

# Map mode to result field
_MODE_MAP: dict[str, MultipleValuation | None] = {
    "P/E": result.pe,
    "EV/EBITDA": result.ev_ebitda,
    "P/B": result.pb,
    "P/S": result.ps,
}
_selected: MultipleValuation | None = _MODE_MAP.get(_mode)


# ── Metrics row ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price", f"${result.current_price:,.2f}")

if _selected:
    c2.metric("Implied Price", f"${_selected.implied_price:,.2f}")
    delta_str = f"{_selected.upside_pct:+.1%}"
    c3.metric("Upside / Downside", delta_str)
    c4.metric(f"Median {_selected.method}", f"{_selected.median_multiple:.2f}x")
else:
    c2.metric("Implied Price", "N/A")
    c3.metric("Upside / Downside", "N/A")
    c4.metric(f"Median {_mode}", "N/A")


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_summary, tab_peers, tab_sensitivity = st.tabs(
    ["Valuation Summary", "Peer Comparison", "Sensitivity"]
)


# ── Tab 1: Valuation Summary (Football Field) ───────────────────────────────
with tab_summary:
    methods: list[tuple[str, MultipleValuation | None]] = [
        ("P/E", result.pe),
        ("EV/EBITDA", result.ev_ebitda),
        ("P/B", result.pb),
        ("P/S", result.ps),
    ]
    available = [(name, val) for name, val in methods if val is not None]

    if not available:
        st.warning("No multiples could be calculated with available data.")
    else:
        fig = go.Figure()

        labels: list[str] = []
        for name, val in available:
            labels.append(name)
            sens = val.sensitivity
            if sens:
                prices = [s.implied_price for s in sens]
                lo, hi = min(prices), max(prices)
            else:
                lo = hi = val.implied_price

            # Range bar (min to max from sensitivity)
            fig.add_trace(
                go.Bar(
                    x=[hi - lo],
                    y=[name],
                    base=[lo],
                    orientation="h",
                    marker_color=BLUE,
                    opacity=0.3,
                    name=f"{name} range",
                    showlegend=False,
                    hovertemplate=f"{name}: ${lo:,.0f} — ${hi:,.0f}<extra></extra>",
                )
            )

            # Implied price marker (median)
            fig.add_trace(
                go.Scatter(
                    x=[val.implied_price],
                    y=[name],
                    mode="markers",
                    marker={"size": 14, "color": BLUE, "symbol": "diamond"},
                    name=f"{name} implied",
                    showlegend=False,
                    hovertemplate=f"{name} implied: ${val.implied_price:,.2f}<extra></extra>",
                )
            )

        # Current price reference line
        fig.add_vline(
            x=result.current_price,
            line_dash="dash",
            line_color=RED,
            line_width=2,
            annotation_text=f"Current ${result.current_price:,.0f}",
            annotation_position="top",
        )

        fig.update_layout(
            title="Football Field — Implied Price Ranges",
            xaxis_title="Implied Share Price ($)",
            height=300 + 60 * len(available),
            showlegend=False,
            barmode="overlay",
        )
        st.plotly_chart(fig, width="stretch")

        # Summary table
        summary_rows = []
        for name, val in available:
            summary_rows.append(
                {
                    "Multiple": name,
                    "Median": f"{val.median_multiple:.2f}x",
                    "Mean": f"{val.mean_multiple:.2f}x",
                    "Metric": f"{val.target_metric_label}: {val.target_metric_value:,.2f}",
                    "Implied Price": f"${val.implied_price:,.2f}",
                    "Upside": f"{val.upside_pct:+.1%}",
                }
            )
        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            width="stretch",
        )


# ── Tab 2: Peer Comparison ──────────────────────────────────────────────────
with tab_peers:
    if _selected is None:
        st.warning(f"No data for {_mode} — target metric may be missing.")
    else:
        # Bar chart: selected multiple across peers + target
        all_tickers: list[str] = []
        all_values: list[float] = []
        all_colors: list[str] = []

        # Target's own multiple
        target_own: float | None = None
        if _mode == "P/E":
            target_own = _target.forward_pe if _use_forward_pe else _target.trailing_pe
        elif _mode == "EV/EBITDA":
            target_own = _target.ev_to_ebitda
        elif _mode == "P/B":
            target_own = _target.price_to_book
        elif _mode == "P/S":
            target_own = _target.price_to_sales

        if target_own is not None:
            all_tickers.append(f"{_target.ticker} (target)")
            all_values.append(target_own)
            all_colors.append(RED)

        for i, pm in enumerate(_selected.peers):
            if pm.value is not None and pm.value > 0:
                all_tickers.append(pm.ticker)
                all_values.append(pm.value)
                all_colors.append(_PEER_COLORS[i % len(_PEER_COLORS)])

        if all_tickers:
            fig_peers = go.Figure()
            fig_peers.add_trace(
                go.Bar(
                    x=all_tickers,
                    y=all_values,
                    marker_color=all_colors,
                    text=[f"{v:.1f}x" for v in all_values],
                    textposition="outside",
                )
            )

            # Median line
            fig_peers.add_hline(
                y=_selected.median_multiple,
                line_dash="dash",
                line_color=OVERLAY1,
                line_width=1.5,
                annotation_text=f"Median {_selected.median_multiple:.2f}x",
                annotation_position="top right",
            )

            fig_peers.update_layout(
                title=f"{_selected.method} — Peer Comparison",
                yaxis_title=_selected.method,
                height=450,
                showlegend=False,
            )
            st.plotly_chart(fig_peers, width="stretch")
        else:
            st.info("No valid peer multiples to display.")

        # Peer fundamentals table
        st.subheader("Peer Fundamentals")
        fund_rows = []
        for p in _peers:
            fund_rows.append(
                {
                    "Ticker": p.ticker,
                    "Name": p.name,
                    "Sector": p.sector,
                    "Price": f"${p.current_price:,.2f}" if p.current_price else "N/A",
                    "Mkt Cap": f"${p.market_cap / 1e9:,.1f}B"
                    if p.market_cap
                    else "N/A",
                    "P/E": f"{p.trailing_pe:.1f}" if p.trailing_pe else "N/A",
                    "EV/EBITDA": f"{p.ev_to_ebitda:.1f}" if p.ev_to_ebitda else "N/A",
                    "P/B": f"{p.price_to_book:.1f}" if p.price_to_book else "N/A",
                    "P/S": f"{p.price_to_sales:.1f}" if p.price_to_sales else "N/A",
                }
            )
        # Add target row
        fund_rows.insert(
            0,
            {
                "Ticker": f"{_target.ticker} (target)",
                "Name": _target.name,
                "Sector": _target.sector,
                "Price": f"${_target.current_price:,.2f}"
                if _target.current_price
                else "N/A",
                "Mkt Cap": f"${_target.market_cap / 1e9:,.1f}B"
                if _target.market_cap
                else "N/A",
                "P/E": f"{_target.trailing_pe:.1f}" if _target.trailing_pe else "N/A",
                "EV/EBITDA": f"{_target.ev_to_ebitda:.1f}"
                if _target.ev_to_ebitda
                else "N/A",
                "P/B": f"{_target.price_to_book:.1f}"
                if _target.price_to_book
                else "N/A",
                "P/S": f"{_target.price_to_sales:.1f}"
                if _target.price_to_sales
                else "N/A",
            },
        )
        st.dataframe(
            pd.DataFrame(fund_rows),
            hide_index=True,
            width="stretch",
        )


# ── Tab 3: Sensitivity ──────────────────────────────────────────────────────
with tab_sensitivity:
    if _selected is None:
        st.warning(f"No data for {_mode} — cannot show sensitivity.")
    elif not _selected.sensitivity:
        st.info("Insufficient peer data for sensitivity analysis.")
    else:
        sens = _selected.sensitivity

        # Sensitivity table
        sens_rows = []
        for s in sens:
            sens_rows.append(
                {
                    f"{_selected.method} Multiple": f"{s.multiple:.2f}x",
                    "Implied Price": f"${s.implied_price:,.2f}",
                    "Upside / Downside": f"{s.upside_pct:+.1%}",
                }
            )
        st.dataframe(
            pd.DataFrame(sens_rows),
            hide_index=True,
            width="stretch",
        )

        # Sensitivity chart
        multiples = [s.multiple for s in sens]
        prices = [s.implied_price for s in sens]
        upsides = [s.upside_pct * 100 for s in sens]

        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Bar(
                x=[f"{m:.2f}x" for m in multiples],
                y=prices,
                marker_color=[GREEN if u >= 0 else RED for u in upsides],
                text=[f"${p:,.0f}" for p in prices],
                textposition="outside",
                name="Implied Price",
            )
        )

        # Current price reference
        fig_sens.add_hline(
            y=result.current_price,
            line_dash="dash",
            line_color=RED,
            line_width=2,
            annotation_text=f"Current ${result.current_price:,.0f}",
            annotation_position="top right",
        )

        # Median marker
        fig_sens.add_hline(
            y=_selected.implied_price,
            line_dash="dot",
            line_color=BLUE,
            line_width=1.5,
            annotation_text=f"Median implied ${_selected.implied_price:,.0f}",
            annotation_position="bottom right",
        )

        fig_sens.update_layout(
            title=f"Sensitivity — {_selected.method}",
            xaxis_title=f"{_selected.method} Multiple",
            yaxis_title="Implied Share Price ($)",
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_sens, width="stretch")


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center'>"
    "Made with \u2764 by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>"
    "maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
