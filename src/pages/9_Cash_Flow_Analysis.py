"""Cash Flow Analysis — mode-based IRR metric explorer."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from investment_portfolio.irr import (
    calculate_airr,
    calculate_freq,
    calculate_girr,
    calculate_horizon_irr,
    calculate_irr,
    calculate_mirr,
    calculate_npv,
    calculate_npv_profile,
    calculate_pairwise_irr,
    calculate_xirr,
)
from investment_portfolio.theme import (
    BLUE,
    GREEN,
    GREEN_ALPHA,
    OVERLAY1,
    RED,
    RED_ALPHA,
    SAPPHIRE,
    TEAL,
)

st.set_page_config(
    page_title="Cash Flow Analysis",
    page_icon=":money_with_wings:",
    layout="wide",
)

st.title("Cash Flow Analysis")

# ── Constants ────────────────────────────────────────────────────────────────
_MIN_CASH_FLOWS = 2
_MAUVE = "#c6a0f6"

_MODES: dict[str, str] = {
    "IRR": "Internal Rate of Return",
    "XIRR": "Extended IRR (irregular dates)",
    "MIRR": "Modified IRR",
    "GIRR": "Generalised IRR",
    "FREQ": "FREQ / ERR",
    "AIRR": "Average IRR",
    "Horizon IRR": "Horizon IRR",
    "Pairwise IRR": "Pairwise IRR",
}

_MODE_HELP: dict[str, str] = {
    "IRR": (
        "Finds **all** discount rates where NPV = 0. "
        "Uses Descartes' rule to predict the number of roots "
        "and a fine sweep + Brent's method to locate each one.\n\n"
        "**No parameters** — IRR is fully determined by the cash flows."
    ),
    "XIRR": (
        "Extended IRR for **irregularly spaced** cash flows. "
        "Uses day-count fraction (Actual/365) instead of equal "
        "periods. Solved via Newton's method with bisection fallback.\n\n"
        "**No parameters** — enter dates alongside each cash flow in "
        "the editor."
    ),
    "MIRR": (
        "Separates the **finance rate** (for negative CFs) from "
        "the **reinvestment rate** (for positive CFs), producing "
        "a single unambiguous rate. Always unique.\n\n"
        "**Parameters:**\n"
        "- **Finance rate** — the cost of borrowing to cover negative "
        "cash flows (e.g. your loan rate or WACC).\n"
        "- **Reinvestment rate** — the rate at which positive cash flows "
        "are reinvested (e.g. a savings rate or market return)."
    ),
    "GIRR": (
        "**Kulakov & Kastro (2015)** — builds a project balance "
        "that switches between an investment rate and a known "
        "finance rate. Solves for the investment rate that makes "
        "the terminal balance zero. Always unique.\n\n"
        "**Parameters:**\n"
        "- **Finance rate** — the external rate applied when the "
        "project balance is non-negative (excess cash reinvested "
        "externally). The solver finds the investment rate for "
        "periods when capital is tied up in the project."
    ),
    "FREQ": (
        "**Teichroew, Robichek & Montalbano (1965)** — builds an "
        "account balance using a two-rate model. Solves for the "
        "rate applied when capital is tied up in the project. "
        "For conventional CFs, FREQ equals IRR.\n\n"
        "**Parameters:**\n"
        "- **Borrowing rate** — the external rate applied when the "
        "account balance is non-negative. Differs from GIRR only "
        "when you set a different external rate."
    ),
    "AIRR": (
        "**Magni (2010)** — always exists, even when IRR does "
        "not. Combines NPV and a capital schedule to produce a "
        "weighted average holding-period rate. Scale-aware.\n\n"
        "**Parameters:**\n"
        "- **Cost of capital** — the opportunity cost / discount rate "
        "used to compute NPV and the present value of invested capital. "
        "AIRR > cost of capital if and only if NPV > 0.\n"
        "- **Depreciation method** — how the capital schedule is built. "
        "*Straight-line* amortises the initial investment evenly over "
        "all periods. *IRR-implied* uses the first IRR root to derive "
        "the schedule from the project balance equation."
    ),
    "Horizon IRR": (
        "Truncates the project at a chosen period, adds a "
        "terminal/residual value, and computes the IRR of that "
        "shortened stream. Useful when far-future CFs are "
        "uncertain.\n\n"
        "**Parameters:**\n"
        "- **Horizon period** — the period at which the project is "
        "liquidated. Cash flows beyond this are ignored.\n"
        "- **Terminal value** — the residual / salvage / liquidation "
        "value received at the horizon (added to the last cash flow)."
    ),
    "Pairwise IRR": (
        "Computes the IRR of the **incremental** cash flows "
        "(A \u2212 B) between two mutually exclusive projects. "
        "If the pairwise IRR exceeds the cost of capital, "
        "project A is preferred.\n\n"
        "**Parameters:**\n"
        "- **Cost of capital** — the hurdle rate for the decision rule. "
        "If the pairwise IRR exceeds this rate, the incremental "
        "investment in project A (over B) is worthwhile."
    ),
}

_DEFAULT_CF = [-10_000.0, 3_000.0, 4_200.0, 6_800.0]
_DEFAULT_CF_B = [-20_000.0, 25_000.0, 8_000.0]


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_cf_editor(
    *,
    init_cf: list[float],
    use_dates: bool,
    key: str,
    label: str = "Cash Flows",
) -> pd.DataFrame:
    """Build a data editor for cash flows, optionally with dates."""
    st.markdown(f"**{label}**")
    if use_dates:
        _months_per_year = 12
        dates = [
            f"2023-{1 + i:02d}-01"
            if i < _months_per_year
            else f"2024-{1 + i - _months_per_year:02d}-01"
            for i in range(len(init_cf))
        ]
        df = pd.DataFrame({"Date": dates, "Amount": init_cf})
    else:
        df = pd.DataFrame(
            {
                "Period": list(range(len(init_cf))),
                "Amount": init_cf,
            }
        )
    return st.data_editor(
        df,
        num_rows="dynamic",
        width="stretch",
        key=key,
    )


def _extract_cf(
    df: pd.DataFrame,
) -> tuple[float, ...] | None:
    """Extract cash flows from a data editor DataFrame."""
    amounts = df["Amount"].dropna().tolist()
    if len(amounts) < _MIN_CASH_FLOWS:
        return None
    return tuple(float(x) for x in amounts)


def _extract_dates(
    df: pd.DataFrame,
    n: int,
) -> tuple[str, ...] | None:
    """Extract dates from a data editor if present."""
    if "Date" not in df.columns:
        return None
    dates = df["Date"].dropna().tolist()
    if len(dates) != n:
        return None
    return tuple(str(d) for d in dates)


def _period_label(
    t: int,
    dates: tuple[str, ...] | None,
) -> str:
    """Return a date string if dates are provided, else period number."""
    if dates and t < len(dates):
        return dates[t]
    return str(t)


def _period_col(
    dates: tuple[str, ...] | None,
) -> str:
    """Return the column header for the time axis."""
    return "Date" if dates else "Period"


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Mode")
    mode = st.selectbox(
        "Analysis method",
        list(_MODES.keys()),
        format_func=lambda k: f"{k} — {_MODES[k]}",
    )

    st.header("Cash Flows")
    use_dates = mode == "XIRR"

    is_pairwise = mode == "Pairwise IRR"

    if not is_pairwise:
        edited_df = _build_cf_editor(
            init_cf=_DEFAULT_CF,
            use_dates=use_dates,
            key="cf_main",
        )

    # ── Mode-specific parameters ─────────────────────────────────────────
    _HAS_PARAMS = {"MIRR", "GIRR", "FREQ", "AIRR", "Horizon IRR", "Pairwise IRR"}
    if mode in _HAS_PARAMS:
        st.header("Parameters")

    # Parameters that certain modes need
    finance_rate = 0.10
    reinvestment_rate = 0.10
    cost_of_capital = 0.10
    dep_key = "straight-line"
    horizon_period = 1
    terminal_value = 0.0

    if mode == "MIRR":
        finance_rate = st.number_input(
            "Finance rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Rate for financing negative cash flows.",
        )
        reinvestment_rate = st.number_input(
            "Reinvestment rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Rate for reinvesting positive cash flows.",
        )
    elif mode == "GIRR":
        finance_rate = st.number_input(
            "Finance rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Rate applied during borrowing periods (B < 0).",
        )
    elif mode == "FREQ":
        cost_of_capital = st.number_input(
            "Borrowing rate",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Rate applied during borrowing periods (A < 0).",
        )
    elif mode == "AIRR":
        cost_of_capital = st.number_input(
            "Cost of capital",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Opportunity cost rate (r).",
        )
        depreciation_method = st.selectbox(
            "Depreciation method",
            ["Straight-line", "IRR-implied"],
            help=(
                "How to build the capital schedule. "
                "Straight-line is always available; "
                "IRR-implied uses the first IRR root."
            ),
        )
        dep_key = (
            "straight-line" if depreciation_method == "Straight-line" else "irr-implied"
        )
    elif mode == "Horizon IRR":
        horizon_period = st.number_input(
            "Horizon period",
            min_value=1,
            max_value=100,
            value=2,
            step=1,
            help="Period at which the project is liquidated.",
        )
        terminal_value = st.number_input(
            "Terminal value",
            value=0.0,
            step=100.0,
            format="%.2f",
            help="Residual / liquidation value at horizon.",
        )
    elif mode == "Pairwise IRR":
        cost_of_capital = st.number_input(
            "Cost of capital",
            min_value=0.0,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f",
            help="Hurdle rate for the decision rule.",
        )

    run = st.button("Run", type="primary")


# ── Pairwise: cash flow editors in main area ─────────────────────────────────
if is_pairwise:
    pw1, pw2 = st.columns(2)
    with pw1:
        edited_df = _build_cf_editor(
            init_cf=_DEFAULT_CF,
            use_dates=use_dates,
            key="cf_pw_a",
            label="Project A",
        )
    with pw2:
        edited_df_b = _build_cf_editor(
            init_cf=_DEFAULT_CF_B,
            use_dates=use_dates,
            key="cf_pw_b",
            label="Project B",
        )


# ── Computation ──────────────────────────────────────────────────────────────
if run:
    cash_flows = _extract_cf(edited_df)
    if cash_flows is None:
        st.error("Enter at least 2 cash flows.")
        st.stop()
        msg = "unreachable: st.stop should halt execution"
        raise RuntimeError(msg)

    dates = _extract_dates(edited_df, len(cash_flows)) if use_dates else None

    with st.spinner("Analysing\u2026"):
        # Always compute IRR + NPV profile (foundation for every mode)
        irr_result = calculate_irr(cash_flows=cash_flows)
        npv_profile = calculate_npv_profile(cash_flows=cash_flows)

        # Mode-specific computation
        xirr_result = None
        mirr_result = None
        girr_result = None
        freq_result = None
        airr_result = None
        horizon_irr_result = None
        pw_result = None
        cf_b: tuple[float, ...] | None = None

        if mode == "XIRR":
            if dates is None:
                st.warning("XIRR requires dates for every cash flow.")
            else:
                try:
                    xirr_result = calculate_xirr(
                        cash_flows=cash_flows, dates=dates,
                    )
                except ValueError:
                    st.warning("XIRR solver could not find a root.")

        elif mode == "MIRR":
            has_pos = any(x > 0 for x in cash_flows)
            has_neg = any(x < 0 for x in cash_flows)
            if has_pos and has_neg:
                mirr_result = calculate_mirr(
                    cash_flows=cash_flows,
                    finance_rate=finance_rate,
                    reinvestment_rate=reinvestment_rate,
                )
            else:
                st.warning("MIRR requires both positive and negative cash flows.")

        elif mode == "GIRR":
            try:
                girr_result = calculate_girr(
                    cash_flows=cash_flows,
                    finance_rate=finance_rate,
                )
            except ValueError:
                st.warning("GIRR solver could not find a root.")

        elif mode == "FREQ":
            try:
                freq_result = calculate_freq(
                    cash_flows=cash_flows,
                    borrowing_rate=cost_of_capital,
                )
            except ValueError:
                st.warning("FREQ solver could not find a root.")

        elif mode == "AIRR":
            try:
                airr_result = calculate_airr(
                    cash_flows=cash_flows,
                    cost_of_capital=cost_of_capital,
                    depreciation=dep_key,
                )
            except ValueError:
                st.warning("AIRR: PV of capital is zero (degenerate).")

        elif mode == "Horizon IRR":
            n_periods = len(cash_flows) - 1
            if 1 <= horizon_period <= n_periods:
                try:
                    horizon_irr_result = calculate_horizon_irr(
                        cash_flows=cash_flows,
                        horizon=horizon_period,
                        terminal_value=terminal_value,
                    )
                except ValueError:
                    st.warning("Horizon IRR solver could not find a root.")
            else:
                st.warning(f"Horizon must be between 1 and {n_periods}.")

        elif mode == "Pairwise IRR":
            cf_b = _extract_cf(edited_df_b)
            if cf_b is None:
                st.error("Project B needs at least 2 cash flows.")
                st.stop()
                msg = "unreachable: st.stop should halt execution"
                raise RuntimeError(msg)
            pw_result = calculate_pairwise_irr(
                cash_flows_a=cash_flows,
                cash_flows_b=cf_b,
                cost_of_capital=cost_of_capital,
            )

    st.session_state["irr_has_run"] = True
    st.session_state["irr_data"] = {
        "mode": mode,
        "cash_flows": cash_flows,
        "dates": dates,
        "irr_result": irr_result,
        "npv_profile": npv_profile,
        "xirr_result": xirr_result,
        "mirr_result": mirr_result,
        "girr_result": girr_result,
        "freq_result": freq_result,
        "airr_result": airr_result,
        "horizon_irr_result": horizon_irr_result,
        "pw_result": pw_result,
        "cf_b": cf_b,
        "finance_rate": finance_rate,
        "reinvestment_rate": reinvestment_rate,
        "cost_of_capital": cost_of_capital,
    }


# ── State check ──────────────────────────────────────────────────────────────
_has_run: bool = (
    st.session_state.get("irr_has_run", False) and "irr_data" in st.session_state
)


# ── Educational expander ─────────────────────────────────────────────────────
with st.expander(
    f"What is {_MODES.get(mode, mode)}?",
    expanded=(not _has_run),
):
    st.markdown(_MODE_HELP.get(mode, ""))
    st.markdown("""\
**General background** — IRR is the discount rate where NPV = 0.
Pitfalls include multiple roots (Descartes' rule), no solution,
unrealistic reinvestment assumptions, and scale independence.
Each variant above resolves one or more of these issues.
""")

if not _has_run:
    st.info("Select a mode, enter cash flows, and click **Run**.")
    st.stop()


# ── Retrieve data ────────────────────────────────────────────────────────────
_d = st.session_state["irr_data"]
_mode: str = _d["mode"]
cash_flows = _d["cash_flows"]
irr_result = _d["irr_result"]
npv_profile = _d["npv_profile"]
xirr_result = _d["xirr_result"]
mirr_result = _d["mirr_result"]
girr_result = _d["girr_result"]
freq_result = _d["freq_result"]
airr_result = _d["airr_result"]
horizon_irr_result = _d["horizon_irr_result"]
pw_result = _d["pw_result"]
_dates: tuple[str, ...] | None = _d.get("dates")


# ── Metrics row ──────────────────────────────────────────────────────────────
def _fmt_rate(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val:.2%}"


# Row 1: always show IRR basics
c1, c2, c3 = st.columns(3)
c1.metric(
    "Primary IRR",
    _fmt_rate(irr_result.primary_irr),
)
c2.metric("NPV(0%)", f"{irr_result.npv_at_zero:,.2f}")
c3.metric(
    "Sign Changes",
    str(irr_result.sign_info.num_sign_changes),
)

# Row 2: mode-specific headline metric
if _mode == "XIRR" and xirr_result:
    st.metric("XIRR", f"{xirr_result.rate:.4%}")
elif _mode == "MIRR" and mirr_result:
    st.metric("MIRR", f"{mirr_result.rate:.4%}")
elif _mode == "GIRR" and girr_result:
    st.metric("GIRR", f"{girr_result.rate:.4%}")
elif _mode == "FREQ" and freq_result:
    st.metric("FREQ", f"{freq_result.rate:.4%}")
elif _mode == "AIRR" and airr_result:
    st.metric("AIRR", f"{airr_result.rate:.4%}")
elif _mode == "Horizon IRR" and horizon_irr_result:
    st.metric(
        "Horizon IRR",
        _fmt_rate(horizon_irr_result.rate),
    )
elif _mode == "Pairwise IRR" and pw_result:
    st.metric("Pairwise IRR", _fmt_rate(pw_result.rate))


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_npv, tab_timeline, tab_details = st.tabs(
    ["NPV Profile", "Cash Flow Timeline", "Detailed Results"]
)

# ── Tab 1: NPV Profile ──────────────────────────────────────────────────────
with tab_npv:
    fig = go.Figure()

    # NPV curve
    fig.add_trace(
        go.Scatter(
            x=npv_profile.rates * 100,
            y=npv_profile.npv_values,
            mode="lines",
            name="NPV",
            line={"width": 2.5, "color": BLUE},
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=OVERLAY1, line_width=1)

    # Shaded regions
    pos_mask = npv_profile.npv_values >= 0
    neg_mask = npv_profile.npv_values < 0
    if np.any(pos_mask):
        fig.add_trace(
            go.Scatter(
                x=npv_profile.rates[pos_mask] * 100,
                y=npv_profile.npv_values[pos_mask],
                fill="tozeroy",
                mode="none",
                fillcolor=GREEN_ALPHA,
                name="NPV > 0",
                showlegend=False,
            )
        )
    if np.any(neg_mask):
        fig.add_trace(
            go.Scatter(
                x=npv_profile.rates[neg_mask] * 100,
                y=npv_profile.npv_values[neg_mask],
                fill="tozeroy",
                mode="none",
                fillcolor=RED_ALPHA,
                name="NPV < 0",
                showlegend=False,
            )
        )

    # IRR root markers
    for i, zc in enumerate(npv_profile.zero_crossings):
        fig.add_trace(
            go.Scatter(
                x=[zc * 100],
                y=[0],
                mode="markers+text",
                marker={
                    "size": 12,
                    "color": RED,
                    "symbol": "diamond",
                },
                text=[f"IRR {zc:.2%}"],
                textposition="top center",
                name=f"IRR root {i + 1}",
                showlegend=i == 0,
            )
        )

    # Mode-specific marker on zero line
    _marker: tuple[str, float | None, str, str] | None = None
    if _mode == "MIRR" and mirr_result:
        _marker = ("MIRR", mirr_result.rate, GREEN, "square")
    elif _mode == "GIRR" and girr_result:
        _marker = ("GIRR", girr_result.rate, TEAL, "triangle-up")
    elif _mode == "FREQ" and freq_result:
        _marker = ("FREQ", freq_result.rate, SAPPHIRE, "star")
    elif _mode == "AIRR" and airr_result:
        _marker = ("AIRR", airr_result.rate, OVERLAY1, "cross")
    elif _mode == "Horizon IRR" and horizon_irr_result:
        _marker = (
            "Horizon",
            horizon_irr_result.rate,
            _MAUVE,
            "pentagon",
        )

    if _marker is not None and _marker[1] is not None:
        label, rate_val, color, symbol = _marker
        fig.add_trace(
            go.Scatter(
                x=[rate_val * 100],
                y=[0],
                mode="markers+text",
                marker={
                    "size": 12,
                    "color": color,
                    "symbol": symbol,
                },
                text=[f"{label} {rate_val:.2%}"],
                textposition="bottom center",
                name=label,
            )
        )

    # Pairwise: overlay both NPV curves + incremental
    if _mode == "Pairwise IRR" and pw_result and _d["cf_b"]:
        profile_b = calculate_npv_profile(cash_flows=_d["cf_b"])
        profile_inc = calculate_npv_profile(
            cash_flows=pw_result.incremental_flows,
        )
        # Rename Project A trace
        fig.data[0].name = "Project A"
        fig.add_trace(
            go.Scatter(
                x=profile_b.rates * 100,
                y=profile_b.npv_values,
                mode="lines",
                name="Project B",
                line={"width": 2, "color": RED},
            )
        )
        fig.add_trace(
            go.Scatter(
                x=profile_inc.rates * 100,
                y=profile_inc.npv_values,
                mode="lines",
                name="Incremental (A\u2212B)",
                line={"width": 2, "color": TEAL, "dash": "dash"},
            )
        )
        if pw_result.rate is not None:
            fig.add_trace(
                go.Scatter(
                    x=[pw_result.rate * 100],
                    y=[0],
                    mode="markers+text",
                    marker={
                        "size": 12,
                        "color": TEAL,
                        "symbol": "diamond",
                    },
                    text=[f"Pairwise {pw_result.rate:.2%}"],
                    textposition="top center",
                    name="Pairwise IRR",
                )
            )

    fig.update_layout(
        title="NPV Profile",
        xaxis_title="Discount Rate (%)",
        yaxis_title="Net Present Value",
        height=500,
        legend={"orientation": "h", "y": -0.15},
    )
    st.plotly_chart(fig, width="stretch")

    # Contextual callout
    n_roots = len(irr_result.roots)
    si = irr_result.sign_info
    if n_roots == 0:
        st.warning(
            "No IRR root exists \u2014 NPV does not cross zero in the search range."
        )
    elif n_roots == 1 and si.is_conventional:
        st.success(
            "Conventional cash flow with a unique IRR = "
            f"**{irr_result.primary_irr:.2%}**."
        )
    elif n_roots > 1:
        st.warning(
            f"**{n_roots} IRR roots** found "
            f"(sign pattern: `{si.sign_pattern}`). "
            "Standard IRR is ambiguous."
        )


# ── Tab 2: Cash Flow Timeline ───────────────────────────────────────────────
with tab_timeline:
    cf_arr = np.array(cash_flows)
    x_axis: list[str] | list[int] = (
        list(_dates) if _dates else list(range(len(cash_flows)))
    )
    x_label = "Date" if _dates else "Period"
    colors = [GREEN if x >= 0 else RED for x in cash_flows]
    cumulative = np.cumsum(cf_arr)

    fig_tl = go.Figure()
    fig_tl.add_trace(
        go.Bar(
            x=x_axis,
            y=list(cash_flows),
            marker_color=colors,
            name="Cash Flow",
            text=[f"{x:,.0f}" for x in cash_flows],
            textposition="outside",
        )
    )
    fig_tl.add_trace(
        go.Scatter(
            x=x_axis,
            y=cumulative.tolist(),
            mode="lines+markers",
            name="Cumulative",
            line={"width": 2, "color": SAPPHIRE, "dash": "dot"},
            marker={"size": 6},
            yaxis="y2",
        )
    )
    fig_tl.update_layout(
        title="Cash Flow Timeline",
        xaxis_title=x_label,
        yaxis_title="Cash Flow",
        yaxis2={
            "title": "Cumulative",
            "overlaying": "y",
            "side": "right",
        },
        height=400,
        legend={"orientation": "h", "y": -0.15},
    )
    st.plotly_chart(fig_tl, width="stretch")


# ── Tab 3: Detailed Results ─────────────────────────────────────────────────
with tab_details:
    # Always show: IRR roots + sign analysis
    st.subheader("IRR Roots")
    if irr_result.roots:
        root_data = []
        for i, r in enumerate(irr_result.roots):
            npv_check = calculate_npv(cash_flows=cash_flows, rate=r)
            root_data.append(
                {
                    "Root #": i + 1,
                    "Rate (%)": f"{r:.4%}",
                    "NPV check": f"{npv_check:.6f}",
                }
            )
        st.dataframe(
            pd.DataFrame(root_data),
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No IRR roots found in the search range.")

    if xirr_result:
        st.subheader("XIRR")
        st.metric("XIRR", f"{xirr_result.rate:.4%}")
        st.caption("Day count basis: actual/365")

    # Sign change analysis
    st.subheader("Sign Change Analysis")
    si = irr_result.sign_info
    st.markdown(
        f"**Pattern:** `{si.sign_pattern}`  \n"
        f"**Sign changes:** {si.num_sign_changes}  \n"
        f"**Max positive roots:** {si.max_positive_roots}  \n"
        f"**Conventional:** "
        f"{'Yes' if si.is_conventional else 'No'}"
    )

    # Mode-specific breakdown
    st.divider()

    if _mode == "MIRR" and mirr_result:
        st.subheader("MIRR Breakdown")
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric(
            "FV of positives",
            f"{mirr_result.fv_positives:,.2f}",
        )
        mc2.metric(
            "|PV of negatives|",
            f"{abs(mirr_result.pv_negatives):,.2f}",
        )
        mc3.metric("Periods (n)", str(mirr_result.num_periods))
        st.latex(
            r"\text{MIRR} = "
            r"\left(\frac{FV_{+}}{|PV_{-}|}\right)"
            r"^{1/n} - 1 = "
            f"{mirr_result.rate:.4%}"
        )

    elif _mode == "GIRR" and girr_result:
        st.subheader("GIRR Project Balance")
        _pcol = _period_col(_dates)
        bal_data = []
        for t, b in enumerate(girr_result.balances):
            phase = "Investment" if b >= 0 else "Borrowing"
            bal_data.append(
                {
                    _pcol: _period_label(t, _dates),
                    "Balance": f"{b:,.2f}",
                    "Phase": phase,
                }
            )
        st.dataframe(
            pd.DataFrame(bal_data),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"Investment rate (solved): {girr_result.rate:.4%} "
            f"| Finance rate (given): "
            f"{girr_result.finance_rate:.2%}"
        )

    elif _mode == "FREQ" and freq_result:
        st.subheader("FREQ Account Balance")
        _pcol = _period_col(_dates)
        fbal_data = []
        for t, b in enumerate(freq_result.balances):
            phase = "Earning" if b >= 0 else "Borrowing"
            fbal_data.append(
                {
                    _pcol: _period_label(t, _dates),
                    "Balance": f"{b:,.2f}",
                    "Phase": phase,
                }
            )
        st.dataframe(
            pd.DataFrame(fbal_data),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            f"FREQ (solved): {freq_result.rate:.4%} "
            f"| Borrowing rate (given): "
            f"{freq_result.borrowing_rate:.2%}"
        )

    elif _mode == "AIRR" and airr_result:
        st.subheader("AIRR Breakdown")
        ac1, ac2, ac3 = st.columns(3)
        ac1.metric("NPV", f"{airr_result.npv:,.2f}")
        ac2.metric(
            "PV(Capital)",
            f"{airr_result.pv_capital:,.2f}",
        )
        ac3.metric(
            "Depreciation",
            airr_result.depreciation_method,
        )
        _pcol = _period_col(_dates)
        airr_data = []
        for t in range(len(airr_result.capital_schedule)):
            ppr = (
                f"{airr_result.per_period_rates[t]:.4%}"
                if t < len(airr_result.per_period_rates)
                else ""
            )
            airr_data.append(
                {
                    _pcol: _period_label(t, _dates),
                    "Capital c_t": (f"{airr_result.capital_schedule[t]:,.2f}"),
                    "Per-period rate": ppr,
                }
            )
        st.dataframe(
            pd.DataFrame(airr_data),
            hide_index=True,
            width="stretch",
        )
        st.latex(
            r"\text{AIRR} = r + \frac{NPV(r)}{PV(C)} = "
            f"{airr_result.rate:.4%}"
        )

    elif _mode == "Horizon IRR" and horizon_irr_result:
        st.subheader("Horizon IRR Breakdown")
        hc1, hc2, hc3 = st.columns(3)
        hc1.metric(
            "Horizon period",
            str(horizon_irr_result.horizon),
        )
        hc2.metric(
            "Terminal value",
            f"{horizon_irr_result.terminal_value:,.2f}",
        )
        hc3.metric(
            "Horizon IRR",
            _fmt_rate(horizon_irr_result.rate),
        )
        st.markdown("**Truncated cash flows (with terminal value):**")
        _pcol = _period_col(_dates)
        trunc_data = [
            {_pcol: _period_label(t, _dates), "Amount": f"{cf:,.2f}"}
            for t, cf in enumerate(horizon_irr_result.truncated_flows)
        ]
        st.dataframe(
            pd.DataFrame(trunc_data),
            hide_index=True,
            width="stretch",
        )

    elif _mode == "Pairwise IRR" and pw_result:
        st.subheader("Pairwise IRR")
        if pw_result.rate is not None:
            if pw_result.prefer_a:
                st.success(
                    f"Pairwise IRR ({pw_result.rate:.2%}) "
                    f"> cost of capital "
                    f"({pw_result.cost_of_capital:.2%}) "
                    "\u2192 **Prefer Project A**"
                )
            else:
                st.info(
                    f"Pairwise IRR ({pw_result.rate:.2%}) "
                    f"\u2264 cost of capital "
                    f"({pw_result.cost_of_capital:.2%}) "
                    "\u2192 **Prefer Project B**"
                )
        else:
            st.warning("No pairwise IRR found for the incremental cash flows.")

        st.markdown("**Incremental cash flows (A \u2212 B):**")
        _pcol = _period_col(_dates)
        inc_data = [
            {_pcol: _period_label(t, _dates), "Amount": f"{v:,.2f}"}
            for t, v in enumerate(pw_result.incremental_flows)
        ]
        st.dataframe(
            pd.DataFrame(inc_data),
            hide_index=True,
            width="stretch",
        )

    elif _mode == "IRR":
        # No extra breakdown beyond roots + sign analysis
        if irr_result.roots:
            st.info(
                f"Primary IRR: "
                f"**{_fmt_rate(irr_result.primary_irr)}** "
                f"({len(irr_result.roots)} root(s) found)."
            )


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align: center'>"
    "Made with \u2764 by Maurycy Blaszczak"
    " (<a href='https://maurycyblaszczak.com/'>"
    "maurycyblaszczak.com</a>)</div>",
    unsafe_allow_html=True,
)
