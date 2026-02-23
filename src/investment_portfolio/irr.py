"""Cash flow analysis — IRR / XIRR / MIRR / GIRR / FREQ / AIRR / Horizon IRR."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import brentq

from investment_portfolio._validation import validate_finite

__all__ = [
    "AIRRResult",
    "FREQResult",
    "GIRRResult",
    "HorizonIRRResult",
    "IRRResult",
    "MIRRResult",
    "NPVProfile",
    "PairwiseIRRResult",
    "SignChangeInfo",
    "XIRRResult",
    "analyze_sign_changes",
    "calculate_airr",
    "calculate_freq",
    "calculate_girr",
    "calculate_horizon_irr",
    "calculate_irr",
    "calculate_mirr",
    "calculate_npv",
    "calculate_npv_profile",
    "calculate_pairwise_irr",
    "calculate_xirr",
    "find_all_irr_roots",
]


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SignChangeInfo:
    """Descartes' rule sign change analysis for a cash flow stream."""

    num_sign_changes: int
    max_positive_roots: int
    is_conventional: bool
    sign_pattern: str


@dataclass(frozen=True, slots=True)
class IRRResult:
    """Internal Rate of Return result with all roots."""

    roots: tuple[float, ...]
    primary_irr: float | None
    npv_at_zero: float
    sign_info: SignChangeInfo


@dataclass(frozen=True, slots=True)
class XIRRResult:
    """Extended IRR for irregular cash flow dates."""

    rate: float
    npv_at_zero: float
    day_count_basis: int


@dataclass(frozen=True, slots=True)
class MIRRResult:
    """Modified Internal Rate of Return."""

    rate: float
    finance_rate: float
    reinvestment_rate: float
    fv_positives: float
    pv_negatives: float
    num_periods: int


@dataclass(frozen=True, slots=True)
class GIRRResult:
    """Generalised IRR (Kulakov & Kastro 2015)."""

    rate: float
    finance_rate: float
    balances: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FREQResult:
    """FREQ / ERR (Teichroew et al. 1965)."""

    rate: float
    borrowing_rate: float
    balances: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class AIRRResult:
    """Average Internal Rate of Return (Magni 2010)."""

    rate: float
    cost_of_capital: float
    npv: float
    pv_capital: float
    per_period_rates: tuple[float, ...]
    capital_schedule: tuple[float, ...]
    depreciation_method: str


@dataclass(frozen=True, slots=True)
class HorizonIRRResult:
    """Horizon IRR — return assuming liquidation at a given period."""

    rate: float | None
    horizon: int
    terminal_value: float
    truncated_flows: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PairwiseIRRResult:
    """Pairwise (incremental) IRR for mutually exclusive projects."""

    rate: float | None
    incremental_flows: tuple[float, ...]
    prefer_a: bool
    cost_of_capital: float


@dataclass(frozen=True, slots=True)
class NPVProfile:
    """NPV as a function of discount rate, with zero crossings."""

    rates: np.ndarray
    npv_values: np.ndarray
    zero_crossings: tuple[float, ...]


# ── Validation ───────────────────────────────────────────────────────────────

_ROOT_TOL: Final = 1e-9
_DERIV_EPS: Final = 1e-15
_RATE_MIN_DEFAULT: Final = -0.50
_RATE_MAX_DEFAULT: Final = 2.00
_SWEEP_POINTS: Final = 2_000
_NPV_PROFILE_POINTS: Final = 500
_DAY_COUNT_BASIS: Final = 365
_MIN_ELEMENTS: Final = 2


def _validate_cash_flows(*, cash_flows: tuple[float, ...]) -> np.ndarray:
    """Convert and validate a cash flow sequence.

    Args:
        cash_flows: Cash flow amounts per period.

    Returns:
        1-D numpy array of cash flows.

    Raises:
        ValueError: If fewer than 2 elements or non-finite values.
    """
    arr = np.asarray(cash_flows, dtype=np.float64)
    if arr.ndim != 1 or arr.size < _MIN_ELEMENTS:
        msg = "cash_flows must be a 1-D sequence with at least 2 elements"
        raise ValueError(msg)
    validate_finite(arr=arr, name="cash_flows")
    return arr


def _validate_dated_cash_flows(
    *,
    cash_flows: tuple[float, ...],
    dates: tuple[str, ...],
) -> tuple[np.ndarray, list[dt.date]]:
    """Validate and parse dated cash flows.

    Args:
        cash_flows: Cash flow amounts.
        dates: ISO-format date strings matching cash_flows.

    Returns:
        Tuple of (cash_flows array, parsed dates list).

    Raises:
        ValueError: If lengths mismatch or dates not ascending.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    if len(dates) != len(cash_flows):
        msg = "dates and cash_flows must have the same length"
        raise ValueError(msg)
    parsed = [dt.date.fromisoformat(d) for d in dates]
    for i in range(1, len(parsed)):
        if parsed[i] <= parsed[i - 1]:
            msg = "dates must be strictly ascending"
            raise ValueError(msg)
    return cf, parsed


# ── Core functions ───────────────────────────────────────────────────────────


def calculate_npv(*, cash_flows: tuple[float, ...], rate: float) -> float:
    """Compute Net Present Value of periodic cash flows.

    Args:
        cash_flows: Cash flow amounts per period (period 0, 1, ..., n).
        rate: Discount rate per period.

    Returns:
        Net present value.

    Raises:
        ValueError: If cash_flows invalid or rate equals -1.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    if rate == -1.0:
        msg = "rate cannot be -1"
        raise ValueError(msg)
    periods = np.arange(len(cf))
    return float(np.sum(cf / (1.0 + rate) ** periods))


def analyze_sign_changes(*, cash_flows: tuple[float, ...]) -> SignChangeInfo:
    """Analyse sign changes for Descartes' rule of signs.

    Args:
        cash_flows: Cash flow amounts per period.

    Returns:
        SignChangeInfo with count, max roots, conventionality, and pattern.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    nonzero = cf[cf != 0]
    if len(nonzero) < _MIN_ELEMENTS:
        return SignChangeInfo(
            num_sign_changes=0,
            max_positive_roots=0,
            is_conventional=False,
            sign_pattern=" ".join("+" if x > 0 else "-" for x in cf),
        )
    signs = np.sign(nonzero)
    changes = int(np.sum(signs[:-1] != signs[1:]))
    pattern = " ".join("+" if x > 0 else ("-" if x < 0 else "0") for x in cf)
    is_conventional = changes == 1
    return SignChangeInfo(
        num_sign_changes=changes,
        max_positive_roots=changes,
        is_conventional=is_conventional,
        sign_pattern=pattern,
    )


def find_all_irr_roots(
    *,
    cash_flows: tuple[float, ...],
    rate_min: float = _RATE_MIN_DEFAULT,
    rate_max: float = _RATE_MAX_DEFAULT,
) -> tuple[float, ...]:
    """Find all IRR roots in [rate_min, rate_max] via sign-change sweep.

    Sweeps the NPV function over a fine grid, detects sign flips, and
    uses Brent's method to refine each bracket to machine precision.

    Args:
        cash_flows: Cash flow amounts per period.
        rate_min: Lower bound of search range (default -0.50).
        rate_max: Upper bound of search range (default 2.00).

    Returns:
        Tuple of IRR roots sorted ascending, possibly empty.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    rates = np.linspace(rate_min, rate_max, _SWEEP_POINTS)
    periods = np.arange(len(cf))
    # Vectorised NPV at all rates: shape (n_rates,)
    discount = (1.0 + rates[:, np.newaxis]) ** periods[np.newaxis, :]
    npvs = np.sum(cf[np.newaxis, :] / discount, axis=1)

    # Find sign flips
    sign_changes = np.where(npvs[:-1] * npvs[1:] < 0)[0]
    roots: list[float] = []
    for idx in sign_changes:
        try:
            root = brentq(
                lambda r, _cf=cf, _p=periods: float(np.sum(_cf / (1.0 + r) ** _p)),
                rates[idx],
                rates[idx + 1],
                xtol=_ROOT_TOL,
            )
            # Deduplicate: skip if too close to an existing root
            if not any(abs(root - r) < _ROOT_TOL * 100 for r in roots):
                roots.append(root)
        except ValueError:
            continue

    # Also check for NPV ≈ 0 at grid points (exact zeros)
    zero_mask = np.abs(npvs) < _ROOT_TOL
    for idx in np.where(zero_mask)[0]:
        r = float(rates[idx])
        if not any(abs(r - existing) < _ROOT_TOL * 100 for existing in roots):
            roots.append(r)

    return tuple(sorted(roots))


def calculate_irr(
    *,
    cash_flows: tuple[float, ...],
    rate_min: float = _RATE_MIN_DEFAULT,
    rate_max: float = _RATE_MAX_DEFAULT,
) -> IRRResult:
    """Compute IRR with all roots and sign change analysis.

    The primary IRR is the smallest non-negative root, or the largest
    root if all roots are negative.

    Args:
        cash_flows: Cash flow amounts per period.
        rate_min: Lower bound of search range.
        rate_max: Upper bound of search range.

    Returns:
        IRRResult with all roots, primary IRR, NPV(0), and sign info.
    """
    roots = find_all_irr_roots(
        cash_flows=cash_flows, rate_min=rate_min, rate_max=rate_max
    )
    sign_info = analyze_sign_changes(cash_flows=cash_flows)
    npv_at_zero = calculate_npv(cash_flows=cash_flows, rate=0.0)

    primary: float | None = None
    if roots:
        non_negative = [r for r in roots if r >= 0]
        primary = min(non_negative) if non_negative else max(roots)

    return IRRResult(
        roots=roots,
        primary_irr=primary,
        npv_at_zero=npv_at_zero,
        sign_info=sign_info,
    )


# ── XIRR ─────────────────────────────────────────────────────────────────────


def _xnpv(
    *,
    cash_flows: np.ndarray,
    dates: list[dt.date],
    rate: float,
) -> float:
    """Compute NPV for irregularly spaced cash flows.

    Args:
        cash_flows: Array of cash flow amounts.
        dates: Corresponding dates.
        rate: Annual discount rate.

    Returns:
        Net present value using actual/365 day count.
    """
    d0 = dates[0]
    day_fractions = np.array([(d - d0).days / _DAY_COUNT_BASIS for d in dates])
    return float(np.sum(cash_flows / (1.0 + rate) ** day_fractions))


def calculate_xirr(
    *,
    cash_flows: tuple[float, ...],
    dates: tuple[str, ...],
    guess: float = 0.10,
) -> XIRRResult:
    """Compute XIRR for irregularly dated cash flows.

    Uses Newton's method with bisection fallback, similar to the
    implied_volatility solver in black_scholes.py.

    Args:
        cash_flows: Cash flow amounts.
        dates: ISO-format date strings (must be ascending).
        guess: Initial rate guess (default 0.10).

    Returns:
        XIRRResult with rate and NPV(0).

    Raises:
        ValueError: If no solution found.
    """
    cf, parsed = _validate_dated_cash_flows(cash_flows=cash_flows, dates=dates)
    npv_at_zero = _xnpv(cash_flows=cf, dates=parsed, rate=0.0)

    def f(r: float) -> float:
        return _xnpv(cash_flows=cf, dates=parsed, rate=r)

    # Newton's method
    rate = guess
    for _ in range(100):
        npv_val = f(rate)
        if abs(npv_val) < _ROOT_TOL:
            return XIRRResult(
                rate=rate,
                npv_at_zero=npv_at_zero,
                day_count_basis=_DAY_COUNT_BASIS,
            )
        # Numerical derivative
        h = 1e-7
        deriv = (f(rate + h) - f(rate - h)) / (2.0 * h)
        if abs(deriv) < _DERIV_EPS:
            break
        rate = rate - npv_val / deriv
        rate = max(rate, -0.999)
        rate = min(rate, 10.0)

    # Bisection fallback
    lo, hi = -0.999, 10.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        msg = "XIRR: no solution found in [-0.999, 10.0]"
        raise ValueError(msg)
    try:
        root = brentq(f, lo, hi, xtol=_ROOT_TOL)
    except ValueError as exc:
        msg = "XIRR: solver failed"
        raise ValueError(msg) from exc
    return XIRRResult(
        rate=root,
        npv_at_zero=npv_at_zero,
        day_count_basis=_DAY_COUNT_BASIS,
    )


# ── MIRR ─────────────────────────────────────────────────────────────────────


def calculate_mirr(
    *,
    cash_flows: tuple[float, ...],
    finance_rate: float,
    reinvestment_rate: float,
) -> MIRRResult:
    """Compute Modified Internal Rate of Return.

    Assumes negative CFs are financed at finance_rate and positive CFs
    are reinvested at reinvestment_rate.

    Args:
        cash_flows: Cash flow amounts per period.
        finance_rate: Rate for financing negative cash flows.
        reinvestment_rate: Rate for reinvesting positive cash flows.

    Returns:
        MIRRResult with rate and component values.

    Raises:
        ValueError: If no positive or no negative cash flows.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    n = len(cf) - 1

    positives = np.where(cf > 0, cf, 0.0)
    negatives = np.where(cf < 0, cf, 0.0)

    if np.sum(positives) == 0:
        msg = "MIRR requires at least one positive cash flow"
        raise ValueError(msg)
    if np.sum(negatives) == 0:
        msg = "MIRR requires at least one negative cash flow"
        raise ValueError(msg)

    # FV of positives at reinvestment rate
    periods = np.arange(len(cf))
    fv_pos = float(np.sum(positives * (1.0 + reinvestment_rate) ** (n - periods)))

    # PV of negatives at finance rate
    pv_neg = float(np.sum(negatives / (1.0 + finance_rate) ** periods))

    rate = (fv_pos / abs(pv_neg)) ** (1.0 / n) - 1.0

    return MIRRResult(
        rate=rate,
        finance_rate=finance_rate,
        reinvestment_rate=reinvestment_rate,
        fv_positives=fv_pos,
        pv_negatives=pv_neg,
        num_periods=n,
    )


# ── GIRR (Kulakov & Kastro 2015) ─────────────────────────────────────────────


def _girr_terminal_balance(
    *,
    cash_flows: np.ndarray,
    r_inv: float,
    r_fin: float,
) -> float:
    """Compute terminal project balance for GIRR.

    Args:
        cash_flows: Array of cash flows.
        r_inv: Investment rate (applied when balance < 0, i.e. capital
            is tied up in the project).
        r_fin: Finance/reinvestment rate (applied when balance >= 0,
            i.e. excess cash is reinvested externally).

    Returns:
        Terminal balance B_n.
    """
    balance = float(cash_flows[0])
    for t in range(1, len(cash_flows)):
        r = r_inv if balance < 0 else r_fin
        balance = balance * (1.0 + r) + float(cash_flows[t])
    return balance


def _girr_balances(
    *,
    cash_flows: np.ndarray,
    r_inv: float,
    r_fin: float,
) -> tuple[float, ...]:
    """Compute all project balances for GIRR.

    Args:
        cash_flows: Array of cash flows.
        r_inv: Investment rate (applied when balance < 0).
        r_fin: Finance/reinvestment rate (applied when balance >= 0).

    Returns:
        Tuple of balances B_0, B_1, ..., B_n.
    """
    balances = [float(cash_flows[0])]
    for t in range(1, len(cash_flows)):
        r = r_inv if balances[-1] < 0 else r_fin
        balances.append(balances[-1] * (1.0 + r) + float(cash_flows[t]))
    return tuple(balances)


def calculate_girr(
    *,
    cash_flows: tuple[float, ...],
    finance_rate: float,
) -> GIRRResult:
    """Compute Generalised IRR (Kulakov & Kastro 2015).

    Uses a two-rate project balance where the investment rate (r_inv) is
    the unknown.  r_inv is applied when the balance is negative (capital
    tied up in the project); r_fin is applied when the balance is
    non-negative (excess cash reinvested externally).  B_n is monotonic
    in r_inv when r_fin is fixed, guaranteeing a unique root.

    For any project starting with CF_0 < 0, B_0 < 0, so r_inv is always
    used — making GIRR truly generalized.

    Args:
        cash_flows: Cash flow amounts per period.
        finance_rate: Rate applied during reinvestment periods (B >= 0).

    Returns:
        GIRRResult with unique rate and balance schedule.

    Raises:
        ValueError: If solver fails to find a root.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)

    def f(r_inv: float) -> float:
        return _girr_terminal_balance(cash_flows=cf, r_inv=r_inv, r_fin=finance_rate)

    root = brentq(f, -0.999, 10.0, xtol=_ROOT_TOL)

    balances = _girr_balances(cash_flows=cf, r_inv=root, r_fin=finance_rate)

    return GIRRResult(
        rate=root,
        finance_rate=finance_rate,
        balances=balances,
    )


# ── FREQ / ERR (Teichroew et al. 1965) ──────────────────────────────────────


def _freq_terminal_balance(
    *,
    cash_flows: np.ndarray,
    freq: float,
    r_borrow: float,
) -> float:
    """Compute terminal account balance for FREQ.

    Args:
        cash_flows: Array of cash flows.
        freq: Rate applied when balance < 0 (capital tied up in
            the project).
        r_borrow: Reinvestment/borrowing rate applied when balance >= 0
            (excess cash reinvested externally).

    Returns:
        Terminal account balance A_n.
    """
    balance = float(cash_flows[0])
    for t in range(1, len(cash_flows)):
        r = freq if balance < 0 else r_borrow
        balance = balance * (1.0 + r) + float(cash_flows[t])
    return balance


def _freq_balances(
    *,
    cash_flows: np.ndarray,
    freq: float,
    r_borrow: float,
) -> tuple[float, ...]:
    """Compute all account balances for FREQ.

    Args:
        cash_flows: Array of cash flows.
        freq: Rate applied when balance < 0.
        r_borrow: Reinvestment/borrowing rate applied when balance >= 0.

    Returns:
        Tuple of balances A_0, A_1, ..., A_n.
    """
    balances = [float(cash_flows[0])]
    for t in range(1, len(cash_flows)):
        r = freq if balances[-1] < 0 else r_borrow
        balances.append(balances[-1] * (1.0 + r) + float(cash_flows[t]))
    return tuple(balances)


def calculate_freq(
    *,
    cash_flows: tuple[float, ...],
    borrowing_rate: float,
) -> FREQResult:
    """Compute FREQ / ERR (Teichroew et al. 1965).

    Uses a two-rate account balance where FREQ is the unknown rate
    applied when the balance is negative (capital tied up in the
    project), and borrowing_rate is applied when the balance is
    non-negative (excess cash reinvested externally).

    For any project starting with CF_0 < 0, A_0 < 0, so FREQ is
    always used.  GIRR and FREQ differ because they use different
    given rates (finance_rate vs borrowing_rate), producing different
    results for mixed projects where the balance alternates sign.

    Args:
        cash_flows: Cash flow amounts per period.
        borrowing_rate: Rate applied during reinvestment periods (A >= 0).

    Returns:
        FREQResult with unique rate and balance schedule.

    Raises:
        ValueError: If solver fails to find a root.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)

    def f(freq: float) -> float:
        return _freq_terminal_balance(cash_flows=cf, freq=freq, r_borrow=borrowing_rate)

    root = brentq(f, -0.999, 10.0, xtol=_ROOT_TOL)

    balances = _freq_balances(cash_flows=cf, freq=root, r_borrow=borrowing_rate)

    return FREQResult(
        rate=root,
        borrowing_rate=borrowing_rate,
        balances=balances,
    )


# ── AIRR (Magni 2010) ───────────────────────────────────────────────────────


def calculate_airr(
    *,
    cash_flows: tuple[float, ...],
    cost_of_capital: float,
    depreciation: str = "straight-line",
) -> AIRRResult:
    """Compute Average Internal Rate of Return (Magni 2010).

    AIRR always exists, even when IRR does not. It depends on the cost
    of capital and a capital schedule (depreciation method).

    Shortcut formula: AIRR = r + NPV(r) / PV(C)
    where PV(C) = sum of c_{t-1} / (1+r)^t for t in 1..n.

    Args:
        cash_flows: Cash flow amounts per period.
        cost_of_capital: Opportunity cost rate (r).
        depreciation: Capital schedule method — "straight-line" or "irr-implied".

    Returns:
        AIRRResult with rate and full breakdown.

    Raises:
        ValueError: If PV of capital is zero (degenerate case).
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    n = len(cf) - 1
    r = cost_of_capital
    npv = calculate_npv(cash_flows=tuple(cf), rate=r)

    # Build capital schedule
    if depreciation == "irr-implied":
        # Try to use IRR for the capital schedule
        roots = find_all_irr_roots(cash_flows=tuple(cf))
        if roots:
            irr_rate = roots[0]
            schedule: list[float] = [float(cf[0])]
            for t in range(1, n):
                c_prev = schedule[-1]
                schedule.append(c_prev * (1.0 + irr_rate) - float(cf[t]))
            capital = schedule
        else:
            # Fall back to straight-line
            c0 = abs(float(cf[0]))
            capital = [c0 * (1.0 - t / n) for t in range(n)]
    else:
        # Straight-line depreciation
        c0 = abs(float(cf[0]))
        capital = [c0 * (1.0 - t / n) for t in range(n)]

    # PV of capital: sum c_{t-1} / (1+r)^t for t=1..n
    pv_capital = sum(capital[t] / (1.0 + r) ** (t + 1) for t in range(n))

    if abs(pv_capital) < _DERIV_EPS:
        msg = "AIRR: PV of capital is zero (degenerate case)"
        raise ValueError(msg)

    airr_rate = r + npv / pv_capital

    # Per-period rates: r_t = CF_t / c_{t-1} + 1 (if capital != 0)
    per_period: list[float] = []
    for t in range(1, n + 1):
        c_prev = capital[t - 1] if t - 1 < len(capital) else 0.0
        if abs(c_prev) > _DERIV_EPS:
            per_period.append(float(cf[t]) / c_prev + 1.0 - 1.0)
        else:
            per_period.append(0.0)

    return AIRRResult(
        rate=airr_rate,
        cost_of_capital=r,
        npv=npv,
        pv_capital=pv_capital,
        per_period_rates=tuple(per_period),
        capital_schedule=tuple(capital),
        depreciation_method=depreciation,
    )


# ── Horizon IRR ──────────────────────────────────────────────────────────────


def calculate_horizon_irr(
    *,
    cash_flows: tuple[float, ...],
    horizon: int,
    terminal_value: float,
) -> HorizonIRRResult:
    """Compute Horizon IRR — the return assuming liquidation at *horizon*.

    Truncates the cash flow stream at *horizon* and adds *terminal_value*
    to the last period, then solves for the IRR of this modified stream.

    This is useful when a project has uncertain far-future cash flows and
    the analyst prefers to assume a sale/liquidation at a specific point.

    Args:
        cash_flows: Cash flow amounts per period.
        horizon: Period at which to truncate (1 <= horizon < n).
        terminal_value: Residual / liquidation value added at horizon.

    Returns:
        HorizonIRRResult with the horizon IRR and truncated stream.

    Raises:
        ValueError: If horizon is out of range.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    n = len(cf) - 1
    if horizon < 1 or horizon > n:
        msg = f"horizon must be in [1, {n}], got {horizon}"
        raise ValueError(msg)

    truncated = list(cf[: horizon + 1].copy())
    truncated[horizon] += terminal_value
    truncated_tuple = tuple(float(x) for x in truncated)

    roots = find_all_irr_roots(cash_flows=truncated_tuple)
    primary: float | None = None
    if roots:
        non_negative = [r for r in roots if r >= 0]
        primary = min(non_negative) if non_negative else roots[0]

    return HorizonIRRResult(
        rate=primary,
        horizon=horizon,
        terminal_value=terminal_value,
        truncated_flows=truncated_tuple,
    )


# ── Pairwise IRR ─────────────────────────────────────────────────────────────


def calculate_pairwise_irr(
    *,
    cash_flows_a: tuple[float, ...],
    cash_flows_b: tuple[float, ...],
    cost_of_capital: float,
) -> PairwiseIRRResult:
    """Compute pairwise (incremental) IRR for project comparison.

    The incremental stream is (A - B). If pairwise IRR > cost of capital,
    project A is preferred; otherwise project B.

    Args:
        cash_flows_a: Cash flows for project A.
        cash_flows_b: Cash flows for project B.
        cost_of_capital: Hurdle rate for decision rule.

    Returns:
        PairwiseIRRResult with incremental IRR and preference.
    """
    a = np.array(cash_flows_a, dtype=np.float64)
    b = np.array(cash_flows_b, dtype=np.float64)

    # Pad shorter array with zeros
    max_len = max(len(a), len(b))
    a_padded = np.zeros(max_len)
    b_padded = np.zeros(max_len)
    a_padded[: len(a)] = a
    b_padded[: len(b)] = b

    incremental = tuple(float(x) for x in (a_padded - b_padded))
    roots = find_all_irr_roots(cash_flows=incremental)

    primary: float | None = None
    if roots:
        non_negative = [r for r in roots if r >= 0]
        primary = min(non_negative) if non_negative else roots[0]

    prefer_a = primary is not None and primary > cost_of_capital

    return PairwiseIRRResult(
        rate=primary,
        incremental_flows=incremental,
        prefer_a=prefer_a,
        cost_of_capital=cost_of_capital,
    )


# ── NPV profile ─────────────────────────────────────────────────────────────


def calculate_npv_profile(
    *,
    cash_flows: tuple[float, ...],
    rate_min: float = _RATE_MIN_DEFAULT,
    rate_max: float = _RATE_MAX_DEFAULT,
    num_points: int = _NPV_PROFILE_POINTS,
) -> NPVProfile:
    """Compute NPV at many discount rates for plotting.

    Args:
        cash_flows: Cash flow amounts per period.
        rate_min: Lowest rate to evaluate.
        rate_max: Highest rate to evaluate.
        num_points: Number of evaluation points (default 500).

    Returns:
        NPVProfile with rates, NPV values, and zero crossings.
    """
    cf = _validate_cash_flows(cash_flows=cash_flows)
    rates = np.linspace(rate_min, rate_max, num_points)
    periods = np.arange(len(cf))
    discount = (1.0 + rates[:, np.newaxis]) ** periods[np.newaxis, :]
    npv_values = np.sum(cf[np.newaxis, :] / discount, axis=1)

    # Find zero crossings via sign flips
    crossings: list[float] = []
    sign_flips = np.where(npv_values[:-1] * npv_values[1:] < 0)[0]
    for idx in sign_flips:
        try:
            root = brentq(
                lambda r, _cf=cf, _p=periods: float(np.sum(_cf / (1.0 + r) ** _p)),
                float(rates[idx]),
                float(rates[idx + 1]),
                xtol=_ROOT_TOL,
            )
            crossings.append(root)
        except ValueError:
            continue

    return NPVProfile(
        rates=rates,
        npv_values=npv_values,
        zero_crossings=tuple(crossings),
    )
