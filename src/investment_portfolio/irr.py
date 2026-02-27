"""Cash flow analysis — IRR / XIRR / MIRR / GIRR / AIRR / Horizon IRR."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy.optimize import brentq

from investment_portfolio._validation import validate_finite

__all__ = [
    "AIRRResult",
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
    day_count_convention: str


@dataclass(frozen=True, slots=True)
class MIRRResult:
    """Modified Internal Rate of Return."""

    rate: float
    finance_rate: float
    reinvestment_rate: float
    fv_positives: float
    pv_negatives: float
    time_span: float


@dataclass(frozen=True, slots=True)
class GIRRResult:
    """Generalised IRR (Kulakov & Kastro 2015)."""

    rate: float
    finance_rate: float
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
_SOLVER_LO: Final = -0.999
_SOLVER_HI: Final = 10.0
_SOLVER_FLOOR: Final = -1.0
_SOLVER_CEIL: Final = 100.0
_SWEEP_POINTS: Final = 2_000
_PROFILE_LO: Final = -0.50
_PROFILE_HI: Final = 2.00
_NPV_PROFILE_POINTS: Final = 500
_DAY_COUNT_CONVENTION: Final = "actual/actual"
_MIN_ELEMENTS: Final = 2


def _year_fraction(d1: dt.date, d2: dt.date) -> float:
    """Year fraction using actual/actual (ISDA) convention.

    Each calendar year contributes ``days_in_segment / days_in_year``,
    so Jan 1 to Jan 1 of adjacent years always equals exactly 1.0
    regardless of leap years.

    Args:
        d1: Start date.
        d2: End date (must be >= *d1*).

    Returns:
        Year fraction (0.0 when *d1* == *d2*).
    """
    if d1 >= d2:
        return 0.0
    total = 0.0
    current = d1
    while current < d2:
        year_start = dt.date(current.year, 1, 1)
        year_end = dt.date(current.year + 1, 1, 1)
        days_in_year = (year_end - year_start).days  # 365 or 366
        segment_end = min(year_end, d2)
        total += (segment_end - current).days / days_in_year
        current = segment_end
    return total


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


def _validate_solver_bounds(
    *,
    rate_min: float,
    rate_max: float,
) -> tuple[float, float]:
    """Validate and return solver bounds.

    Args:
        rate_min: Lower bound (must be > -1).
        rate_max: Upper bound (must be <= 100).

    Returns:
        Tuple of (rate_min, rate_max).

    Raises:
        ValueError: If bounds are invalid.
    """
    if rate_min <= _SOLVER_FLOOR:
        msg = f"rate_min must be > {_SOLVER_FLOOR}, got {rate_min}"
        raise ValueError(msg)
    if rate_max > _SOLVER_CEIL:
        msg = f"rate_max must be <= {_SOLVER_CEIL}, got {rate_max}"
        raise ValueError(msg)
    if rate_min >= rate_max:
        msg = f"rate_min ({rate_min}) must be < rate_max ({rate_max})"
        raise ValueError(msg)
    return rate_min, rate_max


def _resolve_periods(
    *,
    cash_flows: tuple[float, ...],
    dates: tuple[str, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate inputs and return discount exponents.

    When *dates* is ``None``, exponents are ``0, 1, 2, ...`` (equal
    periods).  When *dates* is provided, exponents are actual/365
    year fractions from the first date.

    Args:
        cash_flows: Cash flow amounts.
        dates: Optional ISO-format date strings.

    Returns:
        Tuple of (validated cash-flow array, periods array).
    """
    if dates is not None:
        cf, parsed = _validate_dated_cash_flows(cash_flows=cash_flows, dates=dates)
        d0 = parsed[0]
        periods = np.array(
            [_year_fraction(d0, d) for d in parsed],
            dtype=np.float64,
        )
    else:
        cf = _validate_cash_flows(cash_flows=cash_flows)
        periods = np.arange(len(cf), dtype=np.float64)
    return cf, periods


# ── Core functions ───────────────────────────────────────────────────────────


def calculate_npv(
    *,
    cash_flows: tuple[float, ...],
    rate: float,
    dates: tuple[str, ...] | None = None,
) -> float:
    """Compute Net Present Value.

    When *dates* is ``None``, uses equal-period discounting.  When
    *dates* is provided, uses actual/365 year fractions.

    Args:
        cash_flows: Cash flow amounts per period (period 0, 1, ..., n).
        rate: Discount rate (per period or annualised when dates given).
        dates: Optional ISO-format date strings for irregular spacing.

    Returns:
        Net present value.

    Raises:
        ValueError: If cash_flows invalid or rate equals -1.
    """
    cf, periods = _resolve_periods(cash_flows=cash_flows, dates=dates)
    if rate == -1.0:
        msg = "rate cannot be -1"
        raise ValueError(msg)
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
    dates: tuple[str, ...] | None = None,
    rate_min: float = _SOLVER_LO,
    rate_max: float = _SOLVER_HI,
) -> tuple[float, ...]:
    """Find all IRR roots in [rate_min, rate_max] via sign-change sweep.

    Sweeps the NPV function over a fine grid, detects sign flips, and
    uses Brent's method to refine each bracket to machine precision.

    When *dates* is provided, uses actual/365 year fractions so the
    returned roots are annualised rates.

    Args:
        cash_flows: Cash flow amounts per period.
        dates: Optional ISO-format date strings for irregular spacing.
        rate_min: Lower bound of search range (default -0.999).
        rate_max: Upper bound of search range (default 10.0).

    Returns:
        Tuple of IRR roots sorted ascending, possibly empty.
    """
    cf, periods = _resolve_periods(cash_flows=cash_flows, dates=dates)
    rates = np.linspace(rate_min, rate_max, _SWEEP_POINTS)
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

    # Also check for NPV ~= 0 at grid points (exact zeros)
    zero_mask = np.abs(npvs) < _ROOT_TOL
    for idx in np.where(zero_mask)[0]:
        r = float(rates[idx])
        if not any(abs(r - existing) < _ROOT_TOL * 100 for existing in roots):
            roots.append(r)

    return tuple(sorted(roots))


def calculate_irr(
    *,
    cash_flows: tuple[float, ...],
    dates: tuple[str, ...] | None = None,
    rate_min: float = _SOLVER_LO,
    rate_max: float = _SOLVER_HI,
) -> IRRResult:
    """Compute IRR with all roots and sign change analysis.

    The primary IRR is the smallest non-negative root, or the largest
    root if all roots are negative.  When *dates* is provided, roots
    are annualised rates using actual/365 day count.

    Args:
        cash_flows: Cash flow amounts per period.
        dates: Optional ISO-format date strings for irregular spacing.
        rate_min: Lower bound of search range.
        rate_max: Upper bound of search range.

    Returns:
        IRRResult with all roots, primary IRR, NPV(0), and sign info.
    """
    roots = find_all_irr_roots(
        cash_flows=cash_flows, dates=dates, rate_min=rate_min, rate_max=rate_max
    )
    sign_info = analyze_sign_changes(cash_flows=cash_flows)
    npv_at_zero = calculate_npv(cash_flows=cash_flows, rate=0.0, dates=dates)

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
        Net present value using actual/actual day count.
    """
    d0 = dates[0]
    day_fractions = np.array([_year_fraction(d0, d) for d in dates])
    return float(np.sum(cash_flows / (1.0 + rate) ** day_fractions))


def calculate_xirr(
    *,
    cash_flows: tuple[float, ...],
    dates: tuple[str, ...],
    guess: float = 0.10,
    rate_min: float = _SOLVER_LO,
    rate_max: float = _SOLVER_HI,
) -> XIRRResult:
    """Compute XIRR for irregularly dated cash flows.

    Uses Newton's method with bisection fallback.

    Args:
        cash_flows: Cash flow amounts.
        dates: ISO-format date strings (must be ascending).
        guess: Initial rate guess (default 0.10).
        rate_min: Lower solver bound (default -0.999, must be > -1).
        rate_max: Upper solver bound (default 10.0, must be <= 100).

    Returns:
        XIRRResult with rate and NPV(0).

    Raises:
        ValueError: If no solution found or bounds are invalid.
    """
    cf, parsed = _validate_dated_cash_flows(cash_flows=cash_flows, dates=dates)
    lo, hi = _validate_solver_bounds(rate_min=rate_min, rate_max=rate_max)
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
                day_count_convention=_DAY_COUNT_CONVENTION,
            )
        # Numerical derivative
        h = 1e-7
        deriv = (f(rate + h) - f(rate - h)) / (2.0 * h)
        if abs(deriv) < _DERIV_EPS:
            break
        rate = rate - npv_val / deriv
        rate = max(rate, lo)
        rate = min(rate, hi)

    # Bisection fallback
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        msg = f"XIRR: no solution found in [{lo}, {hi}]"
        raise ValueError(msg)
    try:
        root = brentq(f, lo, hi, xtol=_ROOT_TOL)
    except ValueError as exc:
        msg = "XIRR: solver failed"
        raise ValueError(msg) from exc
    return XIRRResult(
        rate=root,
        npv_at_zero=npv_at_zero,
        day_count_convention=_DAY_COUNT_CONVENTION,
    )


# ── MIRR ─────────────────────────────────────────────────────────────────────


def calculate_mirr(
    *,
    cash_flows: tuple[float, ...],
    finance_rate: float,
    reinvestment_rate: float,
    dates: tuple[str, ...] | None = None,
) -> MIRRResult:
    """Compute Modified Internal Rate of Return.

    Assumes negative CFs are financed at *finance_rate* and positive
    CFs are reinvested at *reinvestment_rate*.  When *dates* is
    provided, uses actual/365 year fractions so the returned rate is
    annualised.

    Args:
        cash_flows: Cash flow amounts per period.
        finance_rate: Rate for financing negative cash flows.
        reinvestment_rate: Rate for reinvesting positive cash flows.
        dates: Optional ISO-format date strings for irregular spacing.

    Returns:
        MIRRResult with rate and component values.

    Raises:
        ValueError: If no positive or no negative cash flows.
    """
    cf, periods = _resolve_periods(cash_flows=cash_flows, dates=dates)
    time_span = float(periods[-1])  # n for equal periods, year fraction for dates

    positives = np.where(cf > 0, cf, 0.0)
    negatives = np.where(cf < 0, cf, 0.0)

    if np.sum(positives) == 0:
        msg = "MIRR requires at least one positive cash flow"
        raise ValueError(msg)
    if np.sum(negatives) == 0:
        msg = "MIRR requires at least one negative cash flow"
        raise ValueError(msg)

    # FV of positives at reinvestment rate
    fv_pos = float(
        np.sum(positives * (1.0 + reinvestment_rate) ** (time_span - periods))
    )

    # PV of negatives at finance rate
    pv_neg = float(np.sum(negatives / (1.0 + finance_rate) ** periods))

    rate = (fv_pos / abs(pv_neg)) ** (1.0 / time_span) - 1.0

    return MIRRResult(
        rate=rate,
        finance_rate=finance_rate,
        reinvestment_rate=reinvestment_rate,
        fv_positives=fv_pos,
        pv_negatives=pv_neg,
        time_span=time_span,
    )


# ── GIRR (Kulakov & Kastro 2015) ─────────────────────────────────────────────


def _girr_terminal_balance(
    *,
    cash_flows: np.ndarray,
    r_inv: float,
    r_fin: float,
    steps: np.ndarray | None = None,
) -> float:
    """Compute terminal project balance for GIRR.

    Args:
        cash_flows: Array of cash flows.
        r_inv: Investment rate (applied when balance < 0, i.e. capital
            is tied up in the project).
        r_fin: Finance/reinvestment rate (applied when balance >= 0,
            i.e. excess cash is reinvested externally).
        steps: Time steps between consecutive periods.  ``None`` means
            unit steps (equal periods).

    Returns:
        Terminal balance B_n.
    """
    balance = float(cash_flows[0])
    for t in range(1, len(cash_flows)):
        r = r_inv if balance < 0 else r_fin
        s = float(steps[t - 1]) if steps is not None else 1.0
        balance = balance * (1.0 + r) ** s + float(cash_flows[t])
    return balance


def _girr_balances(
    *,
    cash_flows: np.ndarray,
    r_inv: float,
    r_fin: float,
    steps: np.ndarray | None = None,
) -> tuple[float, ...]:
    """Compute all project balances for GIRR.

    Args:
        cash_flows: Array of cash flows.
        r_inv: Investment rate (applied when balance < 0).
        r_fin: Finance/reinvestment rate (applied when balance >= 0).
        steps: Time steps between consecutive periods.  ``None`` means
            unit steps (equal periods).

    Returns:
        Tuple of balances B_0, B_1, ..., B_n.
    """
    balances = [float(cash_flows[0])]
    for t in range(1, len(cash_flows)):
        r = r_inv if balances[-1] < 0 else r_fin
        s = float(steps[t - 1]) if steps is not None else 1.0
        balances.append(balances[-1] * (1.0 + r) ** s + float(cash_flows[t]))
    return tuple(balances)


def calculate_girr(
    *,
    cash_flows: tuple[float, ...],
    finance_rate: float,
    dates: tuple[str, ...] | None = None,
    rate_min: float = _SOLVER_LO,
    rate_max: float = _SOLVER_HI,
) -> GIRRResult:
    """Compute Generalised IRR (Kulakov & Kastro 2015).

    Uses a two-rate project balance where the investment rate (r_inv) is
    the unknown.  r_inv is applied when the balance is negative (capital
    tied up in the project); r_fin is applied when the balance is
    non-negative (excess cash reinvested externally).  B_n is monotonic
    in r_inv when r_fin is fixed, guaranteeing a unique root.

    This subsumes the FREQ / ERR model (Teichroew, Robichek &
    Montalbano, 1965) which is mathematically identical.

    When *dates* is provided, compounding uses actual/365 year fractions
    between consecutive dates and the returned rate is annualised.

    Args:
        cash_flows: Cash flow amounts per period.
        finance_rate: Rate applied during reinvestment periods (B >= 0).
        dates: Optional ISO-format date strings for irregular spacing.
        rate_min: Lower solver bound (default -0.999, must be > -1).
        rate_max: Upper solver bound (default 10.0, must be <= 100).

    Returns:
        GIRRResult with unique rate and balance schedule.

    Raises:
        ValueError: If solver fails or bounds are invalid.
    """
    cf, periods = _resolve_periods(cash_flows=cash_flows, dates=dates)
    lo, hi = _validate_solver_bounds(rate_min=rate_min, rate_max=rate_max)
    steps = np.diff(periods)

    def f(r_inv: float) -> float:
        return _girr_terminal_balance(
            cash_flows=cf, r_inv=r_inv, r_fin=finance_rate, steps=steps
        )

    root = brentq(f, lo, hi, xtol=_ROOT_TOL)

    balances = _girr_balances(
        cash_flows=cf, r_inv=root, r_fin=finance_rate, steps=steps
    )

    return GIRRResult(
        rate=root,
        finance_rate=finance_rate,
        balances=balances,
    )


# ── AIRR (Magni 2010) ───────────────────────────────────────────────────────


def _build_capital_schedule(
    *,
    cf: np.ndarray,
    n: int,
    depreciation: str,
    steps: np.ndarray | None = None,
    dates: tuple[str, ...] | None = None,
) -> tuple[list[float], str]:
    """Build a capital schedule for AIRR, guaranteeing non-zero PV(C).

    If the primary method yields a degenerate (all-zero) schedule,
    falls back to straight-line over the sum of absolute cash flows.

    Args:
        cf: Validated cash flow array.
        n: Number of periods (len(cf) - 1).
        depreciation: Requested method ("straight-line" or "irr-implied").
        steps: Time steps between consecutive periods for compounding
            in the IRR-implied schedule.
        dates: Forwarded to ``find_all_irr_roots`` when building the
            IRR-implied schedule.

    Returns:
        Tuple of (capital schedule list, effective depreciation method name).
    """
    if depreciation == "irr-implied":
        roots = find_all_irr_roots(cash_flows=tuple(cf), dates=dates)
        if roots:
            irr_rate = roots[0]
            schedule: list[float] = [abs(float(cf[0]))]
            for t in range(1, n):
                s = float(steps[t - 1]) if steps is not None else 1.0
                c_prev = schedule[-1]
                schedule.append(c_prev * (1.0 + irr_rate) ** s - float(cf[t]))
            if any(abs(c) > _DERIV_EPS for c in schedule):
                return schedule, "irr-implied"
        # Fall through to straight-line

    c0 = abs(float(cf[0]))
    if c0 > _DERIV_EPS:
        return [c0 * (1.0 - t / n) for t in range(n)], "straight-line"

    # CF_0 ~= 0: anchor on sum of absolute outflows so PV(C) > 0.
    c0 = float(np.sum(np.abs(cf)))
    return [c0 * (1.0 - t / n) for t in range(n)], "straight-line (abs-sum)"


def calculate_airr(
    *,
    cash_flows: tuple[float, ...],
    cost_of_capital: float,
    depreciation: str = "straight-line",
    dates: tuple[str, ...] | None = None,
) -> AIRRResult:
    """Compute Average Internal Rate of Return (Magni 2010).

    AIRR always exists, even when IRR does not.  It depends on the cost
    of capital and a capital schedule (depreciation method).

    Shortcut formula: AIRR = r + NPV(r) / PV(C)
    where PV(C) = sum of c_{t-1} / (1+r)^{period_t} for t in 1..n.

    When *dates* is provided, discounting uses actual/365 year fractions
    and the returned rate is annualised.

    Args:
        cash_flows: Cash flow amounts per period.
        cost_of_capital: Opportunity cost rate (r).
        depreciation: Capital schedule method — "straight-line" or
            "irr-implied".
        dates: Optional ISO-format date strings for irregular spacing.

    Returns:
        AIRRResult with rate and full breakdown.
    """
    cf, periods = _resolve_periods(cash_flows=cash_flows, dates=dates)
    n = len(cf) - 1
    r = cost_of_capital
    npv = calculate_npv(cash_flows=tuple(cf), rate=r, dates=dates)
    steps = np.diff(periods)

    capital, effective_dep = _build_capital_schedule(
        cf=cf,
        n=n,
        depreciation=depreciation,
        steps=steps,
        dates=dates,
    )

    # PV of capital: sum c_{t-1} / (1+r)^{period_{t}} for t=1..n
    pv_capital = sum(capital[t] / (1.0 + r) ** periods[t + 1] for t in range(n))

    if abs(pv_capital) < _DERIV_EPS:
        # Truly degenerate (all CFs are zero) -- AIRR equals cost of capital.
        airr_rate = r
        pv_capital = 0.0
    else:
        airr_rate = r + npv / pv_capital

    # Per-period rates (Magni 2010):
    #   r_t = (CF_t + c_t - c_{t-1}) / c_{t-1}
    # where c_n = 0 (project is fully depreciated at terminal period).
    per_period: list[float] = []
    for t in range(1, n + 1):
        c_prev = capital[t - 1] if t - 1 < len(capital) else 0.0
        c_curr = capital[t] if t < len(capital) else 0.0
        if abs(c_prev) > _DERIV_EPS:
            income = float(cf[t]) + c_curr - c_prev
            per_period.append(income / c_prev)
        else:
            per_period.append(0.0)

    return AIRRResult(
        rate=airr_rate,
        cost_of_capital=r,
        npv=npv,
        pv_capital=pv_capital,
        per_period_rates=tuple(per_period),
        capital_schedule=tuple(capital),
        depreciation_method=effective_dep,
    )


# ── Horizon IRR ──────────────────────────────────────────────────────────────


def calculate_horizon_irr(
    *,
    cash_flows: tuple[float, ...],
    horizon: int,
    terminal_value: float,
    dates: tuple[str, ...] | None = None,
    rate_min: float = _SOLVER_LO,
    rate_max: float = _SOLVER_HI,
) -> HorizonIRRResult:
    """Compute Horizon IRR — the return assuming liquidation at *horizon*.

    Truncates the cash flow stream at *horizon* and adds *terminal_value*
    to the last period, then solves for the IRR of this modified stream.

    When *dates* is provided, dates are also truncated and the returned
    rate is annualised using actual/365 day count.

    Args:
        cash_flows: Cash flow amounts per period.
        horizon: Period at which to truncate (1 <= horizon < n).
        terminal_value: Residual / liquidation value added at horizon.
        dates: Optional ISO-format date strings for irregular spacing.
        rate_min: Lower bound of IRR search range (default -0.999).
        rate_max: Upper bound of IRR search range (default 10.0).

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
    truncated_dates = tuple(dates[: horizon + 1]) if dates is not None else None

    roots = find_all_irr_roots(
        cash_flows=truncated_tuple,
        dates=truncated_dates,
        rate_min=rate_min,
        rate_max=rate_max,
    )
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
    rate_min: float = _SOLVER_LO,
    rate_max: float = _SOLVER_HI,
) -> PairwiseIRRResult:
    """Compute pairwise (incremental) IRR for project comparison.

    The incremental stream is always A - B.  The decision rule depends
    on the sign of the NPV of the incremental stream at the cost of
    capital: if positive, A is preferred; otherwise B is preferred.

    Args:
        cash_flows_a: Cash flows for project A.
        cash_flows_b: Cash flows for project B.
        cost_of_capital: Hurdle rate for the decision rule.
        rate_min: Lower bound of IRR search range (default -0.999).
        rate_max: Upper bound of IRR search range (default 10.0).

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

    diff = a_padded - b_padded
    incremental = tuple(float(x) for x in diff)

    roots = find_all_irr_roots(
        cash_flows=incremental,
        rate_min=rate_min,
        rate_max=rate_max,
    )

    primary: float | None = None
    if roots:
        non_negative = [r for r in roots if r >= 0]
        primary = min(non_negative) if non_negative else roots[0]

    # Decision: prefer A when NPV(A - B) at cost of capital is positive.
    inc_npv = calculate_npv(cash_flows=incremental, rate=cost_of_capital)
    prefer_a = inc_npv > 0

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
    dates: tuple[str, ...] | None = None,
    rate_min: float = _PROFILE_LO,
    rate_max: float = _PROFILE_HI,
    num_points: int = _NPV_PROFILE_POINTS,
) -> NPVProfile:
    """Compute NPV at many discount rates for plotting.

    When *dates* is provided, discounting uses actual/365 year fractions.

    Args:
        cash_flows: Cash flow amounts per period.
        dates: Optional ISO-format date strings for irregular spacing.
        rate_min: Lowest rate to evaluate (default -0.50).
        rate_max: Highest rate to evaluate (default 2.00).
        num_points: Number of evaluation points (default 500).

    Returns:
        NPVProfile with rates, NPV values, and zero crossings.
    """
    cf, periods = _resolve_periods(cash_flows=cash_flows, dates=dates)
    rates = np.linspace(rate_min, rate_max, num_points)
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
