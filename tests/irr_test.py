"""Tests for the IRR / XIRR / MIRR / GIRR / AIRR / Horizon IRR module."""

import datetime as dt

import numpy as np
import pytest

from investment_portfolio.irr import (
    AIRRResult,
    GIRRResult,
    HorizonIRRResult,
    IRRResult,
    MIRRResult,
    NPVProfile,
    PairwiseIRRResult,
    SignChangeInfo,
    XIRRResult,
    _build_capital_schedule,
    _girr_balances,
    _girr_terminal_balance,
    _resolve_periods,
    _validate_cash_flows,
    _validate_dated_cash_flows,
    _validate_solver_bounds,
    _xnpv,
    _year_fraction,
    analyze_sign_changes,
    calculate_airr,
    calculate_girr,
    calculate_horizon_irr,
    calculate_irr,
    calculate_mirr,
    calculate_npv,
    calculate_npv_profile,
    calculate_pairwise_irr,
    calculate_xirr,
    find_all_irr_roots,
)

# ── _year_fraction ───────────────────────────────────────────────────────────


class TestYearFraction:
    def test_same_date_returns_zero(self) -> None:
        d = dt.date(2024, 6, 15)
        assert _year_fraction(d, d) == 0.0

    def test_d1_after_d2_returns_zero(self) -> None:
        assert _year_fraction(dt.date(2025, 1, 1), dt.date(2024, 1, 1)) == 0.0

    def test_full_non_leap_year(self) -> None:
        result = _year_fraction(dt.date(2023, 1, 1), dt.date(2024, 1, 1))
        assert result == pytest.approx(1.0)

    def test_full_leap_year(self) -> None:
        result = _year_fraction(dt.date(2024, 1, 1), dt.date(2025, 1, 1))
        assert result == pytest.approx(1.0)

    def test_half_year(self) -> None:
        result = _year_fraction(dt.date(2023, 1, 1), dt.date(2023, 7, 2))
        assert 0.49 < result < 0.51

    def test_cross_year_boundary(self) -> None:
        result = _year_fraction(dt.date(2023, 7, 1), dt.date(2024, 7, 1))
        assert result == pytest.approx(1.0, abs=0.01)

    def test_multi_year_span(self) -> None:
        result = _year_fraction(dt.date(2020, 1, 1), dt.date(2023, 1, 1))
        assert result == pytest.approx(3.0)


# ── _validate_cash_flows ─────────────────────────────────────────────────────


class TestValidateCashFlows:
    def test_valid_cash_flows(self) -> None:
        arr = _validate_cash_flows(cash_flows=(-100.0, 110.0))
        assert arr.shape == (2,)

    def test_too_few_elements_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 elements"):
            _validate_cash_flows(cash_flows=(100.0,))

    def test_empty_tuple_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 elements"):
            _validate_cash_flows(cash_flows=())

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN or Inf"):
            _validate_cash_flows(cash_flows=(-100.0, float("nan")))

    def test_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN or Inf"):
            _validate_cash_flows(cash_flows=(-100.0, float("inf")))


# ── _validate_dated_cash_flows ───────────────────────────────────────────────


class TestValidateDatedCashFlows:
    def test_valid_dated_cash_flows(self) -> None:
        cf, dates = _validate_dated_cash_flows(
            cash_flows=(-100.0, 110.0),
            dates=("2024-01-01", "2025-01-01"),
        )
        assert len(cf) == 2
        assert len(dates) == 2

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            _validate_dated_cash_flows(
                cash_flows=(-100.0, 110.0),
                dates=("2024-01-01",),
            )

    def test_non_ascending_dates_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            _validate_dated_cash_flows(
                cash_flows=(-100.0, 110.0),
                dates=("2025-01-01", "2024-01-01"),
            )

    def test_equal_dates_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly ascending"):
            _validate_dated_cash_flows(
                cash_flows=(-100.0, 110.0),
                dates=("2024-01-01", "2024-01-01"),
            )


# ── _validate_solver_bounds ──────────────────────────────────────────────────


class TestValidateSolverBounds:
    def test_valid_bounds(self) -> None:
        lo, hi = _validate_solver_bounds(rate_min=-0.5, rate_max=5.0)
        assert lo == -0.5
        assert hi == 5.0

    def test_rate_min_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_min must be > -1"):
            _validate_solver_bounds(rate_min=-1.0, rate_max=5.0)

    def test_rate_max_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="rate_max must be <= 100"):
            _validate_solver_bounds(rate_min=-0.5, rate_max=101.0)

    def test_rate_min_ge_rate_max_raises(self) -> None:
        with pytest.raises(ValueError, match=r"rate_min .* must be < rate_max"):
            _validate_solver_bounds(rate_min=5.0, rate_max=5.0)


# ── _resolve_periods ─────────────────────────────────────────────────────────


class TestResolvePeriods:
    def test_without_dates(self) -> None:
        cf, periods = _resolve_periods(
            cash_flows=(-100.0, 50.0, 60.0),
            dates=None,
        )
        assert len(cf) == 3
        np.testing.assert_array_equal(periods, [0.0, 1.0, 2.0])

    def test_with_dates(self) -> None:
        cf, periods = _resolve_periods(
            cash_flows=(-100.0, 110.0),
            dates=("2024-01-01", "2025-01-01"),
        )
        assert len(cf) == 2
        assert periods[0] == 0.0
        assert periods[1] == pytest.approx(1.0, abs=0.01)


# ── calculate_npv ────────────────────────────────────────────────────────────


class TestCalculateNPV:
    def test_zero_rate(self) -> None:
        result = calculate_npv(cash_flows=(-100.0, 50.0, 60.0), rate=0.0)
        assert result == pytest.approx(10.0)

    def test_positive_rate(self) -> None:
        result = calculate_npv(cash_flows=(-100.0, 110.0), rate=0.10)
        assert result == pytest.approx(0.0, abs=0.01)

    def test_rate_minus_one_raises(self) -> None:
        with pytest.raises(ValueError, match="rate cannot be -1"):
            calculate_npv(cash_flows=(-100.0, 110.0), rate=-1.0)

    def test_with_dates(self) -> None:
        result = calculate_npv(
            cash_flows=(-100.0, 110.0),
            rate=0.0,
            dates=("2024-01-01", "2025-01-01"),
        )
        assert result == pytest.approx(10.0)


# ── analyze_sign_changes ────────────────────────────────────────────────────


class TestAnalyzeSignChanges:
    def test_conventional_flow(self) -> None:
        info = analyze_sign_changes(cash_flows=(-100.0, 50.0, 60.0))
        assert isinstance(info, SignChangeInfo)
        assert info.num_sign_changes == 1
        assert info.is_conventional is True

    def test_no_sign_changes(self) -> None:
        info = analyze_sign_changes(cash_flows=(100.0, 200.0, 300.0))
        assert info.num_sign_changes == 0
        assert info.is_conventional is False

    def test_multiple_sign_changes(self) -> None:
        info = analyze_sign_changes(cash_flows=(-100.0, 300.0, -200.0))
        assert info.num_sign_changes == 2
        assert info.is_conventional is False

    def test_sign_pattern_includes_zero(self) -> None:
        info = analyze_sign_changes(cash_flows=(-100.0, 0.0, 110.0))
        assert "0" in info.sign_pattern

    def test_mostly_zeros(self) -> None:
        info = analyze_sign_changes(cash_flows=(0.0, 0.0, 100.0))
        assert info.num_sign_changes == 0
        assert info.max_positive_roots == 0
        assert info.is_conventional is False


# ── find_all_irr_roots ──────────────────────────────────────────────────────


class TestFindAllIRRRoots:
    def test_conventional_single_root(self) -> None:
        roots = find_all_irr_roots(cash_flows=(-100.0, 110.0))
        assert len(roots) == 1
        assert roots[0] == pytest.approx(0.10, abs=0.001)

    def test_no_root(self) -> None:
        roots = find_all_irr_roots(cash_flows=(100.0, 200.0))
        assert len(roots) == 0

    def test_multiple_roots(self) -> None:
        # Non-conventional flow: -100, +230, -132 has 2 roots (10% and 20%)
        roots = find_all_irr_roots(cash_flows=(-100.0, 230.0, -132.0))
        assert len(roots) == 2

    def test_with_dates(self) -> None:
        roots = find_all_irr_roots(
            cash_flows=(-100.0, 110.0),
            dates=("2024-01-01", "2025-01-01"),
        )
        assert len(roots) >= 1
        assert roots[0] == pytest.approx(0.10, abs=0.02)

    def test_custom_bounds(self) -> None:
        roots = find_all_irr_roots(
            cash_flows=(-100.0, 110.0),
            rate_min=-0.5,
            rate_max=5.0,
        )
        assert len(roots) == 1


# ── calculate_irr ────────────────────────────────────────────────────────────


class TestCalculateIRR:
    def test_conventional_irr(self) -> None:
        result = calculate_irr(cash_flows=(-1000.0, 500.0, 600.0))
        assert isinstance(result, IRRResult)
        assert result.primary_irr is not None
        assert result.primary_irr == pytest.approx(0.0639, abs=0.01)

    def test_no_roots(self) -> None:
        result = calculate_irr(cash_flows=(100.0, 200.0))
        assert result.primary_irr is None
        assert result.roots == ()

    def test_npv_at_zero(self) -> None:
        result = calculate_irr(cash_flows=(-100.0, 50.0, 60.0))
        assert result.npv_at_zero == pytest.approx(10.0)

    def test_sign_info_attached(self) -> None:
        result = calculate_irr(cash_flows=(-100.0, 110.0))
        assert isinstance(result.sign_info, SignChangeInfo)
        assert result.sign_info.is_conventional is True

    def test_all_negative_roots_picks_max(self) -> None:
        # Flow where only root is negative: invest 100, get 50 back
        result = calculate_irr(cash_flows=(-100.0, 50.0))
        if result.primary_irr is not None:
            assert result.primary_irr < 0

    def test_with_dates(self) -> None:
        result = calculate_irr(
            cash_flows=(-100.0, 110.0),
            dates=("2024-01-01", "2025-01-01"),
        )
        assert result.primary_irr is not None


# ── _xnpv ────────────────────────────────────────────────────────────────────


class TestXNPV:
    def test_zero_rate(self) -> None:
        cf = np.array([-100.0, 110.0])
        dates = [dt.date(2024, 1, 1), dt.date(2025, 1, 1)]
        result = _xnpv(cash_flows=cf, dates=dates, rate=0.0)
        assert result == pytest.approx(10.0)

    def test_positive_rate(self) -> None:
        cf = np.array([-100.0, 110.0])
        dates = [dt.date(2024, 1, 1), dt.date(2025, 1, 1)]
        result = _xnpv(cash_flows=cf, dates=dates, rate=0.10)
        assert result == pytest.approx(0.0, abs=0.1)


# ── calculate_xirr ───────────────────────────────────────────────────────────


class TestCalculateXIRR:
    def test_simple_xirr(self) -> None:
        result = calculate_xirr(
            cash_flows=(-1000.0, 1100.0),
            dates=("2024-01-01", "2025-01-01"),
        )
        assert isinstance(result, XIRRResult)
        assert result.rate == pytest.approx(0.10, abs=0.02)
        assert result.day_count_convention == "actual/actual"

    def test_multiple_cash_flows(self) -> None:
        result = calculate_xirr(
            cash_flows=(-1000.0, 300.0, 400.0, 500.0),
            dates=("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01"),
        )
        assert isinstance(result, XIRRResult)
        assert result.rate > 0

    def test_npv_at_zero_is_sum(self) -> None:
        result = calculate_xirr(
            cash_flows=(-100.0, 120.0),
            dates=("2024-01-01", "2025-01-01"),
        )
        assert result.npv_at_zero == pytest.approx(20.0)

    def test_no_solution_raises(self) -> None:
        with pytest.raises(ValueError, match="XIRR"):
            calculate_xirr(
                cash_flows=(100.0, 200.0),
                dates=("2024-01-01", "2025-01-01"),
            )

    def test_custom_bounds(self) -> None:
        result = calculate_xirr(
            cash_flows=(-100.0, 110.0),
            dates=("2024-01-01", "2025-01-01"),
            rate_min=-0.5,
            rate_max=5.0,
        )
        assert result.rate == pytest.approx(0.10, abs=0.02)


# ── calculate_mirr ───────────────────────────────────────────────────────────


class TestCalculateMIRR:
    def test_simple_mirr(self) -> None:
        result = calculate_mirr(
            cash_flows=(-1000.0, 500.0, 600.0),
            finance_rate=0.05,
            reinvestment_rate=0.08,
        )
        assert isinstance(result, MIRRResult)
        assert result.finance_rate == 0.05
        assert result.reinvestment_rate == 0.08
        assert result.fv_positives > 0
        assert result.pv_negatives < 0

    def test_mirr_rate_sign(self) -> None:
        result = calculate_mirr(
            cash_flows=(-100.0, 150.0),
            finance_rate=0.05,
            reinvestment_rate=0.10,
        )
        assert result.rate > 0

    def test_no_positive_flows_raises(self) -> None:
        with pytest.raises(ValueError, match="positive cash flow"):
            calculate_mirr(
                cash_flows=(-100.0, -50.0),
                finance_rate=0.05,
                reinvestment_rate=0.08,
            )

    def test_no_negative_flows_raises(self) -> None:
        with pytest.raises(ValueError, match="negative cash flow"):
            calculate_mirr(
                cash_flows=(100.0, 50.0),
                finance_rate=0.05,
                reinvestment_rate=0.08,
            )

    def test_time_span(self) -> None:
        result = calculate_mirr(
            cash_flows=(-100.0, 50.0, 60.0, 70.0),
            finance_rate=0.05,
            reinvestment_rate=0.10,
        )
        assert result.time_span == pytest.approx(3.0)

    def test_with_dates(self) -> None:
        result = calculate_mirr(
            cash_flows=(-1000.0, 600.0, 700.0),
            finance_rate=0.05,
            reinvestment_rate=0.10,
            dates=("2024-01-01", "2025-01-01", "2026-01-01"),
        )
        assert isinstance(result, MIRRResult)
        assert result.rate > 0


# ── _girr helpers ────────────────────────────────────────────────────────────


class TestGIRRHelpers:
    def test_terminal_balance_simple(self) -> None:
        cf = np.array([-100.0, 110.0])
        balance = _girr_terminal_balance(cash_flows=cf, r_inv=0.10, r_fin=0.05)
        # B0 = -100, negative so r_inv=10% applied
        # B1 = -100 * 1.10 + 110 = 0.0
        assert balance == pytest.approx(0.0)

    def test_terminal_balance_with_steps(self) -> None:
        cf = np.array([-100.0, 110.0])
        steps = np.array([1.0])
        balance = _girr_terminal_balance(
            cash_flows=cf, r_inv=0.10, r_fin=0.05, steps=steps
        )
        assert balance == pytest.approx(0.0)

    def test_balances_simple(self) -> None:
        cf = np.array([-100.0, 60.0, 50.0])
        balances = _girr_balances(cash_flows=cf, r_inv=0.10, r_fin=0.05)
        assert len(balances) == 3
        assert balances[0] == pytest.approx(-100.0)

    def test_balances_with_steps(self) -> None:
        cf = np.array([-100.0, 110.0])
        steps = np.array([1.0])
        balances = _girr_balances(cash_flows=cf, r_inv=0.10, r_fin=0.05, steps=steps)
        assert len(balances) == 2
        assert balances[0] == pytest.approx(-100.0)

    def test_balances_positive_balance_uses_r_fin(self) -> None:
        # First CF positive so balance >= 0, should use r_fin
        cf = np.array([100.0, -50.0, -60.0])
        balances = _girr_balances(cash_flows=cf, r_inv=0.10, r_fin=0.05)
        # B0 = 100 (positive), so r_fin=5% used
        # B1 = 100 * 1.05 + (-50) = 55.0
        assert balances[1] == pytest.approx(55.0)


# ── calculate_girr ───────────────────────────────────────────────────────────


class TestCalculateGIRR:
    def test_conventional_flow(self) -> None:
        result = calculate_girr(
            cash_flows=(-1000.0, 500.0, 600.0),
            finance_rate=0.05,
        )
        assert isinstance(result, GIRRResult)
        assert result.finance_rate == 0.05
        assert len(result.balances) == 3

    def test_girr_rate_reasonable(self) -> None:
        result = calculate_girr(
            cash_flows=(-100.0, 110.0),
            finance_rate=0.05,
        )
        # For a simple flow, GIRR should be close to IRR
        assert result.rate == pytest.approx(0.10, abs=0.02)

    def test_with_dates(self) -> None:
        result = calculate_girr(
            cash_flows=(-100.0, 120.0),
            finance_rate=0.05,
            dates=("2024-01-01", "2025-01-01"),
        )
        assert isinstance(result, GIRRResult)
        assert result.rate > 0


# ── _build_capital_schedule ──────────────────────────────────────────────────


class TestBuildCapitalSchedule:
    def test_straight_line(self) -> None:
        cf = np.array([-1000.0, 500.0, 600.0])
        schedule, method = _build_capital_schedule(
            cf=cf, n=2, depreciation="straight-line"
        )
        assert method == "straight-line"
        assert len(schedule) == 2
        assert schedule[0] == pytest.approx(1000.0)
        assert schedule[1] == pytest.approx(500.0)

    def test_irr_implied(self) -> None:
        cf = np.array([-1000.0, 500.0, 600.0])
        schedule, method = _build_capital_schedule(
            cf=cf, n=2, depreciation="irr-implied"
        )
        assert method == "irr-implied"
        assert len(schedule) == 2

    def test_irr_implied_no_roots_falls_back(self) -> None:
        # All positive flows — no IRR root exists
        cf = np.array([100.0, 200.0, 300.0])
        _schedule, method = _build_capital_schedule(
            cf=cf, n=2, depreciation="irr-implied"
        )
        assert method == "straight-line"

    def test_zero_cf0_uses_abs_sum(self) -> None:
        cf = np.array([0.0, 100.0, 200.0])
        schedule, method = _build_capital_schedule(
            cf=cf, n=2, depreciation="straight-line"
        )
        assert "abs-sum" in method
        assert schedule[0] > 0


# ── calculate_airr ───────────────────────────────────────────────────────────


class TestCalculateAIRR:
    def test_straight_line(self) -> None:
        result = calculate_airr(
            cash_flows=(-1000.0, 500.0, 600.0),
            cost_of_capital=0.10,
            depreciation="straight-line",
        )
        assert isinstance(result, AIRRResult)
        assert result.cost_of_capital == 0.10
        assert result.depreciation_method == "straight-line"
        assert len(result.per_period_rates) == 2
        assert len(result.capital_schedule) == 2

    def test_irr_implied(self) -> None:
        result = calculate_airr(
            cash_flows=(-1000.0, 500.0, 600.0),
            cost_of_capital=0.10,
            depreciation="irr-implied",
        )
        assert isinstance(result, AIRRResult)
        assert result.depreciation_method == "irr-implied"

    def test_npv_positive_means_airr_above_coc(self) -> None:
        result = calculate_airr(
            cash_flows=(-100.0, 60.0, 60.0),
            cost_of_capital=0.05,
        )
        assert result.npv > 0
        assert result.rate > result.cost_of_capital

    def test_npv_negative_means_airr_below_coc(self) -> None:
        result = calculate_airr(
            cash_flows=(-100.0, 40.0, 40.0),
            cost_of_capital=0.10,
        )
        assert result.npv < 0
        assert result.rate < result.cost_of_capital

    def test_with_dates(self) -> None:
        result = calculate_airr(
            cash_flows=(-100.0, 60.0, 60.0),
            cost_of_capital=0.05,
            dates=("2024-01-01", "2025-01-01", "2026-01-01"),
        )
        assert isinstance(result, AIRRResult)

    def test_degenerate_zero_pv_capital(self) -> None:
        # All zero cash flows — PV(C) should be ~0, AIRR = cost of capital
        result = calculate_airr(
            cash_flows=(0.0, 0.0, 0.0),
            cost_of_capital=0.10,
        )
        assert result.rate == pytest.approx(0.10)
        assert result.pv_capital == 0.0


# ── calculate_horizon_irr ────────────────────────────────────────────────────


class TestCalculateHorizonIRR:
    def test_simple_horizon(self) -> None:
        result = calculate_horizon_irr(
            cash_flows=(-1000.0, 200.0, 300.0, 400.0, 500.0),
            horizon=2,
            terminal_value=800.0,
        )
        assert isinstance(result, HorizonIRRResult)
        assert result.horizon == 2
        assert result.terminal_value == 800.0
        assert len(result.truncated_flows) == 3

    def test_rate_found(self) -> None:
        result = calculate_horizon_irr(
            cash_flows=(-100.0, 50.0, 60.0, 70.0),
            horizon=1,
            terminal_value=80.0,
        )
        assert result.rate is not None
        assert result.rate > 0

    def test_horizon_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon must be in"):
            calculate_horizon_irr(
                cash_flows=(-100.0, 110.0),
                horizon=0,
                terminal_value=50.0,
            )

    def test_horizon_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match="horizon must be in"):
            calculate_horizon_irr(
                cash_flows=(-100.0, 110.0),
                horizon=2,
                terminal_value=50.0,
            )

    def test_with_dates(self) -> None:
        result = calculate_horizon_irr(
            cash_flows=(-100.0, 50.0, 60.0),
            horizon=1,
            terminal_value=80.0,
            dates=("2024-01-01", "2025-01-01", "2026-01-01"),
        )
        assert isinstance(result, HorizonIRRResult)

    def test_no_root_returns_none(self) -> None:
        # Terminal value too small to produce positive return
        result = calculate_horizon_irr(
            cash_flows=(-1000.0, 10.0, 10.0),
            horizon=1,
            terminal_value=0.0,
        )
        # Rate might be very negative or None if no root in bounds
        assert isinstance(result, HorizonIRRResult)


# ── calculate_pairwise_irr ──────────────────────────────────────────────────


class TestCalculatePairwiseIRR:
    def test_prefer_a(self) -> None:
        result = calculate_pairwise_irr(
            cash_flows_a=(-100.0, 200.0),
            cash_flows_b=(-100.0, 150.0),
            cost_of_capital=0.05,
        )
        assert isinstance(result, PairwiseIRRResult)
        assert result.prefer_a is True
        assert result.cost_of_capital == 0.05

    def test_prefer_b(self) -> None:
        result = calculate_pairwise_irr(
            cash_flows_a=(-100.0, 105.0),
            cash_flows_b=(-100.0, 200.0),
            cost_of_capital=0.05,
        )
        assert result.prefer_a is False

    def test_different_length_flows_padded(self) -> None:
        result = calculate_pairwise_irr(
            cash_flows_a=(-100.0, 50.0, 60.0, 70.0),
            cash_flows_b=(-100.0, 150.0),
            cost_of_capital=0.10,
        )
        assert len(result.incremental_flows) == 4

    def test_incremental_irr_found(self) -> None:
        result = calculate_pairwise_irr(
            cash_flows_a=(-200.0, 300.0),
            cash_flows_b=(-100.0, 120.0),
            cost_of_capital=0.05,
        )
        assert result.rate is not None


# ── calculate_npv_profile ────────────────────────────────────────────────────


class TestCalculateNPVProfile:
    def test_returns_profile(self) -> None:
        result = calculate_npv_profile(
            cash_flows=(-100.0, 50.0, 60.0),
        )
        assert isinstance(result, NPVProfile)
        assert len(result.rates) == 500
        assert len(result.npv_values) == 500

    def test_zero_crossing_found(self) -> None:
        result = calculate_npv_profile(
            cash_flows=(-100.0, 110.0),
        )
        assert len(result.zero_crossings) >= 1
        assert result.zero_crossings[0] == pytest.approx(0.10, abs=0.01)

    def test_custom_range(self) -> None:
        result = calculate_npv_profile(
            cash_flows=(-100.0, 110.0),
            rate_min=-0.2,
            rate_max=0.5,
            num_points=100,
        )
        assert len(result.rates) == 100

    def test_with_dates(self) -> None:
        result = calculate_npv_profile(
            cash_flows=(-100.0, 110.0),
            dates=("2024-01-01", "2025-01-01"),
        )
        assert isinstance(result, NPVProfile)

    def test_no_sign_change(self) -> None:
        result = calculate_npv_profile(
            cash_flows=(100.0, 200.0),
        )
        assert len(result.zero_crossings) == 0
