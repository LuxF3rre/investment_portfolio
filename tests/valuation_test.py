"""Tests for the comparable-multiples valuation module."""

import pytest

from investment_portfolio.fundamentals import Fundamentals
from investment_portfolio.valuation import (
    ComparablesResult,
    _build_sensitivity,
    calculate_comparables,
    calculate_ev_ebitda_valuation,
    calculate_pb_valuation,
    calculate_pe_valuation,
    calculate_ps_valuation,
)


def _make_fund(
    *,
    ticker: str = "TEST",
    name: str = "Test Co",
    current_price: float | None = 150.0,
    eps_ttm: float | None = 6.0,
    eps_forward: float | None = 7.0,
    trailing_pe: float | None = 25.0,
    forward_pe: float | None = 21.4,
    book_value_per_share: float | None = 40.0,
    price_to_book: float | None = 3.75,
    revenue_per_share: float | None = 50.0,
    price_to_sales: float | None = 3.0,
    ebitda: float | None = 10e9,
    ev_to_ebitda: float | None = 15.0,
    shares_outstanding: float | None = 1e9,
    enterprise_value: float | None = 150e9,
    total_debt: float | None = 20e9,
    total_cash: float | None = 10e9,
    market_cap: float | None = 150e9,
    total_revenue: float | None = 50e9,
) -> Fundamentals:
    return Fundamentals(
        ticker=ticker,
        name=name,
        sector="Technology",
        industry="Software",
        current_price=current_price,
        shares_outstanding=shares_outstanding,
        market_cap=market_cap,
        enterprise_value=enterprise_value,
        eps_ttm=eps_ttm,
        eps_forward=eps_forward,
        book_value_per_share=book_value_per_share,
        revenue_per_share=revenue_per_share,
        ebitda=ebitda,
        total_revenue=total_revenue,
        total_debt=total_debt,
        total_cash=total_cash,
        trailing_pe=trailing_pe,
        forward_pe=forward_pe,
        price_to_book=price_to_book,
        ev_to_ebitda=ev_to_ebitda,
        price_to_sales=price_to_sales,
    )


@pytest.fixture
def target() -> Fundamentals:
    return _make_fund(
        ticker="AAPL",
        name="Apple Inc.",
        current_price=180.0,
        eps_ttm=6.0,
        eps_forward=7.0,
        trailing_pe=30.0,
        forward_pe=25.7,
        book_value_per_share=4.0,
        price_to_book=45.0,
        revenue_per_share=24.0,
        price_to_sales=7.5,
        ebitda=130e9,
        ev_to_ebitda=22.0,
        shares_outstanding=15.5e9,
        total_debt=110e9,
        total_cash=60e9,
    )


@pytest.fixture
def peers() -> tuple[Fundamentals, ...]:
    return (
        _make_fund(
            ticker="MSFT",
            name="Microsoft",
            trailing_pe=35.0,
            forward_pe=30.0,
            price_to_book=12.0,
            price_to_sales=12.0,
            ev_to_ebitda=25.0,
        ),
        _make_fund(
            ticker="GOOGL",
            name="Alphabet",
            trailing_pe=25.0,
            forward_pe=22.0,
            price_to_book=6.0,
            price_to_sales=6.0,
            ev_to_ebitda=18.0,
        ),
        _make_fund(
            ticker="META",
            name="Meta Platforms",
            trailing_pe=20.0,
            forward_pe=18.0,
            price_to_book=8.0,
            price_to_sales=8.0,
            ev_to_ebitda=14.0,
        ),
    )


# ── P/E Tests ────────────────────────────────────────────────────────────────


class TestPeValuation:
    def test_happy_path(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_pe_valuation(target=target, peers=peers)
        assert result is not None
        assert result.method == "P/E (TTM)"
        # Median of [35, 25, 20] = 25
        assert result.median_multiple == 25.0
        # Implied = 25 * 6 = 150
        assert result.implied_price == pytest.approx(150.0)
        assert result.upside_pct == pytest.approx((150.0 - 180.0) / 180.0)

    def test_forward_pe(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_pe_valuation(target=target, peers=peers, use_forward=True)
        assert result is not None
        assert result.method == "Forward P/E"
        # Median of [30, 22, 18] = 22
        assert result.median_multiple == 22.0
        # Implied = 22 * 7 = 154
        assert result.implied_price == pytest.approx(154.0)

    def test_none_when_target_eps_missing(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(eps_ttm=None)
        result = calculate_pe_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_target_eps_negative(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(eps_ttm=-2.0)
        result = calculate_pe_valuation(target=t, peers=peers)
        assert result is None

    def test_filters_none_peer_values(
        self,
        target: Fundamentals,
    ) -> None:
        peers_with_none = (
            _make_fund(ticker="A", trailing_pe=20.0),
            _make_fund(ticker="B", trailing_pe=None),
            _make_fund(ticker="C", trailing_pe=30.0),
        )
        result = calculate_pe_valuation(target=target, peers=peers_with_none)
        assert result is not None
        # Median of [20, 30] = 25
        assert result.median_multiple == 25.0

    def test_filters_negative_peer_values(
        self,
        target: Fundamentals,
    ) -> None:
        peers_neg = (
            _make_fund(ticker="A", trailing_pe=-5.0),
            _make_fund(ticker="B", trailing_pe=20.0),
            _make_fund(ticker="C", trailing_pe=30.0),
        )
        result = calculate_pe_valuation(target=target, peers=peers_neg)
        assert result is not None
        # Median of [20, 30] = 25
        assert result.median_multiple == 25.0

    def test_none_when_all_peers_have_none(
        self,
        target: Fundamentals,
    ) -> None:
        peers_all_none = (
            _make_fund(ticker="A", trailing_pe=None),
            _make_fund(ticker="B", trailing_pe=None),
        )
        result = calculate_pe_valuation(target=target, peers=peers_all_none)
        assert result is None

    def test_sensitivity_has_entries(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_pe_valuation(target=target, peers=peers)
        assert result is not None
        assert len(result.sensitivity) > 0
        for row in result.sensitivity:
            assert row.multiple > 0
            assert row.implied_price > 0

    def test_mean_multiple(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_pe_valuation(target=target, peers=peers)
        assert result is not None
        # Mean of [35, 25, 20] = 26.666...
        assert result.mean_multiple == pytest.approx(80.0 / 3.0)


# ── EV/EBITDA Tests ──────────────────────────────────────────────────────────


class TestEvEbitdaValuation:
    def test_happy_path(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_ev_ebitda_valuation(target=target, peers=peers)
        assert result is not None
        assert result.method == "EV/EBITDA"
        # Median of [25, 18, 14] = 18
        assert result.median_multiple == 18.0
        # Implied EV = 18 * 130e9 = 2.34e12
        # Implied equity = 2.34e12 - 110e9 + 60e9 = 2.29e12
        # Implied price = 2.29e12 / 15.5e9 ≈ 147.74
        expected_ev = 18.0 * 130e9
        expected_equity = expected_ev - 110e9 + 60e9
        expected_price = expected_equity / 15.5e9
        assert result.implied_price == pytest.approx(expected_price)

    def test_none_when_ebitda_missing(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(ebitda=None)
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_ebitda_negative(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(ebitda=-5e9)
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_shares_missing(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(shares_outstanding=None)
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is None

    def test_ev_to_equity_conversion(self) -> None:
        """Verify debt/cash correctly adjusts implied equity."""
        t = _make_fund(
            current_price=100.0,
            ebitda=10e9,
            total_debt=5e9,
            total_cash=2e9,
            shares_outstanding=1e9,
        )
        peers = (
            _make_fund(ticker="A", ev_to_ebitda=10.0),
            _make_fund(ticker="B", ev_to_ebitda=12.0),
        )
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is not None
        # Median = 11, EV = 110e9, equity = 110e9 - 5e9 + 2e9 = 107e9
        # Price = 107e9 / 1e9 = 107
        assert result.implied_price == pytest.approx(107.0)

    def test_sensitivity_has_entries(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_ev_ebitda_valuation(target=target, peers=peers)
        assert result is not None
        assert len(result.sensitivity) > 0


# ── P/B Tests ────────────────────────────────────────────────────────────────


class TestPbValuation:
    def test_happy_path(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_pb_valuation(target=target, peers=peers)
        assert result is not None
        assert result.method == "P/B"
        # Median of [12, 6, 8] = 8
        assert result.median_multiple == 8.0
        # Implied = 8 * 4 = 32
        assert result.implied_price == pytest.approx(32.0)

    def test_none_when_bvps_missing(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(book_value_per_share=None)
        result = calculate_pb_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_bvps_negative(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(book_value_per_share=-5.0)
        result = calculate_pb_valuation(target=t, peers=peers)
        assert result is None

    def test_filters_none_peers(
        self,
        target: Fundamentals,
    ) -> None:
        peers = (
            _make_fund(ticker="A", price_to_book=None),
            _make_fund(ticker="B", price_to_book=5.0),
        )
        result = calculate_pb_valuation(target=target, peers=peers)
        assert result is not None
        assert result.median_multiple == 5.0

    def test_none_when_all_peers_invalid(
        self,
        target: Fundamentals,
    ) -> None:
        peers = (
            _make_fund(ticker="A", price_to_book=None),
            _make_fund(ticker="B", price_to_book=-1.0),
        )
        result = calculate_pb_valuation(target=target, peers=peers)
        assert result is None


# ── P/S Tests ────────────────────────────────────────────────────────────────


class TestPsValuation:
    def test_happy_path(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_ps_valuation(target=target, peers=peers)
        assert result is not None
        assert result.method == "P/S"
        # Median of [12, 6, 8] = 8
        assert result.median_multiple == 8.0
        # Implied = 8 * 24 = 192
        assert result.implied_price == pytest.approx(192.0)

    def test_none_when_rps_missing(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(revenue_per_share=None)
        result = calculate_ps_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_rps_negative(
        self,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        t = _make_fund(revenue_per_share=-10.0)
        result = calculate_ps_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_all_peers_invalid(
        self,
        target: Fundamentals,
    ) -> None:
        peers = (
            _make_fund(ticker="A", price_to_sales=None),
            _make_fund(ticker="B", price_to_sales=-2.0),
        )
        result = calculate_ps_valuation(target=target, peers=peers)
        assert result is None


# ── Sensitivity Tests ────────────────────────────────────────────────────────


class TestSensitivity:
    def test_sensitivity_empty_when_no_peers(self) -> None:
        """_build_sensitivity returns () for empty peer list."""
        t = _make_fund(current_price=100.0, eps_ttm=5.0)
        peers: tuple[Fundamentals, ...] = (_make_fund(ticker="A", trailing_pe=None),)
        result = calculate_pe_valuation(target=t, peers=peers)
        assert result is None

    def test_ev_ebitda_sensitivity_uses_implied_price_fn(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        """EV/EBITDA sensitivity rows use the EV-to-equity conversion."""
        result = calculate_ev_ebitda_valuation(target=target, peers=peers)
        assert result is not None
        assert target.ebitda is not None
        for row in result.sensitivity:
            # Each sensitivity row should convert via EV formula
            expected_ev = row.multiple * target.ebitda
            expected_equity = expected_ev - 110e9 + 60e9
            expected_price = expected_equity / 15.5e9
            assert row.implied_price == pytest.approx(expected_price)

    def test_sensitivity_upside_zero_when_current_price_zero(self) -> None:
        """Upside is 0 when current_price is 0 (avoids division by zero)."""
        rows = _build_sensitivity(
            peer_values=[10.0, 20.0],
            metric_value=5.0,
            current_price=0.0,
        )
        assert len(rows) > 0
        for row in rows:
            assert row.upside_pct == 0.0

    def test_sensitivity_returns_empty_for_no_values(self) -> None:
        rows = _build_sensitivity(
            peer_values=[],
            metric_value=5.0,
            current_price=100.0,
        )
        assert rows == ()


# ── EV/EBITDA edge cases ────────────────────────────────────────────────────


class TestEvEbitdaEdgeCases:
    def test_none_when_shares_zero(self) -> None:
        t = _make_fund(shares_outstanding=0.0)
        peers = (_make_fund(ticker="A", ev_to_ebitda=10.0),)
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_current_price_missing(self) -> None:
        t = _make_fund(current_price=None)
        peers = (_make_fund(ticker="A", ev_to_ebitda=10.0),)
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is None

    def test_none_when_all_peers_invalid(self) -> None:
        t = _make_fund(current_price=100.0, ebitda=10e9, shares_outstanding=1e9)
        peers = (
            _make_fund(ticker="A", ev_to_ebitda=None),
            _make_fund(ticker="B", ev_to_ebitda=-5.0),
        )
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is None

    def test_defaults_debt_and_cash_to_zero(self) -> None:
        """When debt/cash are None, defaults to 0."""
        t = _make_fund(
            current_price=100.0,
            ebitda=10e9,
            total_debt=None,
            total_cash=None,
            shares_outstanding=1e9,
        )
        peers = (_make_fund(ticker="A", ev_to_ebitda=10.0),)
        result = calculate_ev_ebitda_valuation(target=t, peers=peers)
        assert result is not None
        # EV = 10 * 10e9 = 100e9, equity = 100e9 - 0 + 0 = 100e9
        # Price = 100e9 / 1e9 = 100
        assert result.implied_price == pytest.approx(100.0)


# ── P/E edge cases ──────────────────────────────────────────────────────────


class TestPeEdgeCases:
    def test_none_when_current_price_missing(self) -> None:
        t = _make_fund(current_price=None, eps_ttm=5.0)
        peers = (_make_fund(ticker="A", trailing_pe=20.0),)
        result = calculate_pe_valuation(target=t, peers=peers)
        assert result is None

    def test_single_peer(self) -> None:
        t = _make_fund(current_price=100.0, eps_ttm=5.0)
        peers = (_make_fund(ticker="A", trailing_pe=20.0),)
        result = calculate_pe_valuation(target=t, peers=peers)
        assert result is not None
        assert result.median_multiple == 20.0
        assert result.mean_multiple == 20.0
        assert result.implied_price == pytest.approx(100.0)


# ── Comparables Tests ────────────────────────────────────────────────────────


class TestCalculateComparables:
    def test_combines_all_methods(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_comparables(target=target, peers=peers)
        assert isinstance(result, ComparablesResult)
        assert result.target_ticker == "AAPL"
        assert result.target_name == "Apple Inc."
        assert result.current_price == 180.0
        assert result.pe is not None
        assert result.ev_ebitda is not None
        assert result.pb is not None
        assert result.ps is not None

    def test_forward_pe_flag(
        self,
        target: Fundamentals,
        peers: tuple[Fundamentals, ...],
    ) -> None:
        result = calculate_comparables(target=target, peers=peers, use_forward_pe=True)
        assert result.pe is not None
        assert result.pe.method == "Forward P/E"

    def test_missing_data_yields_none(self) -> None:
        t = _make_fund(
            current_price=100.0,
            eps_ttm=None,
            ebitda=None,
            book_value_per_share=None,
            revenue_per_share=None,
        )
        peers = (_make_fund(ticker="A"),)
        result = calculate_comparables(target=t, peers=peers)
        assert result.pe is None
        assert result.ev_ebitda is None
        assert result.pb is None
        assert result.ps is None

    def test_partial_data(self) -> None:
        """Only P/E available, others missing."""
        t = _make_fund(
            current_price=100.0,
            eps_ttm=5.0,
            ebitda=None,
            book_value_per_share=None,
            revenue_per_share=None,
        )
        peers = (_make_fund(ticker="A", trailing_pe=20.0),)
        result = calculate_comparables(target=t, peers=peers)
        assert result.pe is not None
        assert result.ev_ebitda is None
        assert result.pb is None
        assert result.ps is None

    def test_current_price_none_sets_zero(self) -> None:
        t = _make_fund(current_price=None)
        peers = (_make_fund(ticker="A"),)
        result = calculate_comparables(target=t, peers=peers)
        assert result.current_price == 0.0
