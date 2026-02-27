"""Tests for the fundamentals data-fetching module."""

import pytest

from investment_portfolio.fundamentals import (
    Fundamentals,
    _build_from_yf,
    _safe_float,
    fetch_fundamentals,
)

_SAMPLE_YF_INFO: dict[str, object] = {
    "shortName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "currentPrice": 180.0,
    "sharesOutstanding": 15_500_000_000,
    "marketCap": 2_790_000_000_000,
    "enterpriseValue": 2_840_000_000_000,
    "trailingEps": 6.13,
    "forwardEps": 7.0,
    "bookValue": 4.38,
    "revenuePerShare": 24.32,
    "ebitda": 130_000_000_000,
    "totalRevenue": 383_000_000_000,
    "totalDebt": 110_000_000_000,
    "totalCash": 60_000_000_000,
    "trailingPE": 29.36,
    "forwardPE": 25.71,
    "priceToBook": 41.1,
    "enterpriseToEbitda": 21.85,
    "priceToSalesTrailing12Months": 7.29,
}


class _FakeTicker:
    """Minimal yfinance Ticker stub."""

    def __init__(self, info: dict[str, object]) -> None:
        self.info = info


@pytest.fixture
def _mock_yf_ticker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``yf.Ticker`` to return a fake info dict."""

    def _ticker(_symbol: str) -> _FakeTicker:
        return _FakeTicker(_SAMPLE_YF_INFO)

    monkeypatch.setattr("investment_portfolio.fundamentals.yf.Ticker", _ticker)


@pytest.fixture
def _mock_yf_ticker_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``yf.Ticker`` to return info with no currentPrice."""

    def _ticker(_symbol: str) -> _FakeTicker:
        return _FakeTicker({"shortName": "Empty"})

    monkeypatch.setattr("investment_portfolio.fundamentals.yf.Ticker", _ticker)


@pytest.fixture
def _mock_yf_ticker_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``yf.Ticker`` to raise."""

    def _ticker(_symbol: str) -> object:
        msg = "network error"
        raise ConnectionError(msg)

    monkeypatch.setattr("investment_portfolio.fundamentals.yf.Ticker", _ticker)


@pytest.fixture
def _mock_yf_ticker_empty_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``yf.Ticker`` to return a completely empty dict."""

    def _ticker(_symbol: str) -> _FakeTicker:
        return _FakeTicker({})

    monkeypatch.setattr("investment_portfolio.fundamentals.yf.Ticker", _ticker)


# ── _safe_float unit tests ──────────────────────────────────────────────────


class TestSafeFloat:
    def test_returns_float_for_int(self) -> None:
        assert _safe_float({"k": 42}, "k") == 42.0

    def test_returns_float_for_float(self) -> None:
        assert _safe_float({"k": 3.14}, "k") == 3.14

    def test_returns_float_for_numeric_string(self) -> None:
        assert _safe_float({"k": "99.5"}, "k") == 99.5

    def test_returns_none_for_missing_key(self) -> None:
        assert _safe_float({}, "k") is None

    def test_returns_none_for_none_value(self) -> None:
        assert _safe_float({"k": None}, "k") is None

    def test_returns_none_for_non_numeric_string(self) -> None:
        assert _safe_float({"k": "not-a-number"}, "k") is None

    def test_returns_none_for_unconvertible_type(self) -> None:
        assert _safe_float({"k": object()}, "k") is None


# ── _build_from_yf unit tests ───────────────────────────────────────────────


class TestBuildFromYf:
    def test_builds_from_full_info(self) -> None:
        f = _build_from_yf(ticker="AAPL", info=_SAMPLE_YF_INFO)
        assert f.ticker == "AAPL"
        assert f.name == "Apple Inc."
        assert f.current_price == 180.0
        assert f.eps_ttm == 6.13

    def test_missing_fields_become_none(self) -> None:
        f = _build_from_yf(ticker="X", info={"currentPrice": 50.0})
        assert f.name == "X"
        assert f.sector == "N/A"
        assert f.eps_ttm is None
        assert f.trailing_pe is None

    def test_defaults_name_to_ticker(self) -> None:
        f = _build_from_yf(ticker="FOO", info={})
        assert f.name == "FOO"


# ── fetch_fundamentals — yfinance succeeds ──────────────────────────────────


@pytest.mark.usefixtures("_mock_yf_ticker")
class TestFetchFundamentalsYfinance:
    def test_returns_fundamentals(self) -> None:
        f = fetch_fundamentals(ticker="AAPL")
        assert isinstance(f, Fundamentals)

    def test_ticker(self) -> None:
        f = fetch_fundamentals(ticker="AAPL")
        assert f.ticker == "AAPL"

    def test_name(self) -> None:
        f = fetch_fundamentals(ticker="AAPL")
        assert f.name == "Apple Inc."

    def test_current_price(self) -> None:
        f = fetch_fundamentals(ticker="AAPL")
        assert f.current_price == 180.0

    def test_eps_ttm(self) -> None:
        f = fetch_fundamentals(ticker="AAPL")
        assert f.eps_ttm == 6.13

    def test_sector(self) -> None:
        f = fetch_fundamentals(ticker="AAPL")
        assert f.sector == "Technology"

    def test_multiples(self) -> None:
        f = fetch_fundamentals(ticker="AAPL")
        assert f.trailing_pe == 29.36
        assert f.forward_pe == 25.71
        assert f.price_to_book == 41.1
        assert f.ev_to_ebitda == 21.85
        assert f.price_to_sales == 7.29


# ── fetch_fundamentals — yfinance returns empty ─────────────────────────────


@pytest.mark.usefixtures("_mock_yf_ticker_empty")
class TestFetchFundamentalsEmpty:
    def test_raises_for_no_current_price(self) -> None:
        with pytest.raises(ValueError, match="no fundamental data returned"):
            fetch_fundamentals(ticker="EMPTY")


@pytest.mark.usefixtures("_mock_yf_ticker_empty_dict")
class TestFetchFundamentalsEmptyDict:
    def test_raises_for_empty_dict(self) -> None:
        with pytest.raises(ValueError, match="no fundamental data returned"):
            fetch_fundamentals(ticker="EMPTY")


# ── fetch_fundamentals — yfinance raises ────────────────────────────────────


@pytest.mark.usefixtures("_mock_yf_ticker_raises")
class TestFetchFundamentalsRaises:
    def test_raises_value_error_on_exception(self) -> None:
        with pytest.raises(ValueError, match="yfinance error"):
            fetch_fundamentals(ticker="BROKEN")
