"""Company fundamentals fetching via yfinance."""

__all__ = ["Fundamentals", "fetch_fundamentals"]

from dataclasses import dataclass

import yfinance as yf


@dataclass(frozen=True, slots=True)
class Fundamentals:
    """Snapshot of a company's key financial data.

    All financial fields are ``float | None`` because not every ticker
    reports every metric.
    """

    ticker: str
    name: str
    sector: str
    industry: str

    # Price & capitalisation
    current_price: float | None
    shares_outstanding: float | None
    market_cap: float | None
    enterprise_value: float | None

    # Per-share metrics
    eps_ttm: float | None
    eps_forward: float | None
    book_value_per_share: float | None
    revenue_per_share: float | None

    # Totals
    ebitda: float | None
    total_revenue: float | None
    total_debt: float | None
    total_cash: float | None

    # Pre-computed multiples (for display)
    trailing_pe: float | None
    forward_pe: float | None
    price_to_book: float | None
    ev_to_ebitda: float | None
    price_to_sales: float | None


def _safe_float(info: dict[str, object], key: str) -> float | None:
    """Extract a numeric value from a yfinance info dict, or ``None``."""
    val = info.get(key)
    if val is None:
        return None
    try:
        result = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result


def _build_from_yf(*, ticker: str, info: dict[str, object]) -> Fundamentals:
    """Construct a ``Fundamentals`` from a yfinance ``Ticker.info`` dict."""
    return Fundamentals(
        ticker=ticker,
        name=str(info.get("shortName", ticker)),
        sector=str(info.get("sector", "N/A")),
        industry=str(info.get("industry", "N/A")),
        current_price=_safe_float(info, "currentPrice"),
        shares_outstanding=_safe_float(info, "sharesOutstanding"),
        market_cap=_safe_float(info, "marketCap"),
        enterprise_value=_safe_float(info, "enterpriseValue"),
        eps_ttm=_safe_float(info, "trailingEps"),
        eps_forward=_safe_float(info, "forwardEps"),
        book_value_per_share=_safe_float(info, "bookValue"),
        revenue_per_share=_safe_float(info, "revenuePerShare"),
        ebitda=_safe_float(info, "ebitda"),
        total_revenue=_safe_float(info, "totalRevenue"),
        total_debt=_safe_float(info, "totalDebt"),
        total_cash=_safe_float(info, "totalCash"),
        trailing_pe=_safe_float(info, "trailingPE"),
        forward_pe=_safe_float(info, "forwardPE"),
        price_to_book=_safe_float(info, "priceToBook"),
        ev_to_ebitda=_safe_float(info, "enterpriseToEbitda"),
        price_to_sales=_safe_float(info, "priceToSalesTrailing12Months"),
    )


def fetch_fundamentals(*, ticker: str) -> Fundamentals:
    """Fetch company fundamentals for a single ticker via yfinance.

    Args:
        ticker: A yfinance-compatible ticker symbol (e.g. ``"AAPL"``).

    Returns:
        A populated ``Fundamentals`` dataclass.

    Raises:
        ValueError: If yfinance returns no usable data.
    """
    try:
        info: dict[str, object] = yf.Ticker(ticker).info
    except Exception as exc:
        msg = f"yfinance error for ticker={ticker!r}"
        raise ValueError(msg) from exc

    if not info or info.get("currentPrice") is None:
        msg = f"no fundamental data returned for ticker={ticker!r}"
        raise ValueError(msg)

    return _build_from_yf(ticker=ticker, info=info)
