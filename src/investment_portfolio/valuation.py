"""Comparable-multiples valuation — pure functions, no I/O."""

__all__ = [
    "ComparablesResult",
    "MultipleValuation",
    "PeerMultiple",
    "SensitivityRow",
    "calculate_comparables",
    "calculate_ev_ebitda_valuation",
    "calculate_pb_valuation",
    "calculate_pe_valuation",
    "calculate_ps_valuation",
]

from dataclasses import dataclass
from statistics import mean, median

from investment_portfolio.fundamentals import Fundamentals


@dataclass(frozen=True, slots=True)
class PeerMultiple:
    """One peer's data for a single valuation multiple."""

    ticker: str
    name: str
    value: float | None


@dataclass(frozen=True, slots=True)
class SensitivityRow:
    """One row of a sensitivity table."""

    multiple: float
    implied_price: float
    upside_pct: float


@dataclass(frozen=True, slots=True)
class MultipleValuation:
    """Result for a single valuation multiple."""

    method: str
    implied_price: float
    current_price: float
    upside_pct: float
    median_multiple: float
    mean_multiple: float
    target_metric_value: float
    target_metric_label: str
    peers: tuple[PeerMultiple, ...]
    sensitivity: tuple[SensitivityRow, ...]


@dataclass(frozen=True, slots=True)
class ComparablesResult:
    """Aggregated comparables valuation across all multiples."""

    target_ticker: str
    target_name: str
    current_price: float
    pe: MultipleValuation | None
    ev_ebitda: MultipleValuation | None
    pb: MultipleValuation | None
    ps: MultipleValuation | None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _percentiles(
    values: list[float],
) -> tuple[float, float, float, float, float]:
    """Return (p5, p25, p50, p75, p95) from a sorted list of values."""
    s = sorted(values)
    n = len(s)

    def _lerp(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return s[lo] + frac * (s[hi] - s[lo])

    return _lerp(0.05), _lerp(0.25), _lerp(0.50), _lerp(0.75), _lerp(0.95)


def _build_sensitivity(
    *,
    peer_values: list[float],
    metric_value: float,
    current_price: float,
    implied_price_fn: object | None = None,
) -> tuple[SensitivityRow, ...]:
    """Build sensitivity rows at percentile multiples + +/-20% around median.

    Args:
        peer_values: Valid peer multiple values.
        metric_value: Target's metric (EPS, BVPS, etc.) — used when
            ``implied_price_fn`` is None (simple multiply).
        current_price: Current share price for upside calculation.
        implied_price_fn: Optional callable ``(multiple) -> implied_price``
            for EV/EBITDA where the mapping is not a simple product.
    """
    if not peer_values:
        return ()

    p5, p25, p50, p75, p95 = _percentiles(peer_values)
    med = median(peer_values)
    targets = sorted({p5, p25, p50, p75, p95, med * 0.8, med * 1.2})

    rows: list[SensitivityRow] = []
    for mult in targets:
        if implied_price_fn is not None:
            imp = implied_price_fn(mult)  # type: ignore[operator]
        else:
            imp = mult * metric_value
        upside = (imp - current_price) / current_price if current_price else 0.0
        rows.append(SensitivityRow(multiple=mult, implied_price=imp, upside_pct=upside))
    return tuple(rows)


# ── P/E Valuation ────────────────────────────────────────────────────────────


def calculate_pe_valuation(
    *,
    target: Fundamentals,
    peers: tuple[Fundamentals, ...],
    use_forward: bool = False,
) -> MultipleValuation | None:
    """Value the target using peer-median P/E ratio.

    Args:
        target: Target company fundamentals.
        peers: Peer company fundamentals.
        use_forward: Use forward P/E and forward EPS instead of trailing.

    Returns:
        Valuation result or ``None`` if data is insufficient.
    """
    eps = target.eps_forward if use_forward else target.eps_ttm
    if eps is None or eps <= 0 or target.current_price is None:
        return None

    peer_multiples: list[PeerMultiple] = []
    valid_values: list[float] = []
    for p in peers:
        pe = p.forward_pe if use_forward else p.trailing_pe
        peer_multiples.append(PeerMultiple(ticker=p.ticker, name=p.name, value=pe))
        if pe is not None and pe > 0:
            valid_values.append(pe)

    if not valid_values:
        return None

    med = median(valid_values)
    avg = mean(valid_values)
    implied = med * eps
    upside = (implied - target.current_price) / target.current_price

    label = "Forward P/E" if use_forward else "P/E (TTM)"
    metric_label = "EPS (Forward)" if use_forward else "EPS (TTM)"

    return MultipleValuation(
        method=label,
        implied_price=implied,
        current_price=target.current_price,
        upside_pct=upside,
        median_multiple=med,
        mean_multiple=avg,
        target_metric_value=eps,
        target_metric_label=metric_label,
        peers=tuple(peer_multiples),
        sensitivity=_build_sensitivity(
            peer_values=valid_values,
            metric_value=eps,
            current_price=target.current_price,
        ),
    )


# ── EV/EBITDA Valuation ─────────────────────────────────────────────────────


def calculate_ev_ebitda_valuation(
    *,
    target: Fundamentals,
    peers: tuple[Fundamentals, ...],
) -> MultipleValuation | None:
    """Value the target using peer-median EV/EBITDA.

    Implied EV = median(EV/EBITDA) * target EBITDA.
    Implied equity = implied EV - total_debt + total_cash.
    Implied price = implied equity / shares_outstanding.

    Args:
        target: Target company fundamentals.
        peers: Peer company fundamentals.

    Returns:
        Valuation result or ``None`` if data is insufficient.
    """
    if (
        target.ebitda is None
        or target.ebitda <= 0
        or target.shares_outstanding is None
        or target.shares_outstanding <= 0
        or target.current_price is None
    ):
        return None

    debt = target.total_debt if target.total_debt is not None else 0.0
    cash = target.total_cash if target.total_cash is not None else 0.0

    peer_multiples: list[PeerMultiple] = []
    valid_values: list[float] = []
    for p in peers:
        peer_multiples.append(
            PeerMultiple(ticker=p.ticker, name=p.name, value=p.ev_to_ebitda)
        )
        if p.ev_to_ebitda is not None and p.ev_to_ebitda > 0:
            valid_values.append(p.ev_to_ebitda)

    if not valid_values:
        return None

    med = median(valid_values)
    avg = mean(valid_values)

    def _implied_price(mult: float) -> float:
        implied_ev = mult * target.ebitda  # type: ignore[operator]
        implied_equity = implied_ev - debt + cash
        return implied_equity / target.shares_outstanding

    implied = _implied_price(med)
    upside = (implied - target.current_price) / target.current_price

    return MultipleValuation(
        method="EV/EBITDA",
        implied_price=implied,
        current_price=target.current_price,
        upside_pct=upside,
        median_multiple=med,
        mean_multiple=avg,
        target_metric_value=target.ebitda,
        target_metric_label="EBITDA",
        peers=tuple(peer_multiples),
        sensitivity=_build_sensitivity(
            peer_values=valid_values,
            metric_value=target.ebitda,
            current_price=target.current_price,
            implied_price_fn=_implied_price,
        ),
    )


# ── P/B Valuation ───────────────────────────────────────────────────────────


def calculate_pb_valuation(
    *,
    target: Fundamentals,
    peers: tuple[Fundamentals, ...],
) -> MultipleValuation | None:
    """Value the target using peer-median Price-to-Book.

    Args:
        target: Target company fundamentals.
        peers: Peer company fundamentals.

    Returns:
        Valuation result or ``None`` if data is insufficient.
    """
    if (
        target.book_value_per_share is None
        or target.book_value_per_share <= 0
        or target.current_price is None
    ):
        return None

    peer_multiples: list[PeerMultiple] = []
    valid_values: list[float] = []
    for p in peers:
        peer_multiples.append(
            PeerMultiple(ticker=p.ticker, name=p.name, value=p.price_to_book)
        )
        if p.price_to_book is not None and p.price_to_book > 0:
            valid_values.append(p.price_to_book)

    if not valid_values:
        return None

    med = median(valid_values)
    avg = mean(valid_values)
    implied = med * target.book_value_per_share
    upside = (implied - target.current_price) / target.current_price

    return MultipleValuation(
        method="P/B",
        implied_price=implied,
        current_price=target.current_price,
        upside_pct=upside,
        median_multiple=med,
        mean_multiple=avg,
        target_metric_value=target.book_value_per_share,
        target_metric_label="BVPS",
        peers=tuple(peer_multiples),
        sensitivity=_build_sensitivity(
            peer_values=valid_values,
            metric_value=target.book_value_per_share,
            current_price=target.current_price,
        ),
    )


# ── P/S Valuation ───────────────────────────────────────────────────────────


def calculate_ps_valuation(
    *,
    target: Fundamentals,
    peers: tuple[Fundamentals, ...],
) -> MultipleValuation | None:
    """Value the target using peer-median Price-to-Sales.

    Args:
        target: Target company fundamentals.
        peers: Peer company fundamentals.

    Returns:
        Valuation result or ``None`` if data is insufficient.
    """
    if (
        target.revenue_per_share is None
        or target.revenue_per_share <= 0
        or target.current_price is None
    ):
        return None

    peer_multiples: list[PeerMultiple] = []
    valid_values: list[float] = []
    for p in peers:
        peer_multiples.append(
            PeerMultiple(ticker=p.ticker, name=p.name, value=p.price_to_sales)
        )
        if p.price_to_sales is not None and p.price_to_sales > 0:
            valid_values.append(p.price_to_sales)

    if not valid_values:
        return None

    med = median(valid_values)
    avg = mean(valid_values)
    implied = med * target.revenue_per_share
    upside = (implied - target.current_price) / target.current_price

    return MultipleValuation(
        method="P/S",
        implied_price=implied,
        current_price=target.current_price,
        upside_pct=upside,
        median_multiple=med,
        mean_multiple=avg,
        target_metric_value=target.revenue_per_share,
        target_metric_label="Revenue/Share",
        peers=tuple(peer_multiples),
        sensitivity=_build_sensitivity(
            peer_values=valid_values,
            metric_value=target.revenue_per_share,
            current_price=target.current_price,
        ),
    )


# ── Aggregate ────────────────────────────────────────────────────────────────


def calculate_comparables(
    *,
    target: Fundamentals,
    peers: tuple[Fundamentals, ...],
    use_forward_pe: bool = False,
) -> ComparablesResult:
    """Run all four comparable-multiples valuations.

    Each field is ``None`` if the corresponding data is insufficient.

    Args:
        target: Target company fundamentals.
        peers: Peer company fundamentals.
        use_forward_pe: Use forward P/E instead of trailing.

    Returns:
        Aggregated ``ComparablesResult``.
    """
    return ComparablesResult(
        target_ticker=target.ticker,
        target_name=target.name,
        current_price=target.current_price or 0.0,
        pe=calculate_pe_valuation(
            target=target, peers=peers, use_forward=use_forward_pe
        ),
        ev_ebitda=calculate_ev_ebitda_valuation(target=target, peers=peers),
        pb=calculate_pb_valuation(target=target, peers=peers),
        ps=calculate_ps_valuation(target=target, peers=peers),
    )
