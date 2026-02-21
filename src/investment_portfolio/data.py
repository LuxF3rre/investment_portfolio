"""Asset-agnostic data fetching via yfinance."""

__all__ = ["fetch_history"]

import pandas as pd
import yfinance as yf


def fetch_history(*, ticker: str, period: str = "max") -> pd.DataFrame:
    """Download historical price data for any yfinance-compatible ticker.

    Args:
        ticker: A yfinance ticker symbol (e.g. ``"AAPL"``, ``"EURUSD=X"``,
            ``"GC=F"``, ``"SPY"``).
        period: Look-back period accepted by yfinance (e.g. ``"1y"``,
            ``"5y"``, ``"max"``).

    Returns:
        DataFrame with a ``DatetimeIndex`` and a single ``"Close"`` column.

    Raises:
        ValueError: If yfinance returns no data for the given ticker/period.
    """
    raw: pd.DataFrame = yf.download(
        ticker,
        period=period,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        msg = f"no data returned for ticker={ticker!r}, period={period!r}"
        raise ValueError(msg)

    df = raw[["Close"]].copy()
    df.columns = ["Close"]
    df.index.name = "Date"
    return df
