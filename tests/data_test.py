"""Tests for the data-fetching module."""

import pandas as pd
import pytest

from investment_portfolio.data import fetch_history


@pytest.fixture
def _mock_yf_download(monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    """Patch ``yf.download`` to return a small DataFrame."""
    index = pd.bdate_range("2024-01-02", periods=5)
    df = pd.DataFrame(
        {"Close": [100.0, 101.0, 102.0, 101.5, 103.0]},
        index=index,
    )

    def _download(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return df

    monkeypatch.setattr("investment_portfolio.data.yf.download", _download)
    return df


@pytest.fixture
def _mock_yf_download_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``yf.download`` to return an empty DataFrame."""

    def _download(*_args: object, **_kwargs: object) -> pd.DataFrame:
        return pd.DataFrame()

    monkeypatch.setattr("investment_portfolio.data.yf.download", _download)


@pytest.mark.usefixtures("_mock_yf_download")
def test_fetch_history_returns_dataframe() -> None:
    df = fetch_history(ticker="AAPL", period="5y")
    assert isinstance(df, pd.DataFrame)
    assert "Close" in df.columns
    assert df.index.name == "Date"


@pytest.mark.usefixtures("_mock_yf_download")
def test_fetch_history_has_correct_length() -> None:
    df = fetch_history(ticker="AAPL")
    assert len(df) == 5


@pytest.mark.usefixtures("_mock_yf_download_empty")
def test_fetch_history_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no data returned"):
        fetch_history(ticker="INVALID")
