"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for deterministic tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_prices() -> pd.Series:
    """Synthetic equity-like price series with DatetimeIndex."""
    rng = np.random.default_rng(42)
    n = 1260  # ~5 years of trading days
    daily_returns = rng.normal(0.0003, 0.015, n)
    prices = 100.0 * np.cumprod(1 + daily_returns)
    index = pd.bdate_range(start="2019-01-02", periods=n)
    return pd.Series(prices, index=index, name="Close")


@pytest.fixture
def sample_fx_rates() -> pd.Series:
    """Synthetic FX-rate-like series with DatetimeIndex."""
    rng = np.random.default_rng(99)
    n = 2000
    daily_returns = rng.normal(0.0, 0.005, n)
    rates = 1.10 * np.cumprod(1 + daily_returns)
    index = pd.bdate_range(start="2017-01-02", periods=n)
    return pd.Series(rates, index=index, name="Close")
