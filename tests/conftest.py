"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest

from investment_portfolio.gpv import GPVModelConfig, KernelType


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


@pytest.fixture
def multi_asset_prices() -> pd.DataFrame:
    """Synthetic multi-asset prices with Cholesky-correlated returns."""
    rng = np.random.default_rng(42)
    n = 1260
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]

    # Target correlation structure
    corr = np.array(
        [
            [1.0, 0.7, 0.5, 0.4],
            [0.7, 1.0, 0.6, 0.5],
            [0.5, 0.6, 1.0, 0.3],
            [0.4, 0.5, 0.3, 1.0],
        ]
    )
    vols = np.array([0.02, 0.018, 0.022, 0.025])
    cov = np.outer(vols, vols) * corr
    chol = np.linalg.cholesky(cov)

    means = np.array([0.0004, 0.0003, 0.0005, 0.0002])
    uncorrelated = rng.standard_normal((n, 4))
    correlated = uncorrelated @ chol.T + means

    prices_arr = 100.0 * np.cumprod(1 + correlated, axis=0)
    index = pd.bdate_range(start="2019-01-02", periods=n)
    return pd.DataFrame(prices_arr, index=index, columns=pd.Index(tickers))


@pytest.fixture
def gpv_config() -> GPVModelConfig:
    """Default GPV model config with exponential kernel."""
    maturities = np.linspace(0.01, 2.0, 50)
    xi_0 = np.full(50, 0.04)  # flat 20% vol
    return GPVModelConfig(
        kernel_type=KernelType.EXPONENTIAL,
        hurst=0.5,
        poly_coeffs=(1.0,),
        xi_0=xi_0,
        maturities=maturities,
        rho=-0.7,
        epsilon=0.1,
    )
