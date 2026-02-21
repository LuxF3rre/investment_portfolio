"""Tests for the Modern Portfolio Theory module."""

import numpy as np
import pandas as pd
import pytest

from investment_portfolio.mpt import (
    PortfolioResult,
    build_efficient_frontier,
    calculate_asset_statistics,
    fetch_multi_history,
    optimize_portfolio,
)


class TestFetchMultiHistory:
    def test_raises_on_fewer_than_two_tickers(self) -> None:
        with pytest.raises(ValueError, match="at least 2 tickers"):
            fetch_multi_history(tickers=["AAPL"])

    def test_correct_columns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rng = np.random.default_rng(0)
        n = 100
        index = pd.bdate_range(start="2023-01-02", periods=n)

        def _mock_fetch(*, ticker: str, period: str = "5y") -> pd.DataFrame:  # noqa: ARG001
            prices = 100 + rng.standard_normal(n).cumsum()
            return pd.DataFrame({"Close": prices}, index=index)

        monkeypatch.setattr("investment_portfolio.mpt.fetch_history", _mock_fetch)
        result = fetch_multi_history(tickers=["AAPL", "MSFT", "GOOGL"])

        assert list(result.columns) == ["AAPL", "MSFT", "GOOGL"]
        assert len(result) == n

    def test_inner_join_drops_non_overlapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        idx_a = pd.bdate_range(start="2023-01-02", periods=50)
        idx_b = pd.bdate_range(start="2023-02-01", periods=50)

        def _mock_fetch(*, ticker: str, period: str = "5y") -> pd.DataFrame:  # noqa: ARG001
            idx = idx_a if ticker == "A" else idx_b
            return pd.DataFrame({"Close": range(len(idx))}, index=idx)

        monkeypatch.setattr("investment_portfolio.mpt.fetch_history", _mock_fetch)
        result = fetch_multi_history(tickers=["A", "B"])

        overlap = idx_a.intersection(idx_b)
        assert len(result) == len(overlap)


class TestOptimizePortfolio:
    def test_weights_sum_to_one(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_portfolio(prices=multi_asset_prices)
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_no_negative_weights_without_short_selling(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_portfolio(
            prices=multi_asset_prices, allow_short_selling=False
        )
        for w in result.weights.values():
            assert w >= -1e-6

    def test_returns_portfolio_result(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_portfolio(prices=multi_asset_prices)
        assert isinstance(result, PortfolioResult)
        assert set(result.weights.keys()) == set(multi_asset_prices.columns)

    def test_works_with_cvar_risk_measure(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_portfolio(prices=multi_asset_prices, rm="CVaR", obj="MinRisk")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_mad_risk_measure(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_portfolio(prices=multi_asset_prices, rm="MAD", obj="MinRisk")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_min_risk_objective(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_portfolio(prices=multi_asset_prices, obj="MinRisk")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_kelly_exact_produces_result(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_portfolio(
            prices=multi_asset_prices, kelly="exact", obj="Sharpe"
        )
        assert isinstance(result, PortfolioResult)


class TestBuildEfficientFrontier:
    def test_correct_number_of_points(self, multi_asset_prices: pd.DataFrame) -> None:
        n_pts = 20
        risks, returns, weights_df = build_efficient_frontier(
            prices=multi_asset_prices, num_points=n_pts
        )
        assert len(risks) == len(returns) == weights_df.shape[1]

    def test_returns_and_risks_match_lengths(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        risks, returns, _ = build_efficient_frontier(
            prices=multi_asset_prices, num_points=30
        )
        assert len(risks) == len(returns)

    def test_weights_sum_to_one_per_point(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        _, _, weights_df = build_efficient_frontier(
            prices=multi_asset_prices, num_points=10
        )
        for col in weights_df.columns:
            total = weights_df[col].sum()
            assert total == pytest.approx(1.0, abs=1e-4)

    def test_works_with_non_mv_risk_measure(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        risks, returns, _weights_df = build_efficient_frontier(
            prices=multi_asset_prices, rm="CVaR", num_points=10
        )
        assert len(risks) > 0
        assert len(returns) > 0


class TestCalculateAssetStatistics:
    def test_correct_keys(self, multi_asset_prices: pd.DataFrame) -> None:
        stats = calculate_asset_statistics(prices=multi_asset_prices)
        assert set(stats.keys()) == set(multi_asset_prices.columns)
        for ticker_stats in stats.values():
            assert "annualized_return" in ticker_stats
            assert "annualized_volatility" in ticker_stats

    def test_positive_volatility(self, multi_asset_prices: pd.DataFrame) -> None:
        stats = calculate_asset_statistics(prices=multi_asset_prices)
        for ticker_stats in stats.values():
            assert ticker_stats["annualized_volatility"] > 0

    def test_float_values(self, multi_asset_prices: pd.DataFrame) -> None:
        stats = calculate_asset_statistics(prices=multi_asset_prices)
        for ticker_stats in stats.values():
            assert isinstance(ticker_stats["annualized_return"], float)
            assert isinstance(ticker_stats["annualized_volatility"], float)
