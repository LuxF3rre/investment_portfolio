"""Tests for the Risk Parity module."""

import numpy as np
import pandas as pd
import pytest

from investment_portfolio.rp import (
    RP_RISK_MEASURES,
    RiskParityResult,
    compare_risk_budgets,
    optimize_risk_parity,
)


class TestRPRiskMeasures:
    def test_count_is_twenty(self) -> None:
        assert len(RP_RISK_MEASURES) == 20

    def test_excludes_non_convex_measures(self) -> None:
        codes = set(RP_RISK_MEASURES.values())
        for excluded in ("WR", "RG", "MDD", "ADD"):
            assert excluded not in codes

    def test_all_codes_are_strings(self) -> None:
        for name, code in RP_RISK_MEASURES.items():
            assert isinstance(name, str)
            assert isinstance(code, str)


class TestOptimizeRiskParity:
    def test_weights_sum_to_one(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices)
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_no_negative_weights_without_short_selling(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_risk_parity(
            prices=multi_asset_prices, allow_short_selling=False
        )
        for w in result.weights.values():
            assert w >= -1e-6

    def test_returns_risk_parity_result(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices)
        assert isinstance(result, RiskParityResult)
        assert set(result.weights.keys()) == set(multi_asset_prices.columns)

    def test_risk_contributions_sum_to_one(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices)
        total_pct = sum(result.risk_contribution_pct.values())
        assert total_pct == pytest.approx(1.0, abs=1e-3)

    def test_equal_budget_near_equal_contributions(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices)
        n = len(result.risk_contribution_pct)
        target = 1.0 / n
        for pct in result.risk_contribution_pct.values():
            assert pct == pytest.approx(target, abs=0.10)

    def test_works_with_cvar(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices, rm="CVaR")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_mad(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices, rm="MAD")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_uci(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices, rm="UCI")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_custom_budget_vector(self, multi_asset_prices: pd.DataFrame) -> None:
        n = multi_asset_prices.shape[1]
        budget = np.array([[0.4], [0.3], [0.2], [0.1]])
        assert budget.shape == (n, 1)
        result = optimize_risk_parity(prices=multi_asset_prices, budget=budget)
        assert isinstance(result, RiskParityResult)
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_invalid_budget_shape_raises(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        bad_budget = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="budget must have shape"):
            optimize_risk_parity(prices=multi_asset_prices, budget=bad_budget)

    def test_non_positive_budget_raises(self, multi_asset_prices: pd.DataFrame) -> None:
        n = multi_asset_prices.shape[1]
        budget = np.zeros((n, 1))
        with pytest.raises(ValueError, match="positive"):
            optimize_risk_parity(prices=multi_asset_prices, budget=budget)

    def test_sharpe_and_return_are_finite(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices)
        assert np.isfinite(result.expected_return)
        assert np.isfinite(result.volatility)
        assert np.isfinite(result.sharpe_ratio)

    def test_budget_dict_matches_tickers(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_risk_parity(prices=multi_asset_prices)
        assert set(result.budget.keys()) == set(multi_asset_prices.columns)
        assert sum(result.budget.values()) == pytest.approx(1.0, abs=1e-6)


class TestCompareBudgets:
    def test_returns_dict_with_correct_labels(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        n = multi_asset_prices.shape[1]
        budgets = {
            "equal": None,
            "custom": np.ones((n, 1)) / n,
        }
        results = compare_risk_budgets(prices=multi_asset_prices, budgets=budgets)
        assert set(results.keys()) == {"equal", "custom"}

    def test_all_results_are_risk_parity_result(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        n = multi_asset_prices.shape[1]
        budgets = {
            "equal": None,
            "tilted": np.array([[0.4], [0.3], [0.2], [0.1]]),
        }
        assert budgets["tilted"].shape == (n, 1)
        results = compare_risk_budgets(prices=multi_asset_prices, budgets=budgets)
        for result in results.values():
            assert isinstance(result, RiskParityResult)
