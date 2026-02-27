"""Tests for the Nested Clustered Optimization module."""

import numpy as np
import pandas as pd
import pytest

from investment_portfolio.nco import (
    NCO_OBJECTIVES,
    NCO_RISK_MEASURES,
    NCOResult,
    optimize_nco,
)


class TestNCORiskMeasures:
    def test_count_is_twenty_four(self) -> None:
        assert len(NCO_RISK_MEASURES) == 24

    def test_all_codes_are_strings(self) -> None:
        for name, code in NCO_RISK_MEASURES.items():
            assert isinstance(name, str)
            assert isinstance(code, str)

    def test_no_rel_suffixed_measures(self) -> None:
        codes = set(NCO_RISK_MEASURES.values())
        for code in codes:
            assert not code.endswith("_Rel")

    def test_no_excluded_measures(self) -> None:
        codes = set(NCO_RISK_MEASURES.values())
        for excluded in ("vol", "VaR", "DaR", "VRG"):
            assert excluded not in codes


class TestNCOObjectives:
    def test_count_is_four(self) -> None:
        assert len(NCO_OBJECTIVES) == 4

    def test_contains_expected_codes(self) -> None:
        codes = set(NCO_OBJECTIVES.values())
        assert codes == {"MinRisk", "Sharpe", "Utility", "ERC"}


class TestOptimizeNCO:
    def test_weights_sum_to_one_minrisk(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, obj="MinRisk")
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_weights_sum_to_one_erc(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, obj="ERC")
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_weights_sum_to_one_sharpe(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, obj="Sharpe")
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_weights_sum_to_one_utility(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, obj="Utility")
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_no_negative_weights(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices)
        for w in result.weights.values():
            assert w >= -1e-6

    def test_returns_nco_result(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices)
        assert isinstance(result, NCOResult)

    def test_tickers_match_input(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices)
        assert set(result.weights.keys()) == set(multi_asset_prices.columns)

    def test_clustering_shape(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices)
        n = multi_asset_prices.shape[1]
        assert result.clustering.shape == (n - 1, 4)

    def test_asset_order_length(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices)
        assert len(result.asset_order) == multi_asset_prices.shape[1]

    def test_k_is_set(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices)
        assert result.k is not None
        assert result.k >= 1

    def test_objective_field_matches_input(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        for obj in ("MinRisk", "Sharpe", "Utility", "ERC"):
            result = optimize_nco(prices=multi_asset_prices, obj=obj)
            assert result.objective == obj

    def test_works_with_cvar(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, rm="CVaR")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_mdd(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, rm="MDD")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_mad(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, rm="MAD")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_ward_linkage(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_nco(prices=multi_asset_prices, linkage="ward")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_spearman_codependence(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_nco(prices=multi_asset_prices, codependence="spearman")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_sharpe_return_volatility_are_finite(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_nco(prices=multi_asset_prices)
        assert np.isfinite(result.expected_return)
        assert np.isfinite(result.volatility)
        assert np.isfinite(result.sharpe_ratio)

    def test_risk_aversion_affects_utility(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        low_aversion = optimize_nco(
            prices=multi_asset_prices, obj="Utility", risk_aversion=0.5
        )
        high_aversion = optimize_nco(
            prices=multi_asset_prices, obj="Utility", risk_aversion=10.0
        )
        # Different risk aversions should produce different allocations
        low_weights = list(low_aversion.weights.values())
        high_weights = list(high_aversion.weights.values())
        assert low_weights != pytest.approx(high_weights, abs=1e-4)
