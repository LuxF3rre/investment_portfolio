"""Tests for the Hierarchical Clustering Portfolio module."""

import numpy as np
import pandas as pd
import pytest

from investment_portfolio.hcp import (
    HC_RISK_MEASURES,
    HCResult,
    optimize_hierarchical_clustering,
)


class TestHCRiskMeasures:
    def test_count_is_thirty_five(self) -> None:
        assert len(HC_RISK_MEASURES) == 35

    def test_includes_compounded_drawdown_measures(self) -> None:
        codes = set(HC_RISK_MEASURES.values())
        for rel in (
            "ADD_Rel",
            "UCI_Rel",
            "DaR_Rel",
            "CDaR_Rel",
            "EDaR_Rel",
            "RLDaR_Rel",
            "MDD_Rel",
        ):
            assert rel in codes

    def test_all_codes_are_strings(self) -> None:
        for name, code in HC_RISK_MEASURES.items():
            assert isinstance(name, str)
            assert isinstance(code, str)


class TestOptimizeHierarchicalClustering:
    def test_weights_sum_to_one_hrp(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices)
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_weights_sum_to_one_herc(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(
            prices=multi_asset_prices, model="HERC"
        )
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_no_negative_weights(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices)
        for w in result.weights.values():
            assert w >= -1e-6

    def test_returns_hc_result(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices)
        assert isinstance(result, HCResult)

    def test_tickers_match_input(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices)
        assert set(result.weights.keys()) == set(multi_asset_prices.columns)

    def test_clustering_shape(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices)
        n = multi_asset_prices.shape[1]
        assert result.clustering.shape == (n - 1, 4)

    def test_asset_order_length(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices)
        assert len(result.asset_order) == multi_asset_prices.shape[1]

    def test_k_is_none_for_hrp(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(
            prices=multi_asset_prices, model="HRP"
        )
        assert result.k is None

    def test_k_is_set_for_herc(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(
            prices=multi_asset_prices, model="HERC"
        )
        assert result.k is not None
        assert result.k >= 1

    def test_model_field_matches_input(self, multi_asset_prices: pd.DataFrame) -> None:
        for model in ("HRP", "HERC"):
            result = optimize_hierarchical_clustering(
                prices=multi_asset_prices, model=model
            )
            assert result.model == model

    def test_works_with_cvar(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices, rm="CVaR")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_mdd(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices, rm="MDD")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_dar(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices, rm="DaR")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_var(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices, rm="VaR")
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_ward_linkage(self, multi_asset_prices: pd.DataFrame) -> None:
        result = optimize_hierarchical_clustering(
            prices=multi_asset_prices, linkage="ward"
        )
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_works_with_spearman_codependence(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_hierarchical_clustering(
            prices=multi_asset_prices, codependence="spearman"
        )
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-4)

    def test_sharpe_and_return_are_finite(
        self, multi_asset_prices: pd.DataFrame
    ) -> None:
        result = optimize_hierarchical_clustering(prices=multi_asset_prices)
        assert np.isfinite(result.expected_return)
        assert np.isfinite(result.volatility)
        assert np.isfinite(result.sharpe_ratio)
