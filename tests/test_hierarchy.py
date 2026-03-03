"""Tests for hierarchical forecasting.

Covers: aggregation, reconciliation, additive consistency.
"""

import numpy as np
import pandas as pd
import pytest

from app.forecasting.data_generator import generate_demand_data, build_hierarchy_matrix
from app.forecasting.hierarchy import aggregate_to_hierarchy, HierarchicalForecaster


@pytest.fixture(scope="module")
def hierarchy_data():
    Y_df, S_df, _, _ = generate_demand_data(seed=42, history_days=60)
    return Y_df, S_df


class TestAggregation:
    def test_aggregate_produces_all_levels(self, hierarchy_data):
        Y_df, S_df = hierarchy_data
        Y_agg, S, tags = aggregate_to_hierarchy(Y_df, S_df)

        # Check all levels present
        assert "Total" in Y_agg["unique_id"].unique()
        for wh in ["NYC", "LAX", "CHI"]:
            assert wh in Y_agg["unique_id"].unique()

    def test_national_equals_sum_of_warehouses(self, hierarchy_data):
        Y_df, S_df = hierarchy_data
        Y_agg, _, tags = aggregate_to_hierarchy(Y_df, S_df)

        # For each date, national = sum of warehouses
        national = Y_agg[Y_agg["unique_id"] == "Total"].set_index("ds")["y"]
        wh_sum = (
            Y_agg[Y_agg["unique_id"].isin(tags["warehouse"])]
            .groupby("ds")["y"].sum()
        )
        common_dates = national.index.intersection(wh_sum.index)
        np.testing.assert_array_almost_equal(
            national.loc[common_dates].values,
            wh_sum.loc[common_dates].values,
            decimal=2,
        )

    def test_summation_matrix_consistency(self, hierarchy_data):
        _, S_df = hierarchy_data
        S, tags = build_hierarchy_matrix(S_df)

        # S matrix: top rows should aggregate bottom rows
        n_bottom = len(tags["sku"])
        # National row sums to n_bottom (all ones)
        assert S[0].sum() == n_bottom


class TestReconciliation:
    def test_bottom_up_produces_all_levels(self, hierarchy_data):
        Y_df, S_df = hierarchy_data
        Y_agg, S, tags = aggregate_to_hierarchy(Y_df, S_df)

        # Create fake base forecasts
        Y_hat = Y_agg[["unique_id", "ds"]].copy()
        Y_hat["y_hat"] = Y_agg["y"] * 1.1  # 10% over-forecast

        reconciler = HierarchicalForecaster(method="bottom_up")
        reconciled = reconciler.reconcile(Y_hat, S, tags)

        assert "y_hat_reconciled" in reconciled.columns
        assert len(reconciled) > 0

    def test_reconciled_non_negative(self, hierarchy_data):
        Y_df, S_df = hierarchy_data
        Y_agg, S, tags = aggregate_to_hierarchy(Y_df, S_df)

        Y_hat = Y_agg[["unique_id", "ds"]].copy()
        Y_hat["y_hat"] = Y_agg["y"] * 0.9

        reconciler = HierarchicalForecaster(method="bottom_up")
        reconciled = reconciler.reconcile(Y_hat, S, tags)

        assert (reconciled["y_hat_reconciled"] >= 0).all()
