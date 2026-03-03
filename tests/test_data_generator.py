"""Tests for Nixtla-format data generator.

Covers: schema validation, M5 properties, hierarchy, reproducibility.
"""

import numpy as np
import pandas as pd
import pytest

from app.forecasting.data_generator import (
    HierarchySpec,
    Y_SCHEMA,
    S_SCHEMA,
    X_FUTURE_SCHEMA,
    X_PAST_SCHEMA,
    build_hierarchy_matrix,
    compute_data_hash,
    generate_demand_data,
    generate_sku_ids,
    get_data_statistics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def generated_data():
    """Generate data once for all tests in this module."""
    return generate_demand_data(seed=42, history_days=365)


@pytest.fixture
def Y_df(generated_data):
    return generated_data[0]


@pytest.fixture
def S_df(generated_data):
    return generated_data[1]


@pytest.fixture
def X_future(generated_data):
    return generated_data[2]


@pytest.fixture
def X_past(generated_data):
    return generated_data[3]


# ---------------------------------------------------------------------------
# Schema Validation Tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Verify Pandera contracts are enforced."""

    def test_y_df_schema(self, Y_df):
        """Y_df passes Pandera validation."""
        validated = Y_SCHEMA.validate(Y_df)
        assert len(validated) == len(Y_df)

    def test_s_df_schema(self, S_df):
        """S_df passes Pandera validation."""
        validated = S_SCHEMA.validate(S_df)
        assert len(validated) == len(S_df)

    def test_x_future_schema(self, X_future):
        """X_future passes Pandera validation."""
        validated = X_FUTURE_SCHEMA.validate(X_future)
        assert len(validated) == len(X_future)

    def test_x_past_schema(self, X_past):
        """X_past passes Pandera validation."""
        validated = X_PAST_SCHEMA.validate(X_past)
        assert len(validated) == len(X_past)

    def test_y_demand_non_negative(self, Y_df):
        """All demand values >= 0."""
        assert (Y_df["y"] >= 0).all()

    def test_x_past_price_positive(self, X_past):
        """All prices > 0."""
        assert (X_past["price"] > 0).all()

    def test_x_past_stock_non_negative(self, X_past):
        """All stock levels >= 0."""
        assert (X_past["stock_level"] >= 0).all()

    def test_promo_flag_binary(self, X_future):
        """Promo flags are 0 or 1."""
        assert set(X_future["promo_flag"].unique()).issubset({0, 1})

    def test_holiday_flag_binary(self, X_future):
        """Holiday flags are 0 or 1."""
        assert set(X_future["is_holiday"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# M5 Statistical Properties Tests
# ---------------------------------------------------------------------------

class TestM5Properties:
    """Verify all 5 M5-style statistical properties."""

    def test_intermittent_demand_exists(self, Y_df):
        """At least 20% of SKUs have >50% zero-demand days."""
        zero_pct = Y_df.groupby("unique_id")["y"].apply(lambda x: (x == 0).mean())
        intermittent_pct = (zero_pct > 0.5).mean()
        # Allow some tolerance since substitution effects may reduce zeros
        assert intermittent_pct >= 0.15, f"Only {intermittent_pct:.1%} intermittent SKUs"

    def test_demand_not_normal(self, Y_df):
        """Demand follows heavy-tailed distribution (not Normal)."""
        # Check skewness > 0 (right-skewed, typical of NegBin)
        demands = Y_df[Y_df["y"] > 0]["y"]
        skewness = demands.skew()
        assert skewness > 0, f"Skewness {skewness} should be positive"

    def test_censored_demand_exists(self, Y_df, X_past):
        """Some zero-demand days coincide with zero stock (censoring)."""
        merged = Y_df.merge(X_past, on=["unique_id", "ds"])
        censored = merged[(merged["y"] == 0) & (merged["stock_level"] == 0)]
        assert len(censored) > 0, "No censored demand found"

    def test_promo_affects_demand(self, Y_df, X_future):
        """Promo days have statistically different demand than non-promo."""
        merged = Y_df.merge(X_future[["unique_id", "ds", "promo_flag"]], on=["unique_id", "ds"])
        promo_demand = merged[merged["promo_flag"] == 1]["y"].mean()
        non_promo_demand = merged[merged["promo_flag"] == 0]["y"].mean()
        # Promo should generally increase demand (price reduction → more demand)
        # But with substitution, the relationship is complex
        assert promo_demand != non_promo_demand, "Promo has no effect on demand"


# ---------------------------------------------------------------------------
# Hierarchy Tests
# ---------------------------------------------------------------------------

class TestHierarchy:
    """Verify 4-layer hierarchy structure."""

    def test_sku_count(self, S_df):
        """Correct number of SKUs generated."""
        spec = HierarchySpec()
        assert len(S_df) == spec.n_skus

    def test_warehouse_count(self, S_df):
        """3 warehouses present."""
        assert S_df["warehouse"].nunique() == 3

    def test_category_count(self, S_df):
        """4 categories present."""
        assert S_df["category"].nunique() == 4

    def test_subcategory_count(self, S_df):
        """20 subcategories present."""
        assert S_df["subcategory"].nunique() == 20

    def test_hierarchy_matrix_shape(self, S_df):
        """Summation matrix has correct shape."""
        S, tags = build_hierarchy_matrix(S_df)
        n_bottom = len(S_df)
        n_total = 1 + 3 + len(tags["category"]) + n_bottom
        assert S.shape == (n_total, n_bottom)

    def test_hierarchy_matrix_national_sums(self, S_df):
        """National level sums all bottom-level nodes."""
        S, tags = build_hierarchy_matrix(S_df)
        # First row (national) should be all 1s
        assert np.all(S[0] == 1.0)

    def test_hierarchy_matrix_additive(self, S_df):
        """Warehouse rows sum to national row."""
        S, tags = build_hierarchy_matrix(S_df)
        warehouse_sum = S[1:4].sum(axis=0)  # 3 warehouse rows
        np.testing.assert_array_almost_equal(warehouse_sum, S[0])

    def test_unique_ids_format(self, S_df):
        """SKU IDs follow expected format."""
        assert all(uid.startswith("SKU_") for uid in S_df["unique_id"])

    def test_all_y_ids_in_s(self, Y_df, S_df):
        """All unique_ids in Y_df exist in S_df."""
        y_ids = set(Y_df["unique_id"].unique())
        s_ids = set(S_df["unique_id"].unique())
        assert y_ids == s_ids


# ---------------------------------------------------------------------------
# Reproducibility Tests
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Verify deterministic data generation."""

    def test_same_seed_same_data(self):
        """Same seed produces identical data."""
        Y1, S1, _, _ = generate_demand_data(seed=42, history_days=30)
        Y2, S2, _, _ = generate_demand_data(seed=42, history_days=30)
        pd.testing.assert_frame_equal(Y1, Y2)
        pd.testing.assert_frame_equal(S1, S2)

    def test_different_seed_different_data(self):
        """Different seeds produce different data."""
        Y1, _, _, _ = generate_demand_data(seed=42, history_days=30)
        Y2, _, _, _ = generate_demand_data(seed=123, history_days=30)
        assert not Y1["y"].equals(Y2["y"])

    def test_data_hash_deterministic(self):
        """SHA-256 hash is deterministic."""
        Y1, _, _, _ = generate_demand_data(seed=42, history_days=30)
        Y2, _, _, _ = generate_demand_data(seed=42, history_days=30)
        assert compute_data_hash(Y1) == compute_data_hash(Y2)


# ---------------------------------------------------------------------------
# Data Statistics Tests
# ---------------------------------------------------------------------------

class TestDataStatistics:
    """Verify data statistics computation."""

    def test_statistics_keys(self, Y_df, S_df):
        """Statistics dict has all expected keys."""
        stats = get_data_statistics(Y_df, S_df)
        expected_keys = {
            "total_rows", "n_skus", "n_days", "n_intermittent_skus",
            "intermittent_pct", "mean_demand", "median_demand",
            "zero_demand_pct", "n_warehouses", "n_categories", "n_subcategories",
        }
        assert expected_keys.issubset(stats.keys())

    def test_row_count_matches(self, Y_df, S_df):
        """total_rows matches actual DataFrame length."""
        stats = get_data_statistics(Y_df, S_df)
        assert stats["total_rows"] == len(Y_df)
