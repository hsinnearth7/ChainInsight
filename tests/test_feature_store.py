"""Tests for Feature Store pattern.

Covers: offline materialization, online serving, consistency, TTL.
"""

import numpy as np
import pandas as pd
import pytest

from app.forecasting.feature_store import (
    DEMAND_FEATURES,
    FeatureStore,
)


@pytest.fixture
def sample_y_df():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    records = []
    for uid in ["SKU_0001", "SKU_0002"]:
        for ds in dates:
            records.append({"unique_id": uid, "ds": ds, "y": round(rng.uniform(5, 50), 2)})
    return pd.DataFrame(records)


@pytest.fixture
def sample_x_future():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    records = []
    for uid in ["SKU_0001", "SKU_0002"]:
        for ds in dates:
            records.append({
                "unique_id": uid, "ds": ds,
                "promo_flag": int(rng.random() < 0.1),
                "is_holiday": 0,
                "temperature": round(rng.normal(60, 10), 1),
            })
    return pd.DataFrame(records)


@pytest.fixture
def feature_store(sample_y_df, sample_x_future):
    fs = FeatureStore()
    fs.materialize_offline(sample_y_df, X_future=sample_x_future)
    return fs


class TestOfflineStore:
    def test_materialization(self, feature_store):
        """Offline store is materialized after calling materialize_offline."""
        assert feature_store.last_update is not None

    def test_lag_features_computed(self, feature_store):
        """Lag features are present in materialized data."""
        features = feature_store.get_training_features()
        assert "lag_1" in features.columns
        assert "lag_7" in features.columns
        assert "rolling_mean_7" in features.columns

    def test_filter_by_sku(self, feature_store):
        """Can filter features by unique_id."""
        features = feature_store.get_training_features(unique_ids=["SKU_0001"])
        assert set(features["unique_id"].unique()) == {"SKU_0001"}

    def test_filter_by_date(self, feature_store):
        features = feature_store.get_training_features(
            start_date=pd.Timestamp("2023-02-01"),
            end_date=pd.Timestamp("2023-02-28"),
        )
        assert features["ds"].min() >= pd.Timestamp("2023-02-01")
        assert features["ds"].max() <= pd.Timestamp("2023-02-28")

    def test_feature_names_list(self, feature_store):
        names = feature_store.feature_names
        assert "lag_1" in names
        assert "unique_id" not in names
        assert "ds" not in names

    def test_not_materialized_raises(self):
        fs = FeatureStore()
        with pytest.raises(ValueError, match="not materialized"):
            fs.get_training_features()


class TestOnlineStore:
    def test_update_and_get(self):
        fs = FeatureStore()
        fs.update_online("SKU_0001", {"lag_1": 42.0, "rolling_mean_7": 35.0})
        result = fs.get_online_features("SKU_0001")
        assert result is not None
        assert result["lag_1"] == 42.0

    def test_missing_returns_none(self):
        fs = FeatureStore()
        assert fs.get_online_features("NONEXISTENT") is None

    def test_update_timestamp(self):
        fs = FeatureStore()
        fs.update_online("SKU_0001", {"lag_1": 10.0})
        result = fs.get_online_features("SKU_0001")
        assert "_updated_at" in result


class TestFeatureDefinitions:
    def test_demand_features_defined(self):
        assert DEMAND_FEATURES.name == "demand_features"
        assert len(DEMAND_FEATURES.features) > 0

    def test_entities(self):
        assert "unique_id" in DEMAND_FEATURES.entities
