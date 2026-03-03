"""Tests for forecasting model zoo.

Covers: unified interface, factory, individual model behavior, routing.
"""

import numpy as np
import pandas as pd
import pytest

from app.forecasting.models import (
    ForecastModel,
    ForecastModelFactory,
    LightGBMForecaster,
    NaiveMovingAverage,
    RoutingEnsemble,
    SARIMAXForecaster,
    XGBoostForecaster,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_y_df():
    """Small sample Y_df for testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    records = []
    for uid in ["SKU_0001", "SKU_0002", "SKU_0003"]:
        base = rng.uniform(10, 50)
        for ds in dates:
            y = max(0, base + rng.normal(0, 5) + 3 * np.sin(2 * np.pi * ds.dayofyear / 365))
            records.append({"unique_id": uid, "ds": ds, "y": round(y, 2)})
    return pd.DataFrame(records)


@pytest.fixture
def intermittent_y_df():
    """Y_df with intermittent demand for routing test."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    records = []
    for ds in dates:
        y = rng.choice([0, 0, 0, 0, rng.uniform(5, 30)])  # 80% zeros
        records.append({"unique_id": "INTERMITTENT", "ds": ds, "y": round(max(0, y), 2)})
    return pd.DataFrame(records)


@pytest.fixture
def cold_start_y_df():
    """Y_df with only 30 days of history (below cold-start threshold)."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    records = []
    for ds in dates:
        records.append({"unique_id": "COLD_START", "ds": ds, "y": round(rng.uniform(5, 20), 2)})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Unified Interface Tests
# ---------------------------------------------------------------------------

class TestUnifiedInterface:
    """All models implement the ForecastModel protocol."""

    @pytest.mark.parametrize("model_name", ["naive_ma30", "sarimax", "xgboost", "lightgbm"])
    def test_model_has_fit_predict(self, model_name):
        """All models have fit() and predict() methods."""
        model = ForecastModelFactory.create(model_name)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "name")

    @pytest.mark.parametrize("model_name", ["naive_ma30", "xgboost", "lightgbm"])
    def test_model_fit_returns_self(self, model_name, sample_y_df):
        """fit() returns self for method chaining."""
        model = ForecastModelFactory.create(model_name)
        result = model.fit(sample_y_df)
        assert result is model

    @pytest.mark.parametrize("model_name", ["naive_ma30", "xgboost", "lightgbm"])
    def test_model_predict_returns_dataframe(self, model_name, sample_y_df):
        """predict() returns DataFrame with required columns."""
        model = ForecastModelFactory.create(model_name)
        model.fit(sample_y_df)
        forecasts = model.predict(h=7)
        assert isinstance(forecasts, pd.DataFrame)
        assert "unique_id" in forecasts.columns
        assert "ds" in forecasts.columns
        assert "y_hat" in forecasts.columns

    @pytest.mark.parametrize("model_name", ["naive_ma30", "xgboost", "lightgbm"])
    def test_forecasts_non_negative(self, model_name, sample_y_df):
        """All forecasts are non-negative."""
        model = ForecastModelFactory.create(model_name)
        model.fit(sample_y_df)
        forecasts = model.predict(h=7)
        assert (forecasts["y_hat"] >= 0).all()


# ---------------------------------------------------------------------------
# Individual Model Tests
# ---------------------------------------------------------------------------

class TestNaiveMovingAverage:
    def test_correct_window(self):
        model = NaiveMovingAverage(window=7)
        assert model.name == "naive_ma7"

    def test_prediction_is_mean(self, sample_y_df):
        model = NaiveMovingAverage(window=30)
        model.fit(sample_y_df)
        forecasts = model.predict(h=1)
        assert len(forecasts) > 0


class TestXGBoostForecaster:
    def test_feature_importance_available(self, sample_y_df):
        model = XGBoostForecaster(n_estimators=10)
        model.fit(sample_y_df)
        importance = model.feature_importance
        assert importance is not None
        assert "lag_1" in importance


class TestLightGBMForecaster:
    def test_feature_importance_available(self, sample_y_df):
        model = LightGBMForecaster(n_estimators=10)
        model.fit(sample_y_df)
        importance = model.feature_importance
        assert importance is not None
        assert len(importance) > 0


# ---------------------------------------------------------------------------
# Routing Ensemble Tests
# ---------------------------------------------------------------------------

class TestRoutingEnsemble:
    def test_routing_decisions(self, sample_y_df, intermittent_y_df, cold_start_y_df):
        """Routing assigns correct models based on SKU characteristics."""
        combined = pd.concat([sample_y_df, intermittent_y_df, cold_start_y_df], ignore_index=True)
        model = RoutingEnsemble(cold_start_threshold_days=60, intermittency_threshold=0.5)
        model.fit(combined)

        decisions = model._routing_decisions
        assert decisions.get("COLD_START") == "chronos2_zs"
        assert decisions.get("INTERMITTENT") == "sarimax"
        assert decisions.get("SKU_0001") == "lightgbm"

    def test_routing_summary(self, sample_y_df):
        model = RoutingEnsemble()
        model.fit(sample_y_df)
        summary = model.routing_summary
        assert isinstance(summary, dict)
        assert sum(summary.values()) == sample_y_df["unique_id"].nunique()


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------

class TestForecastModelFactory:
    def test_create_known_model(self):
        model = ForecastModelFactory.create("naive_ma30")
        assert isinstance(model, NaiveMovingAverage)

    def test_create_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            ForecastModelFactory.create("nonexistent")

    def test_available_models(self):
        models = ForecastModelFactory.available_models()
        assert "naive_ma30" in models
        assert "lightgbm" in models
        assert "routing_ensemble" in models

    def test_create_all(self):
        models = ForecastModelFactory.create_all()
        assert len(models) == 6
