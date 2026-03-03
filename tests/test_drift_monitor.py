"""Tests for drift monitoring.

Covers: data drift (KS), prediction drift (PSI), concept drift (MAPE trend).
"""

import numpy as np
import pandas as pd
import pytest

from app.forecasting.drift_monitor import DriftMonitor


@pytest.fixture
def monitor():
    return DriftMonitor()


class TestDataDrift:
    def test_no_drift_same_data(self, monitor):
        """Identical distributions → no drift."""
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"feature_a": rng.normal(0, 1, 500), "feature_b": rng.normal(10, 2, 500)})
        cur = pd.DataFrame({"feature_a": rng.normal(0, 1, 500), "feature_b": rng.normal(10, 2, 500)})
        results = monitor.check_data_drift(ref, cur)
        drifted = sum(1 for r in results if r.is_drifted)
        assert drifted == 0

    def test_drift_shifted_distribution(self, monitor):
        """Clearly shifted distribution → drift detected."""
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({"feature_a": rng.normal(0, 1, 500)})
        cur = pd.DataFrame({"feature_a": rng.normal(5, 1, 500)})  # mean shifted by 5
        results = monitor.check_data_drift(ref, cur)
        assert len(results) > 0
        assert results[0].is_drifted


class TestPredictionDrift:
    def test_no_drift_same_predictions(self, monitor):
        rng = np.random.default_rng(42)
        ref = rng.normal(100, 10, 500)
        cur = rng.normal(100, 10, 500)
        result = monitor.check_prediction_drift(ref, cur)
        assert not result.is_drifted

    def test_drift_shifted_predictions(self, monitor):
        rng = np.random.default_rng(42)
        ref = rng.normal(100, 10, 500)
        cur = rng.normal(200, 10, 500)  # clearly shifted
        result = monitor.check_prediction_drift(ref, cur)
        assert result.is_drifted


class TestConceptDrift:
    def test_no_drift_low_mape(self, monitor):
        """MAPE below threshold → no concept drift."""
        for i in range(10):
            monitor.record_mape(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), 10.0)
        result = monitor.check_concept_drift()
        assert not result.is_drifted

    def test_drift_high_mape(self, monitor):
        """MAPE above threshold for 7+ days → concept drift."""
        for i in range(10):
            monitor.record_mape(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), 25.0)
        result = monitor.check_concept_drift()
        assert result.is_drifted
        assert result.details["action"] == "AUTO_RETRAIN"

    def test_insufficient_history(self, monitor):
        """Less than 7 days → no drift (insufficient data)."""
        for i in range(3):
            monitor.record_mape(pd.Timestamp("2024-01-01") + pd.Timedelta(days=i), 25.0)
        result = monitor.check_concept_drift()
        assert not result.is_drifted


class TestMonitoringSummary:
    def test_summary_keys(self, monitor):
        summary = monitor.get_monitoring_summary()
        assert "ks_threshold" in summary
        assert "psi_threshold" in summary
        assert "mape_threshold" in summary
