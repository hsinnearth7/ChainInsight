"""Tests for configuration management.

Covers: YAML loading, config accessors, defaults.
"""

import pytest

from app.settings import (
    get_data_config,
    get_eval_config,
    get_model_config,
    get_monitoring_config,
    get_rl_config,
    get_supply_chain_config,
    load_config,
)


class TestConfigLoading:
    def test_load_config_returns_dict(self):
        load_config.cache_clear()
        config = load_config()
        assert isinstance(config, dict)

    def test_config_has_data_section(self):
        load_config.cache_clear()
        config = load_config()
        assert "data" in config

    def test_config_has_model_section(self):
        load_config.cache_clear()
        config = load_config()
        assert "model" in config

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config.cache_clear()
            load_config("/nonexistent/path.yaml")


class TestDataConfig:
    def test_n_skus(self):
        config = get_data_config()
        assert config.get("n_skus") == 200

    def test_n_warehouses(self):
        config = get_data_config()
        assert config.get("n_warehouses") == 3

    def test_seed(self):
        config = get_data_config()
        assert config.get("seed") == 42


class TestModelConfig:
    def test_default_model(self):
        config = get_model_config()
        assert config.get("default") == "lightgbm"

    def test_lightgbm_config(self):
        config = get_model_config("lightgbm")
        assert "n_estimators" in config
        assert "num_leaves" in config

    def test_routing_threshold(self):
        config = get_model_config("routing")
        assert config.get("cold_start_threshold_days") == 60


class TestEvalConfig:
    def test_cv_windows(self):
        config = get_eval_config()
        assert config.get("cv_windows") == 12

    def test_significance_alpha(self):
        config = get_eval_config()
        assert config.get("significance_alpha") == 0.05


class TestMonitoringConfig:
    def test_drift_thresholds(self):
        config = get_monitoring_config()
        assert config.get("drift_threshold_ks") == 0.05
        assert config.get("drift_threshold_psi") == 0.1

    def test_retrain_trigger(self):
        config = get_monitoring_config()
        assert config.get("retrain_trigger_days") == 7
        assert config.get("mape_alert_threshold") == 0.20


class TestRLConfig:
    def test_curriculum_phases(self):
        config = get_rl_config()
        curriculum = config.get("curriculum", [])
        assert len(curriculum) == 3
        assert curriculum[0]["n_products"] == 1
        assert curriculum[2]["stochastic_lead_time"] is True


class TestSupplyChainConfig:
    def test_ordering_cost(self):
        config = get_supply_chain_config()
        assert config.get("ordering_cost") == 50
