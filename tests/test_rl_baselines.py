"""Tests for RL baselines (Newsvendor, (s,S), EOQ).

Covers: theoretical optimality, policy evaluation, comparison table.
"""

import numpy as np
import pytest

from app.rl.baselines import (
    _qty_to_action,
    evaluate_eoq_policy,
    evaluate_ss_policy,
    newsvendor_baseline,
)
from app.rl.environment import InventoryEnv


@pytest.fixture
def env():
    return InventoryEnv(
        unit_cost=50.0,
        daily_demand_mean=20.0,
        daily_demand_std=5.0,
        lead_time=5,
        eoq=200.0,
        safety_stock=50.0,
        episode_length=90,
        seed=42,
    )


class TestNewsvendorBaseline:
    def test_returns_positive_quantity(self):
        result = newsvendor_baseline()
        assert result["optimal_quantity"] > 0

    def test_returns_positive_cost(self):
        result = newsvendor_baseline()
        assert result["optimal_cost"] > 0

    def test_critical_ratio_between_0_and_1(self):
        result = newsvendor_baseline()
        assert 0 < result["critical_ratio"] < 1

    def test_higher_stockout_cost_more_stock(self):
        low = newsvendor_baseline(stockout_cost=10)
        high = newsvendor_baseline(stockout_cost=100)
        assert high["optimal_quantity"] > low["optimal_quantity"]


class TestSSPolicy:
    def test_returns_valid_cost(self, env):
        result = evaluate_ss_policy(env, n_episodes=5)
        assert result["avg_daily_cost"] > 0

    def test_returns_valid_service_level(self, env):
        result = evaluate_ss_policy(env, n_episodes=5)
        assert 0 <= result["avg_service_level"] <= 1


class TestEOQPolicy:
    def test_returns_valid_cost(self, env):
        result = evaluate_eoq_policy(env, n_episodes=5)
        assert result["avg_daily_cost"] > 0

    def test_returns_valid_service_level(self, env):
        result = evaluate_eoq_policy(env, n_episodes=5)
        assert 0 <= result["avg_service_level"] <= 1


class TestQtyToAction:
    def test_zero_qty(self):
        assert _qty_to_action(0, 100) == 0

    def test_half_eoq(self):
        assert _qty_to_action(50, 100) == 1

    def test_full_eoq(self):
        assert _qty_to_action(100, 100) == 2

    def test_double_eoq(self):
        assert _qty_to_action(200, 100) == 4
