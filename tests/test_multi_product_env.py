"""Tests for multi-product inventory environment.

Covers: spaces, reset, step, stochastic lead time, episode completion.
"""

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from app.rl.multi_product_env import MultiProductInventoryEnv


@pytest.fixture
def env():
    return MultiProductInventoryEnv(n_products=3, seed=42)


@pytest.fixture
def stochastic_env():
    return MultiProductInventoryEnv(n_products=3, stochastic_lead_time=True, seed=42)


class TestSpaces:
    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (15,)  # 3 products × 5 dims

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (3,)

    def test_action_space_bounds(self, env):
        assert (env.action_space.low == 0).all()
        assert (env.action_space.high == 1).all()


class TestReset:
    def test_reset_returns_correct_shape(self, env):
        obs, info = env.reset()
        assert obs.shape == (15,)

    def test_reset_obs_in_bounds(self, env):
        obs, _ = env.reset()
        assert (obs >= 0).all()
        assert (obs <= 1).all()

    def test_reset_info_keys(self, env):
        _, info = env.reset()
        assert "day" in info
        assert "stocks" in info
        assert "mean_service_level" in info


class TestStep:
    def test_step_returns_tuple(self, env):
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5  # obs, reward, terminated, truncated, info

    def test_step_obs_shape(self, env):
        env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (15,)

    def test_reward_is_negative(self, env):
        """Reward should be negative (sum of costs)."""
        env.reset()
        _, reward, _, _, _ = env.step(np.array([0.5, 0.5, 0.5]))
        assert reward <= 0

    def test_episode_terminates(self, env):
        env.reset()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            steps += 1
        assert steps == env.episode_length


class TestStochasticLeadTime:
    def test_stochastic_enabled(self, stochastic_env):
        assert stochastic_env.stochastic_lead_time is True

    def test_still_terminates(self, stochastic_env):
        stochastic_env.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = stochastic_env.step(stochastic_env.action_space.sample())
            done = terminated or truncated


class TestEpisodeSummary:
    def test_summary_keys(self, env):
        env.reset()
        for _ in range(env.episode_length):
            env.step(env.action_space.sample())
        summary = env.get_episode_summary()
        assert "total_cost" in summary
        assert "daily_cost" in summary
        assert "mean_service_level" in summary


class TestSeedReproducibility:
    def test_same_seed_same_trajectory(self):
        env1 = MultiProductInventoryEnv(n_products=2, seed=42)
        env2 = MultiProductInventoryEnv(n_products=2, seed=42)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

        action = np.array([0.5, 0.3])
        obs1, r1, _, _, _ = env1.step(action)
        obs2, r2, _, _, _ = env2.step(action)
        np.testing.assert_array_almost_equal(obs1, obs2)
        assert r1 == r2
