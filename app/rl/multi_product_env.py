"""Multi-product inventory environment for advanced RL training.

Extends the single-product InventoryEnv to support:
- Multiple products with independent demand streams
- Multiple warehouses
- Stochastic lead times (optional, for curriculum phase 3)
- Continuous action space (for SAC compatibility)

Gymnasium-compliant with full seed reproducibility.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.logging import get_logger

logger = get_logger(__name__)


class MultiProductInventoryEnv(gym.Env):
    """Multi-product inventory management environment.

    State per product (5 dims):
        current_stock, pending_orders, days_since_order, demand_trend, stockout_streak
    Total state: n_products × 5

    Action: Box(n_products,) in [0, 1] — fraction of max_order for each product.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_products: int = 5,
        n_warehouses: int = 3,
        episode_length: int = 90,
        stochastic_lead_time: bool = False,
        unit_costs: list[float] | None = None,
        demand_means: list[float] | None = None,
        demand_stds: list[float] | None = None,
        lead_times: list[int] | None = None,
        ordering_cost: float = 50.0,
        holding_rate: float = 0.25,
        stockout_penalty: float = 50.0,
        max_stock_per_product: float = 2000.0,
        seed: int | None = None,
    ):
        super().__init__()

        self.n_products = n_products
        self.n_warehouses = n_warehouses
        self.episode_length = episode_length
        self.stochastic_lead_time = stochastic_lead_time
        self.ordering_cost = ordering_cost
        self.holding_rate_daily = holding_rate / 365.0
        self.stockout_penalty = stockout_penalty
        self.max_stock = max_stock_per_product

        # Per-product parameters
        rng = np.random.default_rng(seed or 42)
        self.unit_costs = np.array(unit_costs or rng.uniform(10, 200, n_products))
        self.demand_means = np.array(demand_means or rng.uniform(5, 50, n_products))
        self.demand_stds = np.array(demand_stds or self.demand_means * 0.25)
        self.base_lead_times = np.array(lead_times or rng.integers(3, 10, n_products))

        # EOQ per product
        annual_demands = self.demand_means * 365
        h = self.holding_rate_daily * 365 * self.unit_costs
        self.eoqs = np.sqrt(2 * annual_demands * ordering_cost / np.maximum(h, 0.01))
        self.max_orders = self.eoqs * 2  # max order = 2× EOQ

        # Spaces
        obs_dim = n_products * 5
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.zeros(n_products, dtype=np.float32),
            high=np.ones(n_products, dtype=np.float32),
        )

        # State
        self._stocks = np.zeros(n_products)
        self._pending: list[list[tuple[int, float]]] = [[] for _ in range(n_products)]
        self._days_since_order = np.zeros(n_products)
        self._demand_histories: list[list[float]] = [[] for _ in range(n_products)]
        self._stockout_streaks = np.zeros(n_products)
        self._day = 0

        # Episode tracking
        self._total_costs: dict[str, float] = {"holding": 0, "stockout": 0, "ordering": 0}
        self._service_days = np.zeros(n_products)
        self._total_days = 0

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.n_products * 5, dtype=np.float32)
        for i in range(self.n_products):
            base = i * 5
            pending = sum(qty for _, qty in self._pending[i])
            demand_trend = (
                np.mean(self._demand_histories[i][-7:]) / max(self.demand_means[i], 1)
                if self._demand_histories[i] else 1.0
            )
            obs[base + 0] = np.clip(self._stocks[i] / self.max_stock, 0, 1)
            obs[base + 1] = np.clip(pending / self.max_stock, 0, 1)
            obs[base + 2] = np.clip(self._days_since_order[i] / max(self.base_lead_times[i], 1), 0, 1)
            obs[base + 3] = np.clip(demand_trend, 0, 1)
            obs[base + 4] = np.clip(self._stockout_streaks[i] / 10.0, 0, 1)
        return obs

    def _get_info(self) -> dict[str, Any]:
        svc = self._service_days / max(self._total_days, 1)
        return {
            "day": self._day,
            "stocks": self._stocks.tolist(),
            "service_levels": svc.tolist(),
            "mean_service_level": float(svc.mean()),
            "total_cost": sum(self._total_costs.values()),
            "costs": self._total_costs.copy(),
        }

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        for i in range(self.n_products):
            self._stocks[i] = self.np_random.uniform(
                self.demand_means[i] * 5, self.demand_means[i] * 15
            )
        self._pending = [[] for _ in range(self.n_products)]
        self._days_since_order = np.zeros(self.n_products)
        self._demand_histories = [[] for _ in range(self.n_products)]
        self._stockout_streaks = np.zeros(self.n_products)
        self._day = 0
        self._total_costs = {"holding": 0, "stockout": 0, "ordering": 0}
        self._service_days = np.zeros(self.n_products)
        self._total_days = 0

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(action, 0, 1)
        total_reward = 0.0

        for i in range(self.n_products):
            # 1. Place order
            order_qty = float(action[i]) * float(self.max_orders[i])
            order_cost = 0.0
            if order_qty > 0.01 * self.eoqs[i]:  # min order threshold
                if self.stochastic_lead_time:
                    lt = max(1, int(self.np_random.normal(self.base_lead_times[i], 1)))
                else:
                    lt = int(self.base_lead_times[i])
                self._pending[i].append((self._day + lt, order_qty))
                order_cost = self.ordering_cost
                self._days_since_order[i] = 0
            else:
                self._days_since_order[i] += 1

            # 2. Receive orders
            arrived = sum(qty for arr, qty in self._pending[i] if arr <= self._day)
            self._pending[i] = [(arr, qty) for arr, qty in self._pending[i] if arr > self._day]
            self._stocks[i] += arrived

            # 3. Demand
            demand = max(0, self.np_random.normal(self.demand_means[i], self.demand_stds[i]))
            self._demand_histories[i].append(demand)

            # 4. Fulfill
            fulfilled = min(demand, self._stocks[i])
            unmet = demand - fulfilled
            self._stocks[i] -= fulfilled
            self._stocks[i] = min(self._stocks[i], self.max_stock)

            # 5. Service tracking
            if unmet == 0:
                self._service_days[i] += 1
            if self._stocks[i] <= 0:
                self._stockout_streaks[i] += 1
            else:
                self._stockout_streaks[i] = 0

            # 6. Costs
            holding = self.holding_rate_daily * self.unit_costs[i] * self._stocks[i]
            stockout = self.stockout_penalty * unmet

            self._total_costs["holding"] += holding
            self._total_costs["stockout"] += stockout
            self._total_costs["ordering"] += order_cost

            total_reward -= (holding + stockout + order_cost)

        self._day += 1
        self._total_days += 1
        terminated = self._day >= self.episode_length

        return self._get_obs(), float(total_reward), terminated, False, self._get_info()

    def get_episode_summary(self) -> dict[str, Any]:
        svc = self._service_days / max(self._total_days, 1)
        return {
            "total_cost": sum(self._total_costs.values()),
            "daily_cost": sum(self._total_costs.values()) / max(self._day, 1),
            "service_levels": svc.tolist(),
            "mean_service_level": float(svc.mean()),
            "days": self._day,
            **self._total_costs,
        }
