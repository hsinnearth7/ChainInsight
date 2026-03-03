"""Classical inventory policy baselines for RL benchmarking.

Provides theoretical and heuristic baselines:
- Newsvendor (optimal single-period, Stockpyl): theoretical lower bound
- (s, S) policy: reorder-point + order-up-to-level
- EOQ (Economic Order Quantity): deterministic demand baseline

PPO must beat these to demonstrate RL value-add.
Target: PPO+curriculum within +2% of Newsvendor theoretical optimum.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.logging import get_logger
from app.rl.environment import InventoryEnv

logger = get_logger(__name__)


def newsvendor_baseline(
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0,
    demand_mean: float = 100.0,
    demand_std: float = 20.0,
    lead_time: int = 7,
) -> dict[str, Any]:
    """Compute Newsvendor theoretical optimal using Stockpyl.

    The Newsvendor solution gives the optimal order quantity for a single-period
    problem with uncertain demand — this is the theoretical lower bound on cost.

    Args:
        holding_cost: Holding cost per unit per day.
        stockout_cost: Stockout penalty per unit.
        demand_mean: Mean daily demand.
        demand_std: Standard deviation of daily demand.
        lead_time: Lead time in days.

    Returns:
        Dict with optimal quantity, cost, and critical ratio.
    """
    try:
        from stockpyl.newsvendor import newsvendor_normal

        Q_star, cost_star = newsvendor_normal(
            holding_cost=holding_cost,
            stockout_cost=stockout_cost,
            demand_mean=demand_mean,
            demand_sd=demand_std,
        )

        critical_ratio = stockout_cost / (stockout_cost + holding_cost)

        return {
            "method": "newsvendor_stockpyl",
            "optimal_quantity": float(Q_star),
            "optimal_cost": float(cost_star),
            "critical_ratio": round(critical_ratio, 4),
            "daily_cost_estimate": float(cost_star),
            "source": "Stockpyl newsvendor_normal",
        }

    except ImportError:
        logger.warning("stockpyl_not_installed", fallback="manual_newsvendor")
        # Manual newsvendor calculation
        from scipy.stats import norm

        critical_ratio = stockout_cost / (stockout_cost + holding_cost)
        z_star = norm.ppf(critical_ratio)
        Q_star = demand_mean + z_star * demand_std
        # Expected cost (approximation)
        cost_star = holding_cost * (Q_star - demand_mean) + stockout_cost * demand_std * norm.pdf(z_star)

        return {
            "method": "newsvendor_manual",
            "optimal_quantity": round(float(Q_star), 2),
            "optimal_cost": round(float(cost_star), 2),
            "critical_ratio": round(critical_ratio, 4),
            "daily_cost_estimate": round(float(cost_star), 2),
            "source": "Manual calculation (Stockpyl not installed)",
        }


def evaluate_ss_policy(
    env: InventoryEnv,
    reorder_point: float | None = None,
    order_up_to: float | None = None,
    n_episodes: int = 100,
) -> dict[str, Any]:
    """Evaluate (s, S) policy on the environment.

    (s, S) policy: when stock drops below s, order up to S.

    Args:
        env: InventoryEnv instance.
        reorder_point: Reorder point s. Defaults to safety_stock.
        order_up_to: Order-up-to level S. Defaults to 2 × safety_stock + EOQ.
        n_episodes: Number of evaluation episodes.

    Returns:
        Dict with avg cost, service level, etc.
    """
    s = reorder_point if reorder_point is not None else env.safety_stock
    S = order_up_to if order_up_to is not None else (2 * env.safety_stock + env.eoq)

    total_costs = []
    service_levels = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False

        while not done:
            current_stock = obs[0] * env.max_stock
            pending = obs[1] * env.max_stock

            # (s, S) decision
            if current_stock + pending < s:
                # Order up to S
                order_qty = S - current_stock - pending
                # Map to nearest discrete action
                action = _qty_to_action(order_qty, env.eoq)
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        summary = env.get_episode_summary()
        total_costs.append(summary["total_cost"])
        service_levels.append(summary["service_level"])

    return {
        "method": "(s,S) policy",
        "s": float(s),
        "S": float(S),
        "avg_daily_cost": round(float(np.mean(total_costs)) / env.episode_length, 2),
        "avg_service_level": round(float(np.mean(service_levels)), 4),
        "std_daily_cost": round(float(np.std(total_costs)) / env.episode_length, 2),
        "n_episodes": n_episodes,
    }


def evaluate_eoq_policy(
    env: InventoryEnv,
    n_episodes: int = 100,
) -> dict[str, Any]:
    """Evaluate fixed EOQ policy on the environment.

    EOQ policy: order exactly EOQ whenever stock drops below reorder point.

    Returns:
        Dict with avg cost, service level, etc.
    """
    total_costs = []
    service_levels = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=42 + ep)
        done = False

        while not done:
            current_stock = obs[0] * env.max_stock
            reorder_point = env.daily_demand_mean * env.lead_time

            if current_stock < reorder_point:
                action = 2  # order 1.0 × EOQ
            else:
                action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        summary = env.get_episode_summary()
        total_costs.append(summary["total_cost"])
        service_levels.append(summary["service_level"])

    return {
        "method": "EOQ",
        "eoq": float(env.eoq),
        "avg_daily_cost": round(float(np.mean(total_costs)) / env.episode_length, 2),
        "avg_service_level": round(float(np.mean(service_levels)), 4),
        "std_daily_cost": round(float(np.std(total_costs)) / env.episode_length, 2),
        "n_episodes": n_episodes,
    }


def _qty_to_action(qty: float, eoq: float) -> int:
    """Map continuous order quantity to nearest discrete action."""
    if qty <= 0:
        return 0
    ratio = qty / eoq
    if ratio < 0.25:
        return 0
    elif ratio < 0.75:
        return 1  # 0.5 × EOQ
    elif ratio < 1.25:
        return 2  # 1.0 × EOQ
    elif ratio < 1.75:
        return 3  # 1.5 × EOQ
    else:
        return 4  # 2.0 × EOQ


def build_rl_comparison_table(
    env: InventoryEnv,
    rl_results: dict[str, dict],
) -> list[dict[str, Any]]:
    """Build complete comparison table: classical baselines + RL agents.

    Args:
        env: InventoryEnv for baseline evaluation.
        rl_results: Dict from RLTrainer.get_comparison_data().

    Returns:
        List of dicts for the comparison table.
    """
    table = []

    # Classical baselines
    ss = evaluate_ss_policy(env)
    table.append({
        "algorithm": "(s,S) policy",
        "avg_daily_cost": ss["avg_daily_cost"],
        "service_level": ss["avg_service_level"],
        "training_time": "N/A",
        "category": "classical",
    })

    eoq = evaluate_eoq_policy(env)
    table.append({
        "algorithm": "EOQ",
        "avg_daily_cost": eoq["avg_daily_cost"],
        "service_level": eoq["avg_service_level"],
        "training_time": "N/A",
        "category": "classical",
    })

    nv = newsvendor_baseline(
        holding_cost=env.holding_rate_daily * env.unit_cost,
        stockout_cost=env.stockout_penalty,
        demand_mean=env.daily_demand_mean,
        demand_std=env.daily_demand_std,
    )
    table.append({
        "algorithm": "Newsvendor (theory)",
        "avg_daily_cost": nv["daily_cost_estimate"],
        "service_level": nv["critical_ratio"],
        "training_time": "N/A (theoretical)",
        "category": "theoretical",
    })

    # RL agents
    for name, result in rl_results.items():
        daily_cost = abs(result["final_reward"]) / env.episode_length
        table.append({
            "algorithm": name,
            "avg_daily_cost": round(daily_cost, 2),
            "service_level": round(result.get("final_service_level", 0), 4),
            "training_time": "varies",
            "category": "rl",
        })

    return table
