"""Distributed emergence experiment: the interaction-dependence test.

Runs the full ablation battery on the partially-observable causal-chain task and
computes transparent distributed-emergence metrics. Also runs the individual
solvability test (can any single zone solve the whole task alone?) and a
complete-information single-cell baseline.

Reproducible: deterministic given (seed). Writes results/distributed_seedN.json.

Usage:
    python -m aha.experiments.distributed --seed 0 --episodes 300
    python -m aha.experiments.distributed --seed 0 --episodes 300 --transfer
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any

from aha.cells.config import LPCCConfig
from aha.colony.distributed_colony import DistributedColony
from aha.communication.routed import RouterConfig
from aha.environments.distributed_chain import ZONES, DistributedChainEnv

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")


# --------------------------------------------------------------------------- #
def _make_env(seed: int, **kw) -> DistributedChainEnv:
    return DistributedChainEnv(seed=seed, delay=1, episode_len=40, **kw)


def _zone_actions(env: DistributedChainEnv) -> dict[str, str]:
    return {z: env.zone_action(z) for z in ZONES}


def run_distributed(
    seed: int,
    episodes: int,
    router_cfg: RouterConfig,
    config: LPCCConfig | None = None,
    env_kw: dict | None = None,
    restrict_zones: list[str] | None = None,
) -> dict[str, Any]:
    """Run the distributed colony on the chain task; return metrics + colony."""
    env = _make_env(seed, **(env_kw or {}))
    zones = restrict_zones or list(ZONES)
    colony = DistributedColony(
        zones=zones,
        zone_actions=_zone_actions(env),
        available_actions=env.actions,
        config=config or LPCCConfig(),
        router_config=router_cfg,
        seed=seed,
    )
    total_reward = 0.0
    solved_episodes = 0
    per_ep_reward: list[float] = []
    for ep in range(episodes):
        env.reset()
        done = False
        ep_r = 0.0
        while not done:
            action = colony.decide(env.zone_view)
            step = env.step(action)
            colony.deliver_outcome(lambda z: env.zone_view(z), step.reward, action)
            ep_r += step.reward
            done = step.done
        total_reward += ep_r
        per_ep_reward.append(ep_r)
        if ep_r > 0.5:
            solved_episodes += 1
        env.next_episode()
    return {
        "colony": colony,
        "total_reward": total_reward,
        "solved_episodes": solved_episodes,
        "solve_rate": solved_episodes / episodes,
        "final_100_solve_rate": sum(1 for r in per_ep_reward[-100:] if r > 0.5) / max(1, len(per_ep_reward[-100:])),
        "sent": colony.router.sent_count,
        "delivered": colony.router.delivered_count,
    }


# --------------------------------------------------------------------------- #
# Complete-information single-cell baseline (Outcome test G).
# --------------------------------------------------------------------------- #
def run_complete_info_single(seed: int, episodes: int) -> dict[str, Any]:
    """A single agent that DOES see the full state and can act on any zone.

    This is the "one cell given the complete causal chain" control. It uses the
    same LPCC substrate but a single colony over the full-state observation and
    all zone actions, so it can in principle learn the whole chain alone. If the
    distributed colony matches/exceeds this only via communication, that supports
    the emergence claim; if this single agent dominates, the task is solvable
    centrally.
    """
    from aha.colony.colony import LPCCColony
    from aha.experiments.runner import run_episodes

    env = _make_env(seed)

    class _FullEnvAdapter:
        actions = env.actions

        def __init__(self, e):
            self.e = e

        def reset(self):
            return self.e.reset()

        def step(self, action):
            r = self.e.step(action)
            return r

    adapter = _FullEnvAdapter(_make_env(seed))
    colony = LPCCColony(adapter.actions, config=LPCCConfig(), seed=seed)
    run = run_episodes(colony, adapter, episodes)
    return {"total_reward": run.total_reward}


# --------------------------------------------------------------------------- #
# Individual solvability: can ONE zone alone (no comm) solve the whole task?
# --------------------------------------------------------------------------- #
def run_individual_solvability(seed: int, episodes: int) -> dict[str, Any]:
    results = {}
    for z in ZONES:
        r = run_distributed(
            seed, episodes,
            router_cfg=RouterConfig(enabled=False),
            restrict_zones=[z],
        )
        results[f"zone_{z}_only"] = {
            "total_reward": r["total_reward"],
            "solve_rate": round(r["solve_rate"], 4),
        }
    return results


# --------------------------------------------------------------------------- #
# Transparent distributed-emergence metrics.
# --------------------------------------------------------------------------- #
def compute_metrics(variants: dict[str, dict]) -> dict[str, Any]:
    full = variants["A_full"]["total_reward"]
    no_comm = variants["B_no_comm"]["total_reward"]
    rand = variants["C_random_comm"]["total_reward"]
    delayed = variants["D_delayed_comm"]["total_reward"]
    band = variants["E_limited_bandwidth"]["total_reward"]
    crit = variants["F_critical_pop_removed"]["total_reward"]

    def rel_drop(x):
        return max(0.0, (full - x) / full) if full > 0 else 0.0

    interaction_dependence = rel_drop(no_comm)
    communication_necessity = rel_drop(no_comm)
    random_comm_gap = rel_drop(rand)
    critical_information_loss = rel_drop(crit)

    return {
        "interaction_dependence": round(interaction_dependence, 4),
        "communication_necessity": round(communication_necessity, 4),
        "random_comm_degradation": round(random_comm_gap, 4),
        "delayed_comm_degradation": round(rel_drop(delayed), 4),
        "bandwidth_degradation": round(rel_drop(band), 4),
        "critical_information_loss": round(critical_information_loss, 4),
    }


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Distributed LPCC emergence experiment")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--transfer", action="store_true")
    args = p.parse_args(argv)

    seed, eps = args.seed, args.episodes

    variants: dict[str, dict] = {}

    def record(name, r):
        variants[name] = {
            "total_reward": r["total_reward"],
            "solve_rate": round(r["solve_rate"], 4),
            "final_100_solve_rate": round(r["final_100_solve_rate"], 4),
            "sent": r["sent"],
            "delivered": r["delivered"],
        }

    # A. Full system.
    full = run_distributed(seed, eps, RouterConfig(enabled=True, delay=1))
    record("A_full", full)
    # B. Communication disabled.
    record("B_no_comm", run_distributed(seed, eps, RouterConfig(enabled=False)))
    # C. Random communication.
    record("C_random_comm", run_distributed(seed, eps, RouterConfig(enabled=True, randomize=True, delay=1)))
    # D. Substantial delay.
    record("D_delayed_comm", run_distributed(seed, eps, RouterConfig(enabled=True, delay=6)))
    # E. Limited bandwidth.
    record("E_limited_bandwidth", run_distributed(seed, eps, RouterConfig(enabled=True, delay=1, bandwidth=1)))
    # F. Critical population removed (drop zone B, the middle of the chain).
    record("F_critical_pop_removed", run_distributed(seed, eps, RouterConfig(enabled=True, delay=1), restrict_zones=["A", "C"]))
    # G. Complete-information single-cell baseline.
    variants["G_complete_info_single"] = run_complete_info_single(seed, eps)
    # H. Full comm, no energy cost.
    record("H_full_no_energy_cost", run_distributed(seed, eps, RouterConfig(enabled=True, delay=1, energy_cost=0.0), config=LPCCConfig(enable_energy=False)))

    # Individual solvability.
    individual = run_individual_solvability(seed, eps)

    metrics = compute_metrics(variants)

    # Emergence verdict (operational, all conditions must hold).
    ind_max = max(v["solve_rate"] for v in individual.values())
    verdict = {
        "no_single_zone_solves": ind_max < 0.2,
        "full_colony_solves": variants["A_full"]["solve_rate"] > 0.3,
        "comm_necessary": metrics["communication_necessity"] > 0.2,
        "random_comm_worse": metrics["random_comm_degradation"] > 0.2,
        "critical_info_matters": metrics["critical_information_loss"] > 0.2,
    }
    verdict["interaction_dependent_emergence"] = all(verdict.values())

    payload = {
        "metadata": {
            "experiment": "distributed_emergence",
            "seed": seed,
            "episodes": eps,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "variants": variants,
        "individual_solvability": individual,
        "individual_max_solve_rate": round(ind_max, 4),
        "metrics": metrics,
        "verdict": verdict,
        "information_flow_sample": full["colony"].information_flow_edges()[:40],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"distributed_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print("\n=== DISTRIBUTED EMERGENCE ===")
    for name, v in variants.items():
        print(f"  {name:28s}: reward={v['total_reward']:.1f} solve={v.get('solve_rate','-')}")
    print("  --- individual solvability (no comm, one zone) ---")
    for name, v in individual.items():
        print(f"  {name:28s}: reward={v['total_reward']:.1f} solve={v['solve_rate']}")
    print("  --- metrics ---")
    for k, v in metrics.items():
        print(f"  {k:28s}: {v}")
    print("  --- verdict ---")
    for k, v in verdict.items():
        print(f"  {k:28s}: {v}")
    print(f"  saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
