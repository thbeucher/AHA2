"""Unified reproducible experiment entry point.

Usage:
    python -m aha.experiments.run discovery   --seed 0 --episodes 200
    python -m aha.experiments.run reversal    --seed 0 --episodes 200
    python -m aha.experiments.run transfer    --seed 0 --episodes 200
    python -m aha.experiments.run emergence    --seed 0 --episodes 400
    python -m aha.experiments.run ablations    --seed 0 --episodes 200

Every run writes machine-readable JSON + a human-readable summary to results/.
All randomness is seeded; results are deterministic given (experiment, seed).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from aha.cells.config import LPCCConfig
from aha.colony.colony import LPCCColony
from aha.experiments import baselines
from aha.experiments.runner import run_episodes
from aha.metrics import lpcc_metrics as M

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "results")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _metadata(experiment: str, seed: int, episodes: int, extra: dict | None = None) -> dict:
    md = {
        "experiment": experiment,
        "seed": seed,
        "episodes": episodes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        md.update(extra)
    return md


def _save(experiment: str, seed: int, payload: dict) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{experiment}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def _print_summary(title: str, rows: dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    for k, v in rows.items():
        print(f"  {k:32s}: {v}")


def _rule_tokens_switch() -> set:
    # Full-rule tokens for the HiddenSwitch/Transfer switch environments.
    return {("now:switch", "present"), ("now:door", "open")}


# --------------------------------------------------------------------------- #
# Discovery (Environments 1-2): compare baselines vs AHA variants
# --------------------------------------------------------------------------- #
def exp_discovery(seed: int, episodes: int) -> dict:
    from aha.environments.hidden_rules import HiddenSwitchEnv

    def make_env():
        return HiddenSwitchEnv(seed=seed, delay=2, episode_len=20)

    results: dict[str, Any] = {}

    # Baselines.
    env = make_env()
    rnd = baselines.RandomBaseline(env.actions, seed=seed)
    results["random"] = {"total_reward": baselines.run_baseline(rnd, env, episodes)["total_reward"]}

    env = make_env()
    fixed = baselines.FixedRuleBaseline("switch", "present", "investigate")
    results["fixed_rule"] = {"total_reward": baselines.run_baseline(fixed, env, episodes)["total_reward"]}

    # AHA variants (ablation-style toggles).
    variants = {
        "aha_no_comm": {"enable_communication": False},
        "aha_with_comm": {"enable_communication": True},
        "aha_no_structural": {"enable_structural_plasticity": False},
        "aha_full": {},
    }
    for name, overrides in variants.items():
        env = make_env()
        cfg = LPCCConfig(**overrides)
        colony = LPCCColony(env.actions, config=cfg, seed=seed)
        run = run_episodes(colony, env, episodes)
        summary = M.summarize(colony, run.per_step_action)
        summary["total_reward"] = run.total_reward
        results[name] = summary

    _print_summary("DISCOVERY", {k: v.get("total_reward", v) for k, v in results.items()})
    return results


# --------------------------------------------------------------------------- #
# Rule reversal (falsification test)
# --------------------------------------------------------------------------- #
def exp_reversal(seed: int, episodes: int) -> dict:
    from aha.environments.rule_reversal import RuleReversalEnv

    reversal_ep = episodes // 2
    env = RuleReversalEnv(
        seed=seed, delay=2, episode_len=20, reversal_episode=reversal_ep,
        action_a="investigate", action_b="push",
    )
    cfg = LPCCConfig()
    colony = LPCCColony(env.actions, config=cfg, seed=seed)

    # Track per-episode reward + correct-action accuracy around the reversal.
    ep_rewards: list[float] = []
    ep_correct: list[float] = []

    def on_end(ep, e, c):
        e.next_episode()

    # We run episode-by-episode to capture the correct action and advance phase.
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        correct_hits = 0
        steps = 0
        while not done:
            correct = env.current_correct_action
            decision = colony.decide(obs)
            action = decision.chosen_action
            step = env.step(action)
            colony.deliver_outcome(step.observation, step.reward, action)
            # Score behavioral correctness only when antecedent present.
            if obs.get("switch") == "present":
                correct_hits += int(action == correct)
                steps += 1
            total += step.reward
            obs = step.observation
            done = step.done
        ep_rewards.append(total)
        ep_correct.append(correct_hits / steps if steps else 0.0)
        env.next_episode()

    # Adaptation time: episodes after reversal until correct-rate recovers to
    # 60% of the pre-reversal peak (measured on antecedent-present steps).
    pre = ep_correct[:reversal_ep]
    post = ep_correct[reversal_ep:]
    pre_peak = max(pre) if pre else 0.0
    target = 0.6 * pre_peak
    adaptation_time = None
    for i, v in enumerate(post):
        if v >= target and target > 0:
            adaptation_time = i
            break

    payload = {
        "reversal_episode": reversal_ep,
        "pre_reversal_correct_peak": round(pre_peak, 3),
        "post_reversal_recovery_target": round(target, 3),
        "adaptation_time_episodes": adaptation_time,
        "mean_correct_preA": round(sum(pre) / len(pre), 3) if pre else 0.0,
        "mean_correct_postB": round(sum(post) / len(post), 3) if post else 0.0,
        "ep_correct": [round(x, 3) for x in ep_correct],
        "ep_rewards": ep_rewards,
    }
    _print_summary(
        "RULE REVERSAL (falsification)",
        {
            "reversal_episode": reversal_ep,
            "mean_correct_phaseA": payload["mean_correct_preA"],
            "mean_correct_phaseB": payload["mean_correct_postB"],
            "adaptation_time_episodes": adaptation_time,
        },
    )
    return payload


# --------------------------------------------------------------------------- #
# Transfer (relational structure, changed surface symbols)
# --------------------------------------------------------------------------- #
def exp_transfer(seed: int, episodes: int) -> dict:
    from aha.environments.transfer import make_train_env, make_transfer_env

    # Train on world A.
    train_env = make_train_env(seed=seed, delay=2, episode_len=20)
    cfg = LPCCConfig()
    colony = LPCCColony(train_env.actions, config=cfg, seed=seed)
    train_run = run_episodes(colony, train_env, episodes)

    # Freeze structure by disabling exploration & test on world B.
    test_env = make_transfer_env(seed=seed + 1, delay=2, episode_len=20)
    # Colony keeps its cells; but action vocab differs. We evaluate transfer of
    # the STRUCTURE: does prior experience help vs a fresh colony on world B?
    colony.available_actions = test_env.actions
    transfer_run = run_episodes(colony, test_env, episodes // 2)

    fresh = LPCCColony(test_env.actions, config=LPCCConfig(), seed=seed)
    fresh_env = make_transfer_env(seed=seed + 1, delay=2, episode_len=20)
    fresh_run = run_episodes(fresh, fresh_env, episodes // 2)

    payload = {
        "train_reward": train_run.total_reward,
        "transfer_reward_pretrained": transfer_run.total_reward,
        "transfer_reward_fresh": fresh_run.total_reward,
        "transfer_advantage": transfer_run.total_reward - fresh_run.total_reward,
        "note": (
            "Surface symbols AND action names change between worlds. The symbolic "
            "detector keys on exact tokens, so positive transfer would indicate "
            "structural reuse; ~0 or negative indicates the current detector cannot "
            "transfer relational structure (documented limitation)."
        ),
    }
    _print_summary(
        "TRANSFER",
        {
            "train_reward": payload["train_reward"],
            "transfer_pretrained": payload["transfer_reward_pretrained"],
            "transfer_fresh": payload["transfer_reward_fresh"],
            "transfer_advantage": payload["transfer_advantage"],
        },
    )
    return payload


# --------------------------------------------------------------------------- #
# Emergence (multi-step, requires coordination)
# --------------------------------------------------------------------------- #
def exp_emergence(seed: int, episodes: int) -> dict:
    from aha.environments.emergence import TwoStageEnv

    def make_env():
        return TwoStageEnv(seed=seed, delay=1, episode_len=30)

    full_rule_tokens = {
        ("now:key", "present"),
        ("now:holding", "yes"),
        ("now:door", "open"),
    }

    def run_variant(overrides: dict) -> tuple[LPCCColony, Any]:
        env = make_env()
        cfg = LPCCConfig(**overrides)
        colony = LPCCColony(env.actions, config=cfg, seed=seed)
        run = run_episodes(colony, env, episodes)
        return colony, run

    colony_full, run_full = run_variant({})
    colony_nocomm, run_nocomm = run_variant({"enable_communication": False})

    # Interaction dependence: relative reward drop when communication is removed.
    rf = run_full.total_reward
    rn = run_nocomm.total_reward
    interaction_dependence = max(0.0, min(1.0, (rf - rn) / rf)) if rf > 0 else 0.0

    es = M.emergence_score(colony_full, full_rule_tokens, interaction_dependence)

    # Check: no single cell encodes the full two-stage rule.
    max_cov = 1.0 - M.centralized_encoding_absence(colony_full, full_rule_tokens)

    payload = {
        "reward_full": rf,
        "reward_no_comm": rn,
        "interaction_dependence": round(interaction_dependence, 4),
        "max_single_cell_rule_coverage": round(max_cov, 4),
        "emergence_score": es.as_dict(),
        "full_summary": M.summarize(colony_full, run_full.per_step_action),
        "nocomm_summary": M.summarize(colony_nocomm, run_nocomm.per_step_action),
        "emergence_criteria": {
            "no_single_cell_complete": max_cov < 1.0,
            "depends_on_interaction": interaction_dependence > 0.05,
            "distributed": es.distributed_contribution > 0.1,
        },
    }
    _print_summary(
        "EMERGENCE (multi-step coordination)",
        {
            "reward_full": rf,
            "reward_no_comm": rn,
            "interaction_dependence": payload["interaction_dependence"],
            "max_single_cell_rule_coverage": payload["max_single_cell_rule_coverage"],
            "emergence_combined": es.combined,
        },
    )
    return payload


# --------------------------------------------------------------------------- #
# Ablations (required set A-H)
# --------------------------------------------------------------------------- #
def exp_ablations(seed: int, episodes: int) -> dict:
    from aha.environments.hidden_rules import HiddenSwitchEnv

    def make_env():
        return HiddenSwitchEnv(seed=seed, delay=2, episode_len=20)

    ablations = {
        "H_full": {},
        "A_no_communication": {"enable_communication": False},
        "B_no_delayed_credit": {"enable_delayed_credit": False},
        "C_no_causal": {"enable_causal": False},
        "D_no_structural": {"enable_structural_plasticity": False},
        "E_no_energy": {"enable_energy": False},
        "F_no_novelty": {"enable_novelty": False},
        "G_no_homeostasis": {"enable_homeostasis": False},
    }
    results: dict[str, Any] = {}
    for name, overrides in ablations.items():
        env = make_env()
        cfg = LPCCConfig(**overrides)
        colony = LPCCColony(env.actions, config=cfg, seed=seed)
        run = run_episodes(colony, env, episodes)
        summary = M.summarize(colony, run.per_step_action)
        summary["total_reward"] = run.total_reward
        results[name] = summary

    base = results["H_full"]["total_reward"]
    table = {}
    for name, s in results.items():
        delta = s["total_reward"] - base
        table[name] = {
            "total_reward": s["total_reward"],
            "delta_vs_full": round(delta, 3),
            "pred_acc": s["prediction_accuracy"],
            "population": s["population"],
            "structural_events": s["structural_events"],
        }
    _print_summary(
        "ABLATIONS (reward vs full)",
        {k: v["delta_vs_full"] for k, v in table.items()},
    )
    return {"results": results, "table": table}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
EXPERIMENTS = {
    "discovery": exp_discovery,
    "reversal": exp_reversal,
    "transfer": exp_transfer,
    "emergence": exp_emergence,
    "ablations": exp_ablations,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AHA / LPCC experiment runner")
    parser.add_argument("experiment", choices=list(EXPERIMENTS) + ["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=200)
    args = parser.parse_args(argv)

    to_run = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for exp in to_run:
        fn = EXPERIMENTS[exp]
        payload = fn(args.seed, args.episodes)
        wrapped = {
            "metadata": _metadata(exp, args.seed, args.episodes),
            "results": payload,
        }
        path = _save(exp, args.seed, wrapped)
        print(f"  saved -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
