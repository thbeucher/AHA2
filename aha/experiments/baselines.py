"""Baseline policies for comparison (NOT part of the LPCC substrate).

These are deliberately simple, non-learning references so LPCC results can be
interpreted against (1) random and (2) an oracle-ish hand-coded rule. The
hand-coded baseline is given the correct action for the CURRENT env only and is
NOT adaptive across rule reversal (illustrating the cost of hard-coding).
"""

from __future__ import annotations

import random
from typing import Any


class RandomBaseline:
    def __init__(self, actions: list[str], seed: int = 0):
        self.actions = list(actions)
        self.rng = random.Random(seed)

    def act(self, obs: dict[str, Any]) -> str:
        return self.actions[self.rng.randrange(len(self.actions))]


class FixedRuleBaseline:
    """Hand-coded: if antecedent value present, take the fixed correct action."""

    def __init__(self, antecedent_key: str, antecedent_val: str, correct_action: str, default: str = "observe"):
        self.k = antecedent_key
        self.v = antecedent_val
        self.correct = correct_action
        self.default = default

    def act(self, obs: dict[str, Any]) -> str:
        return self.correct if obs.get(self.k) == self.v else self.default


def run_baseline(policy, env, n_episodes: int) -> dict[str, Any]:
    total = 0.0
    actions: list[str] = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            a = policy.act(obs)
            step = env.step(a)
            actions.append(a)
            total += step.reward
            obs = step.observation
            done = step.done
    return {"total_reward": total, "n_episodes": n_episodes, "actions": actions}