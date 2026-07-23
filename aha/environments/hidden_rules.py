"""Hidden-rule environments with observation -> action -> consequence structure.

The agent sees only symbolic observations. It does NOT see the rule. The core
task family: a "switch" object appears; if the agent performs the correct action
while the switch is present, a delayed consequence (door_open) and reward occur.

Environments implement a common minimal interface:

    obs = env.reset()
    obs_next, reward, done, info = env.step(action)

All environments are deterministic given the seed unless explicitly stochastic.
The `hidden_rule_summary()` returns the ground-truth rule for SCORING ONLY; it is
never passed to the agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class HiddenSwitchEnv:
    """Environment 1/2: deterministic hidden rule + delayed consequence.

    Hidden rule (unknown to agent):
      When `switch=present`, taking action `investigate` causes, `delay` steps
      later, `door=open` and a reward of +1. Any other action leaves door closed.
      A distractor feature toggles randomly and is irrelevant.

    Observations expose: switch, door, distractor, step_phase. The relationship
    switch x investigate -> (delay) door_open must be discovered from experience.
    """

    ACTIONS = ["investigate", "wait", "push", "observe"]

    def __init__(self, seed: int = 0, delay: int = 2, episode_len: int = 20):
        self.rng = random.Random(seed)
        self.delay = delay
        self.episode_len = episode_len
        self._t = 0
        self._switch = False
        self._door = False
        self._distractor = False
        self._pending_open: int | None = None  # timestep at which door opens
        self._pending_reward: int | None = None

    @property
    def actions(self) -> list[str]:
        return list(self.ACTIONS)

    def reset(self) -> dict[str, Any]:
        self._t = 0
        self._door = False
        self._pending_open = None
        self._pending_reward = None
        self._new_switch()
        self._distractor = self.rng.random() < 0.5
        return self._obs()

    def _new_switch(self) -> None:
        # Switch is present ~40% of steps, appearing unpredictably.
        self._switch = self.rng.random() < 0.4

    def _obs(self) -> dict[str, Any]:
        return {
            "switch": "present" if self._switch else "absent",
            "door": "open" if self._door else "closed",
            "distractor": "red" if self._distractor else "blue",
        }

    def step(self, action: str) -> StepResult:
        reward = 0.0
        # Resolve pending delayed consequences scheduled for this step.
        if self._pending_open is not None and self._t >= self._pending_open:
            self._door = True
            self._pending_open = None
        if self._pending_reward is not None and self._t >= self._pending_reward:
            reward += 1.0
            self._pending_reward = None

        # Hidden causal rule: switch present + investigate => schedule consequence.
        if self._switch and action == "investigate":
            self._pending_open = self._t + self.delay
            self._pending_reward = self._t + self.delay

        self._t += 1
        # Environment dynamics: door relaxes closed after being open a while;
        # switch and distractor refresh.
        if self._door and self.rng.random() < 0.5:
            self._door = False
        self._new_switch()
        self._distractor = self.rng.random() < 0.5

        done = self._t >= self.episode_len
        return StepResult(self._obs(), reward, done, {"t": self._t})

    def hidden_rule_summary(self) -> dict[str, Any]:
        """Ground truth for scoring ONLY. Never given to the agent."""
        return {
            "antecedent": {"switch": "present"},
            "action": "investigate",
            "consequent": {"door": "open"},
            "delay": self.delay,
            "reward_on_consequent": 1.0,
        }