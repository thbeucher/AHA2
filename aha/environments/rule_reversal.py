"""Rule-reversal environment for the falsification test.

The hidden rule flips at a configured episode. This is the most important
falsification test: the colony must detect contradiction, decay confidence in
the old rule, and adapt to the new rule.

Phase A (episodes < reversal_episode):
    switch present + action_A  -> (delay) door_open + reward
Phase B (episodes >= reversal_episode):
    switch present + action_B  -> (delay) door_open + reward
    (action_A no longer works)
"""

from __future__ import annotations

import random
from typing import Any

from aha.environments.hidden_rules import StepResult


class RuleReversalEnv:
    ACTIONS = ["investigate", "push", "wait", "observe"]

    def __init__(
        self,
        seed: int = 0,
        delay: int = 2,
        episode_len: int = 20,
        reversal_episode: int = 100,
        action_a: str = "investigate",
        action_b: str = "push",
    ):
        self.rng = random.Random(seed)
        self.delay = delay
        self.episode_len = episode_len
        self.reversal_episode = reversal_episode
        self.action_a = action_a
        self.action_b = action_b
        self._episode = 0
        self._t = 0
        self._switch = False
        self._door = False
        self._distractor = False
        self._pending_open: int | None = None
        self._pending_reward: int | None = None

    @property
    def actions(self) -> list[str]:
        return list(self.ACTIONS)

    @property
    def current_correct_action(self) -> str:
        return self.action_a if self._episode < self.reversal_episode else self.action_b

    def reset(self) -> dict[str, Any]:
        self._t = 0
        self._door = False
        self._pending_open = None
        self._pending_reward = None
        self._switch = self.rng.random() < 0.4
        self._distractor = self.rng.random() < 0.5
        return self._obs()

    def next_episode(self) -> None:
        self._episode += 1

    def _obs(self) -> dict[str, Any]:
        return {
            "switch": "present" if self._switch else "absent",
            "door": "open" if self._door else "closed",
            "distractor": "red" if self._distractor else "blue",
        }

    def step(self, action: str) -> StepResult:
        reward = 0.0
        if self._pending_open is not None and self._t >= self._pending_open:
            self._door = True
            self._pending_open = None
        if self._pending_reward is not None and self._t >= self._pending_reward:
            reward += 1.0
            self._pending_reward = None

        if self._switch and action == self.current_correct_action:
            self._pending_open = self._t + self.delay
            self._pending_reward = self._t + self.delay

        self._t += 1
        if self._door and self.rng.random() < 0.5:
            self._door = False
        self._switch = self.rng.random() < 0.4
        self._distractor = self.rng.random() < 0.5

        done = self._t >= self.episode_len
        return StepResult(self._obs(), reward, done, {"t": self._t, "episode": self._episode})

    def hidden_rule_summary(self) -> dict[str, Any]:
        return {
            "phase": "A" if self._episode < self.reversal_episode else "B",
            "antecedent": {"switch": "present"},
            "action": self.current_correct_action,
            "consequent": {"door": "open"},
            "delay": self.delay,
        }