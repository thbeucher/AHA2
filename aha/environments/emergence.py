"""Multi-step emergence environment: no single cell can solve it alone.

Hidden two-stage rule (unknown to agent):
    Stage 1: when `key=present`, action `grab` -> (delay) `holding=yes`.
    Stage 2: when `holding=yes` AND `door=locked`, action `open` -> (delay)
             `door=open` + reward.

Solving requires a temporally-coordinated sequence discovered by DIFFERENT cells:
  - one cell must learn key+grab -> holding,
  - another must learn holding+open -> door_open+reward.

The useful behavior (grab-then-open) must emerge from communication + action
voting + temporal prediction. No cell is programmed with the full sequence.
"""

from __future__ import annotations

import random
from typing import Any

from aha.environments.hidden_rules import StepResult


class TwoStageEnv:
    ACTIONS = ["grab", "open", "wait", "observe"]

    def __init__(self, seed: int = 0, delay: int = 1, episode_len: int = 30):
        self.rng = random.Random(seed)
        self.delay = delay
        self.episode_len = episode_len
        self._t = 0
        self._key = False
        self._holding = False
        self._door_locked = True
        self._door_open = False
        self._pending_hold: int | None = None
        self._pending_open: int | None = None
        self._pending_reward: int | None = None

    @property
    def actions(self) -> list[str]:
        return list(self.ACTIONS)

    def reset(self) -> dict[str, Any]:
        self._t = 0
        self._key = self.rng.random() < 0.5
        self._holding = False
        self._door_locked = True
        self._door_open = False
        self._pending_hold = None
        self._pending_open = None
        self._pending_reward = None
        return self._obs()

    def _obs(self) -> dict[str, Any]:
        return {
            "key": "present" if self._key else "absent",
            "holding": "yes" if self._holding else "no",
            "door": "open" if self._door_open else ("locked" if self._door_locked else "closed"),
        }

    def step(self, action: str) -> StepResult:
        reward = 0.0
        # Resolve pending consequences.
        if self._pending_hold is not None and self._t >= self._pending_hold:
            self._holding = True
            self._pending_hold = None
        if self._pending_open is not None and self._t >= self._pending_open:
            self._door_open = True
            self._door_locked = False
            self._pending_open = None
        if self._pending_reward is not None and self._t >= self._pending_reward:
            reward += 1.0
            self._pending_reward = None

        # Stage 1.
        if self._key and action == "grab" and not self._holding:
            self._pending_hold = self._t + self.delay
        # Stage 2.
        if self._holding and self._door_locked and action == "open" and not self._door_open:
            self._pending_open = self._t + self.delay
            self._pending_reward = self._t + self.delay

        self._t += 1
        # Key refreshes if not yet grabbed; once holding, key consumed.
        if self._holding:
            self._key = False
        else:
            self._key = self.rng.random() < 0.5

        done = self._t >= self.episode_len or self._door_open
        return StepResult(self._obs(), reward, done, {"t": self._t})

    def hidden_rule_summary(self) -> dict[str, Any]:
        return {
            "stage1": {"antecedent": {"key": "present"}, "action": "grab", "consequent": {"holding": "yes"}},
            "stage2": {"antecedent": {"holding": "yes"}, "action": "open", "consequent": {"door": "open"}},
            "delay": self.delay,
        }