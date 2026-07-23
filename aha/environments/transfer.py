"""Transfer environment: same relational structure, different surface symbols.

Train world uses symbols {switch, investigate, door}. Transfer world uses a
symbol RENAMING (e.g. gadget/prod/gate). The STRUCTURE
   antecedent-present + correct-action -> (delay) consequent
is identical. This tests whether the colony transfers relational structure or
merely memorizes surface tokens.

We implement transfer by symbol substitution maps. The agent's action vocabulary
changes too, so pure token memorization cannot transfer; only structural
re-discovery (or a structure-aware detector) can.
"""

from __future__ import annotations

import random
from typing import Any

from aha.environments.hidden_rules import StepResult


DEFAULT_MAP_A = {
    "antecedent_key": "switch",
    "antecedent_val": "present",
    "antecedent_absent": "absent",
    "consequent_key": "door",
    "consequent_val": "open",
    "consequent_off": "closed",
    "distractor_key": "distractor",
    "distractor_vals": ("red", "blue"),
    "action": "investigate",
}

DEFAULT_MAP_B = {
    "antecedent_key": "gadget",
    "antecedent_val": "lit",
    "antecedent_absent": "dark",
    "consequent_key": "gate",
    "consequent_val": "up",
    "consequent_off": "down",
    "distractor_key": "noise",
    "distractor_vals": ("hum", "buzz"),
    "action": "prod",
}


class TransferEnv:
    """Relational hidden-rule env parameterized by a surface symbol map."""

    def __init__(
        self,
        symbol_map: dict[str, Any],
        other_actions: list[str],
        seed: int = 0,
        delay: int = 2,
        episode_len: int = 20,
    ):
        self.m = symbol_map
        self.rng = random.Random(seed)
        self.delay = delay
        self.episode_len = episode_len
        self._other_actions = other_actions
        self._t = 0
        self._ant = False
        self._con = False
        self._distractor = 0
        self._pending_con: int | None = None
        self._pending_reward: int | None = None

    @property
    def actions(self) -> list[str]:
        acts = [self.m["action"]] + [a for a in self._other_actions if a != self.m["action"]]
        return acts

    def reset(self) -> dict[str, Any]:
        self._t = 0
        self._con = False
        self._pending_con = None
        self._pending_reward = None
        self._ant = self.rng.random() < 0.4
        self._distractor = self.rng.randrange(2)
        return self._obs()

    def _obs(self) -> dict[str, Any]:
        m = self.m
        return {
            m["antecedent_key"]: m["antecedent_val"] if self._ant else m["antecedent_absent"],
            m["consequent_key"]: m["consequent_val"] if self._con else m["consequent_off"],
            m["distractor_key"]: m["distractor_vals"][self._distractor],
        }

    def step(self, action: str) -> StepResult:
        reward = 0.0
        if self._pending_con is not None and self._t >= self._pending_con:
            self._con = True
            self._pending_con = None
        if self._pending_reward is not None and self._t >= self._pending_reward:
            reward += 1.0
            self._pending_reward = None

        if self._ant and action == self.m["action"]:
            self._pending_con = self._t + self.delay
            self._pending_reward = self._t + self.delay

        self._t += 1
        if self._con and self.rng.random() < 0.5:
            self._con = False
        self._ant = self.rng.random() < 0.4
        self._distractor = self.rng.randrange(2)

        done = self._t >= self.episode_len
        return StepResult(self._obs(), reward, done, {"t": self._t})

    def hidden_rule_summary(self) -> dict[str, Any]:
        return {
            "antecedent": {self.m["antecedent_key"]: self.m["antecedent_val"]},
            "action": self.m["action"],
            "consequent": {self.m["consequent_key"]: self.m["consequent_val"]},
            "delay": self.delay,
        }


def make_train_env(seed: int = 0, **kw) -> TransferEnv:
    return TransferEnv(DEFAULT_MAP_A, ["prod", "wait", "observe"], seed=seed, **kw)


def make_transfer_env(seed: int = 0, **kw) -> TransferEnv:
    return TransferEnv(DEFAULT_MAP_B, ["investigate", "wait", "observe"], seed=seed, **kw)