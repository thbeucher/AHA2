"""Deterministic toy GridWorld for substrate smoke tests.

Implements assumptions: A9.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GridWorld:
    """Tiny symbolic embodied environment with key-door structure."""

    width: int = 4
    height: int = 4
    agent_x: int = 0
    agent_y: int = 0
    key_x: int = 1
    key_y: int = 0
    door_x: int = 3
    door_y: int = 3
    has_key: bool = False

    def observe(self) -> dict:
        return {
            "x": self.agent_x,
            "y": self.agent_y,
            "near_key": abs(self.agent_x - self.key_x) + abs(self.agent_y - self.key_y) <= 1 and not self.has_key,
            "has_key": self.has_key,
            "near_door": abs(self.agent_x - self.door_x) + abs(self.agent_y - self.door_y) <= 1,
        }

    def step(self, action: str) -> tuple[dict, float]:
        if action == "investigate":
            self.agent_x = min(self.width - 1, self.agent_x + 1)
        elif action == "exploit":
            self.agent_y = min(self.height - 1, self.agent_y + 1)

        if self.agent_x == self.key_x and self.agent_y == self.key_y:
            self.has_key = True

        reward = 1.0 if self.has_key and self.agent_x == self.door_x and self.agent_y == self.door_y else 0.0
        return self.observe(), reward
