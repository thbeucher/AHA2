"""Resource economy for sparse autonomous computation.

Implements assumptions: A5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ResourceBudget:
    """Energy budget owned by a single agent."""

    energy: float = 10.0
    thinking_cost: float = 0.05
    communication_cost: float = 0.02
    memory_cost: float = 0.01
    prediction_cost: float = 0.03
    retirement_energy: float = 0.1

    def spend(
        self,
        thinking: bool = False,
        communication: int = 0,
        memory: bool = False,
        prediction: bool = False,
    ) -> None:
        cost = 0.0
        if thinking:
            cost += self.thinking_cost
        cost += communication * self.communication_cost
        if memory:
            cost += self.memory_cost
        if prediction:
            cost += self.prediction_cost
        self.energy = max(0.0, self.energy - cost)

    def reward_usefulness(self, amount: float) -> None:
        self.energy += max(0.0, amount)

    @property
    def alive(self) -> bool:
        return self.energy > self.retirement_energy
