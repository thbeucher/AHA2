"""Prediction ledger enforcing strict temporal alignment.

A prediction created at t with horizon h is stored under target_time = t+h.
At each timestep the colony calls `due(now)` to retrieve exactly the predictions
whose target_time == now, then evaluates them against the observed tokens.

This guarantees predictions are NEVER compared to misaligned observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aha.cells.patterns import Token
from aha.prediction.prediction import LocalPrediction


@dataclass
class EvaluatedPrediction:
    """Result of evaluating one aligned prediction."""

    prediction: LocalPrediction
    hit_fraction: float
    supported: bool
    observed_reward: float


@dataclass
class PredictionLedger:
    """Time-indexed store of outstanding predictions."""

    support_threshold: float = 0.5
    _by_target: dict[int, list[LocalPrediction]] = field(default_factory=dict)
    created_count: int = 0
    evaluated_count: int = 0

    def add(self, prediction: LocalPrediction) -> None:
        self._by_target.setdefault(prediction.target_time, []).append(prediction)
        self.created_count += 1

    def outstanding(self) -> int:
        return sum(len(v) for v in self._by_target.values())

    def due(self, now: int) -> list[LocalPrediction]:
        return list(self._by_target.get(now, []))

    def evaluate_due(
        self,
        now: int,
        observed_tokens: set[Token],
        observed_reward: float,
    ) -> list[EvaluatedPrediction]:
        """Evaluate and REMOVE all predictions whose target_time == now."""
        due = self._by_target.pop(now, [])
        results: list[EvaluatedPrediction] = []
        for pred in due:
            hit = pred.hit_fraction(observed_tokens)
            supported = hit >= self.support_threshold
            self.evaluated_count += 1
            results.append(
                EvaluatedPrediction(
                    prediction=pred,
                    hit_fraction=hit,
                    supported=supported,
                    observed_reward=observed_reward,
                )
            )
        return results