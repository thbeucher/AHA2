"""Explicit belief representations for Hypothesis Agents.

Implements assumptions: A1, A2, A3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aha.core.types import EvidenceReport, ObservationContext, Prediction


@dataclass
class StructuredBelief:
    """A replaceable, inspectable hypothesis model.

    The default representation is intentionally simple: a named predicate over
    symbolic observation features plus an expected successor feature map. Later
    phases can replace this with richer causal programs or probabilistic models.
    """

    belief_id: str
    description: str
    predicate: dict[str, Any]
    expected_next: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    uncertainty: float = 0.5
    reliability: float = 0.5
    utility: float = 0.0
    novelty: float = 0.0

    def match(self, context: ObservationContext) -> float:
        """Return fraction of predicate clauses satisfied by observation."""
        if not self.predicate:
            return 0.0
        hits = sum(1 for key, value in self.predicate.items() if context.observation.get(key) == value)
        return hits / len(self.predicate)

    def simulate(self, context: ObservationContext, horizon: int = 1) -> Prediction:
        """Simulate a tiny future consistent with the explicit belief."""
        future = dict(context.observation)
        future.update(self.expected_next)
        return Prediction(
            future_observation=future,
            reward_probability=max(0.0, min(1.0, self.utility)),
            controllability=max(0.0, min(1.0, self.reliability * self.confidence)),
            uncertainty=self.uncertainty,
            horizon=horizon,
            expected_active_beliefs=[self.belief_id],
        )

    def evaluate_prediction(self, prediction: Prediction | None, outcome: dict[str, Any]) -> float:
        """Compute symbolic prediction error against an observed outcome."""
        if prediction is None or not prediction.future_observation:
            return 1.0
        checked = prediction.future_observation
        misses = sum(1 for key, value in checked.items() if outcome.get(key) != value)
        return misses / max(1, len(checked))

    def update_from_evidence(self, report: EvidenceReport) -> None:
        """Revise local scalar summaries without global optimization."""
        delta = 0.12 * report.support - 0.16 * report.contradiction - 0.08 * report.prediction_error
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.uncertainty = max(0.0, min(1.0, self.uncertainty + report.novelty * 0.05 - report.support * 0.04))
        self.reliability = max(0.0, min(1.0, self.reliability + 0.1 * (1.0 - report.prediction_error) - 0.05))
        self.utility = max(0.0, min(1.0, self.utility + 0.05 * report.causal_trace - 0.02 * report.contradiction))
        self.novelty = report.novelty
