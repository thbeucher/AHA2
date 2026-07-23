"""Temporally-aligned local prediction object.

A prediction created at t with horizon h MUST be evaluated at t+h. The ledger
(see ledger.py) enforces this alignment. Predictions carry predicted tokens so
evaluation compares like-with-like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aha.cells.patterns import Token


@dataclass
class LocalPrediction:
    """A single temporally-aligned forecast produced by a LPCC."""

    source_cell_id: str
    hypothesis_id: str
    creation_time: int
    horizon: int
    predicted_tokens: set[Token] = field(default_factory=set)
    predicted_reward: float = 0.0
    predicted_controllability: float = 0.0
    predicted_active_neighbors: list[str] = field(default_factory=list)
    uncertainty: float = 0.5
    prediction_context: set[Token] = field(default_factory=set)
    action: str | None = None

    @property
    def target_time(self) -> int:
        return self.creation_time + self.horizon

    def hit_fraction(self, observed_tokens: set[Token]) -> float:
        """Fraction of predicted tokens observed at target time."""
        if not self.predicted_tokens:
            return 0.0
        hits = sum(1 for t in self.predicted_tokens if t in observed_tokens)
        return hits / len(self.predicted_tokens)