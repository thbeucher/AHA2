"""Explicit local hypothesis maintained by a LPCC.

A hypothesis encodes:
    "Under context C (a set of tokens), if I become active and/or action A occurs,
     future event F (a set of predicted tokens) becomes more likely."

Confidence is DERIVED from evidence (support / contradiction / prediction error /
delayed reward / local causal evidence), never incremented arbitrarily.

This reuses EvidenceStats from aha.cells.state so the whole system shares one
auditable evidence representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aha.cells.patterns import Token
from aha.cells.state import EvidenceStats


@dataclass
class LocalHypothesis:
    """C x A -> F with evidence-derived confidence."""

    hypothesis_id: str
    context_tokens: set[Token] = field(default_factory=set)
    action: str | None = None
    predicted_tokens: set[Token] = field(default_factory=set)
    evidence: EvidenceStats = field(default_factory=EvidenceStats)
    horizon: int = 1

    @property
    def confidence(self) -> float:
        return self.evidence.confidence

    @property
    def uncertainty(self) -> float:
        return self.evidence.uncertainty

    def is_defined(self) -> bool:
        return bool(self.context_tokens) and bool(self.predicted_tokens)

    def context_match(self, signature_tokens: set[Token]) -> float:
        if not self.context_tokens:
            return 0.0
        hits = sum(1 for t in self.context_tokens if t in signature_tokens)
        return hits / len(self.context_tokens)

    def record_outcome(self, predicted_hit_fraction: float, supported: bool) -> None:
        """Record an aligned prediction outcome.

        predicted_hit_fraction in [0,1]: fraction of predicted tokens that were
        actually observed at target time. supported: whether the outcome counts
        as confirming (True) or contradicting (False) the hypothesis.
        """
        self.evidence.prediction_count += 1
        error = 1.0 - predicted_hit_fraction
        self.evidence.prediction_error_sum += error
        if supported:
            self.evidence.supporting_observations += 1
            self.evidence.contradiction_streak = 0
        else:
            self.evidence.contradictory_observations += 1
            self.evidence.contradiction_streak += 1

    def add_delayed_reward_evidence(self, amount: float) -> None:
        self.evidence.delayed_reward_evidence += amount

    def add_causal_evidence(self, amount: float) -> None:
        self.evidence.local_causal_evidence += amount

    def summary(self) -> dict:
        return {
            "id": self.hypothesis_id,
            "action": self.action,
            "n_context": len(self.context_tokens),
            "n_predict": len(self.predicted_tokens),
            "confidence": round(self.confidence, 3),
            "uncertainty": round(self.uncertainty, 3),
            "support": self.evidence.supporting_observations,
            "contradict": self.evidence.contradictory_observations,
            "contradiction_streak": self.evidence.contradiction_streak,
        }