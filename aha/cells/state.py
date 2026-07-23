"""Inspectable state containers for Local Predictive Causal Cells.

The LPCC is a biologically inspired computational abstraction, not a neuron.
All adaptive state is explicit so experiments can audit belief revision,
energy use, temporal credit, and structural lifecycle decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class LifecycleState(StrEnum):
    """Structural lifecycle for activity-dependent plasticity."""

    CANDIDATE = "candidate"
    TESTING = "testing"
    STABLE = "stable"
    SPECIALIZING = "specializing"
    DECLINING = "declining"
    RETIRED = "retired"


@dataclass
class EvidenceStats:
    """Local evidence accumulated by a cell hypothesis."""

    supporting_observations: int = 0
    contradictory_observations: int = 0
    prediction_count: int = 0
    prediction_error_sum: float = 0.0
    delayed_reward_evidence: float = 0.0
    local_causal_evidence: float = 0.0
    contradiction_streak: int = 0

    @property
    def total_observations(self) -> int:
        return self.supporting_observations + self.contradictory_observations

    @property
    def mean_prediction_error(self) -> float:
        if self.prediction_count == 0:
            return 1.0
        return self.prediction_error_sum / self.prediction_count

    @property
    def confidence(self) -> float:
        """Evidence-derived confidence with contradiction sensitivity.

        This is intentionally not an arbitrary incremented scalar. It is a
        smoothed ratio of support to all evidence, discounted by recent
        prediction error and contradiction streak.
        """

        total = self.total_observations
        evidence_ratio = (self.supporting_observations + 1.0) / (total + 2.0)
        error_penalty = 0.5 * self.mean_prediction_error
        contradiction_penalty = min(0.35, 0.07 * self.contradiction_streak)
        causal_bonus = min(0.2, 0.05 * max(0.0, self.local_causal_evidence))
        delayed_bonus = min(0.15, 0.03 * max(0.0, self.delayed_reward_evidence))
        return max(0.0, min(1.0, evidence_ratio - error_penalty - contradiction_penalty + causal_bonus + delayed_bonus))

    @property
    def uncertainty(self) -> float:
        total = self.total_observations
        scarcity = 1.0 / (1.0 + 0.25 * total)
        conflict = 1.0 - abs(self.supporting_observations - self.contradictory_observations) / max(1, total)
        return max(0.0, min(1.0, 0.55 * scarcity + 0.45 * conflict))


@dataclass
class PredictionStats:
    """Auditable local prediction quality counters."""

    created: int = 0
    evaluated: int = 0
    correct: int = 0
    error_sum: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.evaluated == 0:
            return 0.0
        return self.correct / self.evaluated

    @property
    def mean_error(self) -> float:
        if self.evaluated == 0:
            return 1.0
        return self.error_sum / self.evaluated


@dataclass
class ConnectionState:
    """Local plastic relationship between LPCCs."""

    target_cell_id: str
    strength: float = 0.5
    eligibility_trace: float = 0.0
    usefulness: float = 0.0
    coactivation_count: int = 0
    failed_interactions: int = 0

    def decay(self, rate: float = 0.9) -> None:
        self.eligibility_trace *= rate

    def update(self, useful: bool, amount: float = 0.05) -> None:
        self.decay()
        self.eligibility_trace += amount
        if useful:
            self.coactivation_count += 1
            self.usefulness += amount
            self.strength = min(1.0, self.strength + amount * self.eligibility_trace)
        else:
            self.failed_interactions += 1
            self.usefulness -= amount
            self.strength = max(0.0, self.strength - amount * self.eligibility_trace)


@dataclass
class LPCCState:
    """Full inspectable internal state of a Local Predictive Causal Cell."""

    cell_id: str
    age: int = 0
    activation_state: float = 0.0
    membrane_like_state: float = 0.0
    short_term_context: list[dict[str, Any]] = field(default_factory=list)
    slow_context: dict[str, float] = field(default_factory=dict)
    prototype_pattern: dict[str, Any] = field(default_factory=dict)
    temporal_pattern_memory: dict[tuple[str, ...], int] = field(default_factory=dict)
    belief_state: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    uncertainty: float = 0.5
    reliability: float = 0.5
    novelty: float = 0.0
    expected_future: dict[str, Any] = field(default_factory=dict)
    action_tendencies: dict[str, float] = field(default_factory=dict)
    eligibility_trace: float = 0.0
    causal_influence_estimate: float = 0.0
    prediction_statistics: PredictionStats = field(default_factory=PredictionStats)
    evidence: EvidenceStats = field(default_factory=EvidenceStats)
    energy: float = 10.0
    homeostatic_activity_level: float = 0.1
    activation_threshold: float = 0.45
    incoming_connections: dict[str, ConnectionState] = field(default_factory=dict)
    outgoing_connections: dict[str, ConnectionState] = field(default_factory=dict)
    parent_id: str | None = None
    child_ids: list[str] = field(default_factory=list)
    lifecycle_state: LifecycleState = LifecycleState.CANDIDATE
    recent_activity: list[float] = field(default_factory=list)
    structural_event_log: list[str] = field(default_factory=list)

    def bounded_context(self, max_len: int = 8) -> tuple[dict[str, Any], ...]:
        return tuple(self.short_term_context[-max_len:])

    def record_activity(self, activation: float, window: int = 32) -> None:
        self.recent_activity.append(activation)
        if len(self.recent_activity) > window:
            self.recent_activity = self.recent_activity[-window:]

    @property
    def mean_recent_activity(self) -> float:
        if not self.recent_activity:
            return 0.0
        return sum(self.recent_activity) / len(self.recent_activity)

    @property
    def alive(self) -> bool:
        return self.lifecycle_state != LifecycleState.RETIRED and self.energy > 0.05