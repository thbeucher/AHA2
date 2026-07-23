"""Shared explicit types for the AHA substrate.

Implements assumptions: A1, A2, A4, A6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ObservationContext:
    """Information available to an agent at a timestep."""

    observation: dict[str, Any]
    previous_action: str | None = None
    reward: float = 0.0
    timestep: int = 0
    recent_history: tuple[dict[str, Any], ...] = ()


@dataclass
class Prediction:
    """An inspectable forecast produced by a local belief simulator."""

    future_observation: dict[str, Any]
    reward_probability: float
    controllability: float
    uncertainty: float
    horizon: int = 1
    expected_active_beliefs: list[str] = field(default_factory=list)


@dataclass
class ActionProposal:
    """Agent-local intervention proposal for distributed consensus."""

    agent_id: str
    action: str
    expected_utility: float
    expected_information_gain: float
    confidence: float
    risk: float
    rationale: str

    @property
    def vote_weight(self) -> float:
        """Transparent consensus weight; not a policy-network logit."""
        value = self.expected_utility + self.expected_information_gain - self.risk
        return max(0.0, self.confidence) * value


@dataclass
class EvidenceReport:
    """Local evidence assessment for or against a belief."""

    support: float
    contradiction: float
    novelty: float
    prediction_error: float
    causal_trace: float


@dataclass
class AgentDecision:
    """Complete public result of one agent timestep."""

    agent_id: str
    active: bool
    evidence: EvidenceReport
    prediction: Prediction | None
    proposal: ActionProposal | None
    outgoing_messages: list[Any] = field(default_factory=list)
