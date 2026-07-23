"""Stateful autonomous Hypothesis Agent.

Implements assumptions: A1, A2, A3, A4, A5, A7, A8.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from aha.beliefs.base import StructuredBelief
from aha.communication.messages import Message
from aha.core.types import ActionProposal, AgentDecision, EvidenceReport, ObservationContext, Prediction
from aha.economy.resources import ResourceBudget
from aha.memory.store import Episode, EpisodicMemory, SemanticMemory


@dataclass
class HypothesisAgent:
    """A tiny scientist with persistent local state."""

    agent_id: str
    belief: StructuredBelief
    resources: ResourceBudget = field(default_factory=ResourceBudget)
    episodic_memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    semantic_memory: SemanticMemory = field(default_factory=SemanticMemory)
    attention: float = 0.5
    age: int = 0
    activation_history: list[float] = field(default_factory=list)
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    last_prediction: Prediction | None = None
    eligibility_trace: float = 0.0
    alive: bool = True

    def step(self, context: ObservationContext, messages: list[Message], horizon: int = 1) -> AgentDecision:
        """Observe, evaluate, predict, propose, remember, and communicate."""
        if not self.alive:
            return self._inactive_decision()

        self.age += 1
        match_quality = self.belief.match(context)
        self.activation_history.append(match_quality)

        active = match_quality >= self.attention * 0.5
        novelty = 1.0 - match_quality
        prediction_error = self.belief.evaluate_prediction(self.last_prediction, context.observation)
        message_support = min(1.0, sum(m.confidence for m in messages if m.belief_id == self.belief.belief_id) * 0.1)

        self.eligibility_trace = 0.8 * self.eligibility_trace + match_quality * max(0.0, context.reward)
        support = min(1.0, match_quality * (1.0 - prediction_error) + message_support)
        contradiction = (1.0 - match_quality) * prediction_error
        evidence = EvidenceReport(support, contradiction, novelty, prediction_error, self.eligibility_trace)

        prediction = self.belief.simulate(context, horizon) if active else None
        proposal = self._propose_action(prediction) if prediction is not None else None
        outgoing = self._communicate(prediction) if prediction is not None else []

        self.belief.update_from_evidence(evidence)
        self._remember(context, prediction)
        self.resources.spend(thinking=True, communication=len(outgoing), memory=True, prediction=prediction is not None)
        if self.belief.utility > 0.1:
            self.resources.reward_usefulness(0.02 * self.belief.utility)
        self.alive = self.resources.alive and self.belief.confidence > 0.01
        self.last_prediction = prediction

        return AgentDecision(self.agent_id, active, evidence, prediction, proposal, outgoing)

    def should_split(self) -> bool:
        """Phase-1 split hook: specified, not yet executed by the colony."""
        return self.belief.confidence > 0.75 and self.belief.utility > 0.4 and self.belief.uncertainty > 0.55

    def should_retire(self) -> bool:
        """Retire when local evidence and resources no longer justify existence."""
        return not self.alive or (self.age > 10 and self.belief.utility < 0.01 and self.belief.confidence < 0.1)

    def _propose_action(self, prediction: Prediction) -> ActionProposal:
        action = "investigate" if prediction.uncertainty > 0.4 else "exploit"
        return ActionProposal(
            agent_id=self.agent_id,
            action=action,
            expected_utility=self.belief.utility,
            expected_information_gain=prediction.uncertainty,
            confidence=self.belief.confidence,
            risk=prediction.uncertainty * (1.0 - self.belief.reliability),
            rationale=f"Test belief {self.belief.belief_id}: {self.belief.description}",
        )

    def _communicate(self, prediction: Prediction) -> list[Message]:
        if self.belief.confidence < 0.35 and prediction.uncertainty < 0.8:
            return []
        return [
            Message(
                sender_id=self.agent_id,
                receiver_id=None,
                belief_id=self.belief.belief_id,
                confidence=self.belief.confidence,
                prediction_summary=str(prediction.future_observation),
                urgency=prediction.uncertainty,
                request="contradict-or-support" if prediction.uncertainty > 0.5 else None,
            )
        ]

    def _remember(self, context: ObservationContext, prediction: Prediction | None) -> None:
        predicted = prediction.future_observation if prediction else None
        episode = Episode(dict(context.observation), predicted, dict(context.observation), context.timestep)
        self.episodic_memory.add(episode)
        self.semantic_memory.compress_episode(episode)

    def _inactive_decision(self) -> AgentDecision:
        report = EvidenceReport(0.0, 0.0, 1.0, 1.0, self.eligibility_trace)
        return AgentDecision(self.agent_id, False, report, None, None, [])
