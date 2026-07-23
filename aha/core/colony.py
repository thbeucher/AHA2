"""Colony orchestration without layers, gradients, or global loss.

Implements assumptions: A6, A8.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aha.agents.hypothesis_agent import HypothesisAgent
from aha.communication.messages import MessageBus
from aha.core.types import ActionProposal, ObservationContext


@dataclass
class ColonySnapshot:
    """Compact observable colony state for metrics and dashboards."""

    timestep: int
    living_agents: int
    active_agents: int
    chosen_action: str
    mean_confidence: float
    total_energy: float
    communication_overhead: int


@dataclass
class HypothesisColony:
    """Dynamic society of local model builders."""

    agents: dict[str, HypothesisAgent]
    bus: MessageBus = field(default_factory=MessageBus)
    timestep: int = 0
    snapshots: list[ColonySnapshot] = field(default_factory=list)

    def step(self, observation: dict[str, Any], previous_action: str | None = None, reward: float = 0.0) -> str:
        """Run one colony timestep and return a consensus action."""
        context = ObservationContext(observation, previous_action, reward, self.timestep)
        decisions = []
        for agent in list(self.agents.values()):
            messages = self.bus.collect_for(agent.agent_id)
            decision = agent.step(context, messages)
            decisions.append(decision)
            for message in decision.outgoing_messages:
                self.bus.publish(message)

        proposals = [d.proposal for d in decisions if d.proposal is not None]
        action = self.choose_action(proposals)
        self._retire_agents()
        self._record_snapshot(action, decisions)
        self.timestep += 1
        return action

    def choose_action(self, proposals: list[ActionProposal]) -> str:
        """Transparent distributed consensus over local proposals."""
        if not proposals:
            return "observe"
        scores: dict[str, float] = {}
        for proposal in proposals:
            scores[proposal.action] = scores.get(proposal.action, 0.0) + proposal.vote_weight
        return max(scores.items(), key=lambda item: (item[1], item[0]))[0]

    def _retire_agents(self) -> None:
        for agent_id, agent in list(self.agents.items()):
            if agent.should_retire():
                del self.agents[agent_id]

    def _record_snapshot(self, action: str, decisions: list) -> None:
        living = list(self.agents.values())
        mean_conf = sum(a.belief.confidence for a in living) / max(1, len(living))
        energy = sum(a.resources.energy for a in living)
        active = sum(1 for d in decisions if d.active)
        self.snapshots.append(
            ColonySnapshot(
                timestep=self.timestep,
                living_agents=len(living),
                active_agents=active,
                chosen_action=action,
                mean_confidence=mean_conf,
                total_energy=energy,
                communication_overhead=len(self.bus.pending),
            )
        )
