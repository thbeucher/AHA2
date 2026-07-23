from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES: dict[str, str] = {
    "pyproject.toml": '''[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aha-project-vg"
version = "0.1.0"
description = "Autonomous Hypothesis Agents: a non-gradient research substrate for embodied intelligence."
requires-python = ">=3.12"
dependencies = [
  "numpy>=1.26",
  "networkx>=3.2",
  "gymnasium>=0.29",
  "minigrid>=2.3",
  "matplotlib>=3.8",
  "rich>=13.7",
  "typer>=0.12",
  "pydantic>=2.6",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.5"]

[project.scripts]
aha-gridworld = "aha.experiments.run_gridworld:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.ruff]
line-length = 100
target-version = "py312"
''',
    "requirements.txt": '''numpy>=1.26
networkx>=3.2
gymnasium>=0.29
minigrid>=2.3
matplotlib>=3.8
rich>=13.7
typer>=0.12
pydantic>=2.6
pytest>=8.0
''',
    "README.md": '''# Autonomous Hypothesis Agents (AHA)

AHA is a research framework for studying whether intelligence can emerge from a
population of autonomous local hypothesis agents rather than from global
gradient-based parameter optimization.

The elementary computational object is a **Hypothesis Agent**: a persistent,
stateful local model-builder that observes, predicts, intervenes, evaluates
evidence, communicates, specializes, merges, and retires.

The core implementation intentionally avoids policy networks, computational
graphs, global losses, and backpropagation.

## Phase status

This repository currently implements **Phase 1 / early Phase 2**:

- research specification documents,
- high-level architecture,
- explicit assumptions ledger,
- minimal package structure,
- explicit dataclass-based belief/state/message/memory structures,
- a minimal local Hypothesis Agent,
- a minimal Colony coordinator,
- a deterministic toy GridWorld for smoke experiments,
- tests proving the substrate compiles and performs local updates.

Later phases should extend behavior only after each phase compiles and tests pass.

## Quick start

```bash
cd aha_project_vg
python3 -m compileall aha tests
python3 -m pytest
python3 -m aha.experiments.run_gridworld --steps 8
```

## Core rule

If a design choice increases locality, interpretability, continual adaptation,
structural plasticity, causal reasoning, sparse computation, hypothesis formation,
or autonomous self-organization, prefer it over conventional ML convenience.
''',
    "THEORY.md": '''# THEORY: Computational Principles of Autonomous Hypothesis Agents

## Central hypothesis

Intelligence may emerge from maintaining, testing, specializing, communicating,
and discarding explicit hypotheses rather than from optimizing opaque parameters
against a global scalar loss.

The elementary object is not a neuron. It is a **Hypothesis Agent (HA)**: a tiny
scientist with persistent state, explicit beliefs, bounded resources, memory,
predictions, causal estimates, and local survival pressure.

## Non-negotiable commitments

1. **No global loss in the core.** External reward is evidence, not the objective.
2. **No backpropagation in the core.** Learning is local belief revision.
3. **Beliefs are explicit models.** A belief must be inspectable and replaceable.
4. **Agents are active participants.** Agents propose interventions that test
   hypotheses, not merely activations.
5. **The colony is an ecosystem.** Computation is sparse, resource-bounded, and
   structurally plastic.
6. **Prediction and controllability are distinct.** Good forecasting is not
   equivalent to causal influence.
7. **Death is computation.** Retiring low-utility hypotheses is part of learning.

## Minimal local loop

Every timestep, an agent asks:

- What do I believe?
- How certain am I?
- What evidence supports or contradicts me?
- What should happen next?
- What intervention would best test my hypothesis?
- Should I communicate, split, merge, retire, or conserve energy?

## Research questions

- Can explicit local hypotheses support transfer and one-shot adaptation?
- Can dynamic topology reduce catastrophic forgetting?
- Can local causal credit produce useful interventions without policy gradients?
- Can resource pressure yield sparse computation without hand-coded sparsity?
- Which belief representations are interpretable enough for publication-quality
  analysis yet expressive enough for embodied environments?

## Phase 1 scope

Phase 1 establishes vocabulary, invariants, explicit state containers, and a
minimal executable substrate. It is not intended to solve MiniGrid.
''',
    "ARCHITECTURE.md": '''# ARCHITECTURE: Software Instantiation

## Package layout

```text
aha/
  agents/          Hypothesis Agent implementations
  beliefs/         Explicit belief and evidence representations
  communication/   Messages, routing, dynamic usefulness links
  core/            Shared types, config, colony orchestration
  economy/         Energy and resource accounting
  memory/          Episodic and semantic memory
  metrics/         Prediction, controllability, diversity, overhead
  scheduler/       Future activation/resource scheduling policies
  simulation/      Toy and embodied environments
  visualization/   Rich/matplotlib dashboard components
  experiments/     Reproducible experiment entry points
```

## Dependency direction

- Agents depend on beliefs, memory, communication, economy, and core types.
- Colony depends on agents and communication.
- Metrics observe public state; they do not mutate agents.
- Environments do not know about agents.
- Visualization reads snapshots only.

## Current executable loop

1. Environment emits an observation.
2. Colony routes pending messages.
3. Each living agent independently evaluates match quality.
4. Active agents simulate a prediction and propose an action.
5. Colony chooses an action by transparent consensus, not by a policy network.
6. Environment returns consequence.
7. Each agent updates only its own belief, confidence, uncertainty, memory,
   usefulness, causal trace, and energy.
8. Colony retires dead agents and records metrics.

## Assumption references

Every module docstring should reference entries in `ASSUMPTIONS.md`. The initial
code primarily implements A1-A9.
''',
    "ASSUMPTIONS.md": '''# ASSUMPTIONS: Simplifications and Open Questions

Each implementation module should reference the assumptions it currently relies on.

## Active assumptions

- **A1 Explicit small beliefs:** Initial beliefs are structured dictionaries and
  human-readable predicates, not learned latent vectors.
- **A2 Scalar confidence is provisional:** Confidence and uncertainty are simple
  bounded scalars until richer Bayesian or imprecise-probability estimators exist.
- **A3 Local prediction quality:** Agents update reliability from their own
  prediction errors only.
- **A4 Eligibility trace causal credit:** Initial causal contribution is estimated
  by a decaying local eligibility trace. This is not a full counterfactual model.
- **A5 Resource economy is heuristic:** Energy costs are hand-specified in Phase 1.
- **A6 Consensus is transparent voting:** Colony action selection uses weighted
  proposals, not a learned policy.
- **A7 Memory compression is symbolic counting:** Semantic memory initially stores
  repeated episode signatures and outcome counts.
- **A8 Structural plasticity is conservative:** Phase 1 supports retirement hooks;
  splitting and merging are specified but not yet fully implemented.
- **A9 Toy environments first:** GridWorld is used for substrate validation before
  MiniGrid/AnimalAI integration.

## Open questions

- What belief language balances interpretability and expressivity?
- How should agents form genuinely novel hypotheses rather than only specialize
  templates?
- What local causal estimators are publication-worthy beyond eligibility traces?
- How can communication protocols evolve without collapsing into central control?
- Which metrics best distinguish AHA from reinforcement learning baselines?
''',
    "configs/default.json": '''{
  "seed": 7,
  "prediction_horizon": 1,
  "initial_energy": 10.0,
  "thinking_cost": 0.05,
  "communication_cost": 0.02,
  "memory_cost": 0.01,
  "prediction_cost": 0.03,
  "retirement_energy": 0.1,
  "min_confidence": 0.0,
  "max_confidence": 1.0
}
''',
    "docs/PHASE_1_SPEC.md": '''# Phase 1 Research Specification

Phase 1 creates the conceptual and executable foundation only.

## Deliverables

- Theory, architecture, and assumptions documents.
- Explicit data structures for beliefs, evidence, predictions, action proposals,
  memories, messages, resources, and agent state.
- A minimal HA lifecycle with local updates and no gradients.
- A minimal colony with transparent consensus.
- Smoke tests and compilation checks.

## Non-goals

- Solving MiniGrid.
- Implementing deep learning baselines.
- Optimizing performance.
- Hiding beliefs in opaque embeddings.
''',
    "paper/OUTLINE.md": '''# Paper Outline Draft

1. Motivation: intelligence as hypothesis ecology rather than parameter fitting.
2. Related work: Bayesian program learning, active inference, multi-agent systems,
   predictive processing, structural plasticity, scientific discovery systems.
3. Formal object: Hypothesis Agent state and local update loop.
4. Colony dynamics: communication, resource economy, structural plasticity.
5. Experimental substrate: GridWorld, MiniGrid, AnimalAI.
6. Metrics: transfer, one-shot adaptation, forgetting, energy, diversity, causal
   estimation quality, communication overhead.
7. Failure modes and falsification criteria.
''',
    "aha/__init__.py": '''"""Autonomous Hypothesis Agents research substrate."""

__all__ = ["__version__"]
__version__ = "0.1.0"
''',
    "aha/core/types.py": '''"""Shared explicit types for the AHA substrate.

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
''',
    "aha/beliefs/base.py": '''"""Explicit belief representations for Hypothesis Agents.

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
''',
    "aha/memory/store.py": '''"""Agent-owned episodic and semantic memory.

Implements assumptions: A7.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Episode:
    """One local remembered prediction episode."""

    context: dict[str, Any]
    prediction: dict[str, Any] | None
    outcome: dict[str, Any]
    timestamp: int


@dataclass
class EpisodicMemory:
    """Bounded local episodic memory with simple forgetting."""

    capacity: int = 128
    episodes: deque[Episode] = field(default_factory=deque)

    def add(self, episode: Episode) -> None:
        self.episodes.append(episode)
        while len(self.episodes) > self.capacity:
            self.episodes.popleft()

    def retrieve_recent(self, limit: int = 5) -> list[Episode]:
        return list(self.episodes)[-limit:]

    def __len__(self) -> int:
        return len(self.episodes)


@dataclass
class SemanticMemory:
    """Symbolic compression of repeated episodes."""

    counts: Counter[str] = field(default_factory=Counter)

    def compress_episode(self, episode: Episode) -> None:
        signature = self._signature(episode.context, episode.outcome)
        self.counts[signature] += 1

    def retrieve_common(self, limit: int = 5) -> list[tuple[str, int]]:
        return self.counts.most_common(limit)

    @staticmethod
    def _signature(context: dict[str, Any], outcome: dict[str, Any]) -> str:
        ctx = tuple(sorted((str(k), str(v)) for k, v in context.items()))
        out = tuple(sorted((str(k), str(v)) for k, v in outcome.items()))
        return f"ctx={ctx}|out={out}"
''',
    "aha/communication/messages.py": '''"""Local communication messages and routing.

Implements assumptions: A6.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Literal


MessageKind = Literal["belief", "prediction", "urgency", "request"]


@dataclass(frozen=True)
class Message:
    """Inspectably communicated local hypothesis evidence."""

    sender_id: str
    receiver_id: str | None
    belief_id: str
    confidence: float
    prediction_summary: str
    urgency: float
    request: str | None = None
    kind: MessageKind = "belief"


@dataclass
class MessageBus:
    """Sparse message buffer with usefulness accounting."""

    pending: list[Message] = field(default_factory=list)
    usefulness: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))

    def publish(self, message: Message) -> None:
        self.pending.append(message)

    def collect_for(self, agent_id: str) -> list[Message]:
        mine = [m for m in self.pending if m.receiver_id in (None, agent_id)]
        self.pending = [m for m in self.pending if m.receiver_id not in (None, agent_id)]
        return mine

    def mark_useful(self, sender_id: str, receiver_id: str, amount: float = 0.1) -> None:
        self.usefulness[(sender_id, receiver_id)] += amount
''',
    "aha/economy/resources.py": '''"""Resource economy for sparse autonomous computation.

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
''',
    "aha/agents/hypothesis_agent.py": '''"""Stateful autonomous Hypothesis Agent.

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
''',
    "aha/core/colony.py": '''"""Colony orchestration without layers, gradients, or global loss.

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
''',
    "aha/metrics/colony_metrics.py": '''"""Metrics for studying colony behavior.

Implements assumptions: A2, A3, A4.
"""

from __future__ import annotations

from dataclasses import dataclass

from aha.core.colony import HypothesisColony


@dataclass(frozen=True)
class MetricsReport:
    """Initial evaluation summary for Phase 1 smoke experiments."""

    active_agent_count: int
    average_confidence: float
    belief_diversity: int
    total_energy: float
    communication_overhead: int


def summarize_colony(colony: HypothesisColony) -> MetricsReport:
    agents = list(colony.agents.values())
    return MetricsReport(
        active_agent_count=colony.snapshots[-1].active_agents if colony.snapshots else 0,
        average_confidence=sum(a.belief.confidence for a in agents) / max(1, len(agents)),
        belief_diversity=len({a.belief.description for a in agents}),
        total_energy=sum(a.resources.energy for a in agents),
        communication_overhead=len(colony.bus.pending),
    )
''',
    "aha/simulation/gridworld.py": '''"""Deterministic toy GridWorld for substrate smoke tests.

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
''',
    "aha/visualization/dashboard.py": '''"""Live dashboard primitives for AHA colony state.

Implements assumptions: A6 by displaying transparent consensus state.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from aha.core.colony import HypothesisColony


def render_colony_table(colony: HypothesisColony, console: Console | None = None) -> None:
    """Render a Phase-1 live table of local agent state."""
    console = console or Console()
    table = Table(title="AHA Colony")
    table.add_column("Agent")
    table.add_column("Belief")
    table.add_column("Confidence")
    table.add_column("Uncertainty")
    table.add_column("Energy")
    for agent in colony.agents.values():
        table.add_row(
            agent.agent_id,
            agent.belief.description,
            f"{agent.belief.confidence:.2f}",
            f"{agent.belief.uncertainty:.2f}",
            f"{agent.resources.energy:.2f}",
        )
    console.print(table)
''',
    "aha/experiments/run_gridworld.py": '''"""Run a minimal AHA GridWorld smoke experiment."""

from __future__ import annotations

import typer

from aha.agents.hypothesis_agent import HypothesisAgent
from aha.beliefs.base import StructuredBelief
from aha.core.colony import HypothesisColony
from aha.metrics.colony_metrics import summarize_colony
from aha.simulation.gridworld import GridWorld
from aha.visualization.dashboard import render_colony_table

app = typer.Typer(help="AHA toy experiments")


def make_seed_colony() -> HypothesisColony:
    """Create initial hand-specified hypotheses for Phase 1."""
    beliefs = [
        StructuredBelief(
            "b-near-key",
            "Movable useful object may be nearby",
            {"near_key": True},
            {"has_key": True},
        ),
        StructuredBelief(
            "b-door",
            "Door becomes useful after key",
            {"has_key": True},
            {"near_door": True},
            utility=0.2,
        ),
    ]
    agents = {f"ha-{i}": HypothesisAgent(f"ha-{i}", belief) for i, belief in enumerate(beliefs)}
    return HypothesisColony(agents)


@app.command()
def main(steps: int = 8, dashboard: bool = False) -> None:
    env = GridWorld()
    colony = make_seed_colony()
    observation = env.observe()
    previous_action = None
    reward = 0.0
    for _ in range(steps):
        action = colony.step(observation, previous_action, reward)
        observation, reward = env.step(action)
        previous_action = action
    if dashboard:
        render_colony_table(colony)
    print(summarize_colony(colony))


if __name__ == "__main__":
    app()
''',
    "aha/scheduler/__init__.py": '''"""Activation and resource scheduling hooks for future phases."""
''',
    "aha/benchmarks/__init__.py": '''"""Benchmark interfaces; core AHA remains non-gradient."""
''',
    "aha/agents/__init__.py": '''"""Hypothesis Agent implementations."""
''',
    "aha/beliefs/__init__.py": '''"""Explicit belief representations."""
''',
    "aha/communication/__init__.py": '''"""Agent communication primitives."""
''',
    "aha/core/__init__.py": '''"""Core AHA orchestration and shared types."""
''',
    "aha/economy/__init__.py": '''"""Resource economy primitives."""
''',
    "aha/memory/__init__.py": '''"""Agent-owned memory systems."""
''',
    "aha/metrics/__init__.py": '''"""Research metrics for AHA colonies."""
''',
    "aha/simulation/__init__.py": '''"""Embodied simulation environments."""
''',
    "aha/visualization/__init__.py": '''"""Visualization and dashboard modules."""
''',
    "aha/experiments/__init__.py": '''"""Reproducible experiment entry points."""
''',
    "tests/test_agent.py": '''from aha.agents.hypothesis_agent import HypothesisAgent
from aha.beliefs.base import StructuredBelief
from aha.core.types import ObservationContext


def test_agent_updates_local_state_without_gradients():
    belief = StructuredBelief("b1", "near key", {"near_key": True}, {"has_key": True})
    agent = HypothesisAgent("ha1", belief)

    decision = agent.step(ObservationContext({"near_key": True, "has_key": False}, timestep=0), [])

    assert decision.active
    assert decision.prediction is not None
    assert agent.age == 1
    assert len(agent.episodic_memory) == 1
    assert agent.resources.energy < 10.0
''',
    "tests/test_colony.py": '''from aha.experiments.run_gridworld import make_seed_colony
from aha.simulation.gridworld import GridWorld


def test_colony_chooses_transparent_consensus_action():
    env = GridWorld()
    colony = make_seed_colony()

    action = colony.step(env.observe())

    assert action in {"observe", "investigate", "exploit"}
    assert colony.snapshots
    assert colony.snapshots[-1].living_agents >= 1
''',
    "tests/test_memory.py": '''from aha.memory.store import Episode, EpisodicMemory, SemanticMemory


def test_memory_retrieval_and_compression():
    episode = Episode({"x": 1}, {"x": 2}, {"x": 2}, 0)
    episodic = EpisodicMemory(capacity=2)
    semantic = SemanticMemory()

    episodic.add(episode)
    semantic.compress_episode(episode)

    assert episodic.retrieve_recent(1)[0] == episode
    assert semantic.retrieve_common(1)[0][1] == 1
''',
}


def main() -> None:
    for relative_path, content in FILES.items():
        path = ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print(f"created {len(FILES)} files under {ROOT}")


if __name__ == "__main__":
    main()