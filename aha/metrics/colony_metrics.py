"""Metrics for studying colony behavior.

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
