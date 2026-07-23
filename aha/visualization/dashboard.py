"""Live dashboard primitives for AHA colony state.

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
