"""Run a minimal AHA GridWorld smoke experiment."""

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
