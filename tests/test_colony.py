from aha.experiments.run_gridworld import make_seed_colony
from aha.simulation.gridworld import GridWorld


def test_colony_chooses_transparent_consensus_action():
    env = GridWorld()
    colony = make_seed_colony()

    action = colony.step(env.observe())

    assert action in {"observe", "investigate", "exploit"}
    assert colony.snapshots
    assert colony.snapshots[-1].living_agents >= 1
