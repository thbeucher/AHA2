from aha.agents.hypothesis_agent import HypothesisAgent
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
