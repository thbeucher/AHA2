"""Tests for the distributed emergence substrate: strict bottleneck, routed
communication mechanics, information-flow logging, and reproducibility."""

from __future__ import annotations

from aha.cells.config import LPCCConfig
from aha.colony.distributed_colony import DistributedColony
from aha.communication.routed import CausalMessage, MessageRouter, RouterConfig
from aha.environments.distributed_chain import ZONES, DistributedChainEnv


# --- environment bottleneck ------------------------------------------------ #
def test_zone_view_is_strict_subset():
    env = DistributedChainEnv(seed=0)
    env.reset()
    full = env.full_state()
    for z in ZONES:
        view = env.zone_view(z)
        # A zone view exposes exactly one key (its own signal).
        assert len(view) == 1
        # And that key is a subset of the full state.
        for k in view:
            assert k in full


def test_no_single_zone_sees_full_chain():
    env = DistributedChainEnv(seed=0)
    env.reset()
    seen = set()
    for z in ZONES:
        seen |= set(env.zone_view(z).keys())
    # Union of all zone views equals full state, but each individual view is 1 key.
    assert seen == set(env.full_state().keys())
    assert all(len(env.zone_view(z)) == 1 for z in ZONES)


def test_chain_requires_ordered_actions():
    env = DistributedChainEnv(seed=0, delay=1, episode_len=40)
    env.reset()
    order = env._stage_order
    total = 0.0
    # Take the correct ordered actions with waits to let delays resolve.
    for z in order:
        a = env.zone_action(z)
        for _ in range(3):
            step = env.step(a)
            total += step.reward
            if step.done:
                break
    assert total > 0.5  # completing the chain in order yields reward


def test_wrong_action_is_penalized():
    env = DistributedChainEnv(seed=0, delay=1, wrong_action_penalty=0.1)
    env.reset()
    order = env._stage_order
    # Take a WRONG zone action while stage A is live.
    wrong_zone = order[1]  # not the first-stage zone
    step = env.step(env.zone_action(wrong_zone))
    assert step.reward <= 0.0


# --- router mechanics ------------------------------------------------------ #
def _msg(now: int) -> CausalMessage:
    return CausalMessage(
        sender_id="A1", hypothesis_id="H1", proposition=("msg:from:A", "active"),
        confidence=0.8, prediction=None, timestamp=now, validity_horizon=3, uncertainty=0.2,
    )


def test_router_delay_delivery():
    r = MessageRouter(RouterConfig(enabled=True, delay=2), seed=0)
    r.send(_msg(0), None, now=0)
    assert r.deliver(1, ["B1"]) == {"B1": []}  # not yet
    out = r.deliver(2, ["B1"])
    assert len(out["B1"]) == 1


def test_router_disabled_delivers_nothing():
    r = MessageRouter(RouterConfig(enabled=False), seed=0)
    cost = r.send(_msg(0), None, now=0)
    assert cost == 0.0
    assert r.deliver(1, ["B1"]) == {"B1": []}


def test_router_bandwidth_limit():
    r = MessageRouter(RouterConfig(enabled=True, delay=1, bandwidth=1), seed=0)
    r.send(_msg(0), None, now=0)
    r.send(_msg(0), None, now=0)
    r.send(_msg(0), None, now=0)
    out = r.deliver(1, ["B1"])
    assert len(out["B1"]) == 1  # capped


def test_router_randomize_changes_proposition():
    r = MessageRouter(RouterConfig(enabled=True, delay=1, randomize=True), seed=0)
    original = _msg(0)
    r.send(original, None, now=0)
    out = r.deliver(1, ["B1"])["B1"]
    assert len(out) == 1
    assert out[0].proposition != original.proposition or out[0].hypothesis_id == "RANDOM"


# --- distributed colony ---------------------------------------------------- #
def test_distributed_colony_bottleneck_and_flow():
    env = DistributedChainEnv(seed=0, delay=1, episode_len=20)
    zone_actions = {z: env.zone_action(z) for z in ZONES}
    colony = DistributedColony(
        zones=list(ZONES), zone_actions=zone_actions,
        available_actions=env.actions, config=LPCCConfig(enable_energy=False),
        router_config=RouterConfig(enabled=True, delay=1), seed=0,
    )
    env.reset()
    for _ in range(20):
        action = colony.decide(env.zone_view)
        step = env.step(action)
        colony.deliver_outcome(env.zone_view, step.reward, action)
        if step.done:
            break
    # Information-flow log records action events at minimum.
    edges = colony.information_flow_edges()
    assert any(e["kind"] == "action" for e in edges)


def test_distributed_deterministic_given_seed():
    def run():
        env = DistributedChainEnv(seed=5, delay=1, episode_len=20)
        za = {z: env.zone_action(z) for z in ZONES}
        col = DistributedColony(list(ZONES), za, env.actions,
                                config=LPCCConfig(), router_config=RouterConfig(), seed=5)
        env.reset()
        total = 0.0
        for _ in range(40):
            a = col.decide(env.zone_view)
            s = env.step(a)
            col.deliver_outcome(env.zone_view, s.reward, a)
            total += s.reward
            if s.done:
                env.reset()
        return total

    assert run() == run()