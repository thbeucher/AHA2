"""Tests for the LPCC substrate: state transitions, prediction alignment,
delayed eligibility credit, causal estimation, homeostasis, plasticity,
lifecycle, consensus, and ablation wiring."""

from __future__ import annotations

from aha.causal.influence import CausalInfluenceEstimator
from aha.cells.config import LPCCConfig
from aha.cells.lpcc import LPCC, ActionTendency
from aha.cells.patterns import SymbolicTemporalDetector
from aha.cells.state import LifecycleState
from aha.colony.colony import LPCCColony
from aha.colony.consensus import resolve
from aha.hypotheses.candidate_generation import TransitionMiner, specialize
from aha.hypotheses.hypothesis import LocalHypothesis
from aha.prediction.ledger import PredictionLedger
from aha.prediction.prediction import LocalPrediction


def make_cell(**cfg_over):
    hyp = LocalHypothesis(
        hypothesis_id="H1",
        context_tokens={("now:switch", "present")},
        action="investigate",
        predicted_tokens={("now:door", "open")},
        horizon=2,
    )
    return LPCC("C1", hyp, config=LPCCConfig(**cfg_over))


# --- pattern activation ---------------------------------------------------- #
def test_resonance_and_activation():
    cell = make_cell(membrane_decay=0.0)  # no leak: instant activation
    act = cell.sense({"switch": "present"}, [], [], 0.0)
    assert act == 1.0
    cell2 = make_cell(membrane_decay=0.0)
    act2 = cell2.sense({"switch": "absent"}, [], [], 0.0)
    assert act2 == 0.0


def test_undefined_hypothesis_does_not_resonate():
    hyp = LocalHypothesis("H0")  # empty
    cell = LPCC("C0", hyp, config=LPCCConfig(membrane_decay=0.0))
    act = cell.sense({"switch": "present"}, [], [], 0.0)
    assert act == 0.0


# --- prediction creation + alignment --------------------------------------- #
def test_prediction_created_when_active():
    cell = make_cell(membrane_decay=0.0)
    cell.sense({"switch": "present"}, [], [], 0.0)
    pred = cell.predict(now=5)
    assert pred is not None
    assert pred.creation_time == 5
    assert pred.horizon == 2
    assert pred.target_time == 7


def test_ledger_temporal_alignment():
    ledger = PredictionLedger(support_threshold=0.5)
    pred = LocalPrediction("C1", "H1", creation_time=3, horizon=2,
                           predicted_tokens={("now:door", "open")})
    ledger.add(pred)
    # Not due before target.
    assert ledger.evaluate_due(4, {("now:door", "open")}, 0.0) == []
    # Due exactly at target_time=5.
    evals = ledger.evaluate_due(5, {("now:door", "open")}, 1.0)
    assert len(evals) == 1
    assert evals[0].supported
    assert evals[0].hit_fraction == 1.0


def test_contradiction_reduces_confidence():
    cell = make_cell(membrane_decay=0.0)
    ledger = PredictionLedger()
    # Confirm several times.
    for t in range(6):
        cell.sense({"switch": "present"}, [], [], 0.0)
        p = cell.predict(now=t)
        ledger.add(p)
        for ev in ledger.evaluate_due(p.target_time, {("now:door", "open")}, 0.0):
            cell.receive_prediction_outcome(ev)
    conf_after_support = cell.hypothesis.confidence
    # Now contradict repeatedly.
    for t in range(10, 30):
        cell.sense({"switch": "present"}, [], [], 0.0)
        p = cell.predict(now=t)
        ledger.add(p)
        for ev in ledger.evaluate_due(p.target_time, {("now:door", "closed")}, 0.0):
            cell.receive_prediction_outcome(ev)
    assert cell.hypothesis.confidence < conf_after_support
    assert cell.hypothesis.evidence.contradiction_streak > 0


# --- delayed eligibility credit (no backprop) ------------------------------ #
def test_delayed_credit_via_eligibility_trace():
    cell = make_cell(membrane_decay=0.0, enable_energy=False)
    cell.sense({"switch": "present"}, [], [], 0.0)
    cell.register_action_taken("investigate")
    # Reward arrives several steps later; trace has decayed but is nonzero.
    cell.receive_delayed_reward(1.0)
    assert cell.hypothesis.evidence.delayed_reward_evidence > 0.0


def test_delayed_credit_disabled_ablation():
    cell = make_cell(membrane_decay=0.0, enable_delayed_credit=False, enable_energy=False)
    cell.sense({"switch": "present"}, [], [], 0.0)
    cell.register_action_taken("investigate")
    cell.receive_delayed_reward(1.0)
    assert cell.hypothesis.evidence.delayed_reward_evidence == 0.0


# --- causal influence ------------------------------------------------------ #
def test_causal_influence_distinguishes_action():
    est = CausalInfluenceEstimator()
    b = CausalInfluenceEstimator.bucket(frozenset({("now:switch", "present")}))
    # Action A reliably yields outcome 1.0; action B yields 0.0.
    for _ in range(10):
        est.observe(b, "investigate", 1.0)
        est.observe(b, "wait", 0.0)
    assert est.influence(b, "investigate") > est.influence(b, "wait")


# --- homeostasis ----------------------------------------------------------- #
def test_homeostasis_raises_threshold_when_overactive():
    cell = make_cell()
    for _ in range(40):
        cell.state.record_activity(1.0)
    before = cell.state.activation_threshold
    cell.homeostatic_update()
    assert cell.state.activation_threshold > before


# --- connection plasticity ------------------------------------------------- #
def test_connection_plasticity_local():
    cell = make_cell()
    cell.update_connection("C2", useful=True)
    s1 = cell.state.outgoing_connections["C2"].strength
    for _ in range(5):
        cell.update_connection("C2", useful=True)
    s2 = cell.state.outgoing_connections["C2"].strength
    assert s2 >= s1


# --- lifecycle + retirement ------------------------------------------------ #
def test_lifecycle_progression_and_retire_on_energy():
    cell = make_cell()
    cell.state.age = 10
    cell.update_lifecycle()
    assert cell.state.lifecycle_state in (LifecycleState.TESTING, LifecycleState.CANDIDATE)
    cell.state.energy = 0.0
    cell.update_lifecycle()
    assert cell.state.lifecycle_state == LifecycleState.RETIRED
    assert cell.should_retire()


def test_specialize_inherits_documented_subset():
    parent = LocalHypothesis("HP", {("now:switch", "present")}, "investigate",
                             {("now:door", "open")}, horizon=2)
    child = specialize(parent, ("now:distractor", "red"), "HC")
    assert ("now:switch", "present") in child.context_tokens
    assert ("now:distractor", "red") in child.context_tokens
    assert child.action == parent.action
    assert child.predicted_tokens == parent.predicted_tokens
    # Child starts with fresh evidence.
    assert child.evidence.prediction_count == 0


# --- transition miner ------------------------------------------------------ #
def test_miner_discovers_delayed_action_rule():
    miner = TransitionMiner(min_support=2, max_delay=3)
    # switch present + investigate -> (delay 2) door open, repeated.
    for _ in range(5):
        miner.observe({("now:switch", "present")}, "investigate", {("now:switch", "present")})
        miner.observe({("now:switch", "absent")}, "wait", {("now:switch", "absent")})
        miner.observe({("now:switch", "absent")}, "wait", {("now:door", "open"), ("now:switch", "absent")})
    proposals = miner.propose(set())
    assert any(("now:door", "open") in p.predicted_tokens for p in proposals)


# --- consensus ------------------------------------------------------------- #
def test_consensus_prefers_higher_vote_weight():
    import random
    t_hi = ActionTendency("C1", "investigate", 1.0, 0.0, 0.9, 0.1, 0.0, 0.5)
    t_lo = ActionTendency("C2", "wait", 0.1, 0.0, 0.2, 0.1, 0.0, 0.0)
    res = resolve([t_hi, t_lo], ["investigate", "wait", "observe"],
                  random.Random(0), exploration_rate=0.0)
    assert res.chosen_action == "investigate"


def test_consensus_default_when_no_proposals():
    import random
    res = resolve([], ["a", "b"], random.Random(0), exploration_rate=0.0, default_action="observe")
    assert res.chosen_action == "observe"


# --- colony end-to-end determinism ----------------------------------------- #
def test_colony_deterministic_given_seed():
    from aha.environments.hidden_rules import HiddenSwitchEnv
    from aha.experiments.runner import run_episodes

    def run():
        env = HiddenSwitchEnv(seed=3, delay=2, episode_len=15)
        colony = LPCCColony(env.actions, config=LPCCConfig(), seed=3)
        return run_episodes(colony, env, 30).total_reward

    assert run() == run()


def test_colony_discovers_rule_like_hypothesis():
    from aha.environments.hidden_rules import HiddenSwitchEnv
    from aha.experiments.runner import run_episodes

    env = HiddenSwitchEnv(seed=1, delay=2, episode_len=20)
    colony = LPCCColony(env.actions, config=LPCCConfig(), seed=1)
    run_episodes(colony, env, 120)
    found = False
    for c in colony.living_cells():
        h = c.hypothesis
        ctx_has_switch = any("switch" in str(t) for t in h.context_tokens)
        pred_has_door = any("door" in str(t) for t in h.predicted_tokens)
        if ctx_has_switch and pred_has_door and h.action == "investigate":
            found = True
    assert found, "colony did not recruit a switch+investigate->door hypothesis"
