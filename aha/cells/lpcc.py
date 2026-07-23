"""The Local Predictive Causal Cell (LPCC).

This is the primary novel building block. It is NOT a neuron and NOT a
parameter-optimized unit. It maintains an evolving HYPOTHESIS about the world
and runs a closed temporal loop:

  temporal context
    -> pattern resonance
    -> internal hypothesis activation
    -> prediction (temporally aligned)
    -> action tendency
    -> [future observation]
    -> local (indirect) evidence
    -> belief update
    -> structural adaptation

No backpropagation, no gradient descent, no global loss. All updates are local
and use local + indirect evidence. Every mechanism is toggleable for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aha.causal.influence import CausalInfluenceEstimator
from aha.cells.config import LPCCConfig
from aha.cells.patterns import SymbolicTemporalDetector, TemporalSignature
from aha.cells.state import ConnectionState, LifecycleState, LPCCState
from aha.hypotheses.hypothesis import LocalHypothesis
from aha.prediction.ledger import EvaluatedPrediction
from aha.prediction.prediction import LocalPrediction


@dataclass
class ActionTendency:
    """A local action proposal for population consensus (NOT a policy net)."""

    cell_id: str
    action: str
    expected_utility: float
    expected_information_gain: float
    confidence: float
    uncertainty: float
    risk: float
    causal_influence: float

    @property
    def vote_weight(self) -> float:
        """Transparent consensus weight. Explained, not learned."""
        value = (
            self.expected_utility
            + 0.5 * self.expected_information_gain
            + 0.5 * max(0.0, self.causal_influence)
            - self.risk
        )
        return max(0.0, self.confidence) * value


@dataclass
class _PendingTrace:
    """A record of a past activation eligible for delayed credit."""

    hypothesis_id: str
    action: str | None
    context_bucket: int
    trace: float


class LPCC:
    """A single Local Predictive Causal Cell.

    Owns its inspectable LPCCState, one primary LocalHypothesis, a causal
    influence estimator, a pattern detector, and a list of pending eligibility
    traces for delayed credit.
    """

    def __init__(
        self,
        cell_id: str,
        hypothesis: LocalHypothesis,
        config: LPCCConfig | None = None,
        detector: SymbolicTemporalDetector | None = None,
        parent_id: str | None = None,
    ) -> None:
        self.config = config or LPCCConfig()
        self.detector = detector or SymbolicTemporalDetector()
        self.hypothesis = hypothesis
        self.state = LPCCState(
            cell_id=cell_id,
            energy=self.config.initial_energy,
            activation_threshold=self.config.activation_threshold,
            homeostatic_activity_level=self.config.homeostatic_target,
            parent_id=parent_id,
            lifecycle_state=LifecycleState.CANDIDATE,
        )
        self.causal = CausalInfluenceEstimator()
        self._pending: list[_PendingTrace] = []
        self._last_signature: TemporalSignature | None = None

    # ------------------------------------------------------------------ #
    # Perception + activation
    # ------------------------------------------------------------------ #
    def sense(
        self,
        current_obs: dict[str, Any],
        recent_obs: list[dict[str, Any]],
        recent_actions: list[str],
        recent_reward: float,
    ) -> float:
        """Compute resonance and update leaky membrane + activation."""
        sig = self.detector.signature(
            current_obs,
            recent_obs,
            recent_actions,
            recent_reward,
            self.state.activation_state,
        )
        self._last_signature = sig
        resonance = self.detector.resonance(self.hypothesis.context_tokens, sig)

        decay = self.config.membrane_decay if self.config.enable_homeostasis else 0.0
        self.state.membrane_like_state = (
            decay * self.state.membrane_like_state + (1.0 - decay) * resonance
        )

        activation = (
            1.0 if self.state.membrane_like_state >= self.state.activation_threshold else 0.0
        )
        self.state.activation_state = activation
        self.state.age += 1
        self.state.record_activity(activation)

        if self.config.enable_novelty:
            self.state.novelty = (
                max(0.0, 1.0 - resonance) if self.hypothesis.is_defined() else 1.0
            )

        if self.config.enable_energy:
            self.state.energy -= (
                self.config.activation_cost * activation + self.config.memory_cost
            )
        return activation

    def predict(self, now: int) -> LocalPrediction | None:
        """If active, emit a temporally-aligned prediction. Otherwise None."""
        if self.state.activation_state < 0.5 or not self.hypothesis.is_defined():
            return None
        if self.config.enable_energy:
            self.state.energy -= self.config.prediction_cost

        horizon = max(1, min(self.config.max_horizon, self.hypothesis.horizon))
        pred = LocalPrediction(
            source_cell_id=self.state.cell_id,
            hypothesis_id=self.hypothesis.hypothesis_id,
            creation_time=now,
            horizon=horizon,
            predicted_tokens=set(self.hypothesis.predicted_tokens),
            predicted_reward=self.hypothesis.confidence * 0.5,
            predicted_controllability=self.state.reliability * self.hypothesis.confidence,
            uncertainty=self.hypothesis.uncertainty,
            prediction_context=set(self._last_signature.tokens) if self._last_signature else set(),
            action=self.hypothesis.action,
        )
        self.state.prediction_statistics.created += 1
        self.state.expected_future = {"tokens": len(pred.predicted_tokens), "horizon": horizon}
        return pred

    def action_tendency(self) -> ActionTendency | None:
        """Propose an action if active and the hypothesis has an action.

        Utility is driven primarily by REWARD-linked evidence, not by raw token
        prediction confidence. A cell whose action reliably precedes reward (via
        delayed-reward eligibility evidence and/or positive causal influence)
        proposes strongly even when its token prediction is noisy (e.g. transient
        consequences). This is what lets discovered rules actually drive behavior.
        """
        if self.state.activation_state < 0.5 or self.hypothesis.action is None:
            return None
        ctx_bucket = self._context_bucket()
        causal = 0.0
        if self.config.enable_causal:
            causal = self.causal.influence(ctx_bucket, self.hypothesis.action)
        self.state.causal_influence_estimate = causal

        # Reward-linked drive: normalised delayed-reward evidence in [0,1].
        rew_ev = self.hypothesis.evidence.delayed_reward_evidence
        reward_drive = rew_ev / (1.0 + abs(rew_ev)) if rew_ev > 0 else 0.0

        # Expected utility blends reward drive, causal influence and (weakly)
        # prediction confidence. Reward drive dominates so useful actions win.
        expected_utility = (
            0.6 * reward_drive
            + 0.3 * max(0.0, causal)
            + 0.1 * self.hypothesis.confidence
        )
        return ActionTendency(
            cell_id=self.state.cell_id,
            action=self.hypothesis.action,
            expected_utility=expected_utility,
            expected_information_gain=self.hypothesis.uncertainty * self.state.novelty,
            confidence=max(self.hypothesis.confidence, reward_drive, max(0.0, causal)),
            uncertainty=self.hypothesis.uncertainty,
            risk=0.5 * self.hypothesis.uncertainty * (1.0 - self.state.reliability),
            causal_influence=causal,
        )

    def register_action_taken(self, action_taken: str | None) -> None:
        """Record an eligibility trace for the action the colony actually took.

        The trace carries DELAYED credit. It decays over time and is consumed
        when future evidence arrives. This is NOT backpropagation.
        """
        if self.state.activation_state < 0.5:
            self._decay_pending()
            return
        self.state.eligibility_trace = (
            self.config.eligibility_decay * self.state.eligibility_trace
            + self.config.eligibility_gain
        )
        self._pending.append(
            _PendingTrace(
                hypothesis_id=self.hypothesis.hypothesis_id,
                action=action_taken,
                context_bucket=self._context_bucket(),
                trace=1.0,
            )
        )
        self._decay_pending()

    # ------------------------------------------------------------------ #
    # Evidence + belief update (indirect error)
    # ------------------------------------------------------------------ #
    def receive_prediction_outcome(self, ev: EvaluatedPrediction) -> None:
        """Indirect evidence: our own aligned prediction was evaluated.

        No oracle says 'you are wrong'; the cell only observes whether the
        predicted future tokens actually occurred at target time.
        """
        if ev.prediction.hypothesis_id != self.hypothesis.hypothesis_id:
            return
        self.hypothesis.record_outcome(ev.hit_fraction, ev.supported)
        self.state.prediction_statistics.evaluated += 1
        self.state.prediction_statistics.error_sum += 1.0 - ev.hit_fraction
        if ev.supported:
            self.state.prediction_statistics.correct += 1

        if self.config.enable_causal and ev.prediction.action is not None:
            bucket = CausalInfluenceEstimator.bucket(frozenset(ev.prediction.prediction_context))
            self.causal.observe(bucket, ev.prediction.action, ev.hit_fraction)

        if self.config.enable_energy:
            self.state.energy += self.config.predictive_reward_scale * ev.hit_fraction

        self._sync_state()

    def receive_delayed_reward(self, reward: float) -> None:
        """Neuromodulatory-like delayed signal, assigned via eligibility trace.

        Credit is distributed to the hypothesis/action proportionally to the
        (decayed) eligibility trace, NOT via backpropagation.
        """
        if not self.config.enable_delayed_credit or abs(reward) < 1e-9:
            self._decay_pending()
            return
        for pt in self._pending:
            credit = reward * pt.trace
            if pt.hypothesis_id == self.hypothesis.hypothesis_id:
                self.hypothesis.add_delayed_reward_evidence(credit)
                if self.config.enable_causal and pt.action is not None:
                    self.causal.observe(pt.context_bucket, pt.action, max(0.0, credit))
                    infl = self.causal.influence(pt.context_bucket, pt.action)
                    self.hypothesis.add_causal_evidence(infl)
            if self.config.enable_energy:
                self.state.energy += self.config.external_reward_scale * credit
        self._decay_pending()
        self._sync_state()

    # ------------------------------------------------------------------ #
    # Homeostasis + lifecycle + connections
    # ------------------------------------------------------------------ #
    def homeostatic_update(self) -> None:
        """Regulate excitability toward the preferred activity range."""
        if not self.config.enable_homeostasis:
            return
        mean_act = self.state.mean_recent_activity
        target = self.state.homeostatic_activity_level
        # Too active -> raise threshold; too quiet -> lower threshold.
        delta = self.config.threshold_adapt_rate * (mean_act - target)
        self.state.activation_threshold = max(0.05, min(0.95, self.state.activation_threshold + delta))

    def update_connection(self, target_cell_id: str, useful: bool) -> None:
        """Local plastic link update (no global gradient)."""
        conn = self.state.outgoing_connections.get(target_cell_id)
        if conn is None:
            conn = ConnectionState(target_cell_id=target_cell_id)
            self.state.outgoing_connections[target_cell_id] = conn
        conn.update(useful=useful, amount=self.config.connection_learn_rate)

    def update_lifecycle(self) -> None:
        """Advance the structural lifecycle based on local evidence.

        CANDIDATE -> TESTING -> STABLE -> SPECIALIZING -> DECLINING -> RETIRED.
        Every transition has an explicit, logged reason. No random growth.
        """
        s = self.state
        conf = self.hypothesis.confidence
        n_pred = self.hypothesis.evidence.prediction_count
        prev = s.lifecycle_state

        if s.energy <= self.config.retire_energy:
            s.lifecycle_state = LifecycleState.RETIRED
        elif prev == LifecycleState.CANDIDATE and s.age >= self.config.testing_after:
            s.lifecycle_state = LifecycleState.TESTING
        elif prev == LifecycleState.TESTING:
            if conf >= self.config.stable_confidence and n_pred >= self.config.stable_min_predictions:
                s.lifecycle_state = LifecycleState.STABLE
            elif conf < self.config.decline_confidence and n_pred >= self.config.stable_min_predictions:
                s.lifecycle_state = LifecycleState.DECLINING
        elif prev == LifecycleState.STABLE:
            if conf < self.config.decline_confidence:
                s.lifecycle_state = LifecycleState.DECLINING
            elif self.should_split():
                s.lifecycle_state = LifecycleState.SPECIALIZING
        elif prev == LifecycleState.SPECIALIZING:
            if conf < self.config.decline_confidence:
                s.lifecycle_state = LifecycleState.DECLINING
            else:
                s.lifecycle_state = LifecycleState.STABLE
        elif prev == LifecycleState.DECLINING:
            if conf >= self.config.stable_confidence:
                s.lifecycle_state = LifecycleState.STABLE

        if s.lifecycle_state != prev:
            s.structural_event_log.append(
                f"t_age={s.age}:{prev}->{s.lifecycle_state}(conf={conf:.2f},n={n_pred},E={s.energy:.2f})"
            )

    def should_split(self) -> bool:
        """A confident-but-still-uncertain cell should specialize into a child.

        Reason: the hypothesis is useful (confident) yet residual uncertainty
        suggests the context is too broad and a sub-pattern exists.
        """
        if not self.config.enable_structural_plasticity:
            return False
        if len(self.state.child_ids) >= self.config.max_children:
            return False
        if self.state.energy <= self.config.growth_cost + self.config.retire_energy:
            return False
        return (
            self.hypothesis.confidence >= self.config.split_confidence
            and self.hypothesis.uncertainty >= self.config.split_uncertainty
            and self.hypothesis.evidence.prediction_count >= self.config.split_min_predictions
        )

    def should_retire(self) -> bool:
        return self.state.lifecycle_state == LifecycleState.RETIRED or not self.state.alive

    def spend_growth_energy(self) -> None:
        if self.config.enable_energy:
            self.state.energy -= self.config.growth_cost

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _context_bucket(self) -> int:
        tokens = frozenset(self._last_signature.tokens) if self._last_signature else frozenset()
        return CausalInfluenceEstimator.bucket(tokens)

    def _decay_pending(self) -> None:
        for pt in self._pending:
            pt.trace *= self.config.eligibility_decay
        self.state.eligibility_trace *= self.config.eligibility_decay
        # Drop negligible traces to bound memory.
        self._pending = [pt for pt in self._pending if pt.trace > 0.02]

    def _sync_state(self) -> None:
        """Mirror hypothesis-derived quantities into the inspectable state."""
        self.state.confidence = self.hypothesis.confidence
        self.state.uncertainty = self.hypothesis.uncertainty
        self.state.evidence = self.hypothesis.evidence
        acc = self.state.prediction_statistics.accuracy
        self.state.reliability = max(0.0, min(1.0, 0.5 * self.state.reliability + 0.5 * acc))

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        return {
            "cell_id": self.state.cell_id,
            "lifecycle": str(self.state.lifecycle_state),
            "age": self.state.age,
            "energy": round(self.state.energy, 3),
            "activation": self.state.activation_state,
            "threshold": round(self.state.activation_threshold, 3),
            "confidence": round(self.hypothesis.confidence, 3),
            "uncertainty": round(self.hypothesis.uncertainty, 3),
            "reliability": round(self.state.reliability, 3),
            "novelty": round(self.state.novelty, 3),
            "causal": round(self.state.causal_influence_estimate, 3),
            "pred_acc": round(self.state.prediction_statistics.accuracy, 3),
            "action": self.hypothesis.action,
            "hypothesis": self.hypothesis.summary(),
        }
