"""LPCCColony: the population-level orchestrator.

Per-step loop (all local, no global loss/backprop):

  1. Each living LPCC senses the current context (temporal signature).
  2. Active cells emit temporally-aligned predictions -> ledger.
  3. Active cells emit action tendencies.
  4. Consensus resolves the colony action (with epsilon exploration).
  5. Each cell registers an eligibility trace for the chosen action.
  6. Environment step happens OUTSIDE (caller), producing next obs + reward.
  7. deliver_outcome(): ledger evaluates predictions due now; each source cell
     receives its aligned outcome; delayed reward is broadcast and assigned via
     eligibility traces; communication support propagates; miner learns; homeostasis,
     lifecycle, structural plasticity, and retirement run.

The colony NEVER fits all cells jointly. Each cell updates from its own local +
indirect evidence.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from aha.cells.config import LPCCConfig
from aha.cells.lpcc import LPCC
from aha.cells.patterns import SymbolicTemporalDetector, Token, tokenize_observation
from aha.cells.state import LifecycleState
from aha.colony.consensus import ConsensusResult, resolve
from aha.hypotheses.candidate_generation import TransitionMiner, specialize
from aha.hypotheses.hypothesis import LocalHypothesis
from aha.prediction.ledger import PredictionLedger


@dataclass
class ColonyStepRecord:
    """Auditable record of one colony timestep."""

    timestep: int
    chosen_action: str
    was_exploration: bool
    n_living: int
    n_active: int
    n_proposals: int
    mean_confidence: float
    total_energy: float
    outstanding_predictions: int
    messages: int
    reward: float
    structural_events: int


class LPCCColony:
    """A dynamic society of Local Predictive Causal Cells."""

    def __init__(
        self,
        available_actions: list[str],
        config: LPCCConfig | None = None,
        seed: int = 0,
        detector: SymbolicTemporalDetector | None = None,
        max_population: int = 64,
    ) -> None:
        self.config = config or LPCCConfig()
        self.available_actions = list(available_actions)
        self.rng = random.Random(seed)
        self.detector = detector or SymbolicTemporalDetector()
        self.ledger = PredictionLedger()
        self.miner = TransitionMiner(min_support=3)
        self.cells: dict[str, LPCC] = {}
        self.timestep = 0
        self.records: list[ColonyStepRecord] = []
        self.max_population = max_population
        self._id_gen = count(1)
        self._structural_event_count = 0

        # Temporal bookkeeping for miner + prediction context.
        self._recent_obs: list[dict[str, Any]] = []
        self._recent_actions: list[str] = []
        self._prev_signature_tokens: set[Token] = set()
        self._prev_action: str | None = None
        self._last_proposals: list = []
        self._last_active: list[str] = []

    # ------------------------------------------------------------------ #
    # Population management
    # ------------------------------------------------------------------ #
    def add_cell(self, hypothesis: LocalHypothesis, parent_id: str | None = None) -> LPCC:
        cid = f"C{next(self._id_gen)}"
        cell = LPCC(
            cell_id=cid,
            hypothesis=hypothesis,
            config=self.config,
            detector=self.detector,
            parent_id=parent_id,
        )
        self.cells[cid] = cell
        return cell

    def living_cells(self) -> list[LPCC]:
        return [c for c in self.cells.values() if not c.should_retire()]

    # ------------------------------------------------------------------ #
    # Decision phase
    # ------------------------------------------------------------------ #
    def decide(self, observation: dict[str, Any]) -> ConsensusResult:
        """Sense, predict, propose, and resolve a consensus action."""
        recent_reward = 0.0
        active_ids: list[str] = []
        proposals = []

        for cell in self.living_cells():
            act = cell.sense(observation, self._recent_obs, self._recent_actions, recent_reward)
            if act >= 0.5:
                active_ids.append(cell.state.cell_id)
                pred = cell.predict(self.timestep)
                if pred is not None:
                    self.ledger.add(pred)
                tend = cell.action_tendency()
                if tend is not None:
                    proposals.append(tend)

        result = resolve(
            proposals,
            self.available_actions,
            self.rng,
            exploration_rate=self.config.exploration_rate,
            default_action="observe",
        )

        # Register eligibility traces for the chosen action.
        for cell in self.living_cells():
            cell.register_action_taken(result.chosen_action)

        self._last_proposals = proposals
        self._last_active = active_ids
        self._prev_signature_tokens = tokenize_observation(observation, prefix="now")
        self._prev_action = result.chosen_action
        return result

    # ------------------------------------------------------------------ #
    # Outcome phase
    # ------------------------------------------------------------------ #
    def deliver_outcome(
        self,
        next_observation: dict[str, Any],
        reward: float,
        chosen_action: str,
    ) -> None:
        """Feed back the consequence of the chosen action.

        Called after the environment has stepped. Advances colony time.
        """
        now_after = self.timestep + 1
        observed_tokens = self._signature_tokens(next_observation)

        # 1. Evaluate temporally-aligned predictions due at now_after.
        evaluated = self.ledger.evaluate_due(now_after, observed_tokens, reward)
        for ev in evaluated:
            cell = self.cells.get(ev.prediction.source_cell_id)
            if cell is not None:
                cell.receive_prediction_outcome(ev)

        # 2. Broadcast delayed reward; each cell assigns credit via its traces.
        if abs(reward) > 1e-9:
            for cell in self.living_cells():
                cell.receive_delayed_reward(reward)

        # 3. Communication: cells confirming the outcome reinforce plastic links
        #    to co-active neighbors (temporal co-activation + prediction success).
        messages = 0
        if self.config.enable_communication:
            messages = self._propagate_communication(evaluated)

        # 4. Mining: learn transition statistics for candidate recruitment.
        next_now_tokens = tokenize_observation(next_observation, prefix="now")
        self.miner.observe(self._prev_signature_tokens, chosen_action, next_now_tokens)

        # 5. Homeostasis + lifecycle for every cell.
        struct_events = 0
        for cell in self.living_cells():
            cell.homeostatic_update()
            prev_state = cell.state.lifecycle_state
            cell.update_lifecycle()
            if cell.state.lifecycle_state != prev_state:
                struct_events += 1

        # 6. Structural plasticity: splits + recruitment (bounded population).
        if self.config.enable_structural_plasticity:
            struct_events += self._structural_plasticity(next_now_tokens)

        # 7. Retirement.
        self._retire()

        self._structural_event_count += struct_events

        # 8. Update temporal buffers + advance time.
        self._recent_obs.append(dict(next_observation))
        if len(self._recent_obs) > 8:
            self._recent_obs = self._recent_obs[-8:]
        self._recent_actions.append(chosen_action)
        if len(self._recent_actions) > 8:
            self._recent_actions = self._recent_actions[-8:]

        self._record(chosen_action, reward, messages, struct_events)
        self.timestep = now_after

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _signature_tokens(self, observation: dict[str, Any]) -> set[Token]:
        """Full temporal signature tokens for prediction evaluation."""
        sig = self.detector.signature(
            observation,
            self._recent_obs,
            self._recent_actions,
            0.0,
            0.0,
        )
        return set(sig.tokens)

    def _propagate_communication(self, evaluated) -> int:
        """Reinforce plastic links between co-active, mutually-successful cells.

        Local rule: if two cells were active this step and at least one produced
        a supported prediction, strengthen their directed link (useful); else
        weaken. This is temporal co-activation + prediction-usefulness, not a
        gradient.
        """
        supported_sources = {
            ev.prediction.source_cell_id for ev in evaluated if ev.supported
        }
        active = [self.cells[cid] for cid in self._last_active if cid in self.cells]
        messages = 0
        for a in active:
            for b in active:
                if a.state.cell_id == b.state.cell_id:
                    continue
                useful = a.state.cell_id in supported_sources or b.state.cell_id in supported_sources
                a.update_connection(b.state.cell_id, useful=useful)
                messages += 1
        return messages

    def _existing_signatures(self) -> set[frozenset[Token]]:
        sigs: set[frozenset[Token]] = set()
        for c in self.cells.values():
            sigs.add(frozenset(c.hypothesis.context_tokens | c.hypothesis.predicted_tokens))
        return sigs

    def _structural_plasticity(self, next_now_tokens: set[Token]) -> int:
        events = 0
        # (a) Splits: specialize confident-but-uncertain STABLE cells.
        for cell in list(self.living_cells()):
            if cell.state.lifecycle_state != LifecycleState.SPECIALIZING:
                continue
            if len(self.cells) >= self.max_population:
                break
            # Choose a discriminative extra context token from recent arrival.
            extra = self._pick_discriminative_token(cell, next_now_tokens)
            if extra is None:
                continue
            child_id = f"C{next(self._id_gen)}"
            child_hyp = specialize(cell.hypothesis, extra, child_id + "_H")
            cell.spend_growth_energy()
            child = self.add_cell(child_hyp, parent_id=cell.state.cell_id)
            cell.state.child_ids.append(child.state.cell_id)
            cell.state.structural_event_log.append(f"split->{child.state.cell_id} extra={extra}")
            events += 1

        # (b) Recruitment: mine well-supported transitions not yet covered.
        if len(self.cells) < self.max_population:
            proposals = self.miner.propose(self._existing_signatures())
            for hyp in proposals:
                if len(self.cells) >= self.max_population:
                    break
                self.add_cell(hyp)
                events += 1
            if proposals:
                self.miner.reset_counts()
        return events

    def _pick_discriminative_token(self, cell: LPCC, next_now_tokens: set[Token]) -> Token | None:
        candidates = [t for t in next_now_tokens if t not in cell.hypothesis.context_tokens]
        if not candidates:
            return None
        # Deterministic choice for reproducibility.
        return sorted(candidates)[0]

    def _retire(self) -> None:
        for cid, cell in list(self.cells.items()):
            if cell.should_retire():
                del self.cells[cid]

    def _record(self, action: str, reward: float, messages: int, struct_events: int) -> None:
        living = self.living_cells()
        mean_conf = (
            sum(c.hypothesis.confidence for c in living) / len(living) if living else 0.0
        )
        total_energy = sum(c.state.energy for c in living)
        self.records.append(
            ColonyStepRecord(
                timestep=self.timestep,
                chosen_action=action,
                was_exploration=False,
                n_living=len(living),
                n_active=len(self._last_active),
                n_proposals=len(self._last_proposals),
                mean_confidence=mean_conf,
                total_energy=total_energy,
                outstanding_predictions=self.ledger.outstanding(),
                messages=messages,
                reward=reward,
                structural_events=struct_events,
            )
        )

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #
    @property
    def structural_event_count(self) -> int:
        return self._structural_event_count

    def snapshot(self) -> dict[str, Any]:
        living = self.living_cells()
        return {
            "timestep": self.timestep,
            "population": len(self.cells),
            "living": len(living),
            "predictions_created": self.ledger.created_count,
            "predictions_evaluated": self.ledger.evaluated_count,
            "structural_events": self._structural_event_count,
            "cells": [c.snapshot() for c in living],
        }

    def best_hypotheses(self, top: int = 5) -> list[dict[str, Any]]:
        ranked = sorted(
            self.living_cells(),
            key=lambda c: (c.hypothesis.confidence, c.state.prediction_statistics.evaluated),
            reverse=True,
        )
        return [c.snapshot() for c in ranked[:top]]
