"""Distributed colony for the partially-observable causal-chain task.

Each LPCC is assigned ONE zone and receives ONLY:
  - its zone view (bounded local observation),
  - bounded local temporal context,
  - delivered routed messages (folded into its context as msg:* tokens),
  - its own prediction history.

No LPCC receives the full state, the full chain, the full history, or all
messages. This is the strict information bottleneck required to test genuine
interaction-dependent emergence.

Message emission rule (local, not hard-coded solution):
  When a cell is active and its hypothesis predicts an observable change, it may
  broadcast a proposition token summarising its current activation, e.g.
  ("msg:from:<zone>", "active"). Downstream cells can mine these received tokens
  as CONTEXT for their own hypotheses. Thus the chain can only be completed if
  messages actually flow — enabling the ablations to have causal bite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from aha.cells.config import LPCCConfig
from aha.cells.lpcc import LPCC
from aha.cells.patterns import SymbolicTemporalDetector, Token, tokenize_observation
from aha.colony.consensus import resolve
from aha.communication.routed import CausalMessage, DeliveryRecord, MessageRouter, RouterConfig
from aha.hypotheses.candidate_generation import TransitionMiner
from aha.hypotheses.hypothesis import LocalHypothesis
from aha.prediction.ledger import PredictionLedger


@dataclass
class FlowEvent:
    """One edge in the information-flow graph."""

    t: int
    kind: str  # 'discover' | 'send' | 'receive' | 'action' | 'reward'
    cell_id: str | None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZoneCell:
    cell: LPCC
    zone: str
    miner: TransitionMiner
    recent_obs: list[dict[str, Any]] = field(default_factory=list)
    recent_actions: list[str] = field(default_factory=list)
    inbox_tokens: set[Token] = field(default_factory=set)
    prev_view_tokens: set[Token] = field(default_factory=set)


class DistributedColony:
    """Zone-partitioned LPCC population with routed communication."""

    def __init__(
        self,
        zones: list[str],
        zone_actions: dict[str, str],
        available_actions: list[str],
        config: LPCCConfig | None = None,
        router_config: RouterConfig | None = None,
        seed: int = 0,
        cells_per_zone: int = 4,
        max_population: int = 60,
    ):
        self.config = config or LPCCConfig()
        self.zones = list(zones)
        self.zone_actions = dict(zone_actions)
        self.available_actions = list(available_actions)
        self.rng = random.Random(seed)
        self.router = MessageRouter(router_config or RouterConfig(), seed=seed)
        self.ledger = PredictionLedger()
        self.detector = SymbolicTemporalDetector()
        self.timestep = 0
        self.max_population = max_population
        self._id_gen = count(1)
        self.zone_cells: dict[str, list[ZoneCell]] = {z: [] for z in zones}
        self.flow: list[FlowEvent] = []
        self.records: list[dict[str, Any]] = []
        self._structural_events = 0

        # Seed each zone with candidate cells whose action is the zone action.
        for z in zones:
            for _ in range(cells_per_zone):
                self._add_seed_cell(z)

    # ------------------------------------------------------------------ #
    def _add_seed_cell(self, zone: str) -> ZoneCell:
        hid = f"H{next(self._id_gen)}"
        # Start undefined; miner will define context/prediction from experience.
        hyp = LocalHypothesis(hypothesis_id=hid, action=self.zone_actions[zone], horizon=1)
        cid = f"{zone}{next(self._id_gen)}"
        cell = LPCC(cid, hyp, config=self.config, detector=self.detector)
        zc = ZoneCell(cell=cell, zone=zone, miner=TransitionMiner(min_support=2, max_delay=3))
        self.zone_cells[zone].append(zc)
        return zc

    def all_cells(self) -> list[ZoneCell]:
        return [zc for z in self.zones for zc in self.zone_cells[z]]

    def living(self) -> list[ZoneCell]:
        return [zc for zc in self.all_cells() if not zc.cell.should_retire()]

    # ------------------------------------------------------------------ #
    # Decision phase: each cell sees ONLY its zone view + inbox tokens.
    # ------------------------------------------------------------------ #
    def decide(self, zone_view_fn) -> str:
        """zone_view_fn(zone) -> dict observation for that zone only."""
        # 1. Deliver messages queued for this timestep.
        receiver_ids = [zc.cell.state.cell_id for zc in self.living()]
        delivered = self.router.deliver(self.timestep, receiver_ids)

        proposals = []
        active_records: list[tuple[ZoneCell, float]] = []

        for zc in self.living():
            view = zone_view_fn(zc.zone)
            # Fold delivered messages into this cell's local context tokens.
            msg_tokens: set[Token] = set()
            for m in delivered.get(zc.cell.state.cell_id, []):
                if not m.expired(self.timestep):
                    msg_tokens.add(m.proposition)
                    self.flow.append(FlowEvent(self.timestep, "receive", zc.cell.state.cell_id,
                                               {"from": m.sender_id, "prop": m.proposition}))
            zc.inbox_tokens = msg_tokens

            # Build an augmented observation: zone view + message tokens as
            # pseudo-observation features (so the detector's 'now:' signature
            # includes received propositions -> usable as hypothesis context).
            aug = dict(view)
            for (k, v) in msg_tokens:
                aug[k] = v

            tend_before = self._peek_tendency(zc.cell)
            act = zc.cell.sense(aug, zc.recent_obs, zc.recent_actions, 0.0)
            if act >= 0.5:
                pred = zc.cell.predict(self.timestep)
                if pred is not None:
                    self.ledger.add(pred)
                tend = zc.cell.action_tendency()
                if tend is not None:
                    proposals.append(tend)
                    active_records.append((zc, tend.vote_weight))
            zc._aug_obs = aug  # stash for outcome mining
            zc._tend_before = tend_before

        # 2. Consensus action.
        result = resolve(proposals, self.available_actions, self.rng,
                         exploration_rate=self.config.exploration_rate, default_action="observe")

        # 3. Register eligibility traces + emit messages from active cells.
        for zc in self.living():
            zc.cell.register_action_taken(result.chosen_action)
            self._maybe_emit(zc)

        self._last_result = result
        self._active_records = active_records
        self.flow.append(FlowEvent(self.timestep, "action", None, {"action": result.chosen_action}))
        return result.chosen_action

    def _peek_tendency(self, cell: LPCC) -> float:
        t = cell.action_tendency()
        return t.vote_weight if t else 0.0

    def _maybe_emit(self, zc: ZoneCell) -> None:
        """Active cells broadcast a proposition token summarising activation."""
        if zc.cell.state.activation_state < 0.5:
            return
        prop: Token = (f"msg:from:{zc.zone}", "active")
        msg = CausalMessage(
            sender_id=zc.cell.state.cell_id,
            hypothesis_id=zc.cell.hypothesis.hypothesis_id,
            proposition=prop,
            confidence=zc.cell.hypothesis.confidence,
            prediction=next(iter(zc.cell.hypothesis.predicted_tokens), None),
            timestamp=self.timestep,
            validity_horizon=3,
            uncertainty=zc.cell.hypothesis.uncertainty,
        )
        cost = self.router.send(msg, receiver_id=None, now=self.timestep)  # broadcast within router rules
        if cost and self.config.enable_energy:
            zc.cell.state.energy -= cost
        if cost:
            self.flow.append(FlowEvent(self.timestep, "send", zc.cell.state.cell_id, {"prop": prop}))

    # ------------------------------------------------------------------ #
    # Outcome phase.
    # ------------------------------------------------------------------ #
    def deliver_outcome(self, next_zone_view_fn, reward: float, chosen_action: str) -> None:
        now_after = self.timestep + 1

        # Evaluate aligned predictions per cell against its OWN next zone view
        # augmented with any still-valid inbox tokens (its local world).
        for zc in self.living():
            nv = next_zone_view_fn(zc.zone)
            aug = dict(nv)
            for (k, v) in zc.inbox_tokens:
                aug[k] = v
            observed = self._sig_tokens(zc, aug)
            for ev in self.ledger.evaluate_due(now_after, observed, reward):
                cell = self._find_cell(ev.prediction.source_cell_id)
                if cell is not None:
                    cell.receive_prediction_outcome(ev)

        # Delayed reward broadcast (each cell assigns credit via its own traces).
        if abs(reward) > 1e-9:
            for zc in self.living():
                zc.cell.receive_delayed_reward(reward)
            self.flow.append(FlowEvent(now_after, "reward", None, {"reward": reward}))

        # Per-cell mining on its OWN augmented view (bottleneck preserved).
        struct = 0
        for zc in self.living():
            nv = next_zone_view_fn(zc.zone)
            aug_next = dict(nv)
            for (k, v) in zc.inbox_tokens:
                aug_next[k] = v
            prev_now = tokenize_observation(getattr(zc, "_aug_obs", {}), prefix="now")
            next_now = tokenize_observation(aug_next, prefix="now")
            zc.miner.observe(prev_now, chosen_action, next_now)
            zc.cell.homeostatic_update()
            prev_state = zc.cell.state.lifecycle_state
            zc.cell.update_lifecycle()
            if zc.cell.state.lifecycle_state != prev_state:
                struct += 1
            # update temporal buffers
            zc.recent_obs.append(aug_next)
            if len(zc.recent_obs) > 6:
                zc.recent_obs = zc.recent_obs[-6:]
            zc.recent_actions.append(chosen_action)
            if len(zc.recent_actions) > 6:
                zc.recent_actions = zc.recent_actions[-6:]

        # Recruitment: each cell's miner may define/refine its OWN hypothesis
        # (bounded to its zone; context may include received msg:* tokens).
        struct += self._recruit()
        self._structural_events += struct

        self.records.append({
            "t": self.timestep,
            "action": chosen_action,
            "reward": reward,
            "living": len(self.living()),
            "sent": self.router.sent_count,
            "delivered": self.router.delivered_count,
        })
        self.timestep = now_after

    # ------------------------------------------------------------------ #
    def _recruit(self) -> int:
        """Let each zone cell adopt a mined hypothesis if it is still undefined.

        Crucially, mined context tokens may include msg:from:* tokens, so a
        downstream cell can learn 'when I receive B-active AND my signal is live,
        my action advances things' — a rule that REQUIRES communication.
        """
        events = 0
        for zc in self.living():
            if zc.cell.hypothesis.is_defined():
                continue
            proposals = zc.miner.propose(set())
            # Rank proposals so the cell prefers (1) reward-predictive and
            # (2) message-conditioned hypotheses over trivial local transitions.
            # This lets a downstream cell adopt "received-msg + my-signal ->
            # reward" instead of a useless "signal-on -> signal-off" rule.
            def _rank(h) -> tuple:
                predicts_reward = ("rew:sign", "pos") in h.predicted_tokens
                msg_ctx = any(k.startswith("now:msg:") for (k, _v) in h.context_tokens)
                return (predicts_reward, msg_ctx)

            best = None
            best_key = (-1, -1)
            for h in proposals:
                h.action = zc.cell.hypothesis.action  # keep zone action
                pr, mc = _rank(h)
                key = (int(pr), int(mc))
                if key > best_key:
                    best_key = key
                    best = h
            if best is not None:
                # Preserve the cell's evidence-fresh identity but adopt structure.
                best.hypothesis_id = zc.cell.hypothesis.hypothesis_id
                zc.cell.hypothesis = best
                zc.miner.reset_counts()
                self.flow.append(FlowEvent(self.timestep, "discover", zc.cell.state.cell_id,
                                           {"ctx": sorted(best.context_tokens),
                                            "pred": sorted(best.predicted_tokens),
                                            "action": best.action}))
                events += 1
        return events

    def _sig_tokens(self, zc: ZoneCell, aug_obs: dict[str, Any]) -> set[Token]:
        sig = self.detector.signature(aug_obs, zc.recent_obs, zc.recent_actions, 0.0, 0.0)
        return set(sig.tokens)

    def _find_cell(self, cell_id: str) -> LPCC | None:
        for zc in self.all_cells():
            if zc.cell.state.cell_id == cell_id:
                return zc.cell
        return None

    # ------------------------------------------------------------------ #
    # Inspection / metrics support
    # ------------------------------------------------------------------ #
    @property
    def structural_event_count(self) -> int:
        return self._structural_events

    def zone_of(self, cell_id: str) -> str | None:
        for zc in self.all_cells():
            if zc.cell.state.cell_id == cell_id:
                return zc.zone
        return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "timestep": self.timestep,
            "living": len(self.living()),
            "sent": self.router.sent_count,
            "delivered": self.router.delivered_count,
            "structural_events": self._structural_events,
            "cells": [
                {**zc.cell.snapshot(), "zone": zc.zone} for zc in self.living()
            ],
        }

    def information_flow_edges(self) -> list[dict[str, Any]]:
        return [
            {"t": e.t, "kind": e.kind, "cell": e.cell_id, **e.detail} for e in self.flow
        ]
