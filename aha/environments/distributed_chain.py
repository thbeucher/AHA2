"""Partially observable multi-stage causal-chain environment.

RESEARCH PURPOSE
================
Test whether a population of individually-limited LPCCs can solve a task that NO
single LPCC can observe or represent alone.

HIDDEN CAUSAL CHAIN (unknown to agents):
    stageA_action  at zone A  -> sets latent B     (delay dA)
    stageB_action  when B set  -> sets latent C     (delay dB)
    stageC_action  when C set  -> sets latent D + REWARD (delay dC)

STRICT INFORMATION BOTTLENECK (the crucial property):
    The environment exposes THREE disjoint observation views (zones). Each LPCC
    is assigned exactly one zone at construction and receives ONLY that zone's
    observation. No LPCC ever sees the whole state.

    - View A sees: {a_signal, a_local}         (whether stage-A antecedent holds)
    - View B sees: {b_signal, b_local}         (whether latent B holds)
    - View C sees: {c_signal, c_local}         (whether latent C holds)

    latent B/C/D are NOT directly in any single view unless produced by the
    previous stage. Crucially, a view-A cell cannot see whether B is set, so it
    cannot learn A->C->D; it can at best learn "my action changes something I
    cannot see". Only by RECEIVING a message from a view-B cell ("B is set /
    C became reachable") can downstream coordination complete the chain.

    We deliberately make the reward depend on the FULL ordered chain so that no
    single zone's local statistics reveal the action->reward contingency:
    reward only occurs if stageA, then stageB, then stageC actions happen in the
    correct temporal order across DIFFERENT zones.

SELECTIVITY / ENERGY:
    Every primitive action other than the correct stage action is a "wasted"
    action. The colony's energy economy already penalizes activation/prediction/
    communication, so indiscriminate acting is costly (fixing the Phase-1
    reward-farming limitation). We additionally emit a small negative reward for
    an incorrect stage action when that stage is *active*, so "always act" is not
    a strong baseline.

SURFACE SYMBOLS vary across seeds (token renaming) to support transfer tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from aha.environments.hidden_rules import StepResult


# Canonical (role) symbols; surface renamers below map these per seed.
ZONES = ("A", "B", "C")


@dataclass
class _SurfaceMap:
    """Per-seed surface renaming of role tokens -> arbitrary symbols."""

    signal_key: dict[str, str]
    on_val: dict[str, str]
    off_val: dict[str, str]
    action: dict[str, str]

    @staticmethod
    def canonical() -> "_SurfaceMap":
        return _SurfaceMap(
            signal_key={"A": "a_signal", "B": "b_signal", "C": "c_signal"},
            on_val={"A": "hi", "B": "hi", "C": "hi"},
            off_val={"A": "lo", "B": "lo", "C": "lo"},
            action={"A": "actA", "B": "actB", "C": "actC"},
        )

    @staticmethod
    def randomized(seed: int) -> "_SurfaceMap":
        rng = random.Random(seed)
        pool_keys = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "theta", "kappa"]
        pool_vals = ["p", "q", "r", "s", "t", "u", "v", "w"]
        pool_acts = ["do1", "do2", "do3", "do4", "do5", "do6"]
        rng.shuffle(pool_keys)
        rng.shuffle(pool_vals)
        rng.shuffle(pool_acts)
        return _SurfaceMap(
            signal_key={z: pool_keys[i] for i, z in enumerate(ZONES)},
            on_val={z: pool_vals[i] for i, z in enumerate(ZONES)},
            off_val={z: pool_vals[i + 3] for i, z in enumerate(ZONES)},
            action={z: pool_acts[i] for i, z in enumerate(ZONES)},
        )


class DistributedChainEnv:
    """Multi-stage partially observable causal chain.

    The environment is driven by ONE colony action per step, but that action is
    drawn from the union of all zone actions plus neutral actions. Because each
    LPCC only proposes actions for its own zone (it only sees its zone), the
    colony's chosen action reflects which zone's cells currently dominate — and
    getting the ORDER right across zones requires inter-zone coordination.
    """

    NEUTRAL_ACTIONS = ("wait", "observe")

    def __init__(
        self,
        seed: int = 0,
        delay: int = 1,
        episode_len: int = 40,
        stochastic: bool = False,
        reversal_episode: int | None = None,
        randomize_surface: bool = False,
        wrong_action_penalty: float = 0.05,
    ):
        self.rng = random.Random(seed)
        self.delay = delay
        self.episode_len = episode_len
        self.stochastic = stochastic
        self.reversal_episode = reversal_episode
        self.wrong_action_penalty = wrong_action_penalty
        self.surface = (
            _SurfaceMap.randomized(seed) if randomize_surface else _SurfaceMap.canonical()
        )
        self._episode = 0
        self._reset_state()

    # -- action vocabulary -------------------------------------------------- #
    @property
    def actions(self) -> list[str]:
        acts = [self.surface.action[z] for z in ZONES] + list(self.NEUTRAL_ACTIONS)
        return acts

    def zone_action(self, zone: str) -> str:
        return self.surface.action[zone]

    # -- causal-order handling (supports reversal) -------------------------- #
    @property
    def _stage_order(self) -> tuple[str, str, str]:
        """Order in which zone actions must be taken to complete the chain.

        Reversal swaps the required order of stages A and C, so previously
        correct behavior becomes wrong and must be re-learned.
        """
        if self.reversal_episode is not None and self._episode >= self.reversal_episode:
            return ("C", "B", "A")
        return ("A", "B", "C")

    # -- lifecycle ---------------------------------------------------------- #
    def _reset_state(self) -> None:
        self._t = 0
        # latent chain progress: how many correct ordered stages completed.
        self._progress = 0
        # per-zone signal that its stage is currently "actionable".
        self._active_zone_idx = 0  # index into _stage_order that is currently live
        self._pending: list[tuple[int, str]] = []  # (fire_time, event)
        self._rewarded = False

    def reset(self) -> dict[str, Any]:
        self._reset_state()
        return self.full_state()

    def next_episode(self) -> None:
        self._episode += 1

    # -- observation partitioning ------------------------------------------ #
    def full_state(self) -> dict[str, Any]:
        """Ground-truth full state (used ONLY by the complete-info baseline)."""
        order = self._stage_order
        live_zone = order[self._active_zone_idx] if self._active_zone_idx < len(order) else None
        state: dict[str, Any] = {}
        for z in ZONES:
            key = self.surface.signal_key[z]
            on = self.surface.on_val[z]
            off = self.surface.off_val[z]
            state[key] = on if z == live_zone else off
        return state

    def zone_view(self, zone: str) -> dict[str, Any]:
        """Strict bottleneck: return ONLY this zone's signal token.

        A view-`zone` LPCC sees whether ITS stage is currently live, plus a
        constant local token. It cannot see other zones' signals or the global
        progress, so it cannot observe the full chain.
        """
        key = self.surface.signal_key[zone]
        order = self._stage_order
        live_zone = order[self._active_zone_idx] if self._active_zone_idx < len(order) else None
        on = self.surface.on_val[zone]
        off = self.surface.off_val[zone]
        return {key: on if zone == live_zone else off}

    # -- dynamics ----------------------------------------------------------- #
    def step(self, action: str) -> StepResult:
        reward = 0.0
        # Resolve pending stage completions.
        fired = [ev for (ft, ev) in self._pending if self._t >= ft]
        self._pending = [(ft, ev) for (ft, ev) in self._pending if self._t < ft]
        for ev in fired:
            if ev == "advance":
                self._active_zone_idx += 1
            elif ev == "reward":
                reward += 1.0
                self._rewarded = True

        order = self._stage_order
        live_zone = order[self._active_zone_idx] if self._active_zone_idx < len(order) else None

        if live_zone is not None:
            correct_action = self.surface.action[live_zone]
            zone_actions = {self.surface.action[z] for z in ZONES}
            if action == correct_action:
                # Correct stage action: schedule advance (and reward if last).
                is_last = self._active_zone_idx == len(order) - 1
                if self.stochastic and self.rng.random() < 0.2:
                    pass  # 20% the stage fails to advance (stochastic version)
                else:
                    self._pending.append((self._t + self.delay, "advance"))
                    if is_last:
                        self._pending.append((self._t + self.delay, "reward"))
            elif action in zone_actions:
                # Wrong stage action while a stage is live: penalize (selectivity).
                reward -= self.wrong_action_penalty

        self._t += 1
        done = self._t >= self.episode_len or self._rewarded
        return StepResult(self.full_state(), reward, done, {"t": self._t, "progress": self._active_zone_idx})

    # -- scoring truth ------------------------------------------------------ #
    def hidden_rule_summary(self) -> dict[str, Any]:
        return {
            "chain_order": self._stage_order,
            "delay": self.delay,
            "reward_on_final_stage": 1.0,
            "note": "Each zone view sees only its own signal; full chain hidden.",
        }