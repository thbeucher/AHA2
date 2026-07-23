"""Configuration for LPCC dynamics and colony behavior.

Every mechanism has an explicit toggle so ablations (Phase 12) can disable a
single mechanism and measure the causal effect on system behavior.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LPCCConfig:
    # --- activation / membrane ---
    activation_threshold: float = 0.45
    membrane_decay: float = 0.6  # leaky integration of resonance
    homeostatic_target: float = 0.15  # preferred mean activity
    threshold_adapt_rate: float = 0.02

    # --- eligibility trace (indirect delayed credit) ---
    eligibility_decay: float = 0.75
    eligibility_gain: float = 1.0

    # --- prediction ---
    default_horizon: int = 1
    max_horizon: int = 6

    # --- energy economy ---
    initial_energy: float = 10.0
    activation_cost: float = 0.03
    prediction_cost: float = 0.02
    communication_cost: float = 0.01
    memory_cost: float = 0.005
    growth_cost: float = 1.0
    predictive_reward_scale: float = 0.15
    causal_reward_scale: float = 0.2
    external_reward_scale: float = 0.3

    # --- plasticity ---
    connection_learn_rate: float = 0.05
    connection_decay: float = 0.9

    # --- lifecycle / structural plasticity ---
    testing_after: int = 5
    stable_confidence: float = 0.6
    stable_min_predictions: int = 8
    decline_confidence: float = 0.12
    retire_energy: float = 0.1
    split_confidence: float = 0.65
    split_uncertainty: float = 0.5
    split_min_predictions: int = 12
    max_children: int = 3

    # --- ablation toggles ---
    enable_communication: bool = True
    enable_delayed_credit: bool = True
    enable_causal: bool = True
    enable_structural_plasticity: bool = True
    enable_energy: bool = True
    enable_novelty: bool = True
    enable_homeostasis: bool = True

    # --- exploration (interventional variation for causal estimation) ---
    exploration_rate: float = 0.1