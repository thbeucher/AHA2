"""Population-level action consensus over local action tendencies.

The final action EMERGES from local proposals; there is no centralized policy
network. This module implements a transparent weighted vote plus an epsilon
exploration term that provides the interventional variation needed by the local
causal estimators.
"""

from __future__ import annotations

from dataclasses import dataclass

from aha.cells.lpcc import ActionTendency


@dataclass
class ConsensusResult:
    chosen_action: str
    scores: dict[str, float]
    was_exploration: bool
    n_proposals: int


def resolve(
    proposals: list[ActionTendency],
    available_actions: list[str],
    rng,
    exploration_rate: float = 0.1,
    default_action: str = "observe",
) -> ConsensusResult:
    """Resolve competing action tendencies transparently.

    - Each proposal contributes its vote_weight to its action.
    - With probability exploration_rate, pick a uniformly random available action
      (randomized perturbation => interventional data for causal estimation).
    - Ties broken deterministically by action name for reproducibility.
    """
    scores: dict[str, float] = {a: 0.0 for a in available_actions}
    for p in proposals:
        if p.action in scores:
            scores[p.action] += p.vote_weight
        else:
            scores[p.action] = p.vote_weight

    if rng.random() < exploration_rate and available_actions:
        action = available_actions[rng.randrange(len(available_actions))]
        return ConsensusResult(action, scores, True, len(proposals))

    if not proposals or all(v <= 0.0 for v in scores.values()):
        return ConsensusResult(default_action, scores, False, len(proposals))

    best = max(scores.items(), key=lambda kv: (kv[1], kv[0]))
    return ConsensusResult(best[0], scores, False, len(proposals))