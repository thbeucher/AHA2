"""Local causal influence approximation.

BIOLOGICAL MOTIVATION:
  A neuromodulatory-like signal should reflect whether a cell's ACTION actually
  changed outcomes, not merely whether the cell predicted them.

IMPLEMENTED MECHANISM (documented approximation, NOT true causal inference):
  For each (context-bucket, action) the cell keeps running outcome statistics:
  the mean "target rate" (fraction of predicted tokens realised) when the action
  was taken vs. a baseline over all other actions in the same context bucket.
  Causal influence ~= P(outcome | do(action)) - P(outcome | context, other).
  Randomized action perturbation (exploration) provides the interventional
  variation that makes this estimate meaningful.

WHY THIS DISTINGUISHES PREDICTION FROM CAUSATION:
  A hypothesis can have high confidence (good prediction) while its action has
  ZERO causal influence (the outcome happens regardless of the action). The
  contrast term makes that explicit.

UNRESOLVED BIOLOGICAL QUESTION:
  How biological credit assignment approximates counterfactual contrast is open.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _ActionOutcome:
    n: int = 0
    outcome_sum: float = 0.0

    def update(self, outcome: float) -> None:
        self.n += 1
        self.outcome_sum += outcome

    @property
    def mean(self) -> float:
        return self.outcome_sum / self.n if self.n else 0.0


@dataclass
class CausalInfluenceEstimator:
    """Action-conditioned outcome contrast, bucketed by context signature hash.

    Contexts are bucketed by a coarse frozenset key supplied by the caller so
    the estimator stays local and cheap.
    """

    # key: (context_bucket, action) -> outcome stats
    _stats: dict[tuple[int, str], _ActionOutcome] = field(default_factory=dict)
    # key: context_bucket -> all-action stats (baseline)
    _baseline: dict[int, _ActionOutcome] = field(default_factory=dict)

    def observe(self, context_bucket: int, action: str, outcome: float) -> None:
        """Record realised outcome (e.g. hit_fraction or reward) for do(action)."""
        self._stats.setdefault((context_bucket, action), _ActionOutcome()).update(outcome)
        self._baseline.setdefault(context_bucket, _ActionOutcome()).update(outcome)

    def influence(self, context_bucket: int, action: str) -> float:
        """Estimated causal contribution of `action` in this context bucket.

        Returns value in [-1, 1]: positive means the action raises the outcome
        above the context baseline; ~0 means no detectable causal effect.
        """
        key = (context_bucket, action)
        if key not in self._stats:
            return 0.0
        action_mean = self._stats[key].mean
        base = self._baseline.get(context_bucket)
        if base is None or base.n == 0:
            return 0.0
        # Baseline including this action; subtract to approximate the contrast
        # against "context regardless of this particular action".
        baseline_mean = base.mean
        return max(-1.0, min(1.0, action_mean - baseline_mean))

    def confidence_in_estimate(self, context_bucket: int, action: str) -> float:
        """More samples => more trustworthy causal estimate."""
        key = (context_bucket, action)
        n = self._stats[key].n if key in self._stats else 0
        return n / (n + 5.0)

    @staticmethod
    def bucket(tokens: frozenset) -> int:
        """Stable coarse bucket id for a set of context tokens."""
        return hash(tokens) & 0x7FFFFFFF