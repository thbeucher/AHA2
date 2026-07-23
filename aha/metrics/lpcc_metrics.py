"""Research metrics for the LPCC architecture.

All metrics are transparent and computed from inspectable state. The emergence
score in particular is NOT a vague scalar; it is a documented combination of:
  1. distributed contribution (no single cell dominates the useful behavior),
  2. interaction dependence (ablating communication changes behavior),
  3. absence of centralized encoding (no cell's hypothesis equals the full rule),
each measured explicitly and reported separately as well as combined.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any

from aha.colony.colony import LPCCColony


@dataclass
class PredictionMetrics:
    accuracy: float
    calibration_error: float
    n_evaluated: int


def prediction_metrics(colony: LPCCColony) -> PredictionMetrics:
    """Aggregate prediction accuracy + a simple calibration error.

    Calibration: for each cell, |confidence - realised_accuracy|, averaged over
    cells with evaluated predictions. Well-calibrated cells => low error.
    """
    accs: list[float] = []
    cal: list[float] = []
    n_eval = 0
    for c in colony.cells.values():
        ps = c.state.prediction_statistics
        if ps.evaluated > 0:
            accs.append(ps.accuracy)
            cal.append(abs(c.hypothesis.confidence - ps.accuracy))
            n_eval += ps.evaluated
    return PredictionMetrics(
        accuracy=mean(accs) if accs else 0.0,
        calibration_error=mean(cal) if cal else 1.0,
        n_evaluated=n_eval,
    )


def hypothesis_diversity(colony: LPCCColony) -> float:
    """Fraction of distinct (context|prediction|action) signatures among cells."""
    sigs = set()
    total = 0
    for c in colony.living_cells():
        total += 1
        sig = (
            frozenset(c.hypothesis.context_tokens),
            frozenset(c.hypothesis.predicted_tokens),
            c.hypothesis.action,
        )
        sigs.add(sig)
    return len(sigs) / total if total else 0.0


def hypothesis_redundancy(colony: LPCCColony) -> float:
    """1 - diversity: fraction of cells that duplicate another's signature."""
    return 1.0 - hypothesis_diversity(colony)


def behavioral_stability(actions: list[str], window: int = 50) -> float:
    """Fraction of the most common action in the final window (1.0 = fully stable)."""
    if not actions:
        return 0.0
    tail = actions[-window:]
    counts: dict[str, int] = {}
    for a in tail:
        counts[a] = counts.get(a, 0) + 1
    return max(counts.values()) / len(tail)


def behavioral_complexity(actions: list[str], window: int = 50) -> float:
    """Normalized entropy of the action distribution in the final window."""
    if not actions:
        return 0.0
    import math

    tail = actions[-window:]
    counts: dict[str, int] = {}
    for a in tail:
        counts[a] = counts.get(a, 0) + 1
    n = len(tail)
    probs = [c / n for c in counts.values()]
    ent = -sum(p * math.log(p, 2) for p in probs)
    max_ent = math.log(len(counts), 2) if len(counts) > 1 else 1.0
    return ent / max_ent if max_ent > 0 else 0.0


def distributed_contribution(colony: LPCCColony) -> float:
    """How distributed is predictive labor across cells? (Gini-like, 1=distributed).

    Returns 1 - normalized concentration of evaluated-prediction counts. If one
    cell does all the useful prediction, contribution ~0; if evenly spread, ~1.
    """
    counts = [
        c.state.prediction_statistics.correct
        for c in colony.living_cells()
        if c.state.prediction_statistics.evaluated > 0
    ]
    counts = [c for c in counts if c > 0]
    if len(counts) <= 1:
        return 0.0
    total = sum(counts)
    shares = [c / total for c in counts]
    # Herfindahl concentration -> distribution.
    hhi = sum(s * s for s in shares)
    n = len(counts)
    min_hhi = 1.0 / n
    return max(0.0, min(1.0, (1.0 - hhi) / (1.0 - min_hhi)))


def centralized_encoding_absence(colony: LPCCColony, full_rule_tokens: set) -> float:
    """1.0 if NO single cell encodes the complete rule; lower if one nearly does.

    full_rule_tokens: set of tokens representing the complete antecedent+consequent.
    Measures 1 - max over cells of coverage(cell) where coverage is the fraction
    of full_rule_tokens present in that cell's context+prediction.
    """
    if not full_rule_tokens:
        return 1.0
    max_cov = 0.0
    for c in colony.living_cells():
        cell_tokens = c.hypothesis.context_tokens | c.hypothesis.predicted_tokens
        cov = len(full_rule_tokens & cell_tokens) / len(full_rule_tokens)
        max_cov = max(max_cov, cov)
    return 1.0 - max_cov


@dataclass
class EmergenceScore:
    distributed_contribution: float
    interaction_dependence: float
    centralized_encoding_absence: float
    combined: float

    def as_dict(self) -> dict[str, float]:
        return {
            "distributed_contribution": round(self.distributed_contribution, 4),
            "interaction_dependence": round(self.interaction_dependence, 4),
            "centralized_encoding_absence": round(self.centralized_encoding_absence, 4),
            "combined": round(self.combined, 4),
        }


def emergence_score(
    colony: LPCCColony,
    full_rule_tokens: set,
    interaction_dependence: float,
) -> EmergenceScore:
    """Combine the three transparent components.

    interaction_dependence must be supplied by an ablation comparison (e.g.
    relative reward drop when communication is disabled), in [0,1]. This makes
    the emergence claim FALSIFIABLE: if disabling interactions does not change
    behavior, interaction_dependence ~ 0 and emergence is low.
    """
    dc = distributed_contribution(colony)
    ce = centralized_encoding_absence(colony, full_rule_tokens)
    combined = (dc + interaction_dependence + ce) / 3.0
    return EmergenceScore(dc, interaction_dependence, ce, combined)


def summarize(colony: LPCCColony, actions: list[str]) -> dict[str, Any]:
    pm = prediction_metrics(colony)
    return {
        "prediction_accuracy": round(pm.accuracy, 4),
        "prediction_calibration_error": round(pm.calibration_error, 4),
        "predictions_evaluated": pm.n_evaluated,
        "hypothesis_diversity": round(hypothesis_diversity(colony), 4),
        "hypothesis_redundancy": round(hypothesis_redundancy(colony), 4),
        "behavioral_stability": round(behavioral_stability(actions), 4),
        "behavioral_complexity": round(behavioral_complexity(actions), 4),
        "distributed_contribution": round(distributed_contribution(colony), 4),
        "population": len(colony.cells),
        "living": len(colony.living_cells()),
        "structural_events": colony.structural_event_count,
    }