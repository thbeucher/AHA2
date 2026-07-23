# ASSUMPTIONS: Simplifications and Open Questions

Each implementation module should reference the assumptions it currently relies on.

## Active assumptions

- **A1 Explicit small beliefs:** Initial beliefs are structured dictionaries and
  human-readable predicates, not learned latent vectors.
- **A2 Scalar confidence is provisional:** Confidence and uncertainty are simple
  bounded scalars until richer Bayesian or imprecise-probability estimators exist.
- **A3 Local prediction quality:** Agents update reliability from their own
  prediction errors only.
- **A4 Eligibility trace causal credit:** Initial causal contribution is estimated
  by a decaying local eligibility trace. This is not a full counterfactual model.
- **A5 Resource economy is heuristic:** Energy costs are hand-specified in Phase 1.
- **A6 Consensus is transparent voting:** Colony action selection uses weighted
  proposals, not a learned policy.
- **A7 Memory compression is symbolic counting:** Semantic memory initially stores
  repeated episode signatures and outcome counts.
- **A8 Structural plasticity is conservative:** Phase 1 supports retirement hooks;
  splitting and merging are specified but not yet fully implemented.
- **A9 Toy environments first:** GridWorld is used for substrate validation before
  MiniGrid/AnimalAI integration.

## Open questions

- What belief language balances interpretability and expressivity?
- How should agents form genuinely novel hypotheses rather than only specialize
  templates?
- What local causal estimators are publication-worthy beyond eligibility traces?
- How can communication protocols evolve without collapsing into central control?
- Which metrics best distinguish AHA from reinforcement learning baselines?
