# THEORY: Computational Principles of Autonomous Hypothesis Agents

## Central hypothesis

Intelligence may emerge from maintaining, testing, specializing, communicating,
and discarding explicit hypotheses rather than from optimizing opaque parameters
against a global scalar loss.

The elementary object is not a neuron. It is a **Hypothesis Agent (HA)**: a tiny
scientist with persistent state, explicit beliefs, bounded resources, memory,
predictions, causal estimates, and local survival pressure.

## Non-negotiable commitments

1. **No global loss in the core.** External reward is evidence, not the objective.
2. **No backpropagation in the core.** Learning is local belief revision.
3. **Beliefs are explicit models.** A belief must be inspectable and replaceable.
4. **Agents are active participants.** Agents propose interventions that test
   hypotheses, not merely activations.
5. **The colony is an ecosystem.** Computation is sparse, resource-bounded, and
   structurally plastic.
6. **Prediction and controllability are distinct.** Good forecasting is not
   equivalent to causal influence.
7. **Death is computation.** Retiring low-utility hypotheses is part of learning.

## Minimal local loop

Every timestep, an agent asks:

- What do I believe?
- How certain am I?
- What evidence supports or contradicts me?
- What should happen next?
- What intervention would best test my hypothesis?
- Should I communicate, split, merge, retire, or conserve energy?

## Research questions

- Can explicit local hypotheses support transfer and one-shot adaptation?
- Can dynamic topology reduce catastrophic forgetting?
- Can local causal credit produce useful interventions without policy gradients?
- Can resource pressure yield sparse computation without hand-coded sparsity?
- Which belief representations are interpretable enough for publication-quality
  analysis yet expressive enough for embodied environments?

## Phase 1 scope

Phase 1 establishes vocabulary, invariants, explicit state containers, and a
minimal executable substrate. It is not intended to solve MiniGrid.
