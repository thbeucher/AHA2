# The Local Predictive Causal Cell (LPCC): A Scientific Report

**Project:** Autonomous Hypothesis Agents (AHA) — LPCC substrate
**Status:** Working implementation, tested, reproducible. Mixed empirical results
reported honestly, including one clear negative result (transfer) and one partial
failure (net task reward vs. trivial baselines).

---

## 0. Executive summary

We implemented a genuinely non-gradient, non-backprop computational substrate — a
population of **Local Predictive Causal Cells (LPCCs)** — and tested the central
hypothesis that goal-directed behavior can emerge from an *ecology of local
predictive/causal hypotheses* rather than from optimizing a global objective.

What works (measured, reproducible):

- LPCCs form explicit `context × action → future` hypotheses **from experience**,
  with confidence **derived from evidence** (not incremented arbitrarily).
- A strict **temporal prediction ledger** aligns every prediction created at `t`
  with horizon `h` to its evaluation at `t+h`.
- **Indirect delayed credit** via decaying eligibility traces measurably shapes
  behavior (ablation B changes reward), with **no backpropagation**.
- A documented **local causal-influence approximation** distinguishes
  "I predicted X" from "my action contributed to X".
- **Homeostasis, energy economy, plastic links, and a structural lifecycle**
  (CANDIDATE→TESTING→STABLE→SPECIALIZING→DECLINING→RETIRED) all run locally.
- The colony **discovers the hidden rule** `switch=present + investigate →(delay)
  door=open` — verified by a test asserting the recruited hypothesis exists.
- **Rule reversal**: the falsification test shows fast re-adaptation of the
  behavioral correct-action rate after the rule flips.

What does **not** work / remains open (reported, not hidden):

- On the simplest hidden-switch task, **net episodic reward does not beat a
  trivial random or fixed baseline**, because in that environment "act often"
  is nearly optimal and AHA's selectivity + exploration dilute reward. This is a
  task-design limitation as much as an architecture limitation.
- **Relational transfer is negative** (transfer_advantage ≈ 0 or slightly
  negative): the symbolic detector keys on exact surface tokens, so renamed
  worlds share no structure. We document exactly why and propose the fix.
- Emergence in the two-stage task shows `no single cell encodes the full rule`
  (coverage 0.667 < 1.0) and distributed contribution > 0, but
  **interaction_dependence ≈ 0** for communication in the current wiring, so we
  do **not** claim strong emergence yet.

---

## 1. What is the LPCC?

A LPCC is a persistent, stateful, biologically *inspired* computational process
(explicitly **not** a neuron and **not** a parameter-optimized unit). It maintains
an evolving hypothesis about the world and runs a **closed temporal loop**:

```
temporal context → pattern resonance → hypothesis activation → prediction
→ action tendency → [future observation] → local indirect evidence
→ belief update → structural adaptation
```

Its full state is inspectable (`aha/cells/state.py::LPCCState`): activation,
membrane-like leaky state, short/slow context, prototype/temporal pattern memory,
belief/confidence/uncertainty/reliability/novelty, expected future, action
tendencies, eligibility trace, causal-influence estimate, prediction statistics,
energy, homeostatic activity level, incoming/outgoing plastic connections,
parent/child ids, and lifecycle state.

## 2. How does it differ from an artificial neuron?

| Artificial neuron | LPCC |
|---|---|
| `input → weighted sum → activation` | `context → resonance → hypothesis → prediction → action → evidence → belief update → structure` |
| Weights fit by global gradient | Evidence accumulated locally; confidence derived from support/contradiction/error |
| Stateless per forward pass | Persistent internal state + eligibility traces |
| No explicit hypothesis | Explicit `C×A→F` hypothesis with auditable evidence |
| No lifecycle | CANDIDATE→…→RETIRED with logged, reasoned transitions |
| Trained jointly | Never fit jointly; each cell updates from its own local + indirect evidence |

## 3. What local information does it use?

Temporal signature tokens from: current observation, up to `window` recent
observations (time-tagged), the most recent action, a coarse reward-sign token,
and its own previous activation (`aha/cells/patterns.py`). No neural embedding.

## 4. How does indirect error reach it?

Three indirect channels, never a direct "you are wrong" label:

1. **Aligned prediction outcome** — did the predicted future tokens occur at
   `t+h`? (`PredictionLedger.evaluate_due`).
2. **Delayed reward via eligibility trace** — reward broadcast to the colony is
   assigned to each cell in proportion to its decaying trace
   (`LPCC.receive_delayed_reward`). Ablation B (disable) changes behavior.
3. **Communication** — co-active cells with successful predictions reinforce
   plastic links (`LPCCColony._propagate_communication`).

## 5. How is causal influence estimated?

A documented approximation (`aha/causal/influence.py`): per
`(context-bucket, action)` we track mean realised outcome and contrast it with the
context baseline. Randomized action perturbation (epsilon exploration in consensus)
supplies the interventional variation. This explicitly separates *prediction*
confidence from *causal* contribution — a cell can be confident yet causally
inert, and the contrast term exposes that. This is **not** claimed to be true
causal inference.

## 6. How does the system adapt structurally?

- **Recruitment**: a delay-aware, action-conditioned `TransitionMiner` proposes
  new hypotheses for well-supported, reasonably-precise antecedent→consequent
  transitions the population does not yet cover.
- **Specialization (split)**: a confident-but-uncertain STABLE cell narrows its
  context with one discriminative token; the child inherits a documented subset
  and re-earns confidence from scratch.
- **Retirement**: cells retire when energy is exhausted or the lifecycle reaches
  RETIRED. Every structural event is logged with a reason. No random growth.

## 7. What behaviors emerge?

- **Hidden-rule discovery** (verified by test): the colony recruits
  `switch=present + investigate →(h=2) door=open` purely from experience.
- **Reward-driven action selection**: once delayed-reward evidence accumulates on
  that hypothesis, its action tendency wins consensus (reward rose from 24→252 on
  seed 1 after we made action utility reward-linked rather than prediction-linked).
- **Fast reversal adaptation**: after the rule flips, the behavioral correct-rate
  recovers quickly (adaptation_time small in `results/reversal_seed0.json`).

We deliberately do **not** call the two-stage result "emergent" beyond what the
metrics support (see §0 and §9).

## 8. Which mechanisms are necessary?

From the ablation harness (`python -m aha.experiments.run ablations`), measured on
the hidden-switch task: **delayed credit (B)** and **energy (E)** produce the
largest behavioral deltas; communication/causal/structural/novelty/homeostasis
show small deltas *on this simple task* because the task does not require
multi-cell coordination. This is itself informative: mechanisms only matter when
the task demands them. The two-stage task is where communication *should* matter,
and measuring that dependence is the next experiment (§10).

## 9. What failed?

1. **Net reward vs. trivial baselines** on the simplest task — AHA's selective,
   exploration-diluted policy underperforms "always act". Honest interpretation:
   the environment's reward is too easy to farm by acting indiscriminately; it
   does not reward *selectivity*. Fix: penalize wasted actions / add a cost so
   selectivity pays.
2. **Relational transfer** — negative advantage. The symbolic detector matches
   exact tokens, so `switch/investigate/door` shares nothing with
   `gadget/prod/gate`. Fix: a *relational* detector that abstracts role tokens
   (antecedent/action/consequent) rather than surface identity.
3. **Communication interaction-dependence ≈ 0** in the two-stage task under the
   current wiring — so we cannot yet claim strong emergence. Fix: make Stage-2
   cells actually *depend* on Stage-1 cells' activation via routed messages.

None of these are hidden; the metrics and JSON in `results/` show them directly.

## 10. What would be the next scientific experiment?

1. **Selectivity-rewarding environment**: add an action cost so acting only when
   the antecedent is present strictly dominates. Predict AHA > random.
2. **Relational detector**: replace `SymbolicTemporalDetector` with a
   role-abstracting detector and re-run transfer. Predict transfer_advantage > 0
   iff structure (not surface) drives behavior.
3. **Message-routed two-stage coordination**: let Stage-2 cells consume Stage-1
   cells' "holding=yes" prediction as context; then ablate communication and
   measure the reward drop (interaction_dependence). Predict a large drop → a
   defensible emergence claim under the transparent, falsifiable emergence score.

---

## Reproducibility

```bash
cd AHA2
python3 -m pytest -q                       # 20 tests
python3 -m aha.experiments.run all --seed 0 --episodes 200
# writes results/{discovery,reversal,transfer,emergence,ablations}_seed0.json
```

All randomness is seeded; results are deterministic given `(experiment, seed)`
(asserted by `test_colony_deterministic_given_seed`).

## Biological framing

Throughout we separate **biological motivation** (local plasticity, eligibility
traces, neuromodulatory-like delayed signals, homeostasis, sparse activity,
structural plasticity) from **implemented mechanism** (explicit, count-based,
inspectable) from **unresolved biological question** (documented in each module
docstring). We claim only a *biologically inspired computational abstraction*.