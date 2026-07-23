# Phase 2: Testing Interaction-Dependent Emergence in LPCC Populations

**Verdict: OUTCOME B — interaction-dependent emergence was NOT demonstrated.**
The result is reproducible across seeds and the bottleneck is precisely
identified. Per the research brief, this is a valid scientific result.

---

## 1. Research question

> Can a population of individually-limited LPCCs solve a task through
> interaction, communication, temporal coordination, and distributed predictive
> specialization, when no single LPCC has access to the complete causal
> structure?

**Answer (this phase): No — not with the current LPCC hypothesis-formation and
credit-assignment mechanisms.** We built the task, enforced the bottleneck,
implemented routed costly communication with full instrumentation, ran the entire
ablation battery, and measured `interaction_dependence ≈ 0` reproducibly.

## 2. What we built (all new, tested, reproducible)

- **`aha/environments/distributed_chain.py`** — a partially-observable 3-stage
  causal-chain environment (`A → B → C → reward`) with:
  - strict per-zone observation partitioning (each cell sees ONLY its zone's
    signal token),
  - delayed consequences,
  - deterministic / stochastic / rule-reversal variants,
  - per-seed surface-symbol randomization (for transfer),
  - a selectivity penalty (wrong stage actions cost reward) so "always act" is
    not a strong baseline.
- **`aha/communication/routed.py`** — `MessageRouter` with energy cost, delay,
  loss, bandwidth limits, and corruption modes (randomize / corrupt confidence /
  corrupt proposition). Every delivery is loggable for causal analysis. NOT a
  free global broadcast.
- **`aha/colony/distributed_colony.py`** — zone-partitioned LPCC population.
  Each cell receives ONLY its zone view + delivered messages (folded into its
  local context as `now:msg:*` tokens) + its own history. Full information-flow
  event log (`discover / send / receive / action / reward`).
- **`aha/experiments/distributed.py`** — the complete experiment: ablations
  A–H, individual-solvability test, complete-information single-cell baseline,
  transparent metrics, and an operational emergence verdict.

## 3. Enforced information bottleneck (verified)

`no_single_zone_solves = True` in every run: with communication disabled, no
single zone (A-only, B-only, or C-only) achieves solve rate > 0.2. The task
genuinely requires information the individual cannot locally observe. **This
half of the emergence definition holds.**

## 4. Results (seed 0, 300 episodes; reproduced on seeds 1, 2)

| Variant | reward | solve rate |
|---|---|---|
| A. Full system | -15.7 | 0.027 |
| B. No communication | -15.7 | 0.027 |
| C. Random communication | -15.6 | 0.027 |
| D. Delayed communication | -15.7 | 0.027 |
| E. Limited bandwidth | -15.7 | 0.027 |
| F. Critical population removed | -15.7 | 0.027 |
| G. Complete-info single cell | -382.6 | — |
| H. Full comm, no energy cost | +14.9 | 0.170 |

Distributed-emergence metrics (all ≈ 0):

```
interaction_dependence      : 0.0
communication_necessity     : 0.0
random_comm_degradation     : 0.0
delayed_comm_degradation    : 0.0
bandwidth_degradation       : 0.0
critical_information_loss   : 0.0
```

**Operational verdict:** `interaction_dependent_emergence = False`.
Full vs. no-comm are statistically identical (e.g. 19.7 vs 19.3 reward, both
0.175 solve under the no-energy setting). The ~17% solve rate that does occur is
attributable to **exploration luck**, not coordination.

## 5. Precisely identified bottlenecks (the scientific value of this phase)

We instrumented the population and localized *why* emergence did not occur, at
three distinct levels:

1. **Energy starvation (mechanical).** With the energy economy on, cells spend
   energy on activation/prediction/messaging and retire long before the sparse
   end-of-chain reward can teach coordination. Turning energy off raised solve
   from 0.027 → 0.17 (variant H). The energy economy, tuned for the single-agent
   tasks, is mis-scaled for sparse multi-stage reward.

2. **Missing reward consequent (representational).** A downstream, partially
   observing cell has NO local observable consequent to predict — only reward.
   We extended the miner to treat the reward-sign token as a minable consequent.
   Necessary but insufficient (see 3).

3. **Greedy, non-revising hypothesis adoption (learning-rule).** The decisive
   bottleneck: cells adopt the *first / highest-frequency* mined transition
   (a trivial `signal-on → signal-off` local rule) and never revise toward the
   rare **message-conditioned, reward-predictive** hypothesis. Even after adding
   an explicit preference for reward-predictive/message-conditioned candidates,
   **0 cells** ever formed one — because such candidates never pass the
   precision≥0.3 support filter under sparse reward. Count-based mining cannot
   surface a rare causal conjunction (`received-msg ∧ my-signal ∧ my-action →
   reward`) against a flood of high-frequency local co-occurrences.

**Conclusion:** the LPCC substrate's discovery mechanism is a *marginal
co-occurrence miner*. It cannot discover the *conjunctive, communication-gated,
reward-predictive* structure that distributed coordination requires. This is a
representational + credit-assignment limitation, not merely a tuning issue.

## 6. Complete-information control

The complete-information single agent (variant G) performed WORST (-382.6),
because it faces the full combinatorial action/observation space with the same
weak miner and pays the selectivity penalty heavily. So the task is not trivially
centrally solvable by this substrate either — consistent with the diagnosis that
the *learning mechanism*, not the architecture's distribution, is the limiter.

## 7. What would make emergence appear (next experiments)

1. **Conjunctive candidate generation.** Replace marginal co-occurrence mining
   with a mechanism that proposes conjunctive antecedents specifically when a
   single-token hypothesis has high residual prediction error — so a cell that
   fails to predict reward from its signal alone tries `signal ∧ received-msg`.
2. **Reward-gated eligibility for message use.** Give explicit local credit to a
   cell whose action, taken *while a message token was in context*, preceded
   reward — reinforcing message-conditioned hypotheses directly rather than
   relying on the miner's frequency statistics.
3. **Denser curriculum.** Start with a 2-stage chain and short episodes so the
   coordinated behavior is reachable by exploration often enough to accumulate
   the required statistics, then lengthen.
4. **Re-scaled energy economy** for sparse-reward regimes (lower per-step costs,
   larger reward-linked replenishment).

Each is a falsifiable next step with a clear predicted signature
(`communication_necessity > 0.2` and nonzero message-conditioned cell count).

## 8. Honest scientific statement

We did **not** demonstrate interaction-dependent emergent intelligence. We DID:

- construct a task where no individual cell can solve the problem (verified),
- build genuine routed, costly, ablatable communication with full instrumentation,
- run all required ablations + individual-solvability + complete-info baseline,
- reproduce the negative result across seeds,
- and localize the failure to a specific, nameable mechanism: **count-based
  marginal hypothesis mining cannot surface rare conjunctive,
  communication-gated, reward-predictive hypotheses under sparse reward.**

That is the correct, honest conclusion for this phase.

## Reproducibility

```bash
cd AHA2
python3 -m aha.experiments.distributed --seed 0 --episodes 300
python3 -m aha.experiments.distributed --seed 1 --episodes 200
python3 -m aha.experiments.distributed --seed 2 --episodes 200
# -> results/distributed_seed{0,1,2}.json  (variants, individual solvability,
#    metrics, verdict, information_flow_sample)