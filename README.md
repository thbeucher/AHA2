# Autonomous Hypothesis Agents (AHA)

AHA is a research framework for studying whether intelligence can emerge from a
population of autonomous local hypothesis agents rather than from global
gradient-based parameter optimization.

The elementary computational object is a **Hypothesis Agent**: a persistent,
stateful local model-builder that observes, predicts, intervenes, evaluates
evidence, communicates, specializes, merges, and retires.

The core implementation intentionally avoids policy networks, computational
graphs, global losses, and backpropagation.

## Phase status

The repository now implements the full **Local Predictive Causal Cell (LPCC)**
substrate (Phases 0–12) alongside the original Phase-1 belief-agent scaffold
(kept as a legacy baseline). New modules:

- `aha/cells/`      — LPCC primitive, inspectable state, temporal pattern detector, config
- `aha/hypotheses/` — explicit `C×A→F` hypotheses + delay-aware candidate generation
- `aha/prediction/` — temporally-aligned prediction + ledger (t → t+h)
- `aha/causal/`     — local causal-influence approximation
- `aha/colony/`     — population orchestration, transparent consensus, structural plasticity
- `aha/environments/` — hidden-rule, rule-reversal, transfer, two-stage emergence
- `aha/metrics/`    — prediction/calibration/diversity + transparent emergence score
- `aha/experiments/`— reproducible discovery/reversal/transfer/emergence/ablation CLIs

The core LPCC substrate uses **no backpropagation, no gradient descent, no global
loss, no policy network**. Each cell updates only from local + indirect evidence.

See `paper/LPCC_REPORT.md` for the full scientific write-up (including honest
negative results).

## Quick start

```bash
cd AHA2
python3 -m pytest -q                                   # 20 tests
python3 -m aha.experiments.run all --seed 0 --episodes 200
# machine-readable results land in AHA2/results/*.json
```

Individual experiments:

```bash
python3 -m aha.experiments.run discovery  --seed 0 --episodes 200
python3 -m aha.experiments.run reversal   --seed 0 --episodes 200   # falsification test
python3 -m aha.experiments.run transfer   --seed 0 --episodes 200
python3 -m aha.experiments.run emergence  --seed 0 --episodes 200
python3 -m aha.experiments.run ablations  --seed 0 --episodes 200   # required set A-H
```

## Core rule

If a design choice increases locality, interpretability, continual adaptation,
structural plasticity, causal reasoning, sparse computation, hypothesis formation,
or autonomous self-organization, prefer it over conventional ML convenience.
