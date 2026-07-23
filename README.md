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

This repository currently implements **Phase 1 / early Phase 2**:

- research specification documents,
- high-level architecture,
- explicit assumptions ledger,
- minimal package structure,
- explicit dataclass-based belief/state/message/memory structures,
- a minimal local Hypothesis Agent,
- a minimal Colony coordinator,
- a deterministic toy GridWorld for smoke experiments,
- tests proving the substrate compiles and performs local updates.

Later phases should extend behavior only after each phase compiles and tests pass.

## Quick start

```bash
cd aha_project_vg
python3 -m compileall aha tests
python3 -m pytest
python3 -m aha.experiments.run_gridworld --steps 8
```

## Core rule

If a design choice increases locality, interpretability, continual adaptation,
structural plasticity, causal reasoning, sparse computation, hypothesis formation,
or autonomous self-organization, prefer it over conventional ML convenience.
