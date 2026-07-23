# ARCHITECTURE: Software Instantiation

## Package layout

```text
aha/
  agents/          Hypothesis Agent implementations
  beliefs/         Explicit belief and evidence representations
  communication/   Messages, routing, dynamic usefulness links
  core/            Shared types, config, colony orchestration
  economy/         Energy and resource accounting
  memory/          Episodic and semantic memory
  metrics/         Prediction, controllability, diversity, overhead
  scheduler/       Future activation/resource scheduling policies
  simulation/      Toy and embodied environments
  visualization/   Rich/matplotlib dashboard components
  experiments/     Reproducible experiment entry points
```

## Dependency direction

- Agents depend on beliefs, memory, communication, economy, and core types.
- Colony depends on agents and communication.
- Metrics observe public state; they do not mutate agents.
- Environments do not know about agents.
- Visualization reads snapshots only.

## Current executable loop

1. Environment emits an observation.
2. Colony routes pending messages.
3. Each living agent independently evaluates match quality.
4. Active agents simulate a prediction and propose an action.
5. Colony chooses an action by transparent consensus, not by a policy network.
6. Environment returns consequence.
7. Each agent updates only its own belief, confidence, uncertainty, memory,
   usefulness, causal trace, and energy.
8. Colony retires dead agents and records metrics.

## Assumption references

Every module docstring should reference entries in `ASSUMPTIONS.md`. The initial
code primarily implements A1-A9.
