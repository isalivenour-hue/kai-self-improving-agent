# Architecture Decisions

| ADR | Decision | Status |
| --- | --- | --- |
| [ADR-0001](./docs/adr/0001-decouple-identity-from-hardware.md) | Keep KAI identity independent of machine/environment identity | Accepted |
| [ADR-0002](./docs/adr/0002-separate-events-experiences-and-knowledge.md) | Keep events, experiences, memory, and knowledge distinct | Accepted |
| [ADR-0003](./docs/adr/0003-deterministic-bounded-predictor.md) | Start adaptive prediction with a deterministic, allowlisted, versioned running mean | Accepted |
| [ADR-0004](./docs/adr/0004-deterministic-experiment-framework.md) | Separate versioned experiment definitions from executions, results, and evaluations | Accepted |
| [ADR-0005](./docs/adr/0005-self-model-and-recursive-improvement.md) | Persist a provenance-bearing self-model and evaluate isolated candidates without deployment | Accepted |
| [ADR-0006](./docs/adr/0006-disposable-candidate-workspaces.md) | Use disposable snapshots and a fixed, network-isolated runner for candidate evaluation | Accepted |
