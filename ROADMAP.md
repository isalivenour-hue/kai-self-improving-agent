# KAI Roadmap

## Completed: KAI 0.001 foundation

- Persistent identity, SQLite state, boot, events, memory, goals, and self-report.
- Read-only operating-system, software, resource, CPU, PCI/DRM, and GPU-layer
  discovery.
- Timestamped environment snapshots, a privacy-preserving environment-profile
  fingerprint, epistemic claims, and snapshot comparisons.
- Safe opt-in CPU and bounded read-only disk baseline measurement.

## Completed: Environment + Self + Epistemic models

The implementation is intentionally limited to observing, representing,
verifying, and modeling current state. UNKNOWN remains a first-class result.
Snapshots are persistent and comparable; GPU discovery is separated into PCI,
kernel/driver, runtime visibility, usability, and measurement. This milestone
does not authorize autonomous behavior.

## Completed: Persistent experience + learning foundation

- Extensible, causally linked events with UTC timestamps and transaction-order
  IDs; environment observations, comparisons, and experiences commit together.
- Partial experiences, layered working/episodic/semantic/procedural memory,
  bounded deterministic consolidation, and evidence-bearing verification.
- Explicit predictions, deterministic outcome comparison and numeric error,
  auditable learning records, and truthful `UNKNOWN`/`EXPIRED` states.
- SQLite schema v3 migration and restart handling for incomplete measurements.
- `kai events`, `kai experiences`, `kai memory`, and `kai predict` inspection and
  recording flows. Action event types are only records; no actions are executed.

Repeated observations yield candidate hypotheses only. No model training,
neural network, LLM, agent swarm, or autonomous action system was introduced.

## Completed: KAI 0.002 bounded adaptive prediction

- A standard-library deterministic running-mean model is scoped to the fixed
  CPU baseline workload and privacy-preserving environment-profile key.
- The first explicit baseline run seeds the model without fabricating a
  forecast. Later runs forecast, measure, score signed error (observed minus
  expected), append an audited model version, and generate the next forecast.
- Sample spread and standard error are explicitly UNKNOWN until there are at
  least two observations. Accuracy excludes unresolved forecasts; calibration
  remains INSUFFICIENT DATA until 20 scored predictions and is never inferred
  merely from a hit rate.
- Model versions and update evidence are persistent and transactionally linked
  to predictions, experiences, and events. Rollback appends a version rather
  than rewriting history; stale pending forecasts are expired.
- Only the explicitly invoked allowlisted CPU baseline is measured. No external
  discovered tool, GPU workload, neural network, LLM, autonomous experiment
  selector, or general action executor is present.

See [ADR-0003](./docs/adr/0003-deterministic-bounded-predictor.md) for the
model boundary and update choice.

## Completed: KAI 0.003 deterministic experiment framework

- Versioned experiment definitions are distinct from executions; validated
  results and prediction evaluations are stored as structured, provenance-linked
  records.
- The v4-to-v5 migration preserves legacy execution IDs and linked prediction,
  event, experience, learning, and model-update history while converting valid
  prior measurements into structured result/evaluation records.
- The trusted registry exposes only the fixed CPU-baseline procedure. Procedure
  and input allowlists are enforced in code; the CLI accepts no procedure code,
  arbitrary parameters, or shell commands.
- Results distinguish `MEASURED`, `UNKNOWN`, and `INVALID`; evaluations
  distinguish `NOT_EVALUATED`, `EVALUATED`, `UNKNOWN`, and `INVALID`. Provenance
  includes definition/procedure version, context, prediction, model version,
  source execution, and evidence.
- Evidence tiers distinguish no usable observation, insufficient evidence,
  preliminary evidence, and the engineering `ADEQUATE` threshold (20
  same-context observations with prediction and evaluation). This threshold is
  not statistical significance or calibrated confidence.
- The experiment procedure is repeatable; elapsed wall-clock runtime is not.
  KAI retains unknown/invalid results without updating the model.
- New CLI paths inspect definitions, run an explicitly selected allowlisted
  experiment, inspect execution history, and view a complete execution. Existing
  KAI 0.002 prediction and learning workflows remain supported.

See [ADR-0004](./docs/adr/0004-deterministic-experiment-framework.md). The next
milestone remains unstarted.

## Completed: KAI 0.004 persistent self-model and improvement evidence

- Schema v6 persists immutable versioned self-model snapshots and provenance-
  linked self-observations, including explicit `OBSERVED`, `INFERRED`, and
  `UNKNOWN` states.
- KAI generates a self-description from runtime version, identity, source AST
  inventory, dependencies, models, experiments, capabilities, resource and
  experience state, limitations, failures, and improvement history. Source
  inspection parses source and tests without importing or executing them; test
  inventory is never reported as a test result.
- The append-only improvement ledger links observations, opportunities,
  hypotheses, proposals, isolated algorithm candidates, measured evaluations,
  and decisions. Candidate comparisons cannot mutate production code, model
  parameters, or deployment state.
- `kai improvements demo-cpu` explicitly collects 22 bounded CPU timing
  samples and compares the running mean against a last-observation candidate
  using 20 paired prequential forecasts. Results are local to the measured
  sample, not statistical significance or generalized performance claims.
- Candidate acceptance requires an adequate evidence tier, passing declared
  candidate contract checks, zero reported regressions, and a current measured
  improvement. The full project suite remains a separate release gate.
  Production deployment remains unsupported; there is no autonomous candidate
  selection or self-modification.

See [ADR-0005](./docs/adr/0005-self-model-and-recursive-improvement.md).

## Completed: KAI 0.005 isolated candidate development

- Schema v7 adds immutable evaluation-protocol definitions, workspace
  provenance, modification and execution histories, and protocol/digest
  references on candidate evaluations. The transactional migration preserves
  all prior state.
- Candidate source snapshots live outside production source. The explicit
  editor is limited to `.py` files under `kai/` and `tests/`, records hashes
  and diffs, checks AST syntax and lineage, and rejects unsafe paths, symlinks,
  protected infrastructure, and secret-like data.
- Candidate test/benchmark execution is restricted to fixed procedures inside
  Linux bubblewrap with network isolation, read-only candidate source,
  ephemeral mounts, sanitized environment, bounded output, timeout, and CPU,
  memory, file-size, and descriptor limits. Missing sandbox support is
  `UNKNOWN`; there is no unsandboxed fallback.
- The explicit CPU demonstration uses 30 measured timings across three
  predeclared sequences, two warmups per sequence, and 24 paired prequential
  forecasts. Outcomes are derived from immutable protocol criteria and paired
  evidence; this is not a statistical significance test.
- `kai self doctor` and `kai workspaces` expose readiness and audited workspace
  lifecycle operations. No autonomous code generation, production mutation,
  model replacement, or deployment was added.

See [ADR-0006](./docs/adr/0006-disposable-candidate-workspaces.md).

## Completed: KAI 0.006 local-model engineering tasks

- An opt-in Ollama provider is restricted to loopback, bounded requests, and
  local-only execution. No KAI memory, tools, cloud provider, or autonomous
  action interface is exposed to the model.
- `kai engineer inspect`, `plan`, `run`, `status`, `tasks`, and `task` provide
  a read-only inspection and a persisted, append-only task lifecycle.
- Explicit task runs validate a narrow model plan and one audited candidate
  patch, compare one selected unittest module against baseline and candidate
  inside bubblewrap, and cap repair attempts at two. Production source and
  deployment status remain untouched.
- When Ollama is unavailable the task is blocked. Missing sandbox support
  never falls back to host execution. Without a generic benchmark protocol,
  passing targeted tests is insufficient to claim improvement; outcomes remain
  unknown/inconclusive and request more evidence.
- Schema v8 adds persistent engineering task and stage-event records.

## Completed: KAI 0.007 multi-model gateway

- Extended the existing LLM contract and Ollama adapter with live local model
  discovery, capability declarations, selected-model inspection, and usage
  metadata. Active selection persists only after availability validation.
- Added an optional OpenAI adapter that stays inert unless both explicit
  enablement and `OPENAI_API_KEY` are present. It discovers account-visible
  model IDs instead of assuming a model name; no credentials are stored.
- Added persisted task-category routes, up to three ordered explicit
  fallbacks, local-only enforcement, and separate source-code cloud
  authorization. No cloud fallback is configured automatically.
- Added append-only model request metadata without prompt/response contents,
  and linked engineering planning, code proposals, and advisory candidate
  reviews to the active task routes. Reviews never replace mandatory tests.
- Schema v9 adds model selection, routing, fallback, and request-history tables.
- Tests use mocks; they require no OpenAI credentials, SDK, network request, or
  downloaded Ollama model.

## Completed: KAI 0.008 first evidence-driven benchmark

- Added a versioned, SHA-256 identified benchmark specification for the
  existing deterministic CPU prediction capability.
- Added a chronological adaptation/evaluation split with fixed replay data,
  baseline and candidate source identities, raw predictions and targets, MAE,
  RMSE, prediction counts, and explicit decision-policy outcomes.
- Persisted complete benchmark reports as append-only `BENCHMARK_COMPLETED`
  events without allowing a model to edit the evaluator or thresholds.
- Added deterministic tests for specification validation, dataset
  reproducibility, metric/decision boundaries, evidence generation, and
  persistence.
- This milestone does not deploy candidates, claim statistical significance,
  or prove general intelligence. The fixed CPU-like sequences are synthetic
  replay data, not actual CPU timing measurements. The candidate is deterministic
  benchmark code, not a model-generated patch.

## Completed: KAI 0.010 reusable paired benchmark evaluation

- Added an explicit paired-case contract shared by the CPU predictor reference
  benchmark and additional trusted, in-process benchmark collectors.
- Validated unique case IDs, finite non-negative metric values, declared units,
  missing observations/predictions, sample minimums, and expected case counts.
- Added versioned outcomes for `UNKNOWN`, `INVALID`, `FAILED`,
  `INCONCLUSIVE`, `NO_MEANINGFUL_CHANGE`, `REGRESSED`, `IMPROVED`, and
  `STRONGLY_IMPROVED`; improvement thresholds remain predeclared in the
  hashed specification.
- Preserved the existing CPU CLI and append-only evidence report. The CPU
  replay is deterministic synthetic data, so its improvement result is limited
  to that workload and does not establish real timing accuracy or generalization.
- The registry accepts trusted Python collectors only; no plugin loading,
  generated-code execution, candidate execution, or production mutation was
  added.
- Validation: the focused benchmark tests pass (9/9); the documented full
  suite passes (170 run, 169 passed, one opt-in bubblewrap integration test
  skipped); an isolated `KAI_HOME` CLI run emitted an `IMPROVED` report with 24
  paired cases. The replay result is still not evidence of real-world
  generalization.
- Added a provisional v1 alignment scorecard in `docs/`. The fixed category
  weights in the directive total 95, not the stated 100; the scorecard preserves
  those weights and reports 61/95 (64.21%) pending a versioned rubric decision.

## Candidate next milestone (not started)

Expand evaluation evidence with a separately frozen held-out workload and
independent raw-data re-derivation, then implement the machine-readable
alignment scorecard. Keep production integration, deployment, unbounded agent
behavior, and autonomous candidate selection out of scope.

## KAI 0.009 implementation status

- Engineering contract version 1 is attached to validated plans without
  changing the existing bounded plan shape.
- Model-driven engineering failures now persist typed diagnostics identifying
  the stage, category, bounded error message, and whether a retry could be
  eligible. Local gateway categorization distinguishes generation aborts,
  truncated responses, invalid JSON, context limits, timeouts, and provider
  failures.
- This is a safe diagnostic subset of KAI 0.009. A genuine model-generated
  candidate patch and independent benchmark evaluation remain unverified until
  a local model produces a valid plan and patch.

## KAI 0.009.1 implementation status

- Planning now starts from a read-only repository inventory and deterministic
  source/test pairs that exist on disk. Protected evaluator, security, model
  gateway, database, sandbox, and configuration paths are excluded from the
  planning choices.
- The inventory is hashed and persisted with each planning event. The local
  model receives exact enum choices plus a schema-valid example and cannot
  introduce an invented test module or source path without failing validation.
- A live constrained planning run produced a valid plan with the configured
  Ollama model. This demonstrates improved planning reliability only; it did
  not generate a candidate patch or establish genuine self-improvement.

## Later research directions

1. Provenance-aware contradiction queries and promotion/retraction policies
   grounded in independent evidence.
2. Measured, repeatable experiments for capability verification (for example,
   a separately approved GPU probe only when its runtime and safety assumptions
   are understood).
3. Minimal goals with explicit success/failure criteria and no action executor.
4. Better bounded observation and prediction experiments once scoring criteria
   and a concrete domain are agreed.
5. Add providers beyond Ollama/OpenAI only with concrete capability, privacy,
   and budget tests; extend the shared provider contract rather than runtime
   or engineering orchestration.

Neural-network work remains deferred. Repeated execution, stored state, and
component count alone are not evidence of learning or intelligence.
