# KAI 0.005 architecture notes

The maintained architecture, roadmap, development process, and security model
are documented in the project-root [ARCHITECTURE.md](../ARCHITECTURE.md),
[ROADMAP.md](../ROADMAP.md), [DEVELOPMENT.md](../DEVELOPMENT.md),
[SECURITY.md](../SECURITY.md), and [DECISIONS.md](../DECISIONS.md).

## Runtime boundary

`kai.runtime.KaiRuntime` is the orchestration boundary. It coordinates
environment discovery, the state database, generated self-observation, and
explicit candidate evaluation; it does not execute user-selected programs,
contact remote services, or load a language model.

## State

SQLite schema version 7 is the source of truth because it is built into Python, supports
transactions and integrity checks, and keeps identity, events, memory, and goals
together with timestamped environment snapshots and their epistemic claims.
`schema_info` tracks migrations. Identity receives a random UUID on first
initialization and is not derived from host-specific values.

Events preserve source, payload, timestamp, confidence, epistemic status,
environment snapshot, and causal parent. Event IDs provide committed ordering.
Experience records connect context, observation, belief, prediction, action
record, outcome, evaluation, learning, and source events; any field can be
absent, preserving partial experiences. Predictions store expected and observed
outcomes and error. Learning records preserve before/happened/predicted/error/
changed/evidence and their source experience. Predictive model versions and
experiment/update records are separate, transactionally linked records.

Working, episodic, semantic, and procedural memory have distinct tables or
representations. Confidence is metadata, not proof. Deterministic consolidation
creates only candidate hypotheses linked to their source episodes; it never
silently upgrades them to facts. Event/experience/memory/knowledge/truth remain
separate concepts. Schema migrations are transactional; unsupported newer
schemas and failed integrity checks stop startup with an explicit error.

## Discovery and uncertainty

Environment discovery uses Python's platform APIs, Linux procfs/sysfs, PCI/DRM
metadata, network interface enumeration, filesystem statistics, and PATH
lookup. No external executable is run during discovery. PCI visibility, kernel
driver binding, runtime-library/device-node presence, usability, and measured
performance are distinct states. Missing measurements remain `null` or
`unknown`. CPU and bounded read-only disk baseline measurements run only when
the user explicitly invokes `kai measure`; no GPU workload is run.

## Deliberately deferred

Generic prediction comparison is deterministic and does not update the
domain-specific adaptive model. Missing outcomes remain unknown. Recovery records incomplete measurements after restart
without retrying or repairing them. There is no planner, action executor, LLM
adapter, world model, neural training, semantic-memory index, or self-modifying
mechanism in this milestone. Future subsystems should be introduced behind
explicit interfaces and evaluated with repeatable tests and benchmarks before
being treated as capabilities.

## Experiment definitions and evidence

Experiment definitions, executions, results, and evaluations are separate
records. A definition is a versioned contract; an execution is one explicit
invocation of a statically registered trusted procedure; a result is a
validated structured measurement; and an evaluation relates that result to a
prediction with errors, evidence, and status. KAI migrates existing run IDs and
links to execution records without discarding prior history. The CLI exposes
definition display, explicit run, history, and full execution inspection.

Only `cpu-baseline@1` is currently enabled, bound to the fixed
`cpu-baseline-v1` procedure. The procedure is repeatable but elapsed wall-clock
runtime varies with the environment. Fixed parameters and allowlisted context
are enforced; user-supplied code, shell commands, or arbitrary workloads are
not accepted. Results with invalid structure, units, values, or sample counts
cannot update the model. Missing results remain UNKNOWN; malformed results are
INVALID.

Evidence quality uses `NONE`, `INSUFFICIENT`, `PRELIMINARY`, and `ADEQUATE`.
The ADEQUATE tier requires 20 same-context observations and an available
prediction evaluation. It is an engineering reporting threshold, not
statistical significance or calibrated confidence. Measurement validity,
prediction evaluation, procedure reproducibility, and evidence quality remain
distinct.

## Adaptive predictor

KAI 0.002 introduced one deterministic CPU baseline predictor, keyed by a fixed
task and workload plus the environment profile. A first explicitly requested
experiment seeds a per-context running mean; subsequent runs predict, measure,
score observed-minus-expected error, append a model version, and produce a
follow-up forecast. Welford sample spread is unknown before two samples.
Calibration remains insufficient until the stated minimum scored-prediction
count and is not claimed from a small descriptive hit rate. The experiment is
fixed, bounded, opt-in, and does not run GPU workloads or discovered commands.
See the root [architecture](../ARCHITECTURE.md) and
[ADR-0003](./adr/0003-deterministic-bounded-predictor.md) for math and lifecycle.
See [ADR-0004](./adr/0004-deterministic-experiment-framework.md) for the
definition/execution/result/evaluation model and migration boundary.

## Persistent self-model and improvement ledger

KAI 0.004 persists immutable self-model versions and provenance-bearing
self-observations in schema v6. The generated self-model combines actual runtime
and database state with a static AST inventory of `kai/*.py` and
`tests/test_*.py`. Inspection reads and parses source but does not import or
execute it. Test-function inventory is not test execution; the self-model marks
test execution as `UNKNOWN`.

The improvement ledger persists linked opportunities, hypotheses, proposals,
isolated candidates, evaluations, and decisions. Candidate workspaces are
content-hashed filesystem snapshots outside production source. Explicit Python
edits under `kai/` and `tests/` are recorded with diffs, hashes, and hypothesis
lineage. Static validation rejects unsafe paths, symlinks, protected files,
secret-like data, syntax errors, and size violations.

Candidate test and benchmark execution uses a fixed bubblewrap boundary:
network/PID/IPC/UTS namespaces, read-only source, ephemeral mounts, sanitized
environment, and resource/time/output limits. Only fixed unittest selection
and one fixed CPU evaluation worker run; unavailable sandbox support remains
UNKNOWN with no host-execution fallback.

The explicit CPU demonstration measures 30 bounded samples across three
predeclared sequences and scores 24 paired prequential forecasts. Its immutable
protocol applies a 10% aggregate relative-change and two-of-three directional
consistency rule. Status is re-derived from paired errors and cannot be
substituted; the result is not statistical significance or a general
performance claim. Production source, models, and deployment state remain
unchanged.

See [ADR-0005](./adr/0005-self-model-and-recursive-improvement.md) for the
provenance-bearing self-model and [ADR-0006](./adr/0006-disposable-candidate-workspaces.md)
for the disposable candidate and sandbox boundary.
