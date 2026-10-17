# Development Guide

## Requirements and test commands

KAI uses Python 3.10+ and the standard library only.

```sh
PYTHONPATH=. python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 -m kai --help
PYTHONPATH=. python3 -m kai llm status
PYTHONPATH=. python3 -m kai benchmark cpu-predictor
PYTHONPATH=. python3 -m kai awareness status
PYTHONPATH=. python3 -m kai awareness benchmark
```

### Operational awareness

`kai awareness status` collects a fresh environment/self-model snapshot using
the existing runtime status path and reports values with source, timestamp,
epistemic state, and an initial five-minute freshness limit. The status command
therefore appends the usual environment snapshot and self-model history record.
It does not run tests, discovered tools, model generation, GPU workloads, or
candidate code. User-authored goal/task/prediction contents, raw paths, and
secrets are omitted. It does consult immutable model-request history: a recent
successful request for the exact selected provider/model verifies generation
for five minutes; stale or unrelated requests do not.

`kai awareness benchmark` evaluates sixteen deterministic contract fixtures
for capability evidence, claim discipline, generation outcome freshness, and
observation freshness, then stores the result as an append-only event. The
reported fixture accuracy is conformance, not measured live-system awareness
accuracy, statistical significance, or confidence calibration. The pre-change
baseline had no matching awareness protocol; it must be recorded as not
measurable rather than assigned a score.

### Evidence-driven benchmark

`python3 -m kai benchmark cpu-predictor` runs the KAI 0.008 deterministic
replay benchmark. The first two values in each fixed sequence are adaptation
observations; the remaining values are evaluation targets hidden from each
forecaster until its prediction is produced. Metrics and decisions are
calculated by `kai.benchmark`, not by a language model. The shared paired-case
evaluator validates unique case IDs, finite non-negative values, matching
units, missing predictions, and protocol sample counts. Its versioned decision
states are `UNKNOWN`, `INVALID`, `FAILED`, `INCONCLUSIVE`,
`NO_MEANINGFUL_CHANGE`, `REGRESSED`, `IMPROVED`, and `STRONGLY_IMPROVED`.
`BenchmarkRegistry` accepts trusted in-process collectors that return paired
case dictionaries (`case_id`, `target`, `baseline_prediction`,
`candidate_prediction`, and `unit`) and routes them through the same evaluator:
the shared paired-prediction metric is lower-is-better error, and other metric
families or higher-is-better tasks need a separately versioned evaluator.

```python
from kai.benchmark import (
    BENCHMARK_REGISTRY,
    run_registered_benchmark,
)

report = run_registered_benchmark("cpu-predictor-learning")
```

The CPU predictor is the registered reference benchmark. Adding another
benchmark requires an explicit specification and trusted collector in Python;
the registry does not load plugins, execute generated code, or add a dynamic
CLI command. The full report,
including specification and dataset digests, source hashes, raw predictions,
targets, and decision rationale, is stored in the append-only event history.
The CPU report labels its fixed fixture as `SYNTHETIC_REPLAY`; it is not a live
CPU timing measurement. The result applies only to the declared capability and
dataset.

`IMPROVED` is a protocol decision, not a claim of statistical significance or
generalization. `STRONGLY_IMPROVED` requires the separately predeclared stronger
relative-error threshold. A missing outcome, malformed pair, unit mismatch,
incomplete run, or candidate failure cannot produce an improvement claim.
The replay fixture is synthetic deterministic data; it does not establish that
the predictor improves real CPU timing forecasts.

### Engineering diagnostics

Validated engineering plans carry contract version 1. When a model-driven task
fails, the persisted task result includes a diagnostic with the stage, bounded
failure category, safe bounded message, and retry eligibility. The categories
are evidence labels, not claims about model internals. A diagnostic failure is
never silently converted into a candidate, and retry eligibility does not cause
an automatic retry.

For disposable state:

```sh
KAI_HOME="$(mktemp -d)" PYTHONPATH=. python3 -m kai boot
```

Remove only the exact temporary directory created by that command when finished.
Never point tests at the normal `~/.local/share/kai` data directory.
Local model integration tests must mock the loopback HTTP API; tests must not
download models, require a running Ollama service, or make remote requests.

### Verified local-model planning (KAI 0.009.1)

Before requesting a plan, KAI parses the source tree and verifies deterministic
`kai/<module>.py` plus `tests/test_<module>.py` pairs on disk. Protected paths
such as benchmark/evaluator, security, sandbox, provider, database, and
configuration code are not offered as planning targets. The resulting options
and SHA-256 inventory digest are persisted in the engineering task event.
The model receives only those exact choices and a matching JSON example; the
response must pass the contract, schema, pair-membership, path, and test-file
checks. This stage intentionally stops at a plan and does not imply that a
patch, candidate, benchmark, or production modification exists.

## Schema and state

SQLite schema changes increment `SCHEMA_VERSION` and add a transactional migration in
`kai.database.Database`. Tests must cover migration from the prior schema,
preserved identity, integrity checks, rollback, and unsupported future schemas.
Environment snapshots retain canonical model JSON and claims; the configured
history cap is 100 snapshots. Claims are immutable historical reports; new
evidence belongs to a later snapshot.

Events, experiences, predictions, evaluations, and learning records must be
written transactionally with their causal/source references. Use UTC-aware
ISO-8601 timestamps, but use the committed event ID for order. Preserve partial
experiences and UNKNOWN prediction outcomes rather than manufacturing
completion. A learning record must preserve its before/happened/predicted/error/
changed/evidence/source-experience fields.

The four memory layers have distinct uses: working memory for current context,
episodes for particular experiences/events, semantic memory for claims, and
procedural memory for explicit strategies. Deterministic consolidation may
create an unverified hypothesis only; it must not promote a candidate to FACT.
Do not implement model-specific memory or imply that persistence is learning.

Experiment definitions, executions, results, and evaluations are separate
records. Definitions are versioned contracts for trusted procedures; executions
capture one invocation; results contain validated measurements; evaluations
compare those outcomes with predictions and retain evidence quality and
provenance. Do not conflate deterministic procedure code with reproducible
wall-clock performance. A changed model or hardware profile must not silently
rewrite the definition or KAI identity.

Schema v6 adds append-only, versioned self-model snapshots and observations,
plus an improvement ledger linking opportunities, hypotheses, proposals,
isolated candidates, evaluations, and decisions. Generate the self-model from
current runtime/database/source state; do not hardcode machine capabilities or
claim tests passed merely because test functions were inventoried. Static
source inspection must parse files without importing or executing them.
Candidate acceptance validation must check that the current evaluation's
status, metric direction, test and benchmark status, regression/failure counts,
and evidence quality agree with the persisted source observations.
Every evaluation also needs an immutable baseline snapshot: KAI and
architecture versions, relevant model/experiment versions, configuration,
environment context, evaluation dataset references, and metrics. Validate the
baseline dataset links and paired sample count against evaluation evidence.

Schema v7 adds registered evaluation protocols, disposable candidate
workspaces, audited modifications/executions, and immutable evaluation
protocol/dataset provenance. Candidate source workspaces must be outside the
production tree; permit explicit edits only to Python source under `kai/` and
`tests/`. Validate traversal, symlink, protected-file, secret-like content,
size, syntax, and candidate/hypothesis lineage boundaries. Persist every
operation and never represent a workspace edit as a production change.

Candidate code may execute only through fixed test and benchmark procedures
inside Linux bubblewrap. There is no fallback to host execution. Test sandbox
absence, timeout, output/resource limits, malformed candidate output, failed
tests, and valid success without depending on the developer's host. Evaluation
criteria must be versioned before samples are collected; re-derive every
reported metric and status from immutable paired evidence. Report UNKNOWN if
the sandbox, required measurements, or evidence is incomplete. Keep candidate
acceptance separate from deployment.

Schema v8 adds persisted engineering tasks and append-only stage events. Local
model engineering is explicit: inspect must not initialize or write KAI state;
plan/run require the selected model and any provider-specific authorization,
constrain the target and test module, and use the existing audited
candidate/workspace and sandbox paths. Keep repair attempts bounded. A passing
targeted module is not an improvement claim without benchmark evidence; report
`UNKNOWN` or `INCONCLUSIVE` when evaluation evidence is incomplete. Model and
sandbox unavailability must be explicit, and sandbox failure must never fall
back to host execution.

Schema v9 adds persistent active-model selection, task-specific routes,
ordered bounded fallbacks, and append-only safe request metadata. Extend the
existing provider contract rather than creating a parallel model subsystem.
Every adapter must declare capabilities, report limits honestly, and never
advertise tools, streaming, or cancellation unless implemented. Keep provider
credentials in process environment only; never include prompts, responses,
keys, headers, or raw provider error bodies in logs or request history.
Selection changes must validate provider/model availability before persistence.
Cloud adapters require explicit enablement and authorization, and code context
must remain local unless source sharing is separately authorized. Normal tests
must use mocks and never require live credentials or model downloads.
Ordinary status/report commands must not issue OpenAI API requests; account
model discovery is limited to explicit discovery or selection operations.
Engineering candidate reviews are advisory and must not bypass static checks,
tests, or evaluation.

Preserve the distinctions among KAI identity (persistent UUID), environment
identity (privacy-conscious profile fingerprint), hardware (discovered
devices), capability (layered support/availability/usability/measurement),
observation (a source-reported value), fact (a scoped verified claim),
inference (a conclusion from evidence), hypothesis (a testable proposal), and
unknown (insufficient evidence). Confidence is not calibrated probability.
An inference must not be silently promoted to fact; unknown states must remain
explicit.

## Discovery development

Discovery code must be read-only and platform-aware:

- Read Linux procfs/sysfs only when present; preserve missing measurements as
  `None` or `UNKNOWN`.
- Never run a program only because PATH lookup found it.
- Never infer runtime usability from hardware, driver, library, or node presence.
- Do not include serials, MAC addresses, hostnames, or account data in the
  environment fingerprint.
- Add deterministic unit tests using temporary fake sysfs/procfs trees or pure
  fake models. Tests must not require the developer's hardware.
- Add deterministic event, experience, prediction, error, consolidation,
  uncertainty, recovery, migration, and CLI tests using fixed timestamps.
- Adaptive predictors must define a bounded context key and deterministic update
  rule. Test bootstrap without a fabricated prediction, forecast-to-observation
  comparison, error sign convention, model version history, rollback, stale
  pending forecasts, and transactional failure behavior.
- Report discovered and measured states separately. Uncertainty and calibration
  must remain explicitly unknown/insufficient until their documented sample
  thresholds are reached.
- Performance measurements must be opt-in, bounded, and documented with their
  cache/load limitations.
- Experiment registry tests must cover deterministic listing, duplicate and
  unknown definitions, disabled-procedure boundaries, fixed parameter and
  context validation, malformed/non-finite/incompatible results, evidence-tier
  thresholds, and explicit unknown/invalid states.
- Experiment persistence tests must exercise success and failure, result and
  evaluation provenance, relative-error zero handling, independent context
  isolation, immutable model-version history, transactional rollback, restart
  recovery, and fixtures for v4-to-v5 and v5-to-v6 migrations preserving
  historical IDs and foreign keys.
- CLI tests must cover definition display, explicit execution, history and full
  execution inspection, input rejection, and the legacy learning and prediction
  routes. Tests must use fake measurements rather than depend on local hardware.
- Self-model tests must verify source-tree inspection is non-executing,
  versions/observations are immutable and provenance-linked, runtime state is
  generated rather than hardcoded, and unknown test/capability states remain
  unknown. Improvement-ledger tests must cover lifecycle lineage, evidence
  consistency, status/metric agreement, stale-evaluation rejection,
  regression/failure/inconclusive outcomes, and production/deployment
  non-mutation. The CPU comparison demonstration is opt-in and must be tested
  with deterministic mocked measurements; its real measured run is not a
  hardware-independent test oracle.
- Candidate-development tests must cover v6-to-v7 migration, workspace source
  snapshots and cleanup/preservation, path and symlink rejection, explicit
  audited changes, cross-lineage rejection, immutable protocol registration,
  re-derived repeated metrics, execution record persistence, sandbox-missing
  UNKNOWN, and the runtime self-doctor/CLI paths.
- Environment claims must preserve `OBSERVATION`, `FACT`, `INFERENCE`,
  `HYPOTHESIS`, or `UNKNOWN`, with provenance and evidence; never promote an
  inference automatically.

## Review checklist

Before extending the observe/model layer, verify provenance and claim class,
identity/environment separation, unknown and contradictory states, migration
behavior, snapshot change reporting, and that no operating-system state is
modified. Do not begin the next roadmap milestone without user direction.

Before extending the experience/learning layer, verify event causal parents,
episode source links, rollback on failure, event-ID ordering, explicit
prediction status transitions, and that a measured prediction error does not
silently rewrite a semantic claim.

Before extending adaptive prediction, verify workload allowlisting, stable
context isolation, update/version auditability, rollback semantics, prediction
resolution on failed/interrupted experiments, confidence sample thresholds,
and deterministic fake-environment integration tests.

Before extending the experiment framework, verify immutable versioned
definitions, trusted procedure bindings, validated structured results,
definition/execution provenance, honest UNKNOWN/INVALID handling, migration of
historical IDs and links, reproducibility boundaries, evidence thresholds, and
transaction rollback across all persisted records.

Before changing candidate development, verify the snapshot is outside
production, all candidate paths are confined, test/benchmark procedures remain
fixed, the execution sandbox has no unsandboxed fallback, and persisted
evaluation decisions are reproducible from the registered protocol and paired
source observations.
