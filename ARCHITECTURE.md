# KAI Architecture

## Primary mission: evidence-grounded operational self-awareness

KAI's architecture prioritizes accurate, provenance-bearing awareness of its
own state, environment, capabilities, limitations, uncertainties, errors, and
changes over time. The mission is operational and measurable; it is not a claim
of subjective consciousness. This priority governs new capability selection:
improve the validity, freshness, evidence coverage, and correction of KAI's
self-model before adding unrelated autonomy.

KAI already persists environment snapshots, epistemic claims, self-model
versions, and source-linked observations. `kai.awareness` builds a privacy-safe
health report from those records and current database/provider summaries, marks
unknown and stale observations explicitly, and derives a small set of
evidence-linked gaps. `kai awareness benchmark` evaluates eleven deterministic
contract fixtures and stores the result in append-only event history. Fixture
success validates the reporting rules, not the factual accuracy of every
live-host claim.

The awareness status command uses the existing runtime status capture, which
persists a fresh environment snapshot and self-model version. It does not
execute discovered tools, run the test suite or GPU workloads, invoke model
generation, or authorize actions. Capability availability, permission, and
test verification remain separate dimensions. See
[CORE_AWARENESS_MISSION.md](./docs/CORE_AWARENESS_MISSION.md) for semantics,
benchmark scope, baseline, limitations, and the versioned roadmap.

## Current boundary

KAI's intended architecture remains:

```text
OBSERVE -> REPRESENT -> VERIFY -> MODEL -> REASON -> ACT -> MEASURE -> LEARN
```

KAI 0.007 adds a unified, capability-reporting model gateway, persistent model
selection, explicit task routing, and bounded fallback over supported providers.
Ollama remains the local default. The optional OpenAI adapter is inert unless
cloud use is explicitly enabled with a key. The local model can propose
candidate source edits only through `kai engineer plan/run`; it does not add
autonomous reasoning, production mutation, or deployment. This builds on KAI
0.006's explicitly invoked engineering loop and KAI 0.005's sandboxed
candidate-evaluation boundary, and strengthens
OBSERVE -> REPRESENT -> VERIFY -> MODEL using persistent
events, experiences, environment snapshots, and epistemic evidence. Its narrow
CPU-baseline predictor generates forecasts deterministically from a stored
model. The bounded measure/learn experiment is explicitly user-invoked; the
architecture does not yet include an autonomous REASON/ACT loop. KAI does not
infer off-system hardware presence, execute discovered tools, or run a GPU
workload.

## Persistent self-model and improvement evidence

Schema v9 stores immutable, versioned self-model snapshots and provenance-linked
self-observations, the improvement ledger, disposable workspaces, modification
records, sandbox executions, immutable evaluation protocols, and append-only
engineering task stages, as well as active model selection, task routes,
ordered fallbacks, and privacy-safe model request metadata:

```text
observation -> opportunity -> hypothesis -> proposal -> candidate
            -> disposable workspace -> audited edit -> sandbox test
            -> repeated evaluation -> decision

engineering objective -> repository inspection -> local-model plan
                      -> candidate patch -> advisory model review -> fixed tests
                      -> evidence report

task category -> configured route (or active model) -> capability/policy check
              -> bounded request -> request metadata (no prompt/output)
```

The generated self-model combines the running KAI version and identity,
environment fingerprint, schema, active predictive models, registered
experiments, capability inventory, resource and history summaries, source-tree
AST inventory, dependencies visible through source imports, limitations,
failure records, and improvement state. Each self-observation retains subject,
type, timestamp, source, evidence status, epistemic status, confidence (when
known), KAI version, and environment context. The AST inventory reads and
parses owned source and test files but never imports or executes them. The
test-function count is not a test result: self-observation explicitly reports
test execution as `UNKNOWN` because it does not run the suite.

The provider contract covers health, model discovery, text and structured
generation, reported token use, timeout/error classification, and declared
capabilities. Streaming, tools, and request cancellation are explicitly
unsupported by the current adapters. Ollama discovery is based on the live
`/api/tags` inventory; model context/resource limits remain unknown when the
provider does not report them. Its selected model is unloaded after each KAI
request, with KAI local requests serialized under a process and state-directory
lock. OpenAI uses an optional SDK, a fixed official API origin, configured
model discovery, bounded output, and no automatic SDK retries. Its key remains
in the process environment and is never placed in SQLite or logs. Ordinary
status/report commands do not query OpenAI; explicit model discovery and model
selection perform the account API checks.

Model selection is only persisted after a live availability check. Task routes
override the active model only for their declared category. Fallback is
disabled until the user configures an ordered list (maximum three); each
attempt is finite and the actual provider/model and sanitized failure category
are returned and recorded. Cloud models require explicit enablement and an API
key; cloud coding/review requests additionally require explicit source-sharing
authorization. `--local-only` refuses cloud selection and excludes cloud
fallbacks. No network fallback is inferred from a local failure.

`kai engineer inspect` gathers source/environment/provider/sandbox readiness
without initializing the database or creating a KAI log. `plan` and `run`
require the explicitly selected model to be ready. A plan is restricted to one
existing Python test module and a source file under `kai/` or `tests/`. Task
state, validated plans, candidate lineage, bounded repair counts, and terminal
evidence are persisted with append-only stage events. Model requests receive
only the explicit objective and bounded source context needed for the task;
KAI memory and tools are never exposed. If OpenAI is selected, cloud enablement
and source-sharing authorization are required before code context is sent.

Candidate source is copied to a KAI-owned state directory outside the
production tree. The manager accepts explicit ADD/MODIFY/DELETE operations on
Python files beneath `kai/` and `tests/`, validates syntax, records hashes and
patches, and rejects traversal, symlinks, protected infrastructure, and
secret-like paths/content. Candidate generation occurs only after explicit
user invocation, and candidate code is never automatically deployed.
Bubblewrap execution uses network/PID/IPC/UTS isolation, a read-only snapshot,
ephemeral temporary/home/system mounts, a sanitized environment, resource
limits, timeout, and bounded output. Only fixed unittest selection and one
fixed CPU-predictor worker can execute. If the sandbox is unavailable there is
no unsandboxed fallback.

The engineering task currently runs one selected unittest module on baseline
and candidate. Each generated candidate patch, including repairs, receives an
advisory routed model review that cannot replace static checks or tests. The
task permits at most two repair attempts. It has no generic benchmark
protocol, so even a passing targeted test is insufficient to establish an
improvement; the persisted outcome remains unknown or inconclusive and requests
more evidence. Model unavailability blocks the task. An interrupted or failed
run is recorded explicitly and does not imply a successful candidate.

The CPU demonstration compares a last-observation forecaster with the
running-mean baseline using 30 explicitly requested bounded timings across
three predeclared sequences and 24 paired prequential forecasts. Each sequence
includes two warm-up measurements. A 10% aggregate relative-change threshold
and directional consistency in at least two of three sequences define an
engineering outcome; this is not a statistical significance test or a
generalization claim. `UNKNOWN`, failed, and inconclusive outcomes remain
explicit. The demonstration never changes production source, predictor
parameters, or deployment status.

Acceptance requires passing declared candidate contract checks, zero reported
regressions, adequate linked evidence, and the latest matching measured
evaluation. The project-wide suite remains a separate release gate; the demo
does not run it.
Evaluation protocol definitions are immutable and versioned. The evaluation
re-derives aggregate metrics and the decision from stored paired absolute-error
sequences; its baseline snapshot carries the registered protocol digest, data
links, environment context, model/experiment versions, and a canonical SHA-256
digest. Workspace, edit, execution, evaluation, and decision histories are
append-only; workspace lifecycle status is the only mutable workspace field.

Deployment is deliberately unavailable. A candidate decision is an evidence
record, not a production rollout or a claim of autonomous self-improvement.
Each evaluation preserves an immutable baseline snapshot with KAI/architecture
versions, relevant predictive-model and experiment-definition versions,
configuration, environment context, dataset references, and baseline metrics.
The `evaluations list/show` CLI exposes the baseline and its provenance.
Recent counts in the self-model are bounded to their declared query windows;
they are not silently presented as lifetime totals.

## Identity and environment

- **KAI identity** is a stable random UUID plus principles persisted in SQLite.
  It survives software and hardware changes and is not derived from a machine ID.
- **Environment profile identity** is a SHA-256 digest over CPU/architecture and
  enumerated PCI hardware identifiers. It excludes serial numbers, MAC addresses,
  hostnames, and user identifiers. Equal fingerprints mean matching profiles,
  not proof of the same physical computer.
- **Environment snapshot** is a timestamped structured observation, including
  resources and software visibility. Snapshots are separate from KAI identity
  and retained with a bounded history.
- **Hardware** describes enumerated devices and their observed drivers, kernel
  interfaces, and runtime visibility. PCI visibility does not establish physical
  presence outside the current OS namespace.
- **Capability** is layered: discovered hardware -> kernel support -> driver
  binding -> runtime availability -> usable capability -> measured capability.
  A state at one layer never silently proves a later layer.

## Epistemic model

Environment claims are persisted with classification, source, timestamp,
confidence, evidence, verification status, and contradictory evidence:

- `OBSERVATION`: directly reported by a named local interface or file.
- `FACT`: a narrowly scoped claim verified by its stated authoritative source.
- `INFERENCE`: a conclusion drawn from evidence, such as matching environment
  fingerprints; it is never upgraded to a fact automatically.
- `HYPOTHESIS`: a testable proposed explanation.
- `UNKNOWN`: evidence is insufficient or a capability was not safely tested.

Confidence is not truth. Claim confidence values are evidence-strength
heuristics, not calibrated probabilities. The self-model's confidence is an
evidence-coverage ratio, not a probability. An unavailable value remains
unknown. Claims are snapshot-scoped, so newer observations do not rewrite what
prior snapshots reported.

Tool-path absence means “not discovered on the inspected PATH,” not “absent
from the machine.” CUDA-library absence is similarly limited to the documented
inspected library paths.

## Events, experiences, and knowledge

These records have separate roles and are linked rather than conflated:

- **Event** is one timestamped occurrence with a monotonic database event ID,
  event type, source, optional snapshot and causal parent, payload, confidence,
  and epistemic status. Event types are extensible; the initial catalog is in
  `kai.experience.EVENT_TYPES`.
- **Experience** groups context with any available observation, belief,
  prediction, action record, outcome, evaluation, and learning. Partial
  experiences are valid and remain open until an outcome/evaluation is recorded.
- **Memory** is a queryable representation in one of four layers: working
  memory stores current state; episodic memory is represented by specific
  experience records and their source events; semantic memory stores claims;
  procedural memory stores explicit strategies. The existing legacy memory
  interface writes into semantic memory without silently verifying a claim;
  even a legacy `fact` label is a candidate hypothesis until explicit
  evidence-bearing verification.
- **Knowledge** is still an epistemically classified claim. Repeated exact
  observations can create an unverified `HYPOTHESIS` candidate with links to
  its source episodes. Consolidation never promotes a candidate automatically.
- **Truth** is not a database flag that repetition can set. Verification is an
  explicit evidence-bearing transition; conclusions remain scoped to evidence.

Therefore: **EVENT != EXPERIENCE; EXPERIENCE != MEMORY; MEMORY != KNOWLEDGE;
KNOWLEDGE != TRUTH.** Events preserve what occurred, experiences join related
records, memory organizes records for later use, and epistemic status expresses
how well a claim is supported.

## Prediction and learning loop

`kai predict create` stores a hypothesis, expected outcome, context, confidence,
basis, and optional observation window. `kai predict evaluate` compares an
explicit observed value with the expected value. Numeric comparisons store the
signed difference and absolute error; an optional tolerance can yield
`PARTIALLY_CONFIRMED`. Missing outcomes stay `UNKNOWN` and do not generate a
learning record. An evaluated prediction linked to an experience creates an
auditable learning record containing prior belief, observed outcome,
prediction, error, state change, reason, evidence, confidence, and source
experience. This records a comparison; it is not a trained learner, calibrated
predictor, or claim of intelligence. This generic record/evaluate path is
separate from the domain-specific running-mean model described below.

## Bounded adaptive prediction

The only current adaptive model predicts runtime for one fixed 100,000-iteration
Python CPU loop, keyed by task, workload parameters, and the privacy-preserving
environment profile. Snapshot IDs are provenance, not part of the model key.
With no stored observation, KAI reports `INSUFFICIENT DATA`; an explicitly
requested experiment establishes the first sample without inventing a forecast.
Subsequent experiments forecast from the active model, measure the bounded
workload, score signed error as observed minus expected, update the model, and
persist the next forecast.

The model is a standard-library Welford running mean with sample variance.
Standard error is UNKNOWN before two samples and descriptive thereafter; no
prediction interval is claimed calibrated. Accuracy reports exclude pending,
unknown, and expired forecasts. Empirical accuracy is withheld until five
scored predictions; confidence-versus-accuracy remains `INSUFFICIENT DATA`
until 20. Neither empirical hit rate nor descriptive calibration bins are
called calibrated probabilities. Qualitative resource estimates distinguish this
experiment from GPU capability or general workload performance.

Experiments are opt-in, allowlisted, transactional, and linked to events,
experiences, predictions, and versioned model updates. Rollback appends a new
version restoring prior parameters; it does not erase audit history, and it
expires pending predictions tied to superseded versions. A first seed version
has no prior parameter state to restore. No arbitrary tool execution,
autonomous experiment selection, or neural/LLM learner is present. See
[ADR-0003](./docs/adr/0003-deterministic-bounded-predictor.md).

## Experiment definitions, executions, and evidence

An **experiment definition** is a versioned specification, not a run. It
identifies a trusted procedure, fixed parameters, allowlisted context, expected
outcome shape, and enablement status. An **execution** records one explicit
invocation and its definition/procedure version, context, environment snapshot,
timestamps, terminal state, and failure information. A **result** is a
validated structured measurement; an **evaluation** records its comparison to
the prediction (when one exists), signed/absolute/relative error, evidence, and
evaluation state. These records are separate tables and are transactionally
linked. Schema v5 migrates legacy experiment runs into execution records while
preserving their numeric IDs and foreign-key history.

The first registered experiment is `cpu-baseline@1`, bound to the fixed trusted
`cpu-baseline-v1` procedure. Its integer-loop procedure and parameters are
reproducible, but the measured wall-clock duration is not: load, scheduling,
thermal state, and runtime conditions vary. KAI records measured outcomes as
observations and does not interpret a repeated procedure as a deterministic
runtime or a general performance claim. Result validation requires finite
numeric data, JSON-safe evidence, the declared unit, and the declared sample
count. Failed or unavailable measurements remain `UNKNOWN`; malformed or
incompatible data is `INVALID`. Both states are persisted and cannot update the
predictive model.

Evidence quality is an explicit reporting tier: `NONE` for no usable
measurement, `INSUFFICIENT` for fewer than two observations or inconsistent
context, `PRELIMINARY` for at least two observations that do not satisfy the
adequacy requirements, and `ADEQUATE` only with at least 20 same-context
observations plus a prediction and evaluation. This threshold is an engineering
reporting convention, not statistical significance or calibrated confidence.
Procedure reproducibility, measurement validity, prediction evaluation, and
evidence quality are therefore different properties.

Event IDs define committed ordering and causal references; UTC ISO-8601
timestamps preserve wall-clock time. Timestamps do not replace transactional
order. SQLite transactions make each event/prediction/experience transition
atomic. On restart, KAI records recovery events for measurements left without a
completion/failure event; it records the interruption but does not repair or
retry the measurement.

## Components

- `kai.discovery`: existing small platform/tool discovery helpers.
- `kai.environment_model`: structured CPU, PCI/DRM/GPU, memory, storage,
  software, fingerprint, claim, comparison, and opt-in baseline models.
- `kai.experience`: model-independent event catalog, prediction states, and
  typed event/experience/prediction/learning/experiment record shapes.
- `kai.experiment_registry`: immutable versioned definitions, a static trusted
  procedure registry, fixed input validation, structured result validation,
  and evidence-quality classification.
- `kai.predictive`: deterministic context keys, running-mean update math,
  uncertainty/resource summaries, and prediction statistics.
- `kai.database`: transactional SQLite migrations and persistent identity,
  events, layered memory, experiences, predictions, learning records, goals,
  environment snapshots, epistemic claims, experiment definitions/executions/
  results/evaluations, and model histories.
- `kai.runtime`: boot recovery, snapshot/experience persistence, change
  comparison, measurement records, bounded experiment orchestration,
  persistent self-observation, improvement evaluation, and generated
  self-model assembly.
- `kai.awareness`: evidence-constrained operational capability states,
  observation freshness, deterministic awareness contract fixtures, and a
  privacy-safe report derived from existing persisted self-model/runtime
  records.
- `kai.cli`: user-visible command interface. Discovery is read-only; `measure`
  is an explicit opt-in bounded local measurement.

## Persistence and portability

Schema version 6 migrates earlier versions in place and preserves the existing KAI
UUID. Environment snapshots carry JSON representations alongside normalized
snapshot-scoped epistemic claims. Events, their linked experiences, predictions,
learning records, experiments, and model updates have transactional
relationships. Self-model versions and source observations are append-only;
opportunity status transitions and candidate evaluations/decisions preserve
history. The fingerprint is a
profile key, not an authentication credential. A future migration may change
the canonical profile without changing KAI identity.

## Deliberately absent

There is no planner, action executor, neural training, semantic retrieval
model, autonomous experiment selection, or self-modifying code. An optional
one-shot Ollama provider is available only through explicit CLI invocation. It
connects to loopback, does not retrieve KAI memory, expose tools, or persist
prompts/responses, and does not take part in the deterministic predictor or
candidate evaluation.
`ACTION_*` events are representational only: no system action executor exists.
KAI's self-model is generated from local runtime state and static source
inspection; it is not independent introspection or a consciousness claim.
Candidate algorithm code is represented and compared as a fixed in-process
implementation, not loaded from user-controlled source. SQLite is persistence,
not intelligence.
