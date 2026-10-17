# KAI Core Mission: Operational Self-Awareness First

## Mission and scope

KAI's primary architectural priority is to improve accurate, evidence-grounded
awareness of its own operational state, environment, capabilities, limits,
uncertainties, errors, and changes over time. Feature work should strengthen
this model or address an immediate safety/reliability issue before expanding
unrelated autonomous behavior.

This is an engineering mission about observable software behavior. Operational
self-awareness means a system can report scoped state with provenance,
distinguish observation from inference and unknown, expose stale or conflicting
evidence, and test selected reporting rules. It does not establish subjective
experience, feelings, sentience, or phenomenal consciousness.

This document distinguishes implemented behavior from planned work. The first
implementation slice is intentionally bounded: it adds a provenance- and
freshness-aware report, evidence-derived gaps, and deterministic classification
fixtures. It is not a complete factual audit of every KAI subsystem.

## Awareness model

### State awareness

An awareness field reports its value (which may be `null`), epistemic state,
source, observation time, freshness, and freshness limit. The initial
engineering freshness limit is five minutes for the report's observed fields.
An unparsable or missing timestamp and a timestamp implausibly far in the
future produce `UNKNOWN`; an observation older than its limit is `STALE`.
Unknown remains unknown even when its observation time is current.

The report includes identity name, package and architecture versions, SQLite
schema/integrity, privacy-preserving environment fingerprint, provider
readiness, GPU compute usability, test-execution status, source inventory
digest, and privacy-safe counts of goals, task states, predictions, and recent
failures. The freshness threshold is an initial reporting policy, not a claim
that all facts expire at the same rate; it should be refined with evidence.

`kai awareness status` uses KAI's existing status capture and therefore adds the
normal environment snapshot and self-model version to append-only history. It
does not run a test, benchmark, arbitrary discovered tool, model generation,
or GPU workload. Existing provider status inspection remains subject to the
existing gateway policies.

### Capability awareness

Capability status distinguishes declaration, implementation, current
availability, authorization, and an actual test result:

- `VERIFIED` requires a passed test and positive implementation, availability,
  and permission evidence.
- `IMPLEMENTED_UNVERIFIED` means implementation and availability are supported,
  but a validating test has not run.
- `DECLARED_ONLY`, `NOT_IMPLEMENTED`, `UNAVAILABLE`, `BLOCKED`, `FAILED`, and
  `UNKNOWN` remain distinct.

Provider health is not a successful generation test. A discovered binary is
not evidence that an operation works. A test inventory is not test execution.
The awareness command does not invoke capabilities merely to test them.

### Epistemic and provenance rules

KAI's existing event and claim taxonomy remains authoritative. Awareness
reporting preserves `OBSERVED`, `INFERRED`, and `UNKNOWN`; its fixture evaluator
also distinguishes observation, inference, hypothesis, speculation,
unverified/verified fact, contradiction, and staleness. Missing source/evidence
cannot support a fact. A repeated hypothesis remains a hypothesis. Confidence
is not described as a calibrated probability.

The report is derived from the current persisted self-model, direct database
summaries, provider-status metadata, and environment snapshot. Each exposed
field names its source and timestamp. Existing append-only self-model,
observation, and event records remain the historical source; this milestone
does not add a parallel claim database or rewrite past records.

### Temporal continuity and error awareness

The report identifies the self-model version and observation time, marks its
freshness, and summarizes current records without exposing their private
content. It counts task and prediction states, and recent failures, rather than
presenting historical records as active state. Awareness benchmark results
are persisted as append-only `AWARENESS_BENCHMARK_COMPLETED` events and are
available after runtime restart.

Prediction evaluation and restart recovery already exist in KAI, but a unified
cross-restart awareness reconciliation for every task, goal, claim, and
prediction is not implemented by this milestone. Historical/cached records
must not be presented as fresh without a new observation.

## Bounded cognitive loop

The architectural loop is:

```text
OBSERVE
→ REPRESENT
→ VERIFY
→ UPDATE SELF-MODEL AND ENVIRONMENT MODEL
→ IDENTIFY KNOWLEDGE GAPS
→ FORM HYPOTHESES
→ REASON AND FORECAST
→ CHECK CONSEQUENCES, RISKS, AND PERMISSIONS
→ ACT ONLY WHEN AUTHORIZED
→ MEASURE OUTCOMES
→ COMPARE EXPECTATIONS WITH RESULTS
→ UPDATE KNOWLEDGE
→ EVALUATE THE UPDATE
→ VERIFY IMPROVEMENT
```

The current awareness slice implements observation/reporting, a small set of
evidence-derived gaps, deterministic rule fixtures, and append-only benchmark
history. It does not add an autonomous loop or an action executor. Observing
is not verifying; storing is not learning; a proposed candidate is not a
production change; and passing a fixture contract is not generalization.
Blocked, failed, unknown, stale, and inconclusive states must not be converted
to success.

## Awareness health report

Run:

```sh
PYTHONPATH=. python3 -m kai awareness status
```

The JSON report includes field-level values, evidence state/source/time/freshness,
versioned self-model metadata, privacy-safe operational counts, selected
capability assessments, database integrity, the latest awareness benchmark,
and stable actionable gaps. It reports counts and provenance coverage beside
unknown/stale fields; it does not collapse them into a consciousness score.

The report omits secrets, model credentials, raw paths, memory contents, task
objectives, prediction text, and failure payloads. Aggregate counts are bounded
by the existing history APIs (up to 1,000 task, prediction, and experiment
records). A missing observation stays unknown, and provider readiness is
explicitly not a successful generation.

## Knowledge gaps

The initial report identifies only deficiencies supported by explicit
evidence, including tests not executed by the self-observer, GPU workload
usability remaining unknown, and a missing awareness benchmark. Each gap has a
stable ID, evidence references, severity/rationale, affected component,
qualitative evidence-linked confidence, a corrective experiment, acceptance
criterion, status, and self-model link. Confidence labels are not probabilities.
The current implementation does not ask an LLM to create or verify weaknesses.

## Benchmark methodology and baseline

The baseline was established before implementation from commit
`22465d3bc21c93bf73264b50f86c9055eea8b90b`, with a clean worktree:

- Command: `PYTHONPATH=. python3 -m unittest discover -s tests -v`
- Result: 170 tests run, 169 passed, 1 skipped, no failures or errors.
- A disposable pre-change `kai status` report showed package version `0.1.0`,
  schema version 9, 15 entries in the capability summary, 3 unknown
  capabilities, test execution `UNKNOWN`, and SQLite integrity `ok`.
- No dedicated operational-awareness report or awareness benchmark existed.
  Thus the pre-change awareness benchmark is **UNKNOWN / not measurable**,
  not zero accuracy and not a fabricated historical score.

The first deterministic contract benchmark is run and persisted by:

```sh
PYTHONPATH=. python3 -m kai awareness benchmark
```

Its version 1 fixture set covers eleven outcomes: a verified capability,
implemented-but-untested capability, declared-but-unimplemented capability,
unavailable capability, permission block, unknown availability, repeated
hypothesis, unsupported fact, contradiction, current observation, and stale
observation. It reports exact case count, pass count, case accuracy,
false-verified-claim count, evidence status, and limitations. The pass condition
is conformance to all eleven declared deterministic cases; this is an
engineering contract, not a live-host self-knowledge accuracy score,
statistical-significance test, confidence calibration, or generalization claim.

The post-change run must be compared against this protocol, but because the
protocol did not exist at baseline, its measured fixture score has no
historical counterpart. The honest baseline is unavailable. A future version
must freeze richer independent fixtures before claiming improvements in
live-state accuracy.

### Eight evaluation dimensions

| Dimension | Initial status | Method / missing prerequisite |
|---|---|---|
| 1. State accuracy | Partial fixture coverage | Eleven fixtures exercise evidence and freshness rules; independent live expected-state fixtures and completeness/error metrics remain planned. |
| 2. Capability-claim accuracy | Implemented contract fixtures | Exercises verified, untested, unavailable, blocked, not-implemented, and unknown states; false verified claims are counted. Broader environment and prior-availability cases remain planned. |
| 3. Epistemic discipline | Implemented contract fixtures | Exercises unsupported facts, repeated hypotheses, and contradictions; independent claim/evidence corpora remain planned. |
| 4. Freshness and continuity | Partial | Current/stale/future/missing timestamps are classified and benchmark persistence is restart-tested; comprehensive environment-change and interrupted-task restoration evaluation remains planned. |
| 5. Self-error detection | Planned | Reuse prediction IDs, prior claim/evidence links, correction records, and a conflicting observed outcome; assert history preservation and verified correction. |
| 6. Prediction/calibration | Existing separate evaluation | Reuse prediction error and coverage; calibration remains unavailable until sufficient suitable scored observations and a valid calibration method exist. |
| 7. Knowledge-gap detection | Partial, evidence-derived | The report emits stable gaps from explicit unknown/unrun evidence; benchmark recall/precision on an independently labeled gap fixture set remains planned. |
| 8. Improvement verification | Existing separate framework | Reuse fixed versioned candidate evaluation and held-out evidence; awareness-specific held-out tests and regression coverage remain planned. |

No accuracy target is claimed for live KAI self-reporting. The initial
contract target is exact adherence to its eleven fixtures. Future target
thresholds require a separately frozen baseline and an independent labeled
dataset. Missing benchmark data must remain `UNKNOWN`; insufficient evidence
must remain `INCONCLUSIVE`.

## Safety, privacy, and non-goals

- Operational self-awareness is not a consciousness claim.
- No arbitrary discovered tool is executed to inspect capability.
- No model generation, cloud request, GPU workload, candidate execution, or
  deployment is triggered by the awareness benchmark.
- Existing sandbox, provider authorization, cloud source-sharing, and
  candidate-isolation rules remain unchanged.
- No credentials, secrets, user content, raw filesystem paths, or model prompt
  text are added to awareness reports or fixture results.
- Candidate evaluation is not deployment authorization; no production
  self-modification is implemented.
- This milestone adds no dependency and no database schema migration. It
  reuses SQLite schema v9, current self-model snapshots, existing integrity
  checks, and append-only event storage.

## Implementation status and versioned roadmap

### KAI awareness v1 — implemented first slice

- Field-level provenance, observation time, five-minute freshness, explicit
  unknown/stale states, and privacy-safe report.
- Evidence-constrained capability status and an honest no-generation-verification
  label.
- Deterministic eleven-case fixture benchmark and append-only result record.
- Evidence-derived stable test, GPU, and benchmark gaps.
- CLI status/benchmark commands and tests for classification, stale/unknown
  states, privacy filtering, restart persistence, and CLI integration.

### KAI awareness v1.1 — next

- Define independently labeled live-state fixtures for version, schema,
  configuration, model availability, task lifecycle, and permission state.
- Persist explicit test-run records so self-reporting can distinguish a test
  inventory from an actual run and preserve exact outcomes across restarts.
- Add a gap-detection fixture set with precision/recall and evidence links.
- Add current-vs-cached environment reconciliation and bounded interrupted-task
  continuity checks.

### KAI awareness v2 — planned

- Add contradiction/correction lineage and stale-claim supersession only where
  existing event and epistemic models cannot support the required queries.
- Add self-error evaluation linking past predictions/claims to observed
  outcomes without rewriting historical evidence.
- Evaluate awareness improvements on pre-frozen held-out cases, with the
  existing sandbox and explicit authorization boundaries.
- Add no autonomous action, production mutation, or deployment without a
  separately reviewed design and human authorization model.

Current remaining limitations include no complete claim-correction lineage,
no live-state labeled benchmark, no self-observer test execution, no complete
restored-task reconciliation, and no calibrated confidence procedure. These
remain explicit gaps, not implied capabilities.
