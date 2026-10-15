# ADR-0005: Persist a self-model and evidence-led improvement ledger

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

KAI 0.003 has persistent environment and experience models, an auditable
predictor, and a trusted experiment registry. It does not yet preserve a
structured versioned account of KAI's own capabilities and limitations, nor a
linked history for examining a possible improvement from observation through
evaluation. A collection of command output and an in-memory self-description
would not provide durable provenance. Conversely, executing generated code or
automatically replacing production behavior would exceed the current trust and
autonomy boundary.

The next milestone must support explicit self-observation and a bounded,
measured candidate comparison while retaining UNKNOWN and inconclusive results,
preserving production behavior, and leaving deployment to a future separately
approved design.

This ADR records the KAI 0.004 boundary. KAI 0.005 supersedes its no-source-
execution constraint only for fixed procedures inside disposable bubblewrap
workspaces; [ADR-0006](./0006-disposable-candidate-workspaces.md) defines that
separate boundary. Production mutation and deployment remain unavailable.

## Decision

### Self-model and observations

Schema v6 persists immutable self-model versions and linked self-observations.
Each observation records its subject, type, value, timestamp, source, evidence
status, epistemic status (`OBSERVED`, `INFERRED`, or `UNKNOWN`), optional
confidence, KAI version, and environment context. Self-model versions capture
KAI and architecture versions, schema, stable KAI identity, environment
fingerprint, source snapshot digest, and generated model data.

The runtime generates the self-model from actual runtime/database state and a
read-only AST inventory of owned Python source and test files. AST inspection
does not import or execute project source. A count of test functions is source
inventory, not evidence that tests passed; test execution is therefore reported
as UNKNOWN by self-observation. Persisted self-model history is append-only.
Summaries derived from bounded recent-history queries state their window rather
than implying lifetime totals.

### Improvement evidence

The persistent lifecycle links:

```text
observation -> opportunity -> hypothesis -> proposal -> candidate
            -> evaluation -> decision
```

Candidates in this milestone are fixed algorithm specifications with isolated
measurement/evaluation records. They are not mutable source checkouts and are
not dynamically imported or executed. Acceptance requires a latest matching
evaluation that derives improvement from persisted metrics, passing test
status for the declared candidate contract checks, zero regressions, and
adequate evidence linked to source observations. Running the full project suite
is a separate release gate; the demo does not run that suite.
Evaluation labels inconsistent with metric direction, benchmark/test state,
failures, or evidence quality are rejected. Historical failed and superseded
evaluations remain recorded; stale evaluations cannot authorize a later
decision.

Every evaluation embeds an immutable baseline snapshot containing the KAI and
architecture versions, relevant predictive-model and experiment-definition
versions, configuration, environment context, evaluation dataset references,
and baseline metrics. The evaluation row and its evidence are append-only;
candidate and baseline measurements are kept side-by-side under that same
snapshot. A baseline is therefore fixed for the comparison and cannot be
silently replaced after a decision.

The explicit CPU demonstration compares the existing running-mean forecaster
to a last-observation forecaster using 22 bounded real timing observations and
20 paired prequential out-of-sample forecasts. It reports MAE and the measured
sequence outcome, but it does not claim statistical significance,
generalization, or a production improvement. The demo records failed, unknown,
regressed, and inconclusive outcomes rather than replacing them with a
success-shaped fallback. It does not mutate production source, predictor
parameters, or deployment state. The measurement is explicit user invocation,
not part of ordinary self-observation.

No production deployment path is available. `READY_FOR_DEPLOYMENT` is rejected.
Candidate decisions are ledger records, not autonomous actions.

## Consequences

- KAI can inspect historical self-model versions and evidence-linked
  improvement attempts through the CLI.
- Identity, software, environment, hardware, capability, and epistemic status
  remain distinct. Hardware changes do not create a new KAI identity.
- Unobserved test execution and capabilities remain explicitly unknown.
- The AST inventory offers bounded structural self-observation without
  executing inspected project files.
- A single CPU comparison demonstrates the evidence lifecycle only; it is not
  sufficient to claim a broadly superior algorithm.
- Schema v6 adds durable audit state and migration obligations while leaving
  the existing predictor and experiment workflows intact.
- KAI 0.005 adds only repeated evaluation and a disposable candidate workspace.
  Unrestricted execution, production self-modification, autonomous candidate
  selection, and deployment remain outside this decision.

## Alternatives considered

- **Keep self-description in prose or recompute without persistence:** rejected
  because observations, provenance, and historical changes would be lost.
- **Treat test-source inventory as test success:** rejected because parsing test
  functions does not execute assertions.
- **Promote favorable measurements directly to facts or production code:**
  rejected because a short timing sequence is noisy and not a general
  performance proof.
- **Execute candidate source from the ledger:** rejected because candidate
  isolation and execution sandboxing are not implemented.
- **Automatically deploy or rewrite the running predictor:** rejected because
  deployment, rollback, and authorization require a separate safety design.
