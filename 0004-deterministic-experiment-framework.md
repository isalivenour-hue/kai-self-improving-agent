# ADR-0004: Keep experiments deterministic, explicit, and auditable

- **Status:** Accepted
- **Date:** 2026-10-07

## Context

KAI 0.002 stores opt-in CPU baseline runs, but it does not sufficiently
distinguish a reusable experiment specification from an individual execution.
Overloading a run record with definition, procedure, output, prediction
evaluation, and model provenance makes migration and future experiment
registration ambiguous. Treating a fixed procedure as a deterministic outcome
would also overstate wall-clock evidence.

The next step must preserve the existing predictor and history while making
experiments inspectable and safe. It must not add arbitrary execution,
autonomous selection, external model dependencies, or a new measurement domain.

## Decision

Represent an experiment as four distinct records:

1. **Definition:** a versioned, status-bearing contract for a trusted procedure,
   fixed parameters, allowed context, and expected outcome.
2. **Execution:** one explicit invocation with definition/procedure versions,
   context, timestamps, outcome state, and failure information.
3. **Result:** a validated structured measurement with unit, sample count,
   metadata, and `MEASURED`, `UNKNOWN`, or `INVALID` status.
4. **Evaluation:** the relation to an available prediction, signed/absolute/
   relative errors when defined, evidence provenance, and an explicit
   `NOT_EVALUATED`, `EVALUATED`, `UNKNOWN`, or `INVALID` status.

Definitions bind to registered in-process trusted procedures. The only enabled
procedure is the existing fixed CPU-baseline loop. Parameters and context are
allowlisted; CLI arguments cannot provide code, shell commands, or arbitrary
workloads. Measurement outputs are validated against the definition before
they can update a model. Every run remains explicitly selected by the user.

SQLite schema v5 migrates the legacy run table into execution records without
changing existing run IDs or their event, experience, prediction, learning,
and model-update links. It creates a separate definition table and structured
result/evaluation records. Migration is transactional and retains the legacy
run view for compatibility.

The fixed procedure is repeatable, but elapsed wall-clock time is not. Results
remain observations affected by load, scheduling, thermal conditions, and
runtime state. Missing outcomes remain `UNKNOWN`; malformed or incompatible
results remain `INVALID`; neither can train the model. Evidence quality is
reported with `NONE`, `INSUFFICIENT`, `PRELIMINARY`, and `ADEQUATE` tiers.
`ADEQUATE` is an engineering threshold requiring at least 20 same-context
observations plus an available prediction and evaluation; it is not statistical
significance or calibrated confidence.

## Consequences

- Definitions can be versioned and inspected independently from execution
  history.
- Each result and evaluation preserves its provenance and is auditable through
  the execution and model-update records.
- The current deterministic running-mean predictor and its CLI remain
  compatible; existing run IDs and foreign keys survive the schema migration.
- Non-finite, malformed, mismatched-unit, and mismatched-sample results are
  explicit failures, not success-shaped fallbacks.
- Runtime variation cannot be confused with procedure reproducibility or a
  general performance claim.
- New measurement domains require a new definition, trusted bounded procedure,
  safety review, structured result contract, and deterministic fake-based
  tests. This ADR does not authorize such a domain.

## Alternatives considered

- **Treat each run row as both definition and execution:** rejected because
  versioned specifications and historical invocations have different
  lifecycles and provenance.
- **Accept arbitrary user-provided procedure code or shell commands:** rejected
  because it would violate the explicit, read-only discovery and least-
  authority boundary.
- **Store only raw output JSON:** rejected because it obscures type, unit,
  validity, evaluation state, and evidence quality.
- **Treat a repeated procedure as a repeatable runtime:** rejected because
  elapsed wall-clock duration varies with the environment.
- **Replace legacy history or IDs during migration:** rejected because the
  existing event, prediction, and learning links are useful audit evidence.
