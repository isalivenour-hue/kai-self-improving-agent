# ADR-0003: Start with a deterministic, bounded adaptive predictor

- Status: Accepted
- Date: 2026-10-02

## Context

KAI has persistent events, experiences, explicit predictions, and outcome
evaluations, but those generic records do not themselves update a predictive
model. The next step should demonstrate a complete observe/predict/measure/
compare/update loop without implying general intelligence, requiring model
packages, or granting an action executor broad authority.

## Decision

Implement one domain-specific predictor for the fixed 100,000-iteration Python
CPU baseline. Its context is the task, fixed workload parameters, and
privacy-preserving environment profile; snapshot IDs are retained as evidence
but excluded from the model key. The predictor uses Welford's online running
mean and sample variance. Signed error is observed minus expected.

The first explicitly requested measurement initializes the model; KAI does not
invent a numerical forecast before evidence exists. Later explicit runs
forecast from the current model, execute only the fixed in-process baseline,
persist the measurement and evaluation, append a model version/update record,
and generate a subsequent forecast. State transitions are transactional and
causally linked. Rollback appends a new version restoring prior parameters and
expires pending forecasts made by superseded versions.

Standard error is UNKNOWN with fewer than two samples. Empirical accuracy is
withheld until five scored predictions; calibration status remains INSUFFICIENT
DATA until 20 scored predictions. Neither a confidence label nor an estimated
standard error is a calibrated probability or interval. The experiment is
opt-in and its resource costs are qualitative.

## Consequences

- KAI can test a deterministic adaptive update end-to-end with fixed fake
  observations and no third-party dependencies.
- Model identity is scoped to task/workload/environment profile and remains
  separate from KAI's persistent identity and individual environment snapshots.
- Every forecast, outcome, error, update, and rollback remains inspectable.
- There is no arbitrary experiment runner, GPU workload, neural learner, LLM,
  or autonomous experiment selection.
- The model says nothing about general CPU performance, GPU capability, or
  machine learning throughput. A single sample cannot quantify variability.
- Additional domains require their own explicit contexts, safe measurement
  boundaries, deterministic tests, and evidence-backed calibration criteria.

## Alternatives considered

- **Neural network or external model:** rejected; unnecessary for the first
  closed-loop demonstration and introduces dependencies and opaque behavior.
- **Run arbitrary discovered tools as experiments:** rejected; violates the
  read-only discovery boundary and increases system authority.
- **Predict before any local evidence:** rejected; would manufacture certainty
  rather than preserve insufficient data.
- **Update generic semantic memory from the measurement:** rejected; model
  parameter updates and epistemic knowledge are distinct representations.
