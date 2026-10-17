# ADR-0007: Prioritize evidence-grounded operational self-awareness

- Status: Accepted
- Date: 2026-10-17

## Context

KAI already persists environment snapshots, epistemic claims, versioned
self-models, self-observations, predictions, experiment outcomes, and
improvement evaluations. Those records provide useful foundations, but the
system lacks a unified report that makes provenance, freshness, unknowns,
capability evidence, and actionable gaps visible together. Expanding autonomy
before measuring the accuracy and limits of this model would increase authority
without first improving operational self-knowledge.

Operational self-awareness means accurate, scoped, evidence-linked reporting
and evaluation of software state. It is not a claim about subjective
consciousness.

## Decision

Make reliable, evidence-grounded operational self-awareness KAI's primary
architectural priority. Extend the existing runtime, self-model, event history,
and benchmark contract rather than introducing a parallel state store.

The first implementation adds:

- `kai awareness status`, a privacy-safe report with field-level source,
  observation timestamp, epistemic status, and an explicit freshness policy.
- Capability labels that do not equate declaration, implementation,
  availability, permission, or test execution.
- Evidence-derived knowledge gaps with stable IDs and explicit corrective
  experiments and acceptance criteria.
- A deterministic fixture benchmark persisted through existing append-only
  event history.

No schema migration is introduced. The report reuses schema v9 and existing
self-model, environment-snapshot, integrity-check, and event APIs. Fresh
awareness status uses the existing status-capture path, which appends an
environment snapshot and self-model version. Benchmark fixtures do not query
or execute host tools.

## Consequences

- Unknown and stale observations are visible; there is no aggregate
  consciousness score.
- A provider status or source inventory cannot be described as successful
  model generation or test execution.
- The first benchmark demonstrates deterministic contract conformance only;
  live-state accuracy requires independent expected-state fixtures.
- Existing database history and identity remain compatible without migration.
- More complete continuity reconciliation, correction lineage, independent
  gap metrics, and held-out awareness evaluation remain planned.
- No arbitrary tools, cloud requests, candidate deployment, autonomous
  production mutation, or broader permissions are added.

## Alternatives considered

- **Replace the current self-model or redesign the database:** rejected because
  the versioned observation and append-only event mechanisms already support
  this bounded report, while a migration would add risk without a necessary
  data-model gap.
- **Treat test presence/provider readiness as verified capability:** rejected
  because inventory/discovery does not establish execution success.
- **Create a single numeric awareness or consciousness score:** rejected
  because unknown fields and incomplete coverage make such a score misleading.
- **Use an LLM to generate or confirm its own weaknesses:** rejected because
  gap findings must be reproducible and tied to independent evidence.
- **Add autonomous actions to close gaps:** rejected because observation and
  evidence quality must improve without weakening permission boundaries.
