# ADR-0002: Separate events, experiences, memory, and knowledge

- **Status:** Accepted
- **Date:** 2026-10-06

## Context

KAI needs persistent history and a learning foundation without a neural model or
unrestricted action system. Treating each database row, observation, belief, and
conclusion as the same kind of "memory" would erase causality and epistemic
limits. Repetition can be evidence for a pattern, but cannot by itself make a
claim universally true.

## Decision

- An **event** is an atomic timestamped record with source, payload, optional
  environment snapshot, causal parent, confidence, and epistemic status.
- An **experience** links available context, observation, belief, prediction,
  action record, outcome, evaluation, and learning. It may remain partial.
- **Memory** is organized into working, episodic, semantic, and procedural
  layers; each layer has its own meaning and inspection surface.
- **Knowledge** is an epistemically classified claim with evidence and
  verification state. Deterministic consolidation may only create an
  unverified hypothesis linked to its source experiences.
- **Truth** is not inferred from repetition or set merely by persistence.
  Verification is explicit, evidence-bearing, and auditable.
- Prediction and evaluation are model-independent records. Numeric error is
  stored as observed minus expected plus its absolute magnitude. Missing
  outcomes remain UNKNOWN and do not create a learning record.
- SQLite transaction order (event ID) establishes committed ordering; UTC
  timestamps preserve wall-clock context. Transactions atomically persist
  linked state transitions.

## Consequences

- Historical evidence and causal links survive restarts and model replacement.
- Partial information is representable without fabricating a complete episode.
- Prediction error can be measured and audited, but this is not model training
  or proof that KAI is intelligent.
- Future planners or model providers must consume the same identity, event,
  experience, memory, and epistemic interfaces; none is implemented here.
- Event types remain extensible, while action event types do not implement or
  authorize actions.
