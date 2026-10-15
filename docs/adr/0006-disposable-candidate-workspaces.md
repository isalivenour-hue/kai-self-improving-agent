# ADR-0006: Disposable Candidate Workspaces

- Status: Accepted
- Date: 2026-10-21

## Context

KAI 0.004 could compare an isolated algorithm specification, but it could not
validate a source-level candidate. Executing candidate source creates a distinct
security boundary from observing system state or running a trusted in-process
measurement. KAI must preserve production code and model state, avoid arbitrary
commands, and make incomplete sandbox/evaluation outcomes explicit.

## Decision

Create candidate source only in disposable, KAI-owned filesystem snapshots
outside the production source tree. The editor accepts explicit audited
ADD/MODIFY/DELETE operations on Python files under `kai/` and `tests/`, with
path, symlink, size, syntax, secret-like-content, protected-file, and proposal
lineage checks. It records content hashes and unified diffs.

Execute candidate tests and the single fixed CPU predictor procedure only
inside Linux bubblewrap. The runner uses network and process namespace
isolation, read-only candidate source, sanitized environment, ephemeral mounts,
resource limits, a timeout, and bounded input/output. If the sandbox is
unavailable, KAI records `UNKNOWN` and does not fall back to host execution.
Candidate evaluation uses an immutable, registered repeated-sequence protocol;
metrics and decisions are re-derived from paired errors and linked observations.

Candidate source generation, unrestricted shell or Python execution,
production source/model mutation, and deployment remain out of scope.
Preserved workspaces are review artifacts, not deployed candidates.

## Consequences

- Source-level behavior can be evaluated while production source and
  predictive-model state remain unchanged.
- Snapshot, modification, test, evaluation, and decision provenance are
  persistent and inspectable.
- Candidate execution depends on a Linux host with usable bubblewrap. Other
  environments retain discovery and reporting features but cannot claim a
  candidate test or benchmark succeeded.
- Bubblewrap is a least-privilege local process boundary, not a virtual
  machine or a defense against kernel vulnerabilities or hostile control of
  KAI's state directory.
- Repeated directional criteria are engineering thresholds; they do not
  establish statistical significance or performance outside the measured
  environment.

## Alternatives considered

1. Run candidate code directly in the KAI process or host shell: rejected
   because it would expose production process state and lack a reliable
   filesystem/network/resource boundary.
2. Modify a Git worktree: not required for the current repository, which may
   have no Git metadata. A content-hashed filesystem snapshot provides a
   deterministic source artifact without requiring Git.
3. Use a container runtime or virtual machine: deferred because it would add a
   runtime dependency and operational complexity beyond the fixed, bounded
   candidate procedure.
4. Generate and deploy candidate source automatically: rejected as outside
   this milestone and incompatible with explicit human authorization.
