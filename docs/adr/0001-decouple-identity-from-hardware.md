# ADR-0001: Decouple KAI identity from hardware identity

- **Status:** Accepted
- **Date:** 2026-10-06

## Context

KAI should persist across software updates, hardware replacement, and migration.
Hardware changes are useful evidence about the current environment but do not
mean the persistent system identity has changed. Hostnames, serial numbers,
MAC addresses, and machine IDs are sensitive and can be unavailable or cloned.

## Decision

Persist KAI's identity as an independently generated UUID in the SQLite identity
row. Compute a separate environment-profile fingerprint from architecture, CPU
model/vendor/core profile, and enumerated PCI addresses/vendor/device/class IDs.
Do not use serial numbers, MAC addresses, hostnames, account names, or machine
IDs. Report a fingerprint match as an inference about similar environment
profiles, never as proof of the same physical machine.

## Consequences

- A KAI identity survives machine and environment changes.
- A fingerprint change can trigger environment comparison without changing
  identity.
- Similar machines can have the same profile fingerprint; this is not a unique
  device identity or authentication token.
- Fingerprint input changes require documentation and tests. A future schema
  migration can revise profile semantics without replacing KAI identity.

