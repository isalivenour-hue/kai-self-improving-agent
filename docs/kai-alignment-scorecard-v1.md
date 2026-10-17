# KAI Alignment Scorecard v1

This is the first provisional, evidence-based engineering-progress scorecard.
No earlier scorecard was present, so there is no previous score to compare.
The score measures progress against the declared engineering criteria; it is
not a measurement of intelligence, consciousness, or general capability.

## Scoring method and rubric discrepancy

Each criterion receives one maturity value:

| Maturity | Meaning |
|---:|---|
| 0.00 | Missing, unsafe, or contradicted by evidence |
| 0.25 | Documented or designed, but no demonstrated implementation |
| 0.50 | Implementation exists, but behavior has not been adequately verified |
| 0.75 | Relevant tests passed and behavior is demonstrated within a defined scope |
| 1.00 | Reproducible end-to-end evidence covers behavior, failure paths, limitations, and regressions |

Criterion points equal `criterion weight × maturity`. The listed category
weights in the directive add up to **95**, despite its statement that they
total 100. The assessment preserves every listed category weight and divides
earned points by 95, rather than silently changing the rubric. A future rubric
revision must resolve this discrepancy explicitly and preserve this v1 record.

## Category results

| Category | Possible points | Earned points |
|---|---:|---:|
| Environment observation and capability discovery | 10 | 6.25 |
| Structured representation, epistemic verification, and persistent memory | 10 | 7.50 |
| Modeling, prediction, reasoning, and goal planning | 10 | 7.50 |
| Authorized action execution, tools, and task orchestration | 10 | 7.50 |
| Measurable learning and general benchmark infrastructure | 15 | 11.25 |
| Safe recursive self-improvement and rollback | 15 | 9.50 |
| Model/provider integration and independently verifiable specialist orchestration | 5 | 2.75 |
| Ethical online-income experimentation | 5 | 0.625 |
| Reliable long-running operation and resource management | 5 | 1.875 |
| Testing, reproducibility, security, provenance, and operational auditability | 10 | 6.25 |
| **Total of the listed weights** | **95** | **61.00** |

**Normalized alignment: 61 / 95 × 100 = 64.21%.** This percentage is
provisional and inherits the 95-versus-100 rubric inconsistency.

## Criterion evidence and limitations

| Criterion | Weight | Maturity | Evidence and remaining limitation |
|---|---:|---:|---|
| Structured environment inventory | 5 | 0.75 | [environment_model.py](../kai/environment_model.py), [test_runtime.py](../tests/test_runtime.py); observations are bounded, not continuously monitored. |
| Layered capability verification | 5 | 0.50 | [environment_model.py](../kai/environment_model.py), [runtime.py](../kai/runtime.py); accelerator/runtime usability is not verified for every discovered device. |
| Persistent structured history | 5 | 0.75 | [database.py](../kai/database.py), [test_experience.py](../tests/test_experience.py), [test_runtime.py](../tests/test_runtime.py); full retention and disaster recovery remain incomplete. |
| Epistemic provenance | 5 | 0.75 | [environment_model.py](../kai/environment_model.py), [self_improvement.py](../kai/self_improvement.py), [test_self_improvement.py](../tests/test_self_improvement.py); broad independent contradiction reconciliation is unverified. |
| Bounded prediction | 5 | 0.75 | [predictive.py](../kai/predictive.py), [test_predictive.py](../tests/test_predictive.py); model scope is narrow and does not demonstrate generalization. |
| Bounded goal plans | 5 | 0.75 | [engineering.py](../kai/engineering.py), [test_engineering.py](../tests/test_engineering.py); no general autonomous planner or scheduler exists. |
| Candidate sandbox | 5 | 0.75 | [candidate_workspace.py](../kai/candidate_workspace.py), [candidate_runner.py](../kai/candidate_runner.py), [test_candidate_development.py](../tests/test_candidate_development.py); no production deployment path is supported. |
| Engineering task lifecycle | 5 | 0.75 | [engineering.py](../kai/engineering.py), [test_engineering.py](../tests/test_engineering.py); successful live patch generation and candidate evaluation remain unverified. |
| Causal learning records | 5 | 0.75 | [database.py](../kai/database.py), [predictive.py](../kai/predictive.py), [test_predictive.py](../tests/test_predictive.py); held-out generalization is not established. |
| Versioned reusable benchmark contract | 10 | 0.75 | [benchmark.py](../kai/benchmark.py), [test_benchmark.py](../tests/test_benchmark.py); only the synthetic CPU reference workload exists. |
| Isolated candidates | 8 | 0.75 | [candidate_workspace.py](../kai/candidate_workspace.py), [candidate_runner.py](../kai/candidate_runner.py), [test_candidate_development.py](../tests/test_candidate_development.py); a live promotion path is not demonstrated. |
| Evidence-gated promotion and rollback | 7 | 0.50 | [self_improvement.py](../kai/self_improvement.py), [test_self_improvement.py](../tests/test_self_improvement.py); production promotion and deployed rollback are intentionally unsupported. |
| Provider gateway | 3 | 0.75 | [local_llm.py](../kai/local_llm.py), [test_model_gateway.py](../tests/test_model_gateway.py), [test_local_llm.py](../tests/test_local_llm.py); live provider availability was not proven by unit tests. |
| Independent specialist orchestration | 2 | 0.25 | [ARCHITECTURE.md](../ARCHITECTURE.md); not implemented. |
| Income-experiment safety design | 2.5 | 0.25 | [ROADMAP.md](../ROADMAP.md); design only, no persistent experiment subsystem demonstrated. |
| Independently verified net income | 2.5 | 0.00 | No qualifying experiment or evidence exists. |
| Runtime health reporting | 2.5 | 0.50 | [runtime.py](../kai/runtime.py), [environment_model.py](../kai/environment_model.py), [test_runtime.py](../tests/test_runtime.py); command-driven, not a supervised daemon. |
| Persistent scheduler and recovery | 2.5 | 0.25 | [ROADMAP.md](../ROADMAP.md); lifecycle, graceful shutdown, and stalled-task supervision are not implemented. |
| Automated regression tests | 5 | 0.75 | [DEVELOPMENT.md](../DEVELOPMENT.md), [tests/](../tests/); one host's unit suite does not establish cross-platform or live-provider behavior. |
| Security and provenance | 5 | 0.50 | [SECURITY.md](../SECURITY.md), [candidate_workspace.py](../kai/candidate_workspace.py), [candidate_runner.py](../kai/candidate_runner.py); no independent security assessment was performed. |

## Test evidence used

Environment: Python 3.14.4 on the current Linux checkout.

| Command | Result |
|---|---|
| `PYTHONPATH=. python3 -m unittest discover -s tests -v` (before this change) | Exit 0; 166 tests run, 165 passed, 1 skipped, 0 failures, 0 errors. |
| `PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_benchmark.py' -v` (after this change) | Exit 0; 9 tests passed. |
| `PYTHONPATH=. python3 -m unittest discover -s tests -v` (after this change) | Exit 0; 170 tests run, 169 passed, 1 skipped, 0 failures, 0 errors. The skipped test is the opt-in live bubblewrap integration test (`KAI_RUN_SANDBOX_INTEGRATION=1`). |
| `PYTHONPATH=. python3 -m kai --help` | Exit 0. |
| `KAI_HOME=/tmp/kai-cli-smoke.vmiGnT PYTHONPATH=. python3 -m kai benchmark cpu-predictor` | Exit 0 in disposable state; decision `IMPROVED`, evidence status `SYNTHETIC_REPLAY`, 24 paired cases, event ID 2. |

The sandbox binary `bwrap` was found at `/usr/bin/bwrap`; the optional live
bubblewrap integration test was not enabled. Finding the binary alone is not
evidence that every sandbox restriction works in this environment. Tests and
inspection do not substitute for a live model, held-out benchmark, long-duration
service, or income experiment.

Operational note: one pre-change benchmark CLI smoke was inadvertently run
without a temporary `KAI_HOME`; it appended `BENCHMARK_COMPLETED` event ID 7
to the configured KAI database. The event was verified with a read-only query
and retained to preserve append-only history. No existing event was deleted.
Later CLI validation used an isolated temporary state directory.

## Release readiness

**NOT READY**, independent of the weighted score. Blocking evidence gaps:

- The only registered benchmark is a deterministic synthetic CPU replay; no
  independent held-out workload demonstrates generalization.
- Live candidate generation, evaluation, human approval, and rollback have not
  been demonstrated end to end.
- A supervised long-running runtime and restart-recovery lifecycle are absent.
- No independently verified ethical income experiment has been performed.

## Machine-readable record

See [kai-alignment-scorecard-v1.json](./kai-alignment-scorecard-v1.json) for
criterion definitions, weights, maturity levels, evidence paths, calculation,
and readiness blockers.
