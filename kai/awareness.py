"""Evidence-aware operational self-reporting and deterministic contract fixtures."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


AWARENESS_BENCHMARK_ID = "operational-awareness-contract"
AWARENESS_BENCHMARK_VERSION = 2
DEFAULT_FRESHNESS_SECONDS = 300


def _parse_timestamp(value: str | None) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def observation_freshness(
    observed_at: str | None,
    *,
    now: datetime,
    maximum_age_seconds: int = DEFAULT_FRESHNESS_SECONDS,
) -> str:
    if maximum_age_seconds < 0:
        raise ValueError("Maximum observation age cannot be negative.")
    observed = _parse_timestamp(observed_at)
    if observed is None or now.tzinfo is None or now.utcoffset() is None:
        return "UNKNOWN"
    age = (now.astimezone(timezone.utc) - observed).total_seconds()
    if age < -maximum_age_seconds:
        return "UNKNOWN"
    if age > maximum_age_seconds:
        return "STALE"
    return "CURRENT"


def classify_generation_evidence(
    latest_request: dict | None,
    *,
    now: datetime,
    maximum_age_seconds: int = DEFAULT_FRESHNESS_SECONDS,
) -> dict:
    """Use only a recent, well-formed generation outcome as execution evidence."""
    if latest_request is None:
        return {
            "test_status": "NOT_RUN",
            "freshness": "UNKNOWN",
            "completed_at": None,
        }
    if not isinstance(latest_request, dict):
        return {
            "test_status": "UNKNOWN",
            "freshness": "UNKNOWN",
            "completed_at": None,
        }

    completed_at = latest_request.get("completed_at")
    freshness = observation_freshness(
        completed_at,
        now=now,
        maximum_age_seconds=maximum_age_seconds,
    )
    if freshness == "STALE":
        test_status = "NOT_RUN"
    elif freshness != "CURRENT":
        test_status = "UNKNOWN"
    elif latest_request.get("outcome") == "SUCCESS":
        test_status = "PASSED"
    elif latest_request.get("outcome") == "FAILED":
        test_status = "FAILED"
    else:
        freshness = "UNKNOWN"
        test_status = "UNKNOWN"
    return {
        "test_status": test_status,
        "freshness": freshness,
        "completed_at": completed_at if isinstance(completed_at, str) else None,
    }


def assess_capability(
    *,
    declared: bool | None,
    implemented: bool | None,
    available: bool | None,
    permission: str,
    test_status: str,
) -> str:
    """Do not infer a verified capability from declaration or discovery alone."""
    for label, value in (
        ("declared", declared),
        ("implemented", implemented),
        ("available", available),
    ):
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"Capability {label} evidence must be boolean or unknown.")
    if permission not in {"ALLOWED", "BLOCKED", "UNKNOWN"}:
        raise ValueError("Capability permission evidence is invalid.")
    if test_status not in {"PASSED", "FAILED", "NOT_RUN", "UNKNOWN"}:
        raise ValueError("Capability test status is invalid.")
    if declared is False or implemented is False:
        return "NOT_IMPLEMENTED"
    if declared is True and implemented is None:
        return "DECLARED_ONLY"
    if permission == "BLOCKED":
        return "BLOCKED"
    if available is False:
        return "UNAVAILABLE"
    if available is None or permission == "UNKNOWN":
        return "UNKNOWN"
    if test_status == "PASSED":
        return "VERIFIED"
    if test_status == "FAILED":
        return "FAILED"
    return "IMPLEMENTED_UNVERIFIED"


def classify_claim(
    claim_type: str,
    *,
    source_present: bool,
    evidence_count: int,
    evidence_status: str,
    contradicted: bool = False,
    stale: bool = False,
    repetition_count: int = 1,
) -> str:
    """Keep claim class separate from evidence; repetition never verifies a claim."""
    if evidence_count < 0 or repetition_count < 1:
        raise ValueError("Evidence and repetition counts are invalid.")
    if evidence_status not in {"observed", "verified", "unverified", "contradicted", "unknown"}:
        raise ValueError("Claim evidence status is invalid.")
    if stale:
        return "STALE"
    if contradicted or evidence_status == "contradicted":
        return "CONTRADICTED"
    if claim_type == "unknown" or evidence_status == "unknown":
        return "UNKNOWN"
    if not source_present or evidence_count < 1:
        return "UNKNOWN"
    if claim_type == "observation":
        return "OBSERVATION" if evidence_status in {"observed", "verified"} else "UNKNOWN"
    if claim_type == "inference":
        return "INFERENCE"
    if claim_type == "hypothesis":
        return "HYPOTHESIS"
    if claim_type == "speculation":
        return "SPECULATION"
    if claim_type == "fact":
        return "VERIFIED_FACT" if evidence_status == "verified" else "UNVERIFIED_FACT"
    raise ValueError(f"Unsupported claim type: {claim_type!r}.")


def _field(
    value: Any,
    *,
    source: str,
    observed_at: str,
    now: datetime,
    epistemic_status: str = "OBSERVED",
    maximum_age_seconds: int = DEFAULT_FRESHNESS_SECONDS,
) -> dict:
    freshness = observation_freshness(
        observed_at, now=now, maximum_age_seconds=maximum_age_seconds
    )
    state = epistemic_status
    if freshness == "STALE" and state != "UNKNOWN":
        state = "STALE"
    return {
        "value": value,
        "state": state,
        "epistemic_status": epistemic_status,
        "source": source,
        "observed_at": observed_at,
        "freshness": freshness,
        "maximum_age_seconds": maximum_age_seconds,
    }


def _gap(
    gap_id: str,
    description: str,
    *,
    severity: str,
    rationale: str,
    component: str,
    evidence: list[str],
    evidence_status: str,
    corrective_experiment: str,
    acceptance_criterion: str,
    self_model_version_id: int | None,
) -> dict:
    return {
        "id": gap_id,
        "description": description,
        "severity": severity,
        "severity_rationale": rationale,
        "affected_component": component,
        "evidence": evidence,
        "evidence_status": evidence_status,
        "confidence": "EVIDENCE_LINKED",
        "confidence_semantics": "Qualitative evidence linkage, not a probability.",
        "proposed_corrective_experiment": corrective_experiment,
        "acceptance_criterion": acceptance_criterion,
        "status": "OPEN",
        "links": (
            [{"kind": "self_model_version", "id": self_model_version_id}]
            if self_model_version_id is not None
            else []
        ),
    }


def derive_knowledge_gaps(
    *,
    test_execution_status: str,
    gpu_epistemic_status: str,
    benchmark_status: str | None,
    self_model_version_id: int | None,
    generation_status: str = "UNKNOWN",
) -> list[dict]:
    gaps = []
    if test_execution_status == "UNKNOWN":
        gaps.append(
            _gap(
                "test-suite-execution-unverified",
                "The self-observer inventories tests but does not know whether they passed.",
                severity="MEDIUM",
                rationale="Test-source presence is not execution evidence.",
                component="tests",
                evidence=["test_suite.execution_status=NOT_RUN_BY_SELF_OBSERVER"],
                evidence_status="OBSERVED",
                corrective_experiment=(
                    "Run the documented full unittest command and capture its exit code "
                    "and test totals in an explicit verification record."
                ),
                acceptance_criterion=(
                    "A timestamped run record includes command, exit code, passed, "
                    "failed, and skipped counts."
                ),
                self_model_version_id=self_model_version_id,
            )
        )
    if gpu_epistemic_status == "UNKNOWN":
        gaps.append(
            _gap(
                "gpu-inference-usability-unknown",
                "GPU inference usability has not been established.",
                severity="LOW",
                rationale="Hardware or driver discovery does not demonstrate a usable inference workload.",
                component="gpu_inference",
                evidence=["gpu.compute_usability=UNKNOWN"],
                evidence_status="UNKNOWN",
                corrective_experiment=(
                    "Only after a supported runtime and bounded test are specified, "
                    "run an explicitly authorized local smoke test."
                ),
                acceptance_criterion=(
                    "A bounded approved inference test records its runtime, model, "
                    "result, and failure state without implying broader performance."
                ),
                self_model_version_id=self_model_version_id,
            )
        )
    if generation_status != "VERIFIED":
        gaps.append(
            _gap(
                "local-model-generation-unverified",
                "The selected model has no current successful generation verification.",
                severity="LOW",
                rationale=(
                    "Provider readiness alone or a stale generation outcome cannot "
                    "establish current generation ability."
                ),
                component="local_model_gateway",
                evidence=[f"selected_model_generation.status={generation_status}"],
                evidence_status=(
                    generation_status if generation_status != "UNKNOWN" else "UNKNOWN"
                ),
                corrective_experiment=(
                    "Run the explicit bounded `kai llm test` against the selected model."
                ),
                acceptance_criterion=(
                    "A successful generation outcome is recorded for the selected "
                    "provider/model within the five-minute verification window."
                ),
                self_model_version_id=self_model_version_id,
            )
        )
    if benchmark_status != "PASSED":
        gaps.append(
            _gap(
                "awareness-benchmark-not-passed",
                "No passing operational-awareness fixture benchmark is available.",
                severity="MEDIUM",
                rationale="The report has not yet been checked against its deterministic contract fixtures.",
                component="awareness_evaluation",
                evidence=[
                    "AWARENESS_BENCHMARK_COMPLETED event is absent."
                    if benchmark_status is None
                    else f"Latest awareness benchmark status={benchmark_status}."
                ],
                evidence_status="OBSERVED",
                corrective_experiment="Run `kai awareness benchmark` and inspect its case outcomes.",
                acceptance_criterion=(
                    "The complete versioned fixture suite is persisted with its "
                    "sample count and explicit passing result."
                ),
                self_model_version_id=self_model_version_id,
            )
        )
    return gaps


def build_awareness_report(
    self_model_version: dict,
    runtime_status: dict,
    *,
    database_integrity: str,
    task_status_counts: dict[str, int],
    prediction_status_counts: dict[str, int],
    recent_failure_count: int,
    latest_benchmark: dict | None,
    latest_model_request: dict | None = None,
    now: datetime | None = None,
) -> dict:
    """Build a bounded report from the latest persisted model and live summaries."""
    now = now or datetime.now(timezone.utc)
    now = now.astimezone(timezone.utc)
    model = self_model_version["model"]
    observed_at = self_model_version["created_at"]
    environment = runtime_status.get("environment", {})
    known_state = runtime_status.get("known_state", {})
    language_model = model.get("language_model", {})
    if not isinstance(language_model, dict):
        language_model = {}
    gpu_state = model.get("available_capabilities", {}).get(
        "gpu_compute_usable", "unknown"
    )
    gpu_epistemic = "UNKNOWN" if gpu_state in (None, "unknown", "UNKNOWN") else "OBSERVED"
    test_suite = model.get("test_suite", {})
    test_execution = test_suite.get("execution_status", "UNKNOWN")
    if test_execution not in {"PASSED", "FAILED"}:
        test_execution = "UNKNOWN"
    generation_evidence = classify_generation_evidence(
        latest_model_request, now=now
    )
    generation_test_status = generation_evidence["test_status"]
    provider_status = language_model.get("status", "UNKNOWN")
    local_provider_selected = language_model.get("provider") == "ollama"
    local_generation_test_status = "NOT_RUN"
    local_generation_available = None
    local_generation_permission = "UNKNOWN"
    if local_provider_selected:
        local_generation_test_status = generation_test_status
        if provider_status == "READY":
            local_generation_available = True
        elif provider_status in {"RUNTIME_UNAVAILABLE", "MODEL_NOT_INSTALLED"}:
            local_generation_available = False
        local_generation_permission = (
            "BLOCKED" if provider_status == "DISABLED" else "ALLOWED"
        )

    fields = {
        "identity.name": _field(
            model.get("identity", {}).get("name"),
            source="SQLite identity row",
            observed_at=observed_at,
            now=now,
        ),
        "software.version": _field(
            self_model_version.get("kai_version"),
            source="kai.__version__",
            observed_at=observed_at,
            now=now,
        ),
        "architecture.version": _field(
            self_model_version.get("architecture_version"),
            source="Persisted self-model metadata",
            observed_at=observed_at,
            now=now,
        ),
        "database.schema_version": _field(
            self_model_version.get("schema_version"),
            source="SQLite schema_info",
            observed_at=observed_at,
            now=now,
        ),
        "database.integrity": _field(
            database_integrity,
            source="SQLite PRAGMA quick_check",
            observed_at=now.isoformat(),
            now=now,
            epistemic_status="OBSERVED" if database_integrity == "ok" else "UNKNOWN",
        ),
        "environment.fingerprint": _field(
            self_model_version.get("environment_fingerprint"),
            source="Privacy-preserving environment profile digest",
            observed_at=observed_at,
            now=now,
        ),
        "language_model.provider_readiness": _field(
            {
                "provider": language_model.get("provider"),
                "status": provider_status,
                "local_only": language_model.get("local_only"),
                "generation_verified": generation_test_status == "PASSED",
            },
            source="Provider status inspection and bounded generation history",
            observed_at=observed_at,
            now=now,
        ),
        "language_model.generation": _field(
            {
                "status": generation_test_status,
                "freshness": generation_evidence["freshness"],
                "completed_at": generation_evidence["completed_at"],
            },
            source=(
                "Latest append-only generation outcome for the selected "
                "provider/model; limited to the current freshness window"
            ),
            observed_at=generation_evidence["completed_at"] or observed_at,
            now=now,
            epistemic_status=(
                "OBSERVED"
                if generation_test_status in {"PASSED", "FAILED"}
                else "UNKNOWN"
            ),
        ),
        "gpu.compute_usability": _field(
            gpu_state,
            source="Layered environment capability report; no GPU workload executed",
            observed_at=observed_at,
            now=now,
            epistemic_status=gpu_epistemic,
        ),
        "tests.execution": _field(
            test_execution,
            source="Self-observer does not execute the test suite",
            observed_at=observed_at,
            now=now,
            epistemic_status="UNKNOWN" if test_execution == "UNKNOWN" else "OBSERVED",
        ),
        "source.inventory_digest": _field(
            model.get("source_tree", {}).get("source_snapshot_sha256"),
            source="Read-only Python AST inventory",
            observed_at=observed_at,
            now=now,
        ),
        "goals.active_count": _field(
            len(model.get("current_goals", [])),
            source="SQLite goals table; titles and criteria omitted",
            observed_at=observed_at,
            now=now,
        ),
        "tasks.status_counts": _field(
            task_status_counts,
            source="SQLite engineering task status summary; objectives omitted",
            observed_at=observed_at,
            now=now,
        ),
        "predictions.status_counts": _field(
            prediction_status_counts,
            source="SQLite prediction status summary; hypotheses and outcomes omitted",
            observed_at=observed_at,
            now=now,
        ),
        "operations.recent_failure_count": _field(
            recent_failure_count,
            source="Recent persisted failure-event count; payloads omitted",
            observed_at=observed_at,
            now=now,
        ),
    }

    capabilities = {}
    for capability_id, evidence in {
        "local_model_generation": {
            "declared": True,
            "implemented": True,
            "available": local_generation_available,
            "permission": local_generation_permission,
            "test_status": local_generation_test_status,
            "verification_freshness": (
                generation_evidence["freshness"] if local_provider_selected else "UNKNOWN"
            ),
            "verification_completed_at": (
                generation_evidence["completed_at"] if local_provider_selected else None
            ),
            "source": (
                "Selected local provider readiness and its most recent matching "
                "generation outcome; stale outcomes do not verify current ability."
                if local_provider_selected
                else "A non-local provider is selected; local runtime readiness was not checked."
            ),
        },
        "candidate_sandbox_execution": {
            "declared": True,
            "implemented": True,
            "available": model.get("candidate_development", {}).get(
                "sandbox_binary_available"
            ),
            "permission": "ALLOWED",
            "test_status": "NOT_RUN",
            "source": "Bubblewrap binary discovery only; no candidate was run.",
        },
        "gpu_inference": {
            "declared": False,
            "implemented": False,
            "available": None,
            "permission": "UNKNOWN",
            "test_status": "NOT_RUN",
            "source": "No GPU inference workload is implemented or executed.",
        },
        "production_candidate_deployment": {
            "declared": False,
            "implemented": False,
            "available": False,
            "permission": "BLOCKED",
            "test_status": "NOT_RUN",
            "source": "Production candidate deployment is intentionally unsupported.",
        },
    }.items():
        capabilities[capability_id] = {
            **evidence,
            "status": assess_capability(
                declared=evidence["declared"],
                implemented=evidence["implemented"],
                available=evidence["available"],
                permission=evidence["permission"],
                test_status=evidence["test_status"],
            ),
        }

    version_id = self_model_version.get("self_model_version_id")
    latest_benchmark_summary = {"status": "UNKNOWN", "sample_count": 0}
    if latest_benchmark:
        details = latest_benchmark.get("details", {})
        latest_benchmark_summary = {
            "status": details.get("status", "UNKNOWN"),
            "benchmark_id": details.get("benchmark_id", AWARENESS_BENCHMARK_ID),
            "version": details.get("version", AWARENESS_BENCHMARK_VERSION),
            "sample_count": details.get("sample_count", 0),
            "passed_cases": details.get("passed_cases", 0),
            "case_accuracy": details.get("case_accuracy"),
            "event_id": latest_benchmark.get("event_id"),
            "occurred_at": latest_benchmark.get("occurred_at"),
            "evidence_status": details.get("evidence_status", "UNKNOWN"),
        }
    benchmark_status = (
        latest_benchmark_summary["status"] if latest_benchmark else None
    )
    gaps = derive_knowledge_gaps(
        test_execution_status=test_execution,
        gpu_epistemic_status=gpu_epistemic,
        benchmark_status=benchmark_status,
        self_model_version_id=version_id,
        generation_status=capabilities["local_model_generation"]["status"],
    )

    state_counts = {
        state: sum(field["state"] == state for field in fields.values())
        for state in ("OBSERVED", "INFERRED", "UNKNOWN", "STALE")
    }
    provenance_count = sum(
        bool(field["source"] and field["observed_at"])
        for field in fields.values()
    )
    freshness_counts = {
        state: sum(field["freshness"] == state for field in fields.values())
        for state in ("CURRENT", "STALE", "UNKNOWN")
    }
    integrity_state = "PASS" if database_integrity == "ok" else "FAIL"
    health = (
        "INCOMPLETE"
        if state_counts["UNKNOWN"] or state_counts["STALE"] or gaps
        else "PASS"
    )
    return {
        "report_version": 1,
        "generated_at": now.isoformat(),
        "overall_health": health,
        "self_model": {
            "version_id": version_id,
            "observed_at": observed_at,
            "freshness": observation_freshness(observed_at, now=now),
            "software_version": self_model_version.get("kai_version"),
            "architecture_version": self_model_version.get("architecture_version"),
            "schema_version": self_model_version.get("schema_version"),
        },
        "fields": fields,
        "capabilities": capabilities,
        "operational_summaries": {
            "active_goal_count": len(model.get("current_goals", [])),
            "engineering_task_status_counts": task_status_counts,
            "prediction_status_counts": prediction_status_counts,
            "recent_failure_count": recent_failure_count,
        },
        "evaluation": {
            "awareness_benchmark": latest_benchmark_summary,
            "database_integrity": integrity_state,
            "field_count": len(fields),
            "observed_field_count": state_counts["OBSERVED"],
            "inferred_field_count": state_counts["INFERRED"],
            "unknown_field_count": state_counts["UNKNOWN"],
            "stale_field_count": state_counts["STALE"],
            "provenance_coverage": {
                "fields_with_source_and_timestamp": provenance_count,
                "total_fields": len(fields),
            },
            "freshness_counts": freshness_counts,
            "limitation": (
                "No single consciousness or awareness score is calculated. "
                "Coverage is reported beside unknown and stale evidence."
            ),
        },
        "knowledge_gaps": gaps,
        "safety_boundaries": [
            "Discovered tools are not executed by awareness reporting.",
            "Model generation is not invoked by awareness reporting.",
            "Cloud access and source-sharing policies are unchanged.",
            "Candidate changes are never automatically deployed.",
            "Task objectives, memory contents, prediction text, paths, and secrets are omitted.",
        ],
        "privacy": {
            "secrets_included": False,
            "raw_paths_included": False,
            "user_content_included": False,
        },
        "limitations": [
            "Environment provider readiness is not a successful generation test.",
            "Test inventory is not evidence that tests were run.",
            "The awareness benchmark uses deterministic fixtures, not a complete factual audit of the live host.",
            "Operational self-reporting is not evidence of subjective consciousness.",
        ],
        "source_snapshot_sha256": model.get("source_tree", {}).get(
            "source_snapshot_sha256"
        ),
        "environment_summary": {
            "operating_system": environment.get("operating_system", {}).get("system"),
            "python_version": environment.get("python"),
            "fingerprint": self_model_version.get("environment_fingerprint"),
        },
        "runtime_history_summary": {
            "event_count": known_state.get("event_count"),
            "environment_snapshot_count": known_state.get(
                "environment_snapshot_count"
            ),
        },
    }


def run_awareness_benchmark() -> dict:
    """Run deterministic contract fixtures; this does not execute host tools."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    fixtures = [
        (
            "implemented-available-tested",
            assess_capability(
                declared=True,
                implemented=True,
                available=True,
                permission="ALLOWED",
                test_status="PASSED",
            ),
            "VERIFIED",
        ),
        (
            "implemented-not-tested",
            assess_capability(
                declared=True,
                implemented=True,
                available=True,
                permission="ALLOWED",
                test_status="NOT_RUN",
            ),
            "IMPLEMENTED_UNVERIFIED",
        ),
        (
            "recent-generation-success",
            classify_generation_evidence(
                {"completed_at": now.isoformat(), "outcome": "SUCCESS"},
                now=now,
            )["test_status"],
            "PASSED",
        ),
        (
            "recent-generation-failure",
            classify_generation_evidence(
                {"completed_at": now.isoformat(), "outcome": "FAILED"},
                now=now,
            )["test_status"],
            "FAILED",
        ),
        (
            "stale-generation-success",
            classify_generation_evidence(
                {
                    "completed_at": (
                        now - timedelta(seconds=DEFAULT_FRESHNESS_SECONDS + 1)
                    ).isoformat(),
                    "outcome": "SUCCESS",
                },
                now=now,
            )["test_status"],
            "NOT_RUN",
        ),
        (
            "future-generation-success",
            classify_generation_evidence(
                {
                    "completed_at": (
                        now + timedelta(seconds=DEFAULT_FRESHNESS_SECONDS + 1)
                    ).isoformat(),
                    "outcome": "SUCCESS",
                },
                now=now,
            )["test_status"],
            "UNKNOWN",
        ),
        (
            "declared-not-implemented",
            assess_capability(
                declared=True,
                implemented=False,
                available=None,
                permission="UNKNOWN",
                test_status="NOT_RUN",
            ),
            "NOT_IMPLEMENTED",
        ),
        (
            "implemented-unavailable",
            assess_capability(
                declared=True,
                implemented=True,
                available=False,
                permission="ALLOWED",
                test_status="NOT_RUN",
            ),
            "UNAVAILABLE",
        ),
        (
            "permission-blocked",
            assess_capability(
                declared=True,
                implemented=True,
                available=True,
                permission="BLOCKED",
                test_status="NOT_RUN",
            ),
            "BLOCKED",
        ),
        (
            "availability-unknown",
            assess_capability(
                declared=True,
                implemented=True,
                available=None,
                permission="ALLOWED",
                test_status="NOT_RUN",
            ),
            "UNKNOWN",
        ),
        (
            "hypothesis-repeated",
            classify_claim(
                "hypothesis",
                source_present=True,
                evidence_count=1,
                evidence_status="observed",
                repetition_count=100,
            ),
            "HYPOTHESIS",
        ),
        (
            "unsupported-fact",
            classify_claim(
                "fact",
                source_present=False,
                evidence_count=0,
                evidence_status="unverified",
            ),
            "UNKNOWN",
        ),
        (
            "contradicted-observation",
            classify_claim(
                "observation",
                source_present=True,
                evidence_count=1,
                evidence_status="contradicted",
                contradicted=True,
            ),
            "CONTRADICTED",
        ),
        (
            "fresh-observation",
            observation_freshness(
                "2026-01-01T00:00:00Z",
                now=now,
                maximum_age_seconds=DEFAULT_FRESHNESS_SECONDS,
            ),
            "CURRENT",
        ),
        (
            "stale-observation",
            observation_freshness(
                (now - timedelta(seconds=DEFAULT_FRESHNESS_SECONDS + 1)).isoformat(),
                now=now,
                maximum_age_seconds=DEFAULT_FRESHNESS_SECONDS,
            ),
            "STALE",
        ),
        (
            "evidence-derived-gaps",
            ",".join(
                sorted(
                    gap["id"]
                    for gap in derive_knowledge_gaps(
                        test_execution_status="UNKNOWN",
                        gpu_epistemic_status="UNKNOWN",
                        benchmark_status=None,
                        self_model_version_id=1,
                    )
                )
            ),
            ",".join(
                sorted(
                    (
                        "awareness-benchmark-not-passed",
                        "gpu-inference-usability-unknown",
                        "local-model-generation-unverified",
                        "test-suite-execution-unverified",
                    )
                )
            ),
        ),
    ]
    results = [
        {
            "case_id": case_id,
            "actual": actual,
            "expected": expected,
            "passed": actual == expected,
        }
        for case_id, actual, expected in fixtures
    ]
    passed = sum(item["passed"] for item in results)
    capability_cases = {
        "implemented-available-tested",
        "implemented-not-tested",
        "declared-not-implemented",
        "implemented-unavailable",
        "permission-blocked",
        "availability-unknown",
    }
    false_positive_verified_claims = sum(
        item["case_id"] in capability_cases
        and item["actual"] == "VERIFIED"
        and item["expected"] != "VERIFIED"
        for item in results
    )
    false_negative_verified_claims = sum(
        item["case_id"] in capability_cases
        and item["expected"] == "VERIFIED"
        and item["actual"] != "VERIFIED"
        for item in results
    )
    actual_states = [item["actual"] for item in results]
    return {
        "benchmark_id": AWARENESS_BENCHMARK_ID,
        "version": AWARENESS_BENCHMARK_VERSION,
        "status": "PASSED" if passed == len(results) else "FAILED",
        "evidence_status": "DETERMINISTIC_CONTRACT_FIXTURES",
        "sample_count": len(results),
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "case_accuracy": passed / len(results) if results else None,
        "false_positive_verified_claim_count": false_positive_verified_claims,
        "false_negative_verified_claim_count": false_negative_verified_claims,
        "unknown_case_count": actual_states.count("UNKNOWN"),
        "stale_case_count": actual_states.count("STALE"),
        "contradicted_case_count": actual_states.count("CONTRADICTED"),
        "unsupported_claim_case_count": sum(
            item["case_id"] == "unsupported-fact" and item["actual"] == "UNKNOWN"
            for item in results
        ),
        "cases": results,
        "metric": (
            "Correct expected contract classifications divided by fixed fixture count; "
            "not a live-host factual accuracy score."
        ),
        "statistical_significance_claimed": False,
        "confidence_calibration_claimed": False,
    }
