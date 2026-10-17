import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile

from kai.awareness import (
    assess_capability,
    build_awareness_report,
    classify_generation_evidence,
    classify_claim,
    observation_freshness,
    run_awareness_benchmark,
)
from kai.config import KaiConfig
from kai.database import Database
from kai.runtime import KaiRuntime


class AwarenessContractTests(unittest.TestCase):
    def test_capability_state_requires_execution_evidence_to_be_verified(self):
        states = (
            (
                {
                    "declared": True,
                    "implemented": True,
                    "available": True,
                    "permission": "ALLOWED",
                    "test_status": "PASSED",
                },
                "VERIFIED",
            ),
            (
                {
                    "declared": True,
                    "implemented": True,
                    "available": True,
                    "permission": "ALLOWED",
                    "test_status": "NOT_RUN",
                },
                "IMPLEMENTED_UNVERIFIED",
            ),
            (
                {
                    "declared": True,
                    "implemented": False,
                    "available": None,
                    "permission": "UNKNOWN",
                    "test_status": "NOT_RUN",
                },
                "NOT_IMPLEMENTED",
            ),
            (
                {
                    "declared": True,
                    "implemented": True,
                    "available": False,
                    "permission": "ALLOWED",
                    "test_status": "NOT_RUN",
                },
                "UNAVAILABLE",
            ),
            (
                {
                    "declared": True,
                    "implemented": True,
                    "available": True,
                    "permission": "BLOCKED",
                    "test_status": "NOT_RUN",
                },
                "BLOCKED",
            ),
        )
        for evidence, expected in states:
            with self.subTest(expected=expected):
                self.assertEqual(assess_capability(**evidence), expected)

    def test_unknown_stale_and_future_observations_are_not_called_current(self):
        now = datetime(2026, 10, 17, tzinfo=timezone.utc)
        self.assertEqual(
            observation_freshness("2026-10-16T23:59:00Z", now=now), "CURRENT"
        )
        self.assertEqual(
            observation_freshness("2026-10-16T23:00:00Z", now=now), "STALE"
        )
        self.assertEqual(observation_freshness(None, now=now), "UNKNOWN")
        self.assertEqual(
            observation_freshness("2026-10-18T00:00:00Z", now=now), "UNKNOWN"
        )
        with self.assertRaises(ValueError):
            observation_freshness(None, now=now, maximum_age_seconds=-1)

    def test_generation_evidence_is_scoped_by_outcome_and_freshness(self):
        now = datetime(2026, 10, 17, tzinfo=timezone.utc)
        recent_success = classify_generation_evidence(
            {"completed_at": now.isoformat(), "outcome": "SUCCESS"}, now=now
        )
        self.assertEqual(recent_success["test_status"], "PASSED")
        self.assertEqual(recent_success["freshness"], "CURRENT")

        recent_failure = classify_generation_evidence(
            {"completed_at": now.isoformat(), "outcome": "FAILED"}, now=now
        )
        self.assertEqual(recent_failure["test_status"], "FAILED")

        stale_success = classify_generation_evidence(
            {
                "completed_at": "2026-10-16T23:00:00Z",
                "outcome": "SUCCESS",
            },
            now=now,
        )
        self.assertEqual(stale_success["test_status"], "NOT_RUN")
        self.assertEqual(stale_success["freshness"], "STALE")

        future_success = classify_generation_evidence(
            {
                "completed_at": "2026-10-18T00:00:00Z",
                "outcome": "SUCCESS",
            },
            now=now,
        )
        self.assertEqual(future_success["test_status"], "UNKNOWN")
        self.assertEqual(future_success["freshness"], "UNKNOWN")

        malformed = classify_generation_evidence(
            {"completed_at": now.isoformat(), "outcome": "MAYBE"}, now=now
        )
        self.assertEqual(malformed["test_status"], "UNKNOWN")

    def test_claim_repetition_does_not_promote_hypothesis_or_unsupported_fact(self):
        self.assertEqual(
            classify_claim(
                "hypothesis",
                source_present=True,
                evidence_count=1,
                evidence_status="observed",
                repetition_count=100,
            ),
            "HYPOTHESIS",
        )
        self.assertEqual(
            classify_claim(
                "fact",
                source_present=False,
                evidence_count=0,
                evidence_status="unverified",
            ),
            "UNKNOWN",
        )
        self.assertEqual(
            classify_claim(
                "fact",
                source_present=True,
                evidence_count=1,
                evidence_status="verified",
            ),
            "VERIFIED_FACT",
        )
        self.assertEqual(
            classify_claim(
                "observation",
                source_present=True,
                evidence_count=1,
                evidence_status="contradicted",
            ),
            "CONTRADICTED",
        )

    def test_deterministic_benchmark_reports_all_fixture_outcomes(self):
        result = run_awareness_benchmark()
        self.assertEqual(result["status"], "PASSED")
        self.assertEqual(result["sample_count"], 16)
        self.assertEqual(result["passed_cases"], 16)
        self.assertEqual(result["case_accuracy"], 1.0)
        self.assertEqual(result["false_positive_verified_claim_count"], 0)
        self.assertEqual(result["false_negative_verified_claim_count"], 0)
        self.assertEqual(result["unknown_case_count"], 3)
        self.assertEqual(result["stale_case_count"], 1)
        self.assertEqual(result["unsupported_claim_case_count"], 1)
        self.assertEqual(
            result["evidence_status"], "DETERMINISTIC_CONTRACT_FIXTURES"
        )
        self.assertFalse(result["statistical_significance_claimed"])
        self.assertFalse(result["confidence_calibration_claimed"])
        cases_by_id = {case["case_id"]: case for case in result["cases"]}
        self.assertTrue(cases_by_id["stale-generation-success"]["passed"])
        self.assertEqual(
            cases_by_id["stale-generation-success"]["actual"], "NOT_RUN"
        )

    def test_report_preserves_unknowns_and_omits_private_content(self):
        timestamp = "2026-10-17T00:00:00+00:00"
        version = {
            "self_model_version_id": 4,
            "created_at": timestamp,
            "kai_version": "0.1.0",
            "architecture_version": "0.007",
            "schema_version": 9,
            "environment_fingerprint": "f" * 64,
            "model": {
                "identity": {"name": "KAI"},
                "language_model": {
                    "provider": "ollama",
                    "status": "READY",
                    "local_only": True,
                    "model": "private-model-name",
                    "api_key": "never-include-this-secret",
                },
                "available_capabilities": {"gpu_compute_usable": "unknown"},
                "candidate_development": {"sandbox_binary_available": True},
                "test_suite": {
                    "execution_status": "UNKNOWN",
                    "status": "NOT_RUN_BY_SELF_OBSERVER",
                },
                "source_tree": {
                    "source_snapshot_sha256": "a" * 64,
                    "root": "/private/repository/path",
                },
                "current_goals": [
                    {"title": "private goal text", "status": "active"}
                ],
                "recent_runtime_errors": [
                    {"message": "private error payload"}
                ],
            },
        }
        report = build_awareness_report(
            version,
            {
                "environment": {
                    "operating_system": {"system": "Linux"},
                    "python": "3.14",
                },
                "known_state": {
                    "event_count": 9,
                    "environment_snapshot_count": 2,
                },
            },
            database_integrity="ok",
            task_status_counts={"FAILED": 1},
            prediction_status_counts={"PENDING": 2},
            recent_failure_count=1,
            latest_benchmark=None,
            now=datetime(2026, 10, 17, tzinfo=timezone.utc),
        )
        self.assertEqual(report["overall_health"], "INCOMPLETE")
        self.assertEqual(report["fields"]["tests.execution"]["state"], "UNKNOWN")
        self.assertEqual(
            report["capabilities"]["local_model_generation"]["status"],
            "IMPLEMENTED_UNVERIFIED",
        )
        self.assertFalse(
            report["fields"]["language_model.provider_readiness"]["value"][
                "generation_verified"
            ]
        )
        self.assertEqual(
            report["capabilities"]["gpu_inference"]["status"], "NOT_IMPLEMENTED"
        )
        self.assertGreater(report["evaluation"]["unknown_field_count"], 0)
        self.assertEqual(report["operational_summaries"]["active_goal_count"], 1)
        serialized = json.dumps(report)
        for private_value in (
            "never-include-this-secret",
            "private-model-name",
            "private goal text",
            "private error payload",
            "/private/repository/path",
        ):
            self.assertNotIn(private_value, serialized)

    def test_benchmark_persists_append_only_and_survives_runtime_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            config = KaiConfig(Path(directory))
            first_runtime = KaiRuntime(config, configure_logging=False)
            first_runtime.database.initialize()
            identity_before = first_runtime.database.identity()["identity_id"]
            result = first_runtime.awareness_benchmark()
            self.assertEqual(result["status"], "PASSED")
            self.assertEqual(first_runtime.database.schema_version(), 10)

            restarted_database = Database(config.database_path)
            restarted_database.initialize()
            self.assertEqual(
                restarted_database.identity()["identity_id"], identity_before
            )
            event = restarted_database.recent_events(
                limit=1, event_type="AWARENESS_BENCHMARK_COMPLETED"
            )[0]
            self.assertEqual(event["details"]["sample_count"], 16)
            self.assertEqual(event["details"]["status"], "PASSED")
            self.assertEqual(restarted_database.integrity_check(), "ok")

    def test_awareness_uses_only_fresh_generation_for_selected_model(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = KaiRuntime(
                KaiConfig(Path(directory)), configure_logging=False
            )
            runtime.local_llm_status = lambda: {
                "provider": "ollama",
                "model": "qwen2.5:0.5b",
                "status": "READY",
                "local_only": True,
            }
            runtime.database.initialize()
            now = datetime.now(timezone.utc)

            def record_request(request_id, model_id, completed_at, outcome):
                runtime.database.record_llm_request(
                    {
                        "request_id": request_id,
                        "started_at": completed_at,
                        "completed_at": completed_at,
                        "duration_ms": 10,
                        "task_category": "general",
                        "provider": "ollama",
                        "model_id": model_id,
                        "outcome": outcome,
                        "token_usage": {},
                        "error_category": None,
                        "fallback_from": [],
                    }
                )

            record_request(
                "unrelated-model",
                "qwen2.5:1.5b",
                now.isoformat(),
                "SUCCESS",
            )
            report = runtime.awareness_status()
            self.assertEqual(
                report["capabilities"]["local_model_generation"]["status"],
                "IMPLEMENTED_UNVERIFIED",
            )

            stale_time = (now - timedelta(seconds=301)).isoformat()
            record_request(
                "stale-selected-model",
                "qwen2.5:0.5b",
                stale_time,
                "SUCCESS",
            )
            stale_report = runtime.awareness_status()
            self.assertEqual(
                stale_report["capabilities"]["local_model_generation"][
                    "verification_freshness"
                ],
                "STALE",
            )
            self.assertEqual(
                stale_report["capabilities"]["local_model_generation"]["status"],
                "IMPLEMENTED_UNVERIFIED",
            )

            record_request(
                "recent-selected-model",
                "qwen2.5:0.5b",
                datetime.now(timezone.utc).isoformat(),
                "SUCCESS",
            )
            verified_report = runtime.awareness_status()
            self.assertEqual(
                verified_report["capabilities"]["local_model_generation"]["status"],
                "VERIFIED",
            )
            self.assertFalse(
                any(
                    gap["id"] == "local-model-generation-unverified"
                    for gap in verified_report["knowledge_gaps"]
                )
            )

            record_request(
                "failed-selected-model",
                "qwen2.5:0.5b",
                datetime.now(timezone.utc).isoformat(),
                "FAILED",
            )
            failed_report = runtime.awareness_status()
            self.assertEqual(
                failed_report["capabilities"]["local_model_generation"]["status"],
                "FAILED",
            )

    def test_cloud_generation_does_not_verify_local_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = KaiRuntime(
                KaiConfig(Path(directory), llm_provider="openai"),
                configure_logging=False,
            )
            runtime.local_llm_status = lambda: {
                "provider": "openai",
                "model": "gpt-test",
                "status": "READY",
                "local_only": False,
            }
            runtime.database.initialize()
            now = datetime.now(timezone.utc).isoformat()
            runtime.database.record_llm_request(
                {
                    "request_id": "cloud-generation",
                    "started_at": now,
                    "completed_at": now,
                    "duration_ms": 10,
                    "task_category": "general",
                    "provider": "openai",
                    "model_id": "gpt-test",
                    "outcome": "SUCCESS",
                    "token_usage": {},
                    "error_category": None,
                    "fallback_from": [],
                }
            )

            report = runtime.awareness_status()
            self.assertTrue(
                report["fields"]["language_model.provider_readiness"]["value"][
                    "generation_verified"
                ]
            )
            self.assertEqual(
                report["capabilities"]["local_model_generation"]["status"],
                "UNKNOWN",
            )


if __name__ == "__main__":
    unittest.main()
