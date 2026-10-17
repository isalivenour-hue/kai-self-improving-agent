import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
import tempfile

from kai.awareness import (
    assess_capability,
    build_awareness_report,
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
        self.assertEqual(result["sample_count"], 12)
        self.assertEqual(result["passed_cases"], 12)
        self.assertEqual(result["case_accuracy"], 1.0)
        self.assertEqual(result["false_positive_verified_claim_count"], 0)
        self.assertEqual(result["false_negative_verified_claim_count"], 0)
        self.assertEqual(result["unknown_case_count"], 2)
        self.assertEqual(result["stale_case_count"], 1)
        self.assertEqual(result["unsupported_claim_case_count"], 1)
        self.assertEqual(
            result["evidence_status"], "DETERMINISTIC_CONTRACT_FIXTURES"
        )
        self.assertFalse(result["statistical_significance_claimed"])
        self.assertFalse(result["confidence_calibration_claimed"])

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
            self.assertEqual(first_runtime.database.schema_version(), 9)

            restarted_database = Database(config.database_path)
            restarted_database.initialize()
            self.assertEqual(
                restarted_database.identity()["identity_id"], identity_before
            )
            event = restarted_database.recent_events(
                limit=1, event_type="AWARENESS_BENCHMARK_COMPLETED"
            )[0]
            self.assertEqual(event["details"]["sample_count"], 12)
            self.assertEqual(event["details"]["status"], "PASSED")
            self.assertEqual(restarted_database.integrity_check(), "ok")


if __name__ == "__main__":
    unittest.main()
