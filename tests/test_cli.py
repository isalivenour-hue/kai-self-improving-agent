import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kai.database import Database

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.environment = os.environ.copy()
        self.environment["KAI_HOME"] = self.temporary_directory.name
        self.environment["PYTHONPATH"] = str(PROJECT_ROOT)

    def run_kai(self, *arguments):
        return subprocess.run(
            [sys.executable, "-m", "kai", *arguments],
            cwd=PROJECT_ROOT,
            env=self.environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )

    def test_boot_and_status_cli(self):
        boot = self.run_kai("boot")
        status = self.run_kai("status")
        self.assertEqual(boot.returncode, 0, boot.stderr)
        self.assertEqual(status.returncode, 0, status.stderr)
        self.assertEqual(json.loads(boot.stdout)["message"], "KAI boot completed.")
        report = json.loads(status.stdout)
        self.assertEqual(report["identity"]["name"], "KAI")
        self.assertGreaterEqual(report["known_state"]["event_count"], 5)
        self.assertEqual(report["known_state"]["environment_snapshot_count"], 2)
        self.assertTrue(report["self_model"]["environment_fingerprint"])
        self.assertTrue(report["claims"])

    def test_measure_command_is_explicit_and_reports_no_gpu_workload(self):
        boot = self.run_kai("boot")
        measured = self.run_kai("measure")
        status = self.run_kai("status")
        self.assertEqual(boot.returncode, 0, boot.stderr)
        self.assertEqual(measured.returncode, 0, measured.stderr)
        self.assertEqual(status.returncode, 0, status.stderr)
        result = json.loads(measured.stdout)
        self.assertEqual(result["measurements"]["cpu"]["status"], "measured")
        self.assertTrue(result["measurements"]["cpu"]["not_a_model_or_ML_benchmark"])
        self.assertEqual(result["measurements"]["gpu"]["status"], "not_measured")
        self.assertEqual(
            json.loads(status.stdout)["measurements"]["cpu"]["status"], "measured"
        )

    def test_bounded_adaptive_prediction_cli_workflow(self):
        insufficient = self.run_kai("predict", "next")
        self.assertEqual(insufficient.returncode, 0, insufficient.stderr)
        self.assertEqual(json.loads(insufficient.stdout)["status"], "INSUFFICIENT DATA")

        seed = self.run_kai("learn", "cpu-baseline")
        self.assertEqual(seed.returncode, 0, seed.stderr)
        seed_result = json.loads(seed.stdout)
        self.assertIsNone(seed_result["prediction_id"])
        self.assertEqual(seed_result["model"]["version"], 1)

        learned = self.run_kai("learn", "cpu-baseline")
        self.assertEqual(learned.returncode, 0, learned.stderr)
        learned_result = json.loads(learned.stdout)
        self.assertIsNotNone(learned_result["prediction_id"])
        self.assertEqual(learned_result["model"]["version"], 2)
        self.assertIsNotNone(learned_result["next_prediction"])

        for command in (
            ("predict", "models"),
            ("predict", "status"),
            ("predict", "history"),
            ("experiments",),
            ("learning",),
            ("status",),
            ("report",),
            ("inspect",),
            ("hardware",),
        ):
            with self.subTest(command=command):
                result = self.run_kai(*command)
                self.assertEqual(result.returncode, 0, result.stderr)
                json.loads(result.stdout)

    def test_experiment_definition_run_history_and_inspection_cli(self):
        shown = self.run_kai("experiments", "show", "cpu-baseline")
        self.assertEqual(shown.returncode, 0, shown.stderr)
        definition = json.loads(shown.stdout)
        self.assertEqual(definition["experiment_id"], "cpu-baseline")
        self.assertEqual(definition["version"], 1)
        self.assertTrue(definition["executable"])

        executed = self.run_kai("experiments", "run", "cpu-baseline")
        self.assertEqual(executed.returncode, 0, executed.stderr)
        execution = json.loads(executed.stdout)
        execution_id = execution["execution"]["execution_id"]
        self.assertEqual(execution["execution"]["execution_status"], "SUCCEEDED")
        self.assertEqual(execution["measurement"]["measurement_status"], "MEASURED")

        history = self.run_kai(
            "experiments", "history", "--experiment", "cpu-baseline"
        )
        self.assertEqual(history.returncode, 0, history.stderr)
        self.assertEqual(json.loads(history.stdout)[0]["execution_id"], execution_id)

        inspected = self.run_kai(
            "experiments", "inspect-execution", str(execution_id)
        )
        self.assertEqual(inspected.returncode, 0, inspected.stderr)
        inspected_execution = json.loads(inspected.stdout)
        self.assertEqual(inspected_execution["result"]["unit"], "seconds")
        self.assertIn("evidence_quality", inspected_execution["evaluation"])

        rejected = self.run_kai(
            "experiments",
            "run",
            "cpu-baseline",
            "--parameters",
            '{"iterations":999999999}',
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("unrecognized arguments", rejected.stderr)

    def test_self_and_improvement_inspection_cli(self):
        for command in (
            ("self", "status"),
            ("self", "observations", "--limit", "5"),
            ("self", "capabilities"),
            ("self", "weaknesses"),
            ("self", "failures"),
            ("self", "metrics"),
            ("awareness", "status"),
            ("improvements", "list"),
            ("improvements", "history"),
            ("candidates", "list"),
        ):
            with self.subTest(command=command):
                result = self.run_kai(*command)
                self.assertEqual(result.returncode, 0, result.stderr)
                json.loads(result.stdout)

    def test_awareness_benchmark_cli_persists_result_for_status_report(self):
        benchmark = self.run_kai("awareness", "benchmark")
        self.assertEqual(benchmark.returncode, 0, benchmark.stderr)
        benchmark_result = json.loads(benchmark.stdout)
        self.assertEqual(benchmark_result["status"], "PASSED")
        self.assertEqual(benchmark_result["sample_count"], 12)

        status = self.run_kai("awareness", "status")
        self.assertEqual(status.returncode, 0, status.stderr)
        report = json.loads(status.stdout)
        self.assertEqual(
            report["evaluation"]["awareness_benchmark"]["status"], "PASSED"
        )
        self.assertEqual(
            report["evaluation"]["awareness_benchmark"]["sample_count"], 12
        )

    def test_evaluations_cli_lists_and_shows_baseline_lineage(self):
        database = Database(Path(self.temporary_directory.name) / "kai.sqlite3")
        database.initialize()
        observation = database.record_self_observation(
            {
                "subject": "CPU predictor",
                "observation_type": "measurement.fixture",
                "observed_at": "2026-01-01T00:00:00Z",
                "value": {"mae": 1.0},
                "source": "CLI test fixture",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
                "kai_version": "0.1.0",
                "environment_context": {"profile": "fixture"},
            }
        )
        opportunity = database.create_improvement_opportunity(
            source_observation_ids=[observation["observation_id"]],
            affected_subsystem="cpu_predictor",
            current_behavior="Fixture baseline",
            evidence={"observation_id": observation["observation_id"]},
            severity="LOW",
            expected_impact="Exercise CLI inspection",
            confidence=None,
        )
        hypothesis = database.create_improvement_hypothesis(
            opportunity_id=opportunity["opportunity_id"],
            observed_evidence={"observation_id": observation["observation_id"]},
            interpretation="Test the evaluation CLI.",
            proposed_intervention="Persist a fixture candidate.",
            expected_outcome={"metric": "mae"},
            uncertainty={"status": "fixture"},
            negative_consequences={"risk": "none"},
        )
        proposal = database.create_improvement_proposal(
            hypothesis_id=hypothesis["hypothesis_id"],
            affected_component="tests",
            current_implementation="none",
            proposed_implementation="fixture",
            reason="CLI coverage",
            evidence={"observation_id": observation["observation_id"]},
            expected_benefit="inspectable baseline",
            expected_cost="none",
            risk="none",
            dependencies=[],
            test_requirements=["CLI list and show"],
            benchmark_requirements=[],
            rollback_strategy="No production state is changed.",
        )
        candidate = database.create_kai_candidate(
            candidate_id="cli-evaluation-fixture",
            proposal_id=proposal["proposal_id"],
            candidate_version=1,
            parent_kai_version="0.1.0",
            source_snapshot_ref="fixture-source",
            source_snapshot_sha256="f" * 64,
            candidate_specification={"algorithm": "fixture"},
            isolation_metadata={
                "isolated": True,
                "production_files_modified": False,
                "production_model_mutation": False,
                "mode": "test-only fixture",
            },
        )
        metrics = {"mae": 1.0}
        baseline = {
            "kai_version": "0.1.0",
            "architecture_version": "0.004",
            "predictive_model_versions": [],
            "experiment_definition_versions": [],
            "configuration": {"fixture": True},
            "environment_context": {"profile": "fixture"},
            "evaluation_dataset": {
                "observation_ids": [observation["observation_id"]],
                "observation_count": 1,
                "paired_forecast_count": 0,
                "name": "fixture",
            },
            "metrics": metrics,
        }
        evaluation = database.record_candidate_evaluation(
            candidate_id=candidate["candidate_id"],
            baseline=baseline,
            evaluation_status="INCONCLUSIVE",
            baseline_metrics=metrics,
            candidate_metrics=metrics,
            comparison={
                "status": "INCONCLUSIVE",
                "primary_metric": "mae",
                "metric_direction": "lower",
                "paired_forecast_count": 0,
            },
            test_status="PASSED",
            benchmark_status="COMPLETED",
            regression_count=0,
            failure_count=0,
            evidence_quality="INSUFFICIENT",
            evidence={"source_observation_ids": [observation["observation_id"]]},
            hypothesis_satisfied=None,
        )
        for command in (
            ("evaluations",),
            ("evaluations", "list"),
            ("evaluations", "show", str(evaluation["evaluation_id"])),
        ):
            with self.subTest(command=command):
                result = self.run_kai(*command)
                self.assertEqual(result.returncode, 0, result.stderr)
                payload = json.loads(result.stdout)
                if command[-1] == str(evaluation["evaluation_id"]):
                    self.assertEqual(payload["baseline"]["metrics"], metrics)
                    self.assertEqual(
                        payload["baseline"]["evaluation_dataset"]["observation_ids"],
                        [observation["observation_id"]],
                    )
                else:
                    self.assertEqual(
                        payload[0]["evaluation_id"], evaluation["evaluation_id"]
                    )

    def test_invalid_memory_confidence_is_rejected(self):
        result = self.run_kai(
            "memory", "add", "--kind", "hypothesis", "--content", "test", "--confidence", "2"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("confidence must be between", result.stderr.lower())

    def test_doctor_checks_integrity(self):
        self.run_kai("boot")
        result = self.run_kai("doctor")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["integrity_check"], "ok")

    def test_self_doctor_reports_candidate_sandbox_readiness(self):
        result = self.run_kai("self", "doctor")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        candidate = report["candidate_development"]
        self.assertIn(candidate["sandbox"]["probe_status"], {"READY", "UNKNOWN", "UNAVAILABLE"})
        self.assertFalse(candidate["execution_policy"]["production_mutation"])
        self.assertEqual(report["state_database"]["integrity"], "ok")

    def test_candidate_workspace_list_initializes_v7_state(self):
        result = self.run_kai("workspaces", "list")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])

    def test_events_support_type_and_time_filters(self):
        self.assertEqual(self.run_kai("boot").returncode, 0)
        result = self.run_kai("events", "--type", "SYSTEM_BOOT", "--limit", "1")
        self.assertEqual(result.returncode, 0, result.stderr)
        events = json.loads(result.stdout)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "SYSTEM_BOOT")
        since = events[0]["occurred_at"]
        filtered = self.run_kai("events", "--since", since)
        self.assertEqual(filtered.returncode, 0, filtered.stderr)
        self.assertTrue(json.loads(filtered.stdout))
        types = self.run_kai("events", "--types")
        self.assertIn("ACTION_REQUESTED", json.loads(types.stdout))

    def test_prediction_cli_records_evaluation_learning_and_memory_summary(self):
        created = self.run_kai(
            "predict",
            "create",
            "--hypothesis",
            "The baseline takes about one second",
            "--expected-outcome",
            "1.0",
            "--context",
            '{"iterations": 100000}',
            "--basis",
            "prior observation",
            "--confidence",
            "0.6",
        )
        self.assertEqual(created.returncode, 0, created.stderr)
        prediction = json.loads(created.stdout)["prediction"]
        self.assertEqual(prediction["status"], "PENDING")
        evaluated = self.run_kai(
            "predict",
            "evaluate",
            str(prediction["prediction_id"]),
            "--observed-outcome",
            "1.1",
            "--tolerance",
            "0.2",
        )
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertEqual(json.loads(evaluated.stdout)["status"], "PARTIALLY_CONFIRMED")
        experiences = self.run_kai("experiences", "--limit", "2")
        self.assertEqual(experiences.returncode, 0, experiences.stderr)
        self.assertEqual(json.loads(experiences.stdout)[0]["status"], "COMPLETED")
        memory = self.run_kai("memory", "list", "--limit", "2")
        self.assertEqual(memory.returncode, 0, memory.stderr)
        report = json.loads(memory.stdout)
        self.assertIn("working", report)
        self.assertTrue(report["learning"])

    def test_memory_consolidation_keeps_candidate_unverified(self):
        database = Database(Path(self.temporary_directory.name) / "kai.sqlite3")
        database.initialize()
        for _ in range(2):
            database.create_experience({"fixture": True}, observation="repeat me")
        result = self.run_kai("memory", "consolidate", "--minimum-occurrences", "2")
        self.assertEqual(result.returncode, 0, result.stderr)
        response = json.loads(result.stdout)
        self.assertEqual(response["verification_status"], "candidate")
        self.assertEqual(len(response["created_candidate_memory_ids"]), 1)


if __name__ == "__main__":
    unittest.main()
