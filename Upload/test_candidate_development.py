import hashlib
import os
import sqlite3
import sys
import tempfile
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from kai.candidate_runner import CandidateExecutionResult, CandidateRunner
from kai.candidate_workspace import CandidateWorkspaceManager
from kai.database import Database
from kai.errors import KaiError
from kai.evaluation_protocol import (
    CPU_PREDICTOR_PROTOCOL,
    summarize_repeated_errors,
    validate_protocol_comparison,
)


class CandidateDevelopmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        (self.source / "kai").mkdir(parents=True)
        (self.source / "tests").mkdir()
        (self.source / "kai" / "__init__.py").write_text("", encoding="utf-8")
        (self.source / "kai" / "cli.py").write_text("def main():\n    return 0\n", encoding="utf-8")
        (self.source / "kai" / "predictive.py").write_text(
            "def forecast(history):\n    return history[-1]\n", encoding="utf-8"
        )
        (self.source / "tests" / "test_predictive.py").write_text(
            "import unittest\n", encoding="utf-8"
        )
        self.state = self.root / "state"
        self.database = Database(self.state / "kai.sqlite3")
        self.database.initialize()
        self.observation = self.database.record_self_observation(
            self._observation(1)
        )
        opportunity = self.database.create_improvement_opportunity(
            source_observation_ids=[self.observation["observation_id"]],
            affected_subsystem="predictor",
            current_behavior="running mean",
            evidence={"observation_id": self.observation["observation_id"]},
            severity="MEDIUM",
            expected_impact="reduce forecast error",
            confidence=None,
        )
        self.hypothesis = self.database.create_improvement_hypothesis(
            opportunity_id=opportunity["opportunity_id"],
            observed_evidence={"observation_ids": [self.observation["observation_id"]]},
            interpretation="A last-observation predictor may improve this stream.",
            proposed_intervention="Compare in an isolated workspace.",
            expected_outcome={"metric": "MAE", "direction": "lower"},
            uncertainty={"status": "UNKNOWN"},
            negative_consequences={"risk": "timing noise"},
        )
        proposal = self.database.create_improvement_proposal(
            hypothesis_id=self.hypothesis["hypothesis_id"],
            affected_component="kai.predictive",
            current_implementation="running mean",
            proposed_implementation="last observation",
            reason="bounded test",
            evidence={"observation_id": self.observation["observation_id"]},
            expected_benefit="lower error",
            expected_cost="sensitivity to noise",
            risk="regression",
            dependencies=["Python"],
            test_requirements=["unit tests"],
            benchmark_requirements=["paired errors"],
            rollback_strategy="No production changes.",
        )
        self.candidate = self.database.create_kai_candidate(
            candidate_id="candidate-test-1",
            proposal_id=proposal["proposal_id"],
            candidate_version=1,
            parent_kai_version="0.1.0",
            source_snapshot_ref="fixture",
            source_snapshot_sha256="a" * 64,
            candidate_specification={"algorithm": "last-observation"},
            isolation_metadata={
                "isolated": True,
                "production_files_modified": False,
                "production_model_mutation": False,
            },
        )
        self.manager = CandidateWorkspaceManager(
            self.source, self.state, self.database
        )
        self.workspace = self.manager.create(
            candidate_id=self.candidate["candidate_id"],
            parent_kai_version="0.1.0",
            source_revision=None,
            schema_version=self.database.schema_version(),
            environment_fingerprint="fixture-profile",
            configuration={"database_path": "private/path"},
        )

    @staticmethod
    def _observation(index):
        return {
            "subject": "CPU predictor",
            "observation_type": "measurement.cpu_baseline_runtime",
            "observed_at": f"2026-01-01T00:00:{index:02d}Z",
            "value": {"duration_seconds": 1.0 + index / 100.0},
            "source": "deterministic test fixture",
            "epistemic_status": "OBSERVED",
            "evidence_status": "observed",
            "confidence": 1.0,
            "kai_version": "0.1.0",
            "environment_context": {"profile": "fixture"},
        }

    def test_protocol_decision_is_derived_from_paired_sequences(self):
        baseline = [[1.0] * 8 for _ in range(3)]
        candidate = [[0.5] * 8 for _ in range(3)]
        comparison = summarize_repeated_errors(baseline, candidate)
        self.assertEqual(comparison["status"], "IMPROVED")
        self.assertEqual(validate_protocol_comparison(comparison, CPU_PREDICTOR_PROTOCOL), "IMPROVED")
        comparison["status"] = "REGRESSED"
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_protocol_comparison(comparison, CPU_PREDICTOR_PROTOCOL)
        altered_metric = summarize_repeated_errors(baseline, candidate)
        altered_metric["candidate_metrics"]["median_sequence_mae_seconds"] += 0.1
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_protocol_comparison(altered_metric, CPU_PREDICTOR_PROTOCOL)
        zero_baseline = summarize_repeated_errors(
            [[0.0] * 8 for _ in range(3)],
            [[1.0] * 8 for _ in range(3)],
        )
        self.assertEqual(zero_baseline["status"], "REGRESSED")

    def test_workspace_dry_run_does_not_register_a_workspace(self):
        preview = self.manager.preview(
            candidate_id=self.candidate["candidate_id"],
            configuration={"data_dir": "private"},
        )
        self.assertEqual(preview["status"], "READY")
        self.assertFalse(preview["workspace_created"])
        self.assertTrue(preview["source_tree_sha256"])
        workspaces = self.database.list_candidate_workspaces()
        self.assertEqual(len(workspaces), 1)
        self.assertEqual(workspaces[0]["workspace_id"], self.workspace["workspace_id"])

    def test_workspace_edits_are_audited_and_paths_are_confined(self):
        workspace_id = self.workspace["workspace_id"]
        modification = self.manager.apply_change(
            workspace_id=workspace_id,
            target_file="kai/new_predictor.py",
            operation="ADD",
            content="def forecast(history):\n    return history[-1]\n",
            reason="test the explicit editor",
            hypothesis_id=self.hypothesis["hypothesis_id"],
        )
        self.assertEqual(modification["validation_status"], "VALID")
        self.assertTrue(modification["patch"].startswith("--- /dev/null"))
        self.assertTrue(self.manager.diff(workspace_id)[0]["matches_recorded_result"])
        self.assertEqual(
            self.database.get_candidate_workspace(workspace_id)["events"][-1][
                "event_type"
            ],
            "MODIFIED",
        )
        with self.assertRaises(KaiError):
            self.manager.apply_change(
                workspace_id=workspace_id,
                target_file="../outside.py",
                operation="ADD",
                content="pass\n",
                reason="path traversal must fail",
                hypothesis_id=self.hypothesis["hypothesis_id"],
            )
        self.assertEqual(self.database.list_candidate_modifications(workspace_id)[0]["modification_id"], modification["modification_id"])

    def test_modification_lineage_must_match_candidate_hypothesis(self):
        with self.assertRaisesRegex(KaiError, "lineage"):
            self.database.record_candidate_modification(
                modification_id="lineage-mismatch",
                candidate_id=self.candidate["candidate_id"],
                workspace_id=self.workspace["workspace_id"],
                target_file="kai/new_predictor.py",
                operation="ADD",
                original_sha256=None,
                resulting_sha256=hashlib.sha256(b"pass\n").hexdigest(),
                patch="+pass\n",
                reason="wrong hypothesis",
                hypothesis_id=self.hypothesis["hypothesis_id"] + 99,
                validation_status="VALID",
                timestamp="2026-01-02T00:00:00Z",
            )

    def test_preserved_workspace_is_immutable_and_cleanup_is_scoped(self):
        workspace_id = self.workspace["workspace_id"]
        preserved = self.manager.cleanup(workspace_id, preserve=True)
        self.assertEqual(preserved["status"], "PRESERVED")
        self.assertTrue(
            (self.manager.workspace_root / preserved["relative_path"]).is_dir()
        )
        with self.assertRaisesRegex(KaiError, "immutable"):
            self.manager.apply_change(
                workspace_id=workspace_id,
                target_file="kai/new_predictor.py",
                operation="ADD",
                content="pass\n",
                reason="preserved workspaces cannot be changed",
                hypothesis_id=self.hypothesis["hypothesis_id"],
            )
        cleaned = self.manager.cleanup(
            self._create_second_candidate_workspace_id(), preserve=False
        )
        self.assertEqual(cleaned["status"], "CLEANED")
        self.assertFalse(
            (self.manager.workspace_root / cleaned["relative_path"]).exists()
        )

    def _create_second_candidate_workspace_id(self):
        observation = self.database.record_self_observation(self._observation(31))
        opportunity = self.database.create_improvement_opportunity(
            source_observation_ids=[observation["observation_id"]],
            affected_subsystem="predictor-cleanup-fixture",
            current_behavior="running mean",
            evidence={"observation_id": observation["observation_id"]},
            severity="LOW",
            expected_impact="cleanup test",
            confidence=None,
        )
        hypothesis = self.database.create_improvement_hypothesis(
            opportunity_id=opportunity["opportunity_id"],
            observed_evidence={"observation_ids": [observation["observation_id"]]},
            interpretation="fixture",
            proposed_intervention="fixture",
            expected_outcome={"metric": "MAE"},
            uncertainty={"status": "UNKNOWN"},
            negative_consequences={"risk": "none"},
        )
        proposal = self.database.create_improvement_proposal(
            hypothesis_id=hypothesis["hypothesis_id"],
            affected_component="kai.predictive",
            current_implementation="a",
            proposed_implementation="b",
            reason="fixture",
            evidence={"observation_id": observation["observation_id"]},
            expected_benefit="fixture",
            expected_cost="fixture",
            risk="fixture",
            dependencies=["Python"],
            test_requirements=["fixture"],
            benchmark_requirements=["fixture"],
            rollback_strategy="No production change.",
        )
        candidate = self.database.create_kai_candidate(
            candidate_id="candidate-test-cleanup",
            proposal_id=proposal["proposal_id"],
            candidate_version=1,
            parent_kai_version="0.1.0",
            source_snapshot_ref="fixture",
            source_snapshot_sha256="c" * 64,
            candidate_specification={"algorithm": "fixture"},
            isolation_metadata={
                "isolated": True,
                "production_files_modified": False,
                "production_model_mutation": False,
            },
        )
        workspace = self.manager.create(
            candidate_id=candidate["candidate_id"],
            parent_kai_version="0.1.0",
            source_revision=None,
            schema_version=self.database.schema_version(),
            environment_fingerprint="fixture-profile",
            configuration={},
        )
        return workspace["workspace_id"]

    @patch("kai.candidate_runner.shutil.which", return_value=None)
    def test_missing_sandbox_is_persisted_as_unknown_not_success(self, _which):
        runner = CandidateRunner(self.database, self.manager)
        result = runner.run_tests(
            self.candidate["candidate_id"],
            self.workspace["workspace_id"],
            module="tests.test_predictive",
        )
        self.assertEqual(result.status, "UNKNOWN")
        stored = self.database.get_candidate_execution(result.execution_id)
        self.assertEqual(stored["status"], "UNKNOWN")
        self.assertIsNotNone(stored["reason"])
        last_event = self.database.get_candidate_workspace(
            self.workspace["workspace_id"]
        )["events"][-1]
        self.assertEqual(last_event["event_type"], "TESTED")
        self.assertEqual(last_event["details"]["status"], "UNKNOWN")

    def test_unittest_zero_tests_is_not_reported_as_success(self):
        now = datetime.now(timezone.utc).isoformat()
        result = CandidateExecutionResult(
            execution_id=str(uuid.uuid4()),
            candidate_id=self.candidate["candidate_id"],
            workspace_id=self.workspace["workspace_id"],
            procedure_id="unittest:tests.test_predictive",
            status="SUCCESS",
            started_at=now,
            completed_at=now,
            duration_seconds=0.0,
            exit_code=0,
            stdout="",
            stderr="Ran 0 tests in 0.000s\n\nOK\n",
            environment={"sandbox": "bubblewrap"},
        )
        runner = CandidateRunner(self.database, self.manager)
        with patch.object(runner, "_run", return_value=result):
            checked = runner.run_tests(
                self.candidate["candidate_id"],
                self.workspace["workspace_id"],
                module="tests.test_predictive",
            )
        self.assertEqual(checked.status, "INVALID")
        self.assertIn("at least one", checked.reason)
        self.assertEqual(
            self.database.get_candidate_execution(checked.execution_id)["status"],
            "INVALID",
        )

    def test_registered_protocol_baseline_and_evaluation_are_immutable(self):
        protocol_record = self.database.register_evaluation_protocol(
            CPU_PREDICTOR_PROTOCOL
        )
        observation_ids = [self.observation["observation_id"]]
        for index in range(2, 31):
            observation_ids.append(
                self.database.record_self_observation(self._observation(index))[
                    "observation_id"
                ]
            )
        comparison = summarize_repeated_errors(
            [[1.0] * 8 for _ in range(3)],
            [[0.5] * 8 for _ in range(3)],
        )
        metrics = {
            "mae_seconds": comparison["baseline_metrics"]["mae_seconds"],
            "algorithm": "running-mean",
        }
        candidate_metrics = {
            "mae_seconds": comparison["candidate_metrics"]["mae_seconds"],
            "algorithm": "last-observation",
        }
        baseline = {
            "kai_version": "0.1.0",
            "architecture_version": "0.007",
            "predictive_model_versions": [],
            "experiment_definition_versions": [],
            "configuration": {"protocol": protocol_record["protocol_id"]},
            "environment_context": {"profile": "fixture"},
            "evaluation_dataset": {
                "observation_ids": observation_ids,
                "observation_count": len(observation_ids),
                "paired_forecast_count": 24,
            },
            "metrics": metrics,
            "evaluation_protocol": protocol_record,
        }
        comparison.update(
            {"primary_metric": "mae_seconds", "metric_direction": "lower"}
        )
        evaluation = self.database.record_candidate_evaluation(
            candidate_id=self.candidate["candidate_id"],
            baseline=baseline,
            evaluation_status="IMPROVED",
            baseline_metrics=metrics,
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            test_status="PASSED",
            benchmark_status="COMPLETED",
            regression_count=0,
            failure_count=0,
            evidence_quality="ADEQUATE",
            evidence={"source_observation_ids": observation_ids},
            hypothesis_satisfied=True,
        )
        self.assertEqual(evaluation["evaluation_protocol_id"], protocol_record["protocol_id"])
        self.assertEqual(
            evaluation["baseline_snapshot_sha256"],
            hashlib.sha256(
                __import__("json").dumps(
                    baseline, sort_keys=True, allow_nan=False
                ).encode("utf-8")
            ).hexdigest(),
        )
        unknown_comparison = {
            **comparison,
            "protocol_status": "IMPROVED",
            "status": "UNKNOWN",
        }
        unknown_evaluation = self.database.record_candidate_evaluation(
            candidate_id=self.candidate["candidate_id"],
            baseline=baseline,
            evaluation_status="UNKNOWN",
            baseline_metrics=metrics,
            candidate_metrics=candidate_metrics,
            comparison=unknown_comparison,
            test_status="NOT_RUN",
            benchmark_status="COMPLETED",
            regression_count=0,
            failure_count=0,
            evidence_quality="ADEQUATE",
            evidence={"source_observation_ids": observation_ids},
            hypothesis_satisfied=None,
        )
        self.assertEqual(unknown_evaluation["evaluation_status"], "UNKNOWN")
        with self.database.connection() as connection:
            with self.assertRaises(sqlite3.IntegrityError):
                connection.execute(
                    "UPDATE evaluation_protocols SET definition_json = '{}' "
                    "WHERE protocol_id = ? AND version = ?",
                    (protocol_record["protocol_id"], protocol_record["version"]),
                )

    @unittest.skipUnless(
        os.environ.get("KAI_RUN_SANDBOX_INTEGRATION") == "1",
        "set KAI_RUN_SANDBOX_INTEGRATION=1 for an explicit bubblewrap smoke test",
    )
    def test_real_bubblewrap_worker_executes_candidate_in_isolation(self):
        if sys.platform != "linux":
            self.skipTest("bubblewrap integration is Linux-only")
        self.manager.apply_change(
            workspace_id=self.workspace["workspace_id"],
            target_file="kai/candidate_algorithm.py",
            operation="ADD",
            content=(
                "def forecast(history):\n"
                "    if not history:\n"
                "        raise ValueError('history must not be empty')\n"
                "    return history[-1]\n"
            ),
            reason="Exercise the fixed worker in the explicit sandbox smoke test.",
            hypothesis_id=self.hypothesis["hypothesis_id"],
        )
        sequences = [
            [1.0 + index / 100.0 for index in range(10)]
            for _ in range(CPU_PREDICTOR_PROTOCOL.sequence_count)
        ]
        runner = CandidateRunner(self.database, self.manager)
        result, errors = runner.run_cpu_predictor_evaluation(
            self.candidate["candidate_id"],
            self.workspace["workspace_id"],
            sequences,
            CPU_PREDICTOR_PROTOCOL.warmup_measurements,
        )
        self.assertEqual(result.status, "SUCCESS", result.stderr or result.reason)
        self.assertIsNotNone(errors)
        self.assertEqual(len(errors["candidate_errors"]), 3)


if __name__ == "__main__":
    unittest.main()
