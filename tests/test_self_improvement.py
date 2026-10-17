import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kai.config import KaiConfig
from kai.database import Database
from kai.errors import KaiError
from kai.runtime import KaiRuntime
from kai.self_improvement import (
    CPU_DEMO_MINIMUM_OBSERVATIONS,
    inspect_source_tree,
    score_forecasters,
    validate_improvement_decision,
    validate_self_observation,
)


class SelfImprovementFixtures(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Database(Path(self.temporary_directory.name) / "kai.sqlite3")
        self.database.initialize()
        self.runtime = KaiRuntime(KaiConfig(Path(self.temporary_directory.name)))
        self.runtime.database = self.database

    def _save_self_model(self, timestamp="2026-01-01T00:00:00Z"):
        identity = self.database.identity()
        observation = {
            "subject": "KAI",
            "observation_type": "capability.example",
            "observed_at": timestamp,
            "value": {"state": "available"},
            "source": "fixture source",
            "epistemic_status": "OBSERVED",
            "evidence_status": "observed",
            "confidence": 0.9,
            "kai_version": "0.1.0",
            "environment_context": {"profile": "fixture"},
        }
        return self.database.save_self_model_version(
            identity_id=identity["identity_id"],
            kai_version="0.1.0",
            architecture_version="0.007",
            schema_version=8,
            environment_fingerprint="fixture",
            source_snapshot_sha256="a" * 64,
            epistemic_summary={"observed": ["capability.example"]},
            model={"identity": identity, "capability": {"example": "available"}},
            observations=[observation],
            timestamp=timestamp,
        )

    def _create_improvement_lineage(self):
        observation = self.database.record_self_observation(
            {
                "subject": "CPU baseline predictor",
                "observation_type": "prediction.error",
                "observed_at": "2026-01-01T00:00:00Z",
                "value": {"absolute_error_seconds": 0.5},
                "source": "fixture measured outcome",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
                "kai_version": "0.1.0",
                "environment_context": {"profile": "fixture"},
            }
        )
        opportunity = self.database.create_improvement_opportunity(
            source_observation_ids=[observation["observation_id"]],
            affected_subsystem="cpu_baseline_predictor",
            current_behavior="Uses running mean",
            evidence={"observation_id": observation["observation_id"]},
            severity="MEDIUM",
            expected_impact="Lower mean absolute error",
            confidence=None,
            timestamp="2026-01-01T00:00:01Z",
        )
        hypothesis = self.database.create_improvement_hypothesis(
            opportunity_id=opportunity["opportunity_id"],
            observed_evidence={"observation_ids": [observation["observation_id"]]},
            interpretation="The current model may lag changes.",
            proposed_intervention="Compare a last-observation candidate.",
            expected_outcome={"metric": "MAE", "direction": "lower"},
            uncertainty={"status": "UNKNOWN"},
            negative_consequences={"risk": "Timing noise"},
            timestamp="2026-01-01T00:00:02Z",
        )
        proposal = self.database.create_improvement_proposal(
            hypothesis_id=hypothesis["hypothesis_id"],
            affected_component="kai.predictive",
            current_implementation="running mean",
            proposed_implementation="last observation",
            reason="Compare using measured data",
            evidence={"source_observation_id": observation["observation_id"]},
            expected_benefit="Lower MAE",
            expected_cost="More sensitive to noise",
            risk="Regression on unstable workloads",
            dependencies=["Python standard library"],
            test_requirements=["deterministic unit tests"],
            benchmark_requirements=["paired prediction errors"],
            rollback_strategy="No production state is changed.",
            timestamp="2026-01-01T00:00:03Z",
        )
        candidate = self.database.create_kai_candidate(
            candidate_id="candidate-fixture-1",
            proposal_id=proposal["proposal_id"],
            candidate_version=1,
            parent_kai_version="0.1.0",
            source_snapshot_ref="fixture-source",
            source_snapshot_sha256="b" * 64,
            candidate_specification={"algorithm": "last observation"},
            isolation_metadata={
                "isolated": True,
                "production_files_modified": False,
                "production_model_mutation": False,
            },
            timestamp="2026-01-01T00:00:04Z",
        )
        return observation, opportunity, hypothesis, proposal, candidate

    def _additional_observation_ids(self, count):
        return [
            self.database.record_self_observation(
                {
                    "subject": "CPU baseline predictor",
                    "observation_type": "measurement.cpu_baseline_runtime",
                    "observed_at": f"2026-01-02T00:00:{index:02d}Z",
                    "value": {"duration_seconds": 1.0 + index / 100.0},
                    "source": "fixture measurement",
                    "epistemic_status": "OBSERVED",
                    "evidence_status": "observed",
                    "confidence": 1.0,
                    "kai_version": "0.1.0",
                    "environment_context": {"profile": "fixture"},
                }
            )["observation_id"]
            for index in range(count)
        ]


class SelfModelTests(SelfImprovementFixtures):
    def test_self_model_versions_and_observations_are_historical(self):
        first = self._save_self_model()
        second = self._save_self_model("2026-01-02T00:00:00Z")
        self.assertNotEqual(
            first["self_model_version_id"], second["self_model_version_id"]
        )
        self.assertEqual(self.database.self_model_version_count(), 2)
        self.assertEqual(
            self.database.get_self_model_version()["model"]["capability"]["example"],
            "available",
        )
        observation = second["observations"][0]
        self.assertEqual(observation["source"], "fixture source")
        self.assertEqual(observation["environment_context"], {"profile": "fixture"})
        self.assertEqual(observation["epistemic_status"], "OBSERVED")
        with self.database.connection() as connection:
            with self.assertRaises(Exception):
                connection.execute(
                    "UPDATE self_observations SET source = 'rewritten' WHERE observation_id = ?",
                    (observation["observation_id"],),
                )
            with self.assertRaises(Exception):
                connection.execute(
                    "DELETE FROM self_model_versions WHERE self_model_version_id = ?",
                    (first["self_model_version_id"],),
                )

    def test_observation_preserves_observed_inferred_unknown_and_provenance(self):
        model = self._save_self_model()
        values = (
            ("observed", "OBSERVED", "observed", 0.9),
            ("inferred", "INFERRED", "unverified", 0.6),
            ("unknown", "UNKNOWN", "unknown", None),
        )
        for name, state, evidence_status, confidence in values:
            self.database.record_self_observation(
                {
                    "subject": "GPU",
                    "observation_type": f"capability.{name}",
                    "observed_at": "2026-01-02T00:00:00Z",
                    "value": {"state": name},
                    "source": "fixture",
                    "epistemic_status": state,
                    "evidence_status": evidence_status,
                    "confidence": confidence,
                    "kai_version": "0.1.0",
                    "environment_context": {
                        "fingerprint": model["environment_fingerprint"]
                    },
                }
            )
        rows = self.database.list_self_observations(
            limit=10, subject="GPU"
        )
        self.assertEqual(
            {row["epistemic_status"] for row in rows},
            {"OBSERVED", "INFERRED", "UNKNOWN"},
        )
        unknown = next(row for row in rows if row["epistemic_status"] == "UNKNOWN")
        self.assertIsNone(unknown["confidence"])
        self.assertEqual(unknown["value"]["state"], "unknown")

    def test_invalid_or_naive_self_observation_is_rejected(self):
        base = {
            "subject": "KAI",
            "observation_type": "test",
            "observed_at": "2026-01-01T00:00:00",
            "value": {},
            "source": "fixture",
            "epistemic_status": "OBSERVED",
            "evidence_status": "observed",
            "confidence": 1.0,
            "kai_version": "0.1.0",
            "environment_context": {},
        }
        with self.assertRaises(ValueError):
            validate_self_observation(base)
        base["observed_at"] = "2026-01-01T00:00:00Z"
        base["epistemic_status"] = "FACT"
        with self.assertRaises(ValueError):
            validate_self_observation(base)

    def test_runtime_self_model_is_generated_from_actual_state_and_source(self):
        first = self.runtime.self_status()
        second = self.runtime.self_status()
        model = second["model"]
        self.assertEqual(second["schema_version"], 10)
        self.assertEqual(model["software"]["version"], self.runtime.version)
        self.assertEqual(model["architecture"]["version"], "0.007")
        self.assertTrue(model["source_tree"]["source_snapshot_sha256"])
        self.assertGreater(model["source_tree"]["test_function_count"], 0)
        self.assertEqual(model["test_suite"]["execution_status"], "UNKNOWN")
        self.assertEqual(
            model["test_suite"]["status"], "NOT_RUN_BY_SELF_OBSERVER"
        )
        self.assertIn(
            "recent_experiment_outcomes",
            model["performance_metrics"],
        )
        self.assertEqual(
            model["performance_metrics"]["recent_experiment_outcomes"][
                "history_limit"
            ],
            100,
        )
        self.assertTrue(
            any(
                item["epistemic_status"] == "UNKNOWN"
                for item in second["observations"]
            )
        )
        self.assertGreater(
            second["self_model_version_id"], first["self_model_version_id"]
        )
        self.assertEqual(
            model["self_model_history_count"],
            self.database.self_model_version_count(),
        )
        self.assertFalse(model["improvement_engine"]["production_deployment_available"])
        self.assertFalse(model["improvement_engine"]["automatic_candidate_generation"])


class ImprovementLedgerTests(SelfImprovementFixtures):
    @staticmethod
    def _baseline(metrics, observation_ids, paired_forecast_count):
        return {
            "kai_version": "0.1.0",
            "architecture_version": "0.004",
            "predictive_model_versions": [],
            "experiment_definition_versions": [],
            "configuration": {"fixture": True},
            "environment_context": {"profile": "fixture"},
            "evaluation_dataset": {
                "observation_ids": observation_ids,
                "observation_count": len(observation_ids),
                "paired_forecast_count": paired_forecast_count,
                "name": "fixture dataset",
            },
            "metrics": metrics,
        }

    def test_opportunity_hypothesis_proposal_candidate_lineage_is_persisted(self):
        observation, opportunity, hypothesis, proposal, candidate = (
            self._create_improvement_lineage()
        )
        self.assertEqual(opportunity["status"], "OPEN")
        self.assertEqual(opportunity["source_observation_ids"], [observation["observation_id"]])
        self.assertEqual(hypothesis["epistemic_status"], "HYPOTHESIS")
        self.assertEqual(proposal["rollback_strategy"], "No production state is changed.")
        self.assertEqual(candidate["parent_kai_version"], "0.1.0")
        self.assertEqual(candidate["deployment_status"], "NOT_DEPLOYED")
        self.assertTrue(candidate["isolation_metadata"]["isolated"])
        with self.assertRaises(KaiError):
            self.database.create_kai_candidate(
                candidate_id="unsafe-candidate",
                proposal_id=proposal["proposal_id"],
                candidate_version=2,
                parent_kai_version="0.1.0",
                source_snapshot_ref="fixture",
                source_snapshot_sha256="c" * 64,
                candidate_specification={"algorithm": "unsafe"},
                isolation_metadata={
                    "isolated": True,
                    "production_files_modified": True,
                    "production_model_mutation": False,
                },
            )

    def test_opportunity_lifecycle_is_validated_and_audited(self):
        _, opportunity, *_ = self._create_improvement_lineage()
        current = self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "INVESTIGATING",
            "Reviewing evidence.",
            {"review": "started"},
            "2026-01-01T00:00:05Z",
        )
        self.assertEqual(current["status"], "INVESTIGATING")
        self.assertEqual(len(current["history"]), 2)
        with self.assertRaises(KaiError):
            self.database.update_improvement_opportunity_status(
                opportunity["opportunity_id"],
                "ACCEPTED",
                "Skip evaluation",
            )

    def test_candidate_evaluation_statuses_and_metrics(self):
        _, _, _, _, candidate = self._create_improvement_lineage()
        observation_ids = [1] + self._additional_observation_ids(21)
        statuses = (
            ("STRONGLY_IMPROVED", "ADEQUATE", 0),
            ("REGRESSED", "ADEQUATE", 1),
            ("FAILED", "ADEQUATE", 0),
            ("INCONCLUSIVE", "ADEQUATE", 0),
        )
        saved = []
        for index, (status, quality, regressions) in enumerate(statuses):
            saved.append(
                self.database.record_candidate_evaluation(
                    candidate_id=candidate["candidate_id"],
                    baseline=self._baseline(
                        {"mae": 2.0}, observation_ids, 20
                    ),
                    evaluation_status=status,
                    baseline_metrics={"mae": 2.0},
                    candidate_metrics={
                        "mae": (
                            1.0
                            if status == "STRONGLY_IMPROVED"
                            else 3.0
                            if status == "REGRESSED"
                            else 2.0
                        )
                    },
                    comparison={
                        "status": status,
                        "primary_metric": "mae",
                        "metric_direction": "lower",
                        "paired_forecast_count": 20,
                    },
                    test_status="FAILED" if status == "FAILED" else "PASSED",
                    benchmark_status="FAILED" if status == "FAILED" else "COMPLETED",
                    regression_count=regressions,
                    failure_count=int(status == "FAILED"),
                    evidence_quality=quality,
                    evidence={
                        "source": "fixture",
                        "source_observation_ids": observation_ids,
                    },
                    hypothesis_satisfied=(
                        True if status == "STRONGLY_IMPROVED" else
                        False if status == "REGRESSED" else None
                    ),
                    timestamp=f"2026-01-01T00:00:0{index + 5}Z",
                )
            )
        self.assertEqual(
            [item["evaluation_status"] for item in saved],
            ["STRONGLY_IMPROVED", "REGRESSED", "FAILED", "INCONCLUSIVE"],
        )
        self.assertEqual(
            saved[0]["baseline"]["metrics"],
            saved[0]["baseline_metrics"],
        )
        self.assertEqual(
            saved[0]["baseline"]["evaluation_dataset"]["observation_ids"],
            observation_ids,
        )
        current = self.database.get_kai_candidate(candidate["candidate_id"])
        self.assertEqual(current["evaluation_status"], "INCONCLUSIVE")
        metrics = self.database.improvement_metrics()
        self.assertEqual(metrics["candidate_count"], 1)
        self.assertEqual(metrics["evaluations_by_status"]["FAILED"], 1)
        self.assertEqual(metrics["regression_rate"], 0.25)
        self.assertFalse(metrics.get("success_rate_claimed", False))

    def test_evaluation_rejects_baseline_metric_or_dataset_substitution(self):
        _, _, _, _, candidate = self._create_improvement_lineage()
        observation_ids = [1] + self._additional_observation_ids(21)
        baseline = self._baseline({"mae": 2.0}, observation_ids, 20)
        with self.assertRaises(KaiError):
            self.database.record_candidate_evaluation(
                candidate_id=candidate["candidate_id"],
                baseline={**baseline, "metrics": {"mae": 3.0}},
                evaluation_status="IMPROVED",
                baseline_metrics={"mae": 2.0},
                candidate_metrics={"mae": 1.9},
                comparison={
                    "status": "IMPROVED",
                    "primary_metric": "mae",
                    "metric_direction": "lower",
                    "paired_forecast_count": 20,
                },
                test_status="PASSED",
                benchmark_status="COMPLETED",
                regression_count=0,
                failure_count=0,
                evidence_quality="ADEQUATE",
                evidence={"source_observation_ids": observation_ids},
                hypothesis_satisfied=True,
            )
        tampered = {
            **baseline,
            "evaluation_dataset": {
                **baseline["evaluation_dataset"],
                "observation_ids": observation_ids[:-1],
            },
        }
        with self.assertRaises(KaiError):
            self.database.record_candidate_evaluation(
                candidate_id=candidate["candidate_id"],
                baseline=tampered,
                evaluation_status="IMPROVED",
                baseline_metrics={"mae": 2.0},
                candidate_metrics={"mae": 1.9},
                comparison={
                    "status": "IMPROVED",
                    "primary_metric": "mae",
                    "metric_direction": "lower",
                    "paired_forecast_count": 20,
                },
                test_status="PASSED",
                benchmark_status="COMPLETED",
                regression_count=0,
                failure_count=0,
                evidence_quality="ADEQUATE",
                evidence={"source_observation_ids": observation_ids},
                hypothesis_satisfied=True,
            )

    def test_accept_requires_measured_improvement_tests_and_adequate_evidence(self):
        _, _, _, _, candidate = self._create_improvement_lineage()
        observation_ids = [1] + self._additional_observation_ids(21)
        with self.assertRaises(ValueError):
            validate_improvement_decision(
                "ACCEPT_CANDIDATE",
                {
                    "status": "IMPROVED",
                    "test_status": "PASSED",
                    "evidence_quality": "PRELIMINARY",
                },
                0,
            )
        with self.assertRaises(ValueError):
            validate_improvement_decision(
                "ACCEPT_CANDIDATE",
                {
                    "status": "IMPROVED",
                    "test_status": "PASSED",
                    "evidence_quality": "ADEQUATE",
                },
                1,
            )
        with self.assertRaises(KaiError):
            self.database.record_candidate_evaluation(
                candidate_id=candidate["candidate_id"],
                baseline=self._baseline(
                    {"mae": 1.0}, observation_ids, 20
                ),
                evaluation_status="IMPROVED",
                baseline_metrics={"mae": 1.0},
                candidate_metrics={"mae": 2.0},
                comparison={
                    "status": "IMPROVED",
                    "primary_metric": "mae",
                    "metric_direction": "lower",
                    "paired_forecast_count": 20,
                },
                test_status="PASSED",
                benchmark_status="COMPLETED",
                regression_count=0,
                failure_count=0,
                evidence_quality="ADEQUATE",
                evidence={"source_observation_ids": observation_ids},
                hypothesis_satisfied=True,
            )
        evaluation = self.database.record_candidate_evaluation(
            candidate_id=candidate["candidate_id"],
            baseline=self._baseline(
                {"mae": 2.0}, observation_ids, 20
            ),
            evaluation_status="IMPROVED",
            baseline_metrics={"mae": 2.0},
            candidate_metrics={"mae": 1.9},
            comparison={
                "improvement_fraction": 0.05,
                "primary_metric": "mae",
                "metric_direction": "lower",
                "paired_forecast_count": 20,
                "status": "IMPROVED",
            },
            test_status="PASSED",
            benchmark_status="COMPLETED",
            regression_count=0,
            failure_count=0,
            evidence_quality="ADEQUATE",
            evidence={"source_observation_ids": observation_ids},
            hypothesis_satisfied=True,
        )
        accepted = self.database.record_improvement_decision(
            candidate_id=candidate["candidate_id"],
            evaluation_id=evaluation["evaluation_id"],
            decision="ACCEPT_CANDIDATE",
            reason="Declared evaluation criterion improved with adequate evidence.",
            evidence={"deployment": "not performed"},
        )
        self.assertEqual(accepted["decision"], "ACCEPT_CANDIDATE")
        self.assertEqual(
            self.database.get_kai_candidate(candidate["candidate_id"])[
                "deployment_status"
            ],
            "NOT_DEPLOYED",
        )
        with self.assertRaises(KaiError):
            self.database.record_improvement_decision(
                candidate_id=candidate["candidate_id"],
                evaluation_id=evaluation["evaluation_id"],
                decision="READY_FOR_DEPLOYMENT",
                reason="This milestone has no deployment mechanism.",
                evidence={},
            )

    def test_failed_evaluation_is_immutable_and_success_requires_new_evaluation(self):
        _, _, _, _, candidate = self._create_improvement_lineage()
        failed = self.database.record_candidate_evaluation(
            candidate_id=candidate["candidate_id"],
            baseline=self._baseline({}, [1], 0),
            evaluation_status="FAILED",
            baseline_metrics={},
            candidate_metrics={},
            comparison={
                "reason": "candidate test failed",
                "primary_metric": "mae",
                "metric_direction": "lower",
                "paired_forecast_count": 0,
                "status": "FAILED",
            },
            test_status="FAILED",
            benchmark_status="FAILED",
            regression_count=0,
            failure_count=1,
            evidence_quality="INSUFFICIENT",
            evidence={"failure": "fixture", "source_observation_ids": [1]},
            hypothesis_satisfied=None,
        )
        with self.database.connection() as connection:
            with self.assertRaises(Exception):
                connection.execute(
                    "UPDATE improvement_evaluations SET evaluation_status = 'IMPROVED' "
                    "WHERE evaluation_id = ?",
                    (failed["evaluation_id"],),
                )
        new_eval = self.database.record_candidate_evaluation(
            candidate_id=candidate["candidate_id"],
            baseline=self._baseline({"mae": 1.0}, [1], 0),
            evaluation_status="INCONCLUSIVE",
            baseline_metrics={"mae": 1.0},
            candidate_metrics={"mae": 1.0},
            comparison={
                "reason": "new evidence but still inconclusive",
                "primary_metric": "mae",
                "metric_direction": "lower",
                "paired_forecast_count": 0,
                "status": "INCONCLUSIVE",
            },
            test_status="PASSED",
            benchmark_status="COMPLETED",
            regression_count=0,
            failure_count=0,
            evidence_quality="INSUFFICIENT",
            evidence={"new_evaluation": True, "source_observation_ids": [1]},
            hypothesis_satisfied=None,
        )
        self.assertNotEqual(failed["evaluation_id"], new_eval["evaluation_id"])
        self.assertEqual(
            self.database.get_improvement_evaluation(failed["evaluation_id"])[
                "evaluation_status"
            ],
            "FAILED",
        )

    def test_ledger_history_and_candidate_production_isolation(self):
        _, opportunity, _, _, candidate = self._create_improvement_lineage()
        with self.database.connection() as connection:
            before_models = connection.execute(
                "SELECT COUNT(*) FROM predictive_models"
            ).fetchone()[0]
        result = self.database.improvement_history()
        self.assertEqual(result["metrics"]["candidate_count"], 1)
        self.assertEqual(result["opportunities"][0]["opportunity_id"], opportunity["opportunity_id"])
        self.assertEqual(candidate["deployment_status"], "NOT_DEPLOYED")
        with self.database.connection() as connection:
            after_models = connection.execute(
                "SELECT COUNT(*) FROM predictive_models"
            ).fetchone()[0]
        self.assertEqual(before_models, after_models)

    def test_source_tree_inventory_uses_ast_without_execution(self):
        report = inspect_source_tree(Path(__file__).resolve().parents[1])
        self.assertGreater(report["source_file_count"], 0)
        self.assertGreater(report["test_function_count"], 0)
        self.assertFalse(report["syntax_failures"])
        self.assertEqual(
            report["source_inventory_method"],
            "Python AST parse; source code is not executed.",
        )
        self.assertEqual(len(report["source_snapshot_sha256"]), 64)

    def test_cpu_predictor_metric_comparison_has_real_status_boundaries(self):
        same = score_forecasters([1.0] * CPU_DEMO_MINIMUM_OBSERVATIONS)
        self.assertEqual(same["status"], "INCONCLUSIVE")
        improved = score_forecasters(
            [1.0, 1.0] + [10.0] * (CPU_DEMO_MINIMUM_OBSERVATIONS - 2)
        )
        self.assertEqual(improved["status"], "STRONGLY_IMPROVED")
        self.assertEqual(improved["paired_forecast_count"], 20)
        self.assertEqual(improved["evidence_quality"], "ADEQUATE")
        regressed = score_forecasters(
            [1.0 if index % 2 == 0 else 10.0 for index in range(CPU_DEMO_MINIMUM_OBSERVATIONS)]
        )
        self.assertEqual(regressed["status"], "REGRESSED")
        insufficient = score_forecasters([1.0, 2.0])
        self.assertEqual(insufficient["status"], "INCONCLUSIVE")
        with self.assertRaises(ValueError):
            score_forecasters([1.0] * 21 + [float("nan")])

    def test_deterministic_improvement_demo_persists_actual_measurement_evidence(self):
        with patch(
            "kai.runtime.measure_baseline",
            side_effect=[
                {
                    "captured_at": f"2026-01-01T00:00:{index:02d}Z",
                    "cpu": {
                        "status": "measured",
                        "duration_seconds": value,
                        "iterations": 100000,
                    },
                    "disk_read": {"status": "unknown"},
                }
                for index, value in enumerate(
                    [1.0, 1.0] + [10.0] * (CPU_DEMO_MINIMUM_OBSERVATIONS - 2)
                )
            ],
        ) as measurement:
            result = self.runtime.demonstrate_cpu_predictor_improvement()
        self.assertEqual(measurement.call_count, CPU_DEMO_MINIMUM_OBSERVATIONS)
        self.assertEqual(result["evaluation"]["evaluation_status"], "STRONGLY_IMPROVED")
        self.assertEqual(result["evaluation"]["evidence_quality"], "ADEQUATE")
        self.assertEqual(result["decision"]["decision"], "ACCEPT_CANDIDATE")
        self.assertFalse(result["production_changed"])
        self.assertEqual(
            len(result["evaluation"]["evidence"]["source_observation_ids"]),
            CPU_DEMO_MINIMUM_OBSERVATIONS,
        )
        self.assertEqual(
            result["candidate"]["deployment_status"],
            "NOT_DEPLOYED",
        )
        self.assertEqual(self.database.list_predictive_models(), [])
        self.assertEqual(
            self.database.get_improvement_opportunity(
                result["opportunity"]["opportunity_id"]
            )["status"],
            "ACCEPTED",
        )

    def test_demo_records_regression_inconclusive_and_unknown_measurement(self):
        with patch(
            "kai.runtime.measure_baseline",
            side_effect=[
                {
                    "captured_at": f"2026-01-01T00:00:{index:02d}Z",
                    "cpu": {
                        "status": "measured",
                        "duration_seconds": value,
                        "iterations": 100000,
                    },
                    "disk_read": {"status": "unknown"},
                }
                for index, value in enumerate(
                    [
                        1.0 if index % 2 == 0 else 10.0
                        for index in range(CPU_DEMO_MINIMUM_OBSERVATIONS)
                    ]
                )
            ],
        ):
            regressed = self.runtime.demonstrate_cpu_predictor_improvement()
        self.assertEqual(regressed["evaluation"]["evaluation_status"], "REGRESSED")
        self.assertEqual(regressed["decision"]["decision"], "REJECT")
        with patch(
            "kai.runtime.measure_baseline",
            side_effect=[
                {
                    "captured_at": f"2026-01-02T00:00:{index:02d}Z",
                    "cpu": {
                        "status": "measured",
                        "duration_seconds": 2.0,
                        "iterations": 100000,
                    },
                    "disk_read": {"status": "unknown"},
                }
                for index in range(CPU_DEMO_MINIMUM_OBSERVATIONS)
            ],
        ):
            inconclusive = self.runtime.demonstrate_cpu_predictor_improvement()
        self.assertEqual(inconclusive["evaluation"]["evaluation_status"], "INCONCLUSIVE")
        self.assertEqual(
            inconclusive["decision"]["decision"], "REQUEST_MORE_EVIDENCE"
        )

        unknown_report = {
            "captured_at": "2026-01-03T00:00:00Z",
            "cpu": {"status": "unknown", "reason": "fixture unavailable"},
            "disk_read": {"status": "unknown"},
        }
        with patch("kai.runtime.measure_baseline", return_value=unknown_report):
            unknown = self.runtime.demonstrate_cpu_predictor_improvement()
        self.assertEqual(unknown["evaluation"]["evaluation_status"], "UNKNOWN")
        self.assertEqual(unknown["decision"]["decision"], "REQUEST_MORE_EVIDENCE")
        self.assertEqual(unknown["measurement_observations"][0]["epistemic_status"], "UNKNOWN")

    def test_failed_demo_measurement_is_preserved_without_success_fallback(self):
        with patch(
            "kai.runtime.measure_baseline",
            side_effect=KaiError("fixture measurement failure"),
        ):
            failed = self.runtime.demonstrate_cpu_predictor_improvement()
        self.assertEqual(failed["evaluation"]["evaluation_status"], "FAILED")
        self.assertEqual(failed["decision"]["decision"], "REJECT")
        self.assertEqual(failed["evaluation"]["failure_count"], 1)
        self.assertFalse(failed["production_changed"])


class SelfImprovementMigrationTests(unittest.TestCase):
    def test_v5_migration_preserves_identity_events_and_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v5.sqlite3"
            legacy = Database(path)
            with legacy.connection() as connection:
                connection.execute(
                    "CREATE TABLE schema_info "
                    "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                    "version INTEGER NOT NULL)"
                )
                connection.execute("INSERT INTO schema_info VALUES (1, 0)")
                Database._migrate_to_v1(connection)
                Database._migrate_to_v2(connection)
                Database._migrate_to_v3(connection)
                Database._migrate_to_v4(connection)
                Database._migrate_to_v5(connection)
                identity = connection.execute(
                    "SELECT identity_id FROM identity WHERE singleton = 1"
                ).fetchone()[0]
                connection.execute(
                    "INSERT INTO events (event_id, occurred_at, event_type, summary, "
                    "details_json, source, payload_json) "
                    "VALUES (33, '2025-01-01T00:00:00Z', 'ERROR', 'preserved', '{}', "
                    "'fixture', '{}')"
                )
                connection.execute("UPDATE schema_info SET version = 5")
                connection.commit()
            legacy.initialize()
            self.assertEqual(legacy.schema_version(), 10)
            self.assertEqual(legacy.identity()["identity_id"], identity)
            self.assertEqual(legacy.recent_events()[0]["event_id"], 33)
            self.assertEqual(legacy.integrity_check(), "ok")
            with legacy.connection() as connection:
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
