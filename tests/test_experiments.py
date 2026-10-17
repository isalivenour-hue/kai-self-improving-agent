import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from kai.config import KaiConfig
from kai.database import Database
from kai.errors import KaiError
from kai.experiment_registry import (
    CPU_BASELINE_EXPERIMENT,
    ExperimentDefinition,
    ExperimentRegistry,
    ExperimentResult,
    InvalidExperimentResultError,
    classify_evidence_quality,
    validate_experiment_result,
)
from kai.predictive import cpu_baseline_context
from kai.runtime import KaiRuntime


class ExperimentRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = ExperimentRegistry()

    def test_builtin_registry_is_deterministic_and_rejects_unknown_ids(self):
        definitions = self.registry.list()
        self.assertEqual(
            [(item.experiment_id, item.version) for item in definitions],
            [("cpu-baseline", 1)],
        )
        self.assertEqual(definitions, self.registry.list())
        self.assertTrue(self.registry.contains("cpu-baseline", 1))
        self.assertFalse(self.registry.contains("not-registered"))
        with self.assertRaises(KaiError):
            self.registry.get("not-registered")
        with self.assertRaises(KaiError):
            self.registry.get("cpu-baseline", 2)

    def test_duplicate_registration_and_invalid_definitions_are_rejected(self):
        with self.assertRaises(KaiError):
            self.registry.register(CPU_BASELINE_EXPERIMENT, lambda *_: {})
        with self.assertRaises(ValueError):
            replace(CPU_BASELINE_EXPERIMENT, version=0)
        with self.assertRaises(ValueError):
            replace(CPU_BASELINE_EXPERIMENT, status="UNKNOWN")
        with self.assertRaises(ValueError):
            replace(
                CPU_BASELINE_EXPERIMENT,
                outcome_specification={"not_json": object()},
            )

    def test_registered_definitions_are_isolated_from_mutable_callers(self):
        registry = ExperimentRegistry(include_builtins=False)
        definition = replace(
            CPU_BASELINE_EXPERIMENT,
            parameter_specification={
                key: dict(value)
                for key, value in CPU_BASELINE_EXPERIMENT.parameter_specification.items()
            },
        )
        registry.register(definition, lambda *_: {})
        definition.parameter_specification["iterations"]["value"] = 1
        first_read = registry.get("cpu-baseline")
        self.assertEqual(first_read.parameter_specification["iterations"]["value"], 100000)
        first_read.parameter_specification["iterations"]["value"] = 2
        self.assertEqual(
            registry.get("cpu-baseline").parameter_specification["iterations"]["value"],
            100000,
        )

    def test_disabled_definition_and_unregistered_runner_cannot_execute(self):
        disabled_registry = ExperimentRegistry(include_builtins=False)
        disabled_definition = replace(CPU_BASELINE_EXPERIMENT, status="DISABLED")
        procedure = Mock()
        disabled_registry.register(disabled_definition, procedure)
        with self.assertRaisesRegex(KaiError, "disabled"):
            disabled_registry.execute(
                "cpu-baseline",
                {"iterations": 100000},
                cpu_baseline_context("fixture"),
                {},
                Path("unused"),
            )
        procedure.assert_not_called()

        unbound_registry = ExperimentRegistry(include_builtins=False)
        unbound_registry.register(CPU_BASELINE_EXPERIMENT)
        with self.assertRaisesRegex(KaiError, "not registered"):
            unbound_registry.execute(
                "cpu-baseline",
                {"iterations": 100000},
                cpu_baseline_context("fixture"),
                {},
                Path("unused"),
            )
        unbound_registry.register_procedure(CPU_BASELINE_EXPERIMENT.procedure_id, procedure)
        with self.assertRaisesRegex(KaiError, "already registered"):
            unbound_registry.register_procedure(
                CPU_BASELINE_EXPERIMENT.procedure_id, procedure
            )

    def test_parameters_and_context_are_allowlisted_before_procedure_call(self):
        procedure = Mock(
            return_value=ExperimentResult(
                outcome_value=1.0,
                unit="seconds",
                sample_count=1,
                duration_seconds=1.0,
                metadata={},
                measurement_status="MEASURED",
                raw_result={},
            )
        )
        registry = ExperimentRegistry(include_builtins=False)
        registry.register(CPU_BASELINE_EXPERIMENT, procedure)
        context = cpu_baseline_context("fixture")
        for parameters, candidate_context in (
            ({"iterations": 100001}, context),
            ({"iterations": 100000}, {**context, "hostname": "private"}),
            ({"iterations": 100000}, {**context, "task": "OTHER"}),
        ):
            with self.subTest(parameters=parameters, context=candidate_context):
                with self.assertRaises(KaiError):
                    registry.execute(
                        "cpu-baseline",
                        parameters,
                        candidate_context,
                        {},
                        Path("unused"),
                    )
        procedure.assert_not_called()

    def test_result_validation_rejects_malformed_and_incompatible_values(self):
        invalid_results = (
            {"unit": "seconds"},
            {
                "outcome_value": float("nan"),
                "unit": "seconds",
                "sample_count": 1,
                "measurement_status": "MEASURED",
            },
            {
                "outcome_value": 1.0,
                "unit": "seconds",
                "sample_count": True,
                "measurement_status": "MEASURED",
            },
            {
                "outcome_value": 1.0,
                "unit": "seconds",
                "sample_count": 1,
                "measurement_status": "MEASURED",
                "metadata": {"not_json": object()},
            },
            {
                "outcome_value": None,
                "unit": "seconds",
                "sample_count": 0,
                "measurement_status": "MEASURED",
            },
            {
                "outcome_value": 1.0,
                "unit": "seconds",
                "sample_count": 1,
                "measurement_status": "UNKNOWN",
            },
        )
        for result in invalid_results:
            with self.subTest(result=result), self.assertRaises(
                InvalidExperimentResultError
            ):
                validate_experiment_result(result)

    def test_registry_enforces_definition_result_unit_and_sample_contract(self):
        wrong_result = ExperimentResult(
            outcome_value=1.0,
            unit="milliseconds",
            sample_count=2,
            duration_seconds=1.0,
            metadata={},
            measurement_status="MEASURED",
            raw_result={},
        )
        procedure = Mock(return_value=wrong_result)
        registry = ExperimentRegistry(include_builtins=False)
        registry.register(CPU_BASELINE_EXPERIMENT, procedure)
        with self.assertRaises(InvalidExperimentResultError):
            registry.execute(
                "cpu-baseline",
                {"iterations": 100000},
                cpu_baseline_context("fixture"),
                {},
                Path("unused"),
            )

    def test_evidence_quality_tiers_require_valid_measurements_and_evaluations(self):
        inputs = {
            "context_consistency": True,
            "measurement_status": "MEASURED",
            "prediction_available": True,
            "evaluation_available": True,
        }
        self.assertEqual(
            classify_evidence_quality(sample_count=0, **inputs)[0], "NONE"
        )
        self.assertEqual(
            classify_evidence_quality(sample_count=1, **inputs)[0], "INSUFFICIENT"
        )
        self.assertEqual(
            classify_evidence_quality(sample_count=2, **inputs)[0], "PRELIMINARY"
        )
        self.assertEqual(
            classify_evidence_quality(sample_count=19, **inputs)[0], "PRELIMINARY"
        )
        quality, evidence = classify_evidence_quality(sample_count=20, **inputs)
        self.assertEqual(quality, "ADEQUATE")
        self.assertEqual(evidence["adequate_sample_threshold"], 20)
        self.assertIn("not statistical significance", evidence["threshold_interpretation"])
        self.assertEqual(
            classify_evidence_quality(
                sample_count=20, **{**inputs, "context_consistency": False}
            )[0],
            "INSUFFICIENT",
        )
        self.assertEqual(
            classify_evidence_quality(
                sample_count=20, **{**inputs, "evaluation_available": False}
            )[0],
            "PRELIMINARY",
        )


class ExperimentPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Database(Path(self.temporary_directory.name) / "state.sqlite3")
        self.database.initialize()
        self.context = cpu_baseline_context("fixture-profile")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _complete(self, value, timestamp):
        execution = self.database.begin_cpu_baseline_experiment(
            self.context, timestamp
        )
        result = self.database.complete_cpu_baseline_experiment(
            execution["execution_id"], value, timestamp
        )
        return execution["execution_id"], result

    def test_execution_persists_result_evaluation_and_full_provenance(self):
        execution_id, first = self._complete(2.0, "2026-01-01T00:00:00Z")
        self.assertEqual(first["evaluation_status"], "NOT_EVALUATED")
        self.assertEqual(first["evidence_quality"], "INSUFFICIENT")

        next_execution = self.database.begin_cpu_baseline_experiment(
            self.context, "2026-01-01T00:00:01Z"
        )
        completed = self.database.complete_cpu_baseline_experiment(
            next_execution["execution_id"], 3.0, "2026-01-01T00:00:02Z"
        )
        execution = self.database.get_experiment_execution(
            next_execution["execution_id"]
        )
        self.assertEqual(execution["execution_status"], "SUCCEEDED")
        self.assertEqual(execution["experiment_id"], "cpu-baseline")
        self.assertEqual(execution["experiment_version"], 1)
        self.assertEqual(execution["procedure_id"], "cpu-baseline-v1")
        self.assertEqual(execution["result"]["outcome_value"], 3.0)
        self.assertEqual(execution["result"]["measurement_status"], "MEASURED")
        self.assertEqual(execution["evaluation"]["status"], "EVALUATED")
        self.assertEqual(execution["evaluation"]["expected_value"], 2.0)
        self.assertEqual(execution["evaluation"]["error"], 1.0)
        self.assertEqual(execution["evaluation"]["relative_error"], 0.5)
        self.assertEqual(
            execution["evaluation"]["evidence"]["execution_id"],
            next_execution["execution_id"],
        )
        self.assertEqual(completed["model_update"]["version_after"], 2)
        model_id = completed["model"]["model_id"]
        model_history = self.database.predictive_model_history(model_id)
        self.assertEqual(
            [version["version"] for version in model_history["versions"]], [2, 1]
        )
        self.assertEqual(
            model_history["versions"][1]["parameters"]["mean_seconds"], 2.0
        )
        self.assertEqual(
            model_history["versions"][0]["parameters"]["mean_seconds"], 2.5
        )
        self.assertEqual(model_history["updates"][0]["execution_id"], next_execution["execution_id"])
        self.assertEqual(
            self.database.get_experiment_execution(execution_id)["result"]["unit"],
            "seconds",
        )
        self.assertEqual(self.database.integrity_check(), "ok")
        with self.database.connection() as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_different_contexts_update_separate_models(self):
        self._complete(2.0, "2026-01-01T00:00:00Z")
        other_context = cpu_baseline_context("other-fixture-profile")
        execution = self.database.begin_cpu_baseline_experiment(
            other_context, "2026-01-01T00:00:01Z"
        )
        self.database.complete_cpu_baseline_experiment(
            execution["execution_id"], 4.0, "2026-01-01T00:00:02Z"
        )
        self.assertEqual(len(self.database.list_predictive_models()), 2)
        self.assertEqual(
            self.database.get_cpu_baseline_model(self.context)["parameters"][
                "mean_seconds"
            ],
            2.0,
        )
        self.assertEqual(
            self.database.get_cpu_baseline_model(other_context)["parameters"][
                "mean_seconds"
            ],
            4.0,
        )

    def test_result_persistence_failure_rolls_back_the_entire_model_update(self):
        self._complete(2.0, "2026-01-01T00:00:00Z")
        execution = self.database.begin_cpu_baseline_experiment(
            self.context, "2026-01-01T00:00:01Z"
        )
        with self.database.connection() as connection:
            connection.execute(
                "CREATE TRIGGER reject_experiment_result BEFORE INSERT "
                "ON experiment_results BEGIN SELECT RAISE(ABORT, 'fixture rollback'); END"
            )
            connection.commit()
        with self.assertRaises(KaiError):
            self.database.complete_cpu_baseline_experiment(
                execution["execution_id"], 3.0, "2026-01-01T00:00:02Z"
            )
        with self.database.connection() as connection:
            connection.execute("DROP TRIGGER reject_experiment_result")
            connection.commit()
        after_rollback = self.database.get_experiment_execution(
            execution["execution_id"]
        )
        self.assertEqual(after_rollback["execution_status"], "RUNNING")
        self.assertIsNone(after_rollback["result"])
        model = self.database.get_cpu_baseline_model(self.context)
        self.assertEqual(model["version"], 1)
        self.assertEqual(model["parameters"]["mean_seconds"], 2.0)
        retried = self.database.complete_cpu_baseline_experiment(
            execution["execution_id"], 3.0, "2026-01-01T00:00:03Z"
        )
        self.assertEqual(retried["model"]["version"], 2)

    def test_zero_expected_prediction_omits_relative_error(self):
        self._complete(2.0, "2026-01-01T00:00:00Z")
        execution = self.database.begin_cpu_baseline_experiment(
            self.context, "2026-01-01T00:00:01Z"
        )
        with self.database.connection() as connection:
            prediction_id = connection.execute(
                "SELECT prediction_id FROM experiences WHERE experience_id = ?",
                (execution["experience_id"],),
            ).fetchone()[0]
            connection.execute(
                "UPDATE predictions SET expected_outcome_json = '0' WHERE prediction_id = ?",
                (prediction_id,),
            )
            connection.commit()
        completed = self.database.complete_cpu_baseline_experiment(
            execution["execution_id"], 3.0, "2026-01-01T00:00:02Z"
        )
        self.assertEqual(completed["prediction_error"]["expected_seconds"], 0)
        self.assertIsNone(completed["prediction_error"]["relative_error"])
        stored = self.database.get_experiment_execution(execution["execution_id"])
        self.assertIsNone(stored["evaluation"]["relative_error"])

    def test_failed_execution_retains_unknown_measurement_and_no_model_update(self):
        self._complete(2.0, "2026-01-01T00:00:00Z")
        execution = self.database.begin_cpu_baseline_experiment(
            self.context, "2026-01-01T00:00:01Z"
        )
        self.database.fail_cpu_baseline_experiment(
            execution["execution_id"], "fixture failure", execution_status="UNKNOWN"
        )
        stored = self.database.get_experiment_execution(execution["execution_id"])
        self.assertEqual(stored["execution_status"], "UNKNOWN")
        self.assertEqual(stored["result"]["measurement_status"], "UNKNOWN")
        self.assertEqual(stored["evaluation"]["status"], "UNKNOWN")
        self.assertEqual(stored["output"], {"error": "fixture failure"})
        self.assertEqual(self.database.list_learning_records(), [])

    def test_disabled_persisted_definition_cannot_be_run(self):
        self.database.set_experiment_definition_status("cpu-baseline", "DISABLED", 1)
        with tempfile.TemporaryDirectory() as directory:
            runtime = KaiRuntime(KaiConfig(Path(directory)))
            runtime.database = self.database
            with self.assertRaisesRegex(KaiError, "disabled"):
                runtime.run_experiment("cpu-baseline")
        self.assertEqual(self.database.list_experiment_executions(), [])

    def test_migration_v4_preserves_linked_history_and_identifiers(self):
        path = Path(self.temporary_directory.name) / "legacy-v4.sqlite3"
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
            connection.execute(
                "INSERT INTO experiences (experience_id, created_at, updated_at, status, "
                "context_json, source_event_ids_json) "
                "VALUES (11, '2025-01-01', '2025-01-01', 'COMPLETED', ?, '[15]')",
                (json.dumps({"environment_profile": "legacy"}),),
            )
            connection.execute(
                "INSERT INTO predictive_models VALUES "
                "('legacy-model', 'running_mean', 'cpu_baseline_runtime', 'legacy-context', "
                "1, 'ACTIVE', '2025-01-01', '2025-01-01', '{}', '{}', '{}', '[]')"
            )
            connection.execute(
                "INSERT INTO predictive_model_versions "
                "(model_id, version, created_at, parameters_json, confidence_json, "
                "performance_json, evidence_refs_json, status) "
                "VALUES ('legacy-model', 1, '2025-01-01', '{}', '{}', '{}', '[]', 'ACTIVE')"
            )
            connection.execute(
                "INSERT INTO experiments (experiment_id, experience_id, domain, question, "
                "hypothesis, experiment_type, resource_estimate_json, status, "
                "uncertainty_before_json, started_at, completed_at, output_json) "
                "VALUES (13, 11, 'cpu_baseline_runtime', 'q', 'h', 'measurement', '{}', "
                "'COMPLETED', '{}', '2025-01-01', '2025-01-01', "
                "'{\"measured_runtime_seconds\": 1.5}')"
            )
            connection.execute(
                "INSERT INTO predictions (prediction_id, experience_id, created_at, "
                "context_json, hypothesis, expected_outcome_json, basis, status, "
                "observed_outcome_json, model_id, model_version, experiment_id) "
                "VALUES (12, 11, '2025-01-01', '{}', 'h', '1.0', 'legacy', "
                "'CONFIRMED', '1.5', 'legacy-model', 1, 13)"
            )
            connection.execute(
                "UPDATE experiences SET prediction_id = 12, experiment_id = 13 "
                "WHERE experience_id = 11"
            )
            connection.execute(
                "INSERT INTO learning_records (learning_id, experience_id, created_at, "
                "before_json, happened_json, predicted_json, error_json, changed_json, "
                "why, evidence_json, source_experiences_json) "
                "VALUES (14, 11, '2025-01-01', '{}', '{}', '{}', '{}', '{}', "
                "'legacy learning', '[]', '[11]')"
            )
            connection.execute(
                "INSERT INTO events (event_id, occurred_at, event_type, summary, details_json, "
                "source, epistemic_status, payload_json, experiment_id) "
                "VALUES (15, '2025-01-01', 'LEGACY', 'legacy event', '{}', 'test', "
                "'OBSERVATION', '{}', 13)"
            )
            connection.execute(
                "INSERT INTO predictive_model_updates (update_id, model_id, experience_id, "
                "prediction_id, experiment_id, created_at, version_before, version_after, "
                "update_rule, parameters_before_json, parameters_after_json, "
                "input_evidence_json) VALUES (16, 'legacy-model', 11, 12, 13, "
                "'2025-01-01', 0, 1, 'legacy', '{}', '{}', '{}')"
            )
            connection.commit()

        legacy.initialize()
        self.assertEqual(legacy.schema_version(), 10)
        self.assertEqual(legacy.list_learning_records()[0]["learning_id"], 14)
        self.assertEqual(legacy.get_prediction(12)["prediction_id"], 12)
        self.assertEqual(legacy.get_experiment_execution(13)["execution_id"], 13)
        migrated = legacy.get_experiment_execution(13)
        self.assertEqual(migrated["result"]["outcome_value"], 1.5)
        self.assertEqual(migrated["result"]["measurement_status"], "MEASURED")
        self.assertEqual(legacy.integrity_check(), "ok")
        with legacy.connection() as connection:
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            self.assertEqual(
                connection.execute(
                    "SELECT execution_id FROM experiment_executions"
                ).fetchone()[0],
                13,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT execution_id FROM events WHERE event_id = 15"
                ).fetchone()[0],
                13,
            )
            self.assertEqual(
                connection.execute(
                    "SELECT execution_id FROM predictive_model_updates WHERE update_id = 16"
                ).fetchone()[0],
                13,
            )


class ExperimentRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.runtime = KaiRuntime(KaiConfig(Path(self.temporary_directory.name)))
        self.runtime.database.initialize()
        self.fixture_report = {
            "environment_fingerprint": "fixture-profile",
            "snapshot_id": None,
            "environment": {},
            "self_model": {"identity": self.runtime.database.identity()},
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_unknown_measurement_is_persisted_without_training(self):
        with (
            patch.object(
                self.runtime, "_capture_model", return_value=self.fixture_report
            ),
            patch(
                "kai.experiment_registry.measure_baseline",
                return_value={
                    "captured_at": "2026-01-01T00:00:00Z",
                    "cpu": {"status": "unknown", "reason": "fixture unavailable"},
                    "disk_read": {"status": "unknown"},
                },
            ),
        ):
            result = self.runtime.run_experiment("cpu-baseline")
        self.assertEqual(result["execution"]["execution_status"], "UNKNOWN")
        self.assertEqual(result["execution"]["result"]["measurement_status"], "UNKNOWN")
        self.assertEqual(result["evaluation_status"], "NOT_EVALUATED")
        self.assertIsNone(result["model_update"])
        self.assertEqual(self.runtime.database.list_predictive_models(), [])

    def test_invalid_measurement_is_recorded_before_error_is_reported(self):
        with (
            patch.object(
                self.runtime, "_capture_model", return_value=self.fixture_report
            ),
            patch(
                "kai.experiment_registry.measure_baseline",
                return_value={
                    "captured_at": "2026-01-01T00:00:00Z",
                    "cpu": {
                        "status": "measured",
                        "duration_seconds": float("nan"),
                    },
                    "disk_read": {"status": "unknown"},
                },
            ),
        ):
            with self.assertRaises(InvalidExperimentResultError):
                self.runtime.run_experiment("cpu-baseline")
        execution = self.runtime.database.list_experiment_executions()[0]
        self.assertEqual(execution["execution_status"], "FAILED")
        self.assertEqual(execution["result"]["measurement_status"], "INVALID")
        self.assertEqual(execution["evaluation"]["status"], "INVALID")
        self.assertIn("finite numeric", execution["failure_information"])

    def test_missing_trusted_procedure_is_not_reported_or_started_as_executable(self):
        registry = ExperimentRegistry(include_builtins=False)
        registry.register(CPU_BASELINE_EXPERIMENT)
        self.runtime.experiment_registry = registry
        self.assertFalse(
            self.runtime.show_experiment_definition("cpu-baseline")["executable"]
        )
        with self.assertRaisesRegex(KaiError, "no registered trusted procedure"):
            self.runtime.run_experiment("cpu-baseline")
        self.assertEqual(self.runtime.database.list_experiment_executions(), [])

    def test_malformed_baseline_report_is_recorded_as_invalid(self):
        with (
            patch.object(
                self.runtime, "_capture_model", return_value=self.fixture_report
            ),
            patch("kai.experiment_registry.measure_baseline", return_value={"cpu": None}),
        ):
            with self.assertRaises(InvalidExperimentResultError):
                self.runtime.run_experiment("cpu-baseline")
        execution = self.runtime.database.list_experiment_executions()[0]
        self.assertEqual(execution["result"]["measurement_status"], "INVALID")
        self.assertEqual(execution["evaluation"]["status"], "INVALID")


if __name__ == "__main__":
    unittest.main()
