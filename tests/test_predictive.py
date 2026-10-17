import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kai.config import KaiConfig
from kai.database import Database
from kai.errors import KaiError
from kai.runtime import KaiRuntime
from kai.predictive import (
    CALIBRATION_MINIMUM_SAMPLES,
    CONFIDENCE_MINIMUM_PREDICTIONS,
    confidence_from_statistics,
    cpu_baseline_context,
    initial_parameters,
    prediction_context_key,
    prediction_statistics,
    uncertainty,
    update_running_mean,
    validate_cpu_baseline_context,
)


class PredictiveMathTests(unittest.TestCase):
    def test_context_key_ignores_snapshot_but_isolates_profile_and_workload(self):
        first = cpu_baseline_context("profile-a", 1)
        same_profile_new_snapshot = cpu_baseline_context("profile-a", 99)
        other_profile = cpu_baseline_context("profile-b", 1)
        self.assertEqual(
            prediction_context_key(first),
            prediction_context_key(same_profile_new_snapshot),
        )
        self.assertNotEqual(
            prediction_context_key(first), prediction_context_key(other_profile)
        )

    def test_running_mean_sample_variance_and_unknown_single_sample(self):
        one = initial_parameters(2.0)
        self.assertEqual(uncertainty(one)["status"], "UNKNOWN")
        two = update_running_mean(one, 4.0)
        self.assertEqual(two["observation_count"], 2)
        self.assertEqual(two["mean_seconds"], 3.0)
        self.assertEqual(two["sum_squared_deviations"], 2.0)
        self.assertAlmostEqual(uncertainty(two)["standard_error_seconds"], 1.0)
        three = update_running_mean(two, 3.0)
        self.assertEqual(three["mean_seconds"], 3.0)
        self.assertAlmostEqual(three["sum_squared_deviations"], 2.0)

    def test_nonfinite_and_negative_runtime_is_rejected(self):
        for invalid in (math.nan, math.inf, -1.0, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                initial_parameters(invalid)

    def test_cpu_experiment_context_cannot_change_allowlisted_workload(self):
        context = cpu_baseline_context("fixture-profile")
        validate_cpu_baseline_context(context)
        for key, value in (
            ("task", "ARBITRARY"),
            ("workload_parameters", {"iterations": 1_000_000_000}),
            ("environment_profile", ""),
        ):
            invalid = dict(context)
            invalid[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_cpu_baseline_context(invalid)

    def test_accuracy_calibration_and_insufficient_data_are_explicit(self):
        rows = [
            {
                "status": "CONFIRMED",
                "confidence": 0.8,
                "difference": {"difference": 0.0, "error_magnitude": 0.0},
            },
            {
                "status": "REFUTED",
                "confidence": 0.8,
                "difference": {"difference": -1.0, "error_magnitude": 1.0},
            },
            {"status": "UNKNOWN", "confidence": None, "difference": None},
            {"status": "PENDING", "confidence": None, "difference": None},
        ]
        stats = prediction_statistics(rows)
        self.assertEqual(stats["empirical_accuracy"], 0.5)
        self.assertEqual(stats["mean_signed_error_seconds"], -0.5)
        self.assertEqual(stats["mean_absolute_error_seconds"], 0.5)
        self.assertEqual(
            stats["confidence_vs_observed_accuracy"]["status"], "INSUFFICIENT DATA"
        )
        self.assertEqual(CALIBRATION_MINIMUM_SAMPLES, 20)
        self.assertEqual(CONFIDENCE_MINIMUM_PREDICTIONS, 5)
        enough = prediction_statistics(
            [
                {
                    "status": "CONFIRMED",
                    "confidence": 0.6,
                    "difference": {"difference": 0.0, "error_magnitude": 0.0},
                }
                for _ in range(CALIBRATION_MINIMUM_SAMPLES)
            ]
        )
        self.assertEqual(
            enough["confidence_vs_observed_accuracy"]["status"], "ESTIMATED"
        )
        confidence = confidence_from_statistics(stats)
        self.assertEqual(confidence["status"], "INSUFFICIENT DATA")
        five_scored = prediction_statistics(rows[:1] * CONFIDENCE_MINIMUM_PREDICTIONS)
        self.assertEqual(confidence_from_statistics(five_scored)["status"], "EMPIRICAL")


class PredictiveDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Database(Path(self.temporary_directory.name) / "state.sqlite3")
        self.database.initialize()
        self.context = cpu_baseline_context("deterministic-test-profile")

    def run_experiment(self, seconds, timestamp):
        experiment = self.database.begin_cpu_baseline_experiment(
            self.context, timestamp
        )
        result = self.database.complete_cpu_baseline_experiment(
            experiment["experiment_id"], seconds, timestamp
        )
        return experiment, result

    def test_closed_loop_seeds_predicts_measures_updates_and_predicts_again(self):
        seed_experiment, seed = self.run_experiment(
            2.0, "2026-01-01T00:00:00Z"
        )
        self.assertIsNone(seed_experiment["prediction"])
        self.assertIsNone(seed["prediction_id"])
        self.assertEqual(seed["model_update"]["version_after"], 1)
        self.assertEqual(seed["next_prediction"]["expected_outcome"], 2.0)

        experiment, result = self.run_experiment(
            2.5, "2026-01-01T00:00:01Z"
        )
        self.assertEqual(experiment["prediction"]["expected_outcome"], 2.0)
        self.assertEqual(result["prediction_error"]["signed_error_seconds"], 0.5)
        self.assertEqual(result["prediction_status"], "REFUTED")
        self.assertEqual(result["model_update"]["version_after"], 2)
        self.assertEqual(result["model"]["parameters"]["mean_seconds"], 2.25)
        self.assertEqual(result["next_prediction"]["expected_outcome"], 2.25)
        self.assertEqual(result["next_prediction"]["model_version"], 2)

        followup, next_result = self.run_experiment(
            1.5, "2026-01-01T00:00:02Z"
        )
        self.assertEqual(followup["prediction"]["expected_outcome"], 2.25)
        self.assertEqual(next_result["model"]["parameters"]["mean_seconds"], 2.0)
        self.assertEqual(
            self.database.get_prediction(followup["prediction_id"])["status"],
            "REFUTED",
        )
        self.assertEqual(self.database.integrity_check(), "ok")

    def test_update_rollback_restores_prior_parameters_in_a_new_version(self):
        _, seed = self.run_experiment(2.0, "2026-01-01T00:00:00Z")
        _, update = self.run_experiment(4.0, "2026-01-01T00:00:01Z")
        rolled_back = self.database.rollback_predictive_model_update(
            update["model_update"]["update_id"]
        )
        self.assertEqual(rolled_back["version_before"], 2)
        self.assertEqual(rolled_back["version_after"], 3)
        self.assertEqual(rolled_back["parameters"], seed["model"]["parameters"])
        model = self.database.get_cpu_baseline_model(self.context)
        self.assertEqual(model["version"], 3)
        self.assertEqual(model["parameters"], seed["model"]["parameters"])
        self.assertEqual(
            self.database.get_prediction(update["next_prediction"]["prediction_id"])[
                "status"
            ],
            "EXPIRED",
        )
        self.assertEqual(
            self.database.predictive_model_history(model["model_id"])["updates"][0][
                "reverses_update_id"
            ],
            update["model_update"]["update_id"],
        )

    def test_initial_seed_cannot_restore_a_nonexistent_prior_state(self):
        _, result = self.run_experiment(2.0, "2026-01-01T00:00:00Z")
        with self.assertRaisesRegex(KaiError, "no prior model state"):
            self.database.rollback_predictive_model_update(
                result["model_update"]["update_id"]
            )

    def test_failed_experiment_preserves_prediction_as_unknown(self):
        self.run_experiment(2.0, "2026-01-01T00:00:00Z")
        experiment = self.database.begin_cpu_baseline_experiment(
            self.context, "2026-01-01T00:00:01Z"
        )
        self.database.fail_cpu_baseline_experiment(
            experiment["experiment_id"], "fixed test failure"
        )
        prediction = self.database.get_prediction(experiment["prediction_id"])
        self.assertEqual(prediction["status"], "UNKNOWN")
        self.assertEqual(
            self.database.list_experiments(1)[0]["status"], "FAILED"
        )

    def test_interrupted_experiment_is_recovered_as_unknown(self):
        self.run_experiment(2.0, "2026-01-01T00:00:00Z")
        experiment = self.database.begin_cpu_baseline_experiment(
            self.context, "2026-01-01T00:00:01Z"
        )
        recovered = self.database.recover_interrupted_operations()
        self.assertTrue(recovered)
        self.assertEqual(
            self.database.get_prediction(experiment["prediction_id"])["status"],
            "UNKNOWN",
        )
        self.assertEqual(
            self.database.list_experiments(1)[0]["status"], "INTERRUPTED"
        )
        self.assertEqual(self.database.recover_interrupted_operations(), [])

    def test_completion_failure_rolls_back_model_prediction_and_learning_writes(self):
        experiment = self.database.begin_cpu_baseline_experiment(
            self.context, "2026-01-01T00:00:00Z"
        )
        with self.database.connection() as connection:
            connection.execute(
                "CREATE TRIGGER reject_model_update BEFORE INSERT ON "
                "predictive_model_updates BEGIN SELECT RAISE(ABORT, 'test failure'); END"
            )
            connection.commit()
        with self.assertRaisesRegex(KaiError, "test failure"):
            self.database.complete_cpu_baseline_experiment(
                experiment["experiment_id"], 2.0, "2026-01-01T00:00:01Z"
            )
        self.assertIsNone(self.database.get_cpu_baseline_model(self.context))
        self.assertEqual(
            self.database.list_experiments(1)[0]["status"], "RUNNING"
        )
        self.assertEqual(self.database.list_learning_records(), [])

    def test_v3_to_v6_migration_preserves_identity_and_history(self):
        path = Path(self.temporary_directory.name) / "v3.sqlite3"
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
            identity_id = connection.execute(
                "SELECT identity_id FROM identity WHERE singleton = 1"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO events (occurred_at, event_type, summary, details_json, "
                "source, epistemic_status, payload_json) "
                "VALUES ('2026-01-01T00:00:00Z', 'ERROR', 'retained', '{}', "
                "'test', 'OBSERVATION', '{}')"
            )
            connection.commit()
        legacy.initialize()
        self.assertEqual(legacy.schema_version(), 10)
        self.assertEqual(legacy.identity()["identity_id"], identity_id)
        self.assertEqual(legacy.recent_events()[0]["summary"], "retained")
        self.assertEqual(legacy.integrity_check(), "ok")
        with legacy.connection() as connection:
            self.assertEqual(
                connection.execute("PRAGMA foreign_key_check").fetchall(), []
            )


class PredictiveRuntimeTests(unittest.TestCase):
    def test_runtime_runs_only_mocked_bounded_measurement_and_updates_model(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = KaiRuntime(KaiConfig(Path(directory)))
            runtime.database.initialize()
            fixture_report = {
                "environment_fingerprint": "fake-profile",
                "snapshot_id": None,
                "environment": {},
                "self_model": {
                    "identity": runtime.database.identity(),
                    "environment_fingerprint": "fake-profile",
                },
            }
            measurements = [
                {
                    "captured_at": f"2026-01-01T00:00:0{index}Z",
                    "cpu": {
                        "status": "measured",
                        "duration_seconds": value,
                        "metric": "fixed-loop elapsed wall-clock time",
                        "iterations": 100000,
                        "result_checksum": 100000,
                        "not_a_model_or_ML_benchmark": True,
                    },
                    "disk_read": {"status": "unknown"},
                }
                for index, value in enumerate((2.0, 3.0))
            ]
            with (
                patch.object(runtime, "_capture_model", return_value=fixture_report),
                patch("kai.experiment_registry.measure_baseline", side_effect=measurements),
            ):
                seeded = runtime.run_cpu_baseline_experiment()
                updated = runtime.run_cpu_baseline_experiment()
            self.assertIsNone(seeded["prediction_id"])
            self.assertEqual(updated["prediction_error"]["expected_seconds"], 2.0)
            self.assertEqual(updated["model"]["parameters"]["mean_seconds"], 2.5)
            self.assertEqual(updated["next_prediction"]["expected_outcome"], 2.5)
            self.assertEqual(
                runtime.database.integrity_check(),
                "ok",
            )


if __name__ == "__main__":
    unittest.main()
