import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from kai.benchmark import (
    BENCHMARK_REGISTRY,
    BenchmarkRegistry,
    CPU_PREDICTOR_BENCHMARK,
    benchmark_dataset,
    decide_comparison,
    evaluate_paired_predictions,
    run_registered_benchmark,
    run_cpu_predictor_benchmark,
)
from kai.config import KaiConfig
from kai.runtime import KaiRuntime


class BenchmarkTests(unittest.TestCase):
    def test_specification_is_versioned_and_hashed(self):
        spec = CPU_PREDICTOR_BENCHMARK
        self.assertEqual(spec.benchmark_id, "cpu-predictor-learning")
        self.assertEqual(spec.version, 2)
        self.assertEqual(len(spec.specification_sha256), 64)
        self.assertEqual(spec.specification_sha256, spec.specification_sha256)
        self.assertEqual(spec.to_dict()["sequence_count"], 3)
        with self.assertRaises(ValueError):
            replace(spec, metric_direction="higher")
        with self.assertRaises(ValueError):
            replace(spec, baseline_version="")

    def test_dataset_is_deterministic_and_has_adaptation_split(self):
        first = benchmark_dataset()
        second = benchmark_dataset()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue(all(len(sequence) == 10 for sequence in first))
        self.assertEqual(
            hashlib.sha256(json.dumps(first, separators=(",", ":")).encode()).hexdigest(),
            "ff928fe13d4cfa3f2f2198269a26047a556bfd5f76a26a5f949166f874b072ca",
        )

    def test_known_metric_decisions(self):
        improved = decide_comparison(
            {"mean_absolute_error_seconds": 1.0, "prediction_count": 3, "missing_prediction_count": 0},
            {"mean_absolute_error_seconds": 0.8, "prediction_count": 3, "missing_prediction_count": 0},
        )
        self.assertEqual(improved["status"], "IMPROVED")
        inconclusive = decide_comparison(
            {"mean_absolute_error_seconds": 1.0, "prediction_count": 3, "missing_prediction_count": 0},
            {"mean_absolute_error_seconds": 0.95, "prediction_count": 3, "missing_prediction_count": 0},
        )
        self.assertEqual(inconclusive["status"], "NO_MEANINGFUL_CHANGE")
        failed = decide_comparison(
            {"mean_absolute_error_seconds": 1.0, "prediction_count": 3, "missing_prediction_count": 1},
            {"mean_absolute_error_seconds": 0.1, "prediction_count": 3, "missing_prediction_count": 0},
        )
        self.assertEqual(failed["status"], "FAILED")

    def test_real_deterministic_benchmark_produces_independent_evidence(self):
        report = run_cpu_predictor_benchmark(source_root=Path(__file__).resolve().parents[1])
        self.assertEqual(report["evidence_status"], "SYNTHETIC_REPLAY")
        self.assertEqual(report["correctness"]["status"], "PASSED")
        self.assertEqual(report["baseline"]["prediction_count"], 24)
        self.assertEqual(report["candidate"]["prediction_count"], 24)
        self.assertEqual(report["decision"], "IMPROVED")
        self.assertFalse(report["statistical_significance_claimed"])
        self.assertEqual(len(report["case_results"]), 24)

    def test_generic_decision_policy_distinguishes_all_required_states(self):
        spec = replace(
            CPU_PREDICTOR_BENCHMARK,
            minimum_valid_samples=2,
            expected_sample_count=2,
            strong_improvement_threshold=0.50,
        )

        def run(baseline, candidate, *, unit="seconds", target=0.0):
            return evaluate_paired_predictions(
                spec,
                [
                    {
                        "case_id": f"case-{index}",
                        "target": target,
                        "baseline_prediction": baseline,
                        "candidate_prediction": candidate,
                        "unit": unit,
                    }
                    for index in range(2)
                ],
            )["status"]

        self.assertEqual(evaluate_paired_predictions(spec, [])["status"], "UNKNOWN")
        self.assertEqual(
            run(1.0, 0.0),
            "STRONGLY_IMPROVED",
        )
        self.assertEqual(run(1.0, 0.8), "IMPROVED")
        self.assertEqual(run(1.0, 0.95), "NO_MEANINGFUL_CHANGE")
        self.assertEqual(run(1.0, 1.2), "REGRESSED")
        self.assertEqual(
            evaluate_paired_predictions(
                replace(spec, expected_sample_count=3),
                [
                    {
                        "case_id": "only-case",
                        "target": 1.0,
                        "baseline_prediction": 1.0,
                        "candidate_prediction": 0.0,
                        "unit": "seconds",
                    }
                ],
            )["status"],
            "INCONCLUSIVE",
        )
        self.assertEqual(run(1.0, 0.0, unit="milliseconds"), "INVALID")
        self.assertEqual(
            evaluate_paired_predictions(
                spec,
                [
                    {
                        "case_id": "missing",
                        "target": 1.0,
                        "baseline_prediction": 1.0,
                        "candidate_prediction": None,
                        "unit": "seconds",
                    }
                ],
            )["status"],
            "FAILED",
        )
        self.assertEqual(
            evaluate_paired_predictions(
                spec,
                [
                    {
                        "case_id": "duplicate",
                        "target": 1.0,
                        "baseline_prediction": 1.0,
                        "candidate_prediction": 1.0,
                        "unit": "seconds",
                    }
                ]
                * 2,
            )["status"],
            "INVALID",
        )
        self.assertEqual(
            run(float("nan"), 0.0),
            "INVALID",
        )

    def test_invalid_metric_and_minimum_sample_count_are_explicit(self):
        invalid = decide_comparison(
            {
                "mean_absolute_error_seconds": float("nan"),
                "prediction_count": 2,
                "missing_prediction_count": 0,
            },
            {
                "mean_absolute_error_seconds": 0.0,
                "prediction_count": 2,
                "missing_prediction_count": 0,
            },
        )
        self.assertEqual(invalid["status"], "INVALID")
        too_few = decide_comparison(
            {
                "mean_absolute_error_seconds": 1.0,
                "prediction_count": 0,
                "missing_prediction_count": 0,
            },
            {
                "mean_absolute_error_seconds": 1.0,
                "prediction_count": 0,
                "missing_prediction_count": 0,
            },
        )
        self.assertEqual(too_few["status"], "UNKNOWN")

    def test_registry_reuses_generic_evaluator_for_additional_benchmarks(self):
        spec = replace(
            CPU_PREDICTOR_BENCHMARK,
            benchmark_id="fixture-benchmark",
            primary_metric="mean_absolute_error_items",
            metric_unit="items",
            secondary_metrics=("root_mean_squared_error_items", "prediction_count", "missing_prediction_count"),
            expected_sample_count=1,
        )
        registry = BenchmarkRegistry()
        registry.register(
            spec,
            lambda: [
                {
                    "case_id": "fixture",
                    "target": 0.0,
                    "baseline_prediction": 1.0,
                    "candidate_prediction": 0.5,
                    "unit": "items",
                }
            ],
        )
        result = registry.run("fixture-benchmark")
        self.assertEqual(result["status"], "IMPROVED")
        self.assertEqual(result["evidence_status"], "EVALUATED")
        with self.assertRaises(ValueError):
            registry.register(spec, lambda: [])
        with self.assertRaises(ValueError):
            registry.run("not-registered")

    def test_registered_cpu_benchmark_uses_shared_evaluator(self):
        self.assertEqual(BENCHMARK_REGISTRY.specifications()[0]["benchmark_id"], "cpu-predictor-learning")
        result = run_registered_benchmark("cpu-predictor-learning")
        self.assertEqual(result["status"], "IMPROVED")
        self.assertEqual(result["case_count"], 24)
        self.assertEqual(result["evidence_status"], "EVALUATED")

    def test_cli_runtime_persists_append_only_benchmark_event(self):
        with tempfile.TemporaryDirectory() as directory:
            config = KaiConfig.from_environment()
            config = KaiConfig(
                data_dir=Path(directory),
                llm_provider=config.llm_provider,
                llm_model=config.llm_model,
                llm_timeout_seconds=config.llm_timeout_seconds,
                openai_enabled=False,
                openai_allow_source=False,
            )
            runtime = KaiRuntime(config=config, configure_logging=False)
            result = runtime.run_cpu_predictor_benchmark()
            events = runtime.database.recent_events(event_type="BENCHMARK_COMPLETED")
            self.assertEqual(result["event_id"], events[0]["event_id"])
            self.assertEqual(events[0]["details"]["benchmark"]["specification_sha256"], result["benchmark"]["specification_sha256"])


if __name__ == "__main__":
    unittest.main()
