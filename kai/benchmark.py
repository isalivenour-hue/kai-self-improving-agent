"""Evidence-driven, model-independent benchmarks for KAI predictors.

The first benchmark is a deterministic replay of the existing CPU prediction
capability.  It compares KAI's running-mean predictor with a last-observation
candidate on fixed, chronological data.  The evaluator is independent of any
language model and returns explicit decision states.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from kai import __version__
from kai.predictive import CPU_BASELINE_DOMAIN, CPU_BASELINE_ITERATIONS


BENCHMARK_ID = "cpu-predictor-learning"
BENCHMARK_VERSION = 2
DECISION_POLICY_VERSION = 2
BASELINE_ALGORITHM = "running_mean"
BASELINE_ALGORITHM_VERSION = "kai.predictive:Welford-1"
CANDIDATE_ALGORITHM = "last_observation"
CANDIDATE_ALGORITHM_VERSION = "benchmark.replay:last-observation-1"


@dataclass(frozen=True)
class BenchmarkSpecification:
    benchmark_id: str
    version: int
    capability: str
    baseline_algorithm: str
    baseline_version: str
    candidate_algorithm: str
    candidate_version: str
    dataset_id: str
    dataset_version: int
    dataset_procedure: str
    input_features: tuple[str, ...]
    prediction_target: str
    adaptation_observations_per_sequence: int
    evaluation_observations_per_sequence: int
    sequence_count: int
    primary_metric: str
    metric_direction: str
    metric_unit: str
    secondary_metrics: tuple[str, ...]
    minimum_meaningful_improvement: float
    strong_improvement_threshold: float
    maximum_acceptable_regression: float
    minimum_valid_samples: int
    expected_sample_count: int | None
    decision_policy_version: int
    correctness_requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "benchmark_id",
            "capability",
            "baseline_algorithm",
            "baseline_version",
            "candidate_algorithm",
            "candidate_version",
            "dataset_id",
            "dataset_procedure",
            "prediction_target",
            "primary_metric",
            "metric_unit",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Benchmark {name} must be non-empty.")
        if self.metric_direction != "lower":
            raise ValueError(
                "The paired-prediction error evaluator supports lower-is-better metrics only."
            )
        if self.primary_metric != f"mean_absolute_error_{self.metric_unit}":
            raise ValueError(
                "The paired-prediction evaluator requires a unit-qualified MAE primary metric."
            )
        for name in ("version", "dataset_version", "decision_policy_version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Benchmark {name} must be a positive integer.")
        if self.expected_sample_count is not None and (
            isinstance(self.expected_sample_count, bool)
            or not isinstance(self.expected_sample_count, int)
            or self.expected_sample_count < self.minimum_valid_samples
        ):
            raise ValueError(
                "Expected sample count must be null or at least the valid-sample minimum."
            )
        for name in (
            "sequence_count",
            "adaptation_observations_per_sequence",
            "evaluation_observations_per_sequence",
            "minimum_valid_samples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Benchmark {name} must be a positive integer.")
        for name in (
            "minimum_meaningful_improvement",
            "strong_improvement_threshold",
            "maximum_acceptable_regression",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"Benchmark {name} must be between 0 and 1.")
        if self.strong_improvement_threshold < self.minimum_meaningful_improvement:
            raise ValueError(
                "Strong improvement threshold cannot be below the meaningful threshold."
            )

    def to_dict(self) -> dict:
        return asdict(self) | {
            "input_features": list(self.input_features),
            "secondary_metrics": list(self.secondary_metrics),
            "correctness_requirements": list(self.correctness_requirements),
        }

    @property
    def specification_sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CPU_PREDICTOR_BENCHMARK = BenchmarkSpecification(
    benchmark_id=BENCHMARK_ID,
    version=BENCHMARK_VERSION,
    capability=CPU_BASELINE_DOMAIN,
    baseline_algorithm=BASELINE_ALGORITHM,
    baseline_version=BASELINE_ALGORITHM_VERSION,
    candidate_algorithm=CANDIDATE_ALGORITHM,
    candidate_version=CANDIDATE_ALGORITHM_VERSION,
    dataset_id="cpu-predictor-replay-drift-v1",
    dataset_version=1,
    dataset_procedure=(
        "Fixed synthetic chronological replay of three CPU-runtime-like sequences; "
        "the first two observations adapt the predictor and later observations are "
        "evaluation-only targets."
    ),
    input_features=("sequence_index", "chronological_position", "iterations"),
    prediction_target="runtime_seconds",
    adaptation_observations_per_sequence=2,
    evaluation_observations_per_sequence=8,
    sequence_count=3,
    primary_metric="mean_absolute_error_seconds",
    metric_direction="lower",
    metric_unit="seconds",
    secondary_metrics=(
        "root_mean_squared_error_seconds",
        "prediction_count",
        "missing_prediction_count",
    ),
    minimum_meaningful_improvement=0.10,
    strong_improvement_threshold=0.75,
    maximum_acceptable_regression=0.10,
    minimum_valid_samples=1,
    expected_sample_count=24,
    decision_policy_version=DECISION_POLICY_VERSION,
    correctness_requirements=(
        "all predictions must be finite non-negative numbers",
        "the candidate must not see evaluation targets before predicting",
        "baseline and candidate must use identical evaluation observations",
        "no missing evaluation predictions are allowed",
    ),
)


def benchmark_dataset(spec: BenchmarkSpecification = CPU_PREDICTOR_BENCHMARK) -> list[list[float]]:
    """Return the immutable deterministic replay dataset."""
    if (
        spec.benchmark_id != CPU_PREDICTOR_BENCHMARK.benchmark_id
        or spec.version != CPU_PREDICTOR_BENCHMARK.version
    ):
        raise ValueError("Only the registered CPU predictor benchmark is available.")
    return [
        [0.0100, 0.0110, 0.0120, 0.0130, 0.0140, 0.0150, 0.0160, 0.0170, 0.0180, 0.0190],
        [0.0200, 0.0210, 0.0220, 0.0230, 0.0240, 0.0250, 0.0260, 0.0270, 0.0280, 0.0290],
        [0.0300, 0.0310, 0.0320, 0.0330, 0.0340, 0.0350, 0.0360, 0.0370, 0.0380, 0.0390],
    ]


def _validate_prediction(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Prediction must be numeric.")
    if not math.isfinite(value) or value < 0:
        raise ValueError("Prediction must be finite and non-negative.")
    return float(value)


def _run_forecaster(
    sequence: Sequence[float],
    algorithm: str,
    adaptation_count: int,
) -> tuple[list[float], list[float]]:
    if len(sequence) <= adaptation_count:
        raise ValueError("Sequence is shorter than the adaptation window.")
    observed = [float(value) for value in sequence]
    if any(not math.isfinite(value) or value < 0 for value in observed):
        raise ValueError("Dataset contains an invalid observation.")
    history = observed[:adaptation_count]
    predictions: list[float] = []
    targets: list[float] = []
    for target in observed[adaptation_count:]:
        if algorithm == BASELINE_ALGORITHM:
            prediction = math.fsum(history) / len(history)
        elif algorithm == CANDIDATE_ALGORITHM:
            prediction = history[-1]
        else:
            raise ValueError(f"Unsupported benchmark algorithm: {algorithm}")
        predictions.append(_validate_prediction(prediction))
        targets.append(target)
        history.append(target)
    return predictions, targets


def _metrics(
    predictions: Sequence[float],
    targets: Sequence[float],
    *,
    metric_unit: str = "seconds",
) -> dict:
    if len(predictions) != len(targets):
        raise ValueError("Prediction and target counts must match.")
    if not predictions:
        raise ValueError("At least one prediction is required.")
    errors = [abs(float(p) - float(t)) for p, t in zip(predictions, targets)]
    return {
        f"mean_absolute_error_{metric_unit}": math.fsum(errors) / len(errors),
        f"root_mean_squared_error_{metric_unit}": math.sqrt(
            math.fsum(error * error for error in errors) / len(errors)
        ),
        "prediction_count": len(predictions),
        "missing_prediction_count": 0,
        "predictions": [float(value) for value in predictions],
        "targets": [float(value) for value in targets],
        "absolute_errors_seconds": errors,
    }


def decide_comparison(
    baseline: dict,
    candidate: dict,
    spec: BenchmarkSpecification = CPU_PREDICTOR_BENCHMARK,
) -> dict:
    """Apply the independent, versioned decision policy to validated metrics."""
    required_metrics = (
        spec.primary_metric,
        "prediction_count",
        "missing_prediction_count",
    )
    for label, metrics in (("baseline", baseline), ("candidate", candidate)):
        if not isinstance(metrics, dict) or any(
            key not in metrics for key in required_metrics
        ):
            return {
                "status": "INVALID",
                "reason": f"{label.title()} metrics are incomplete.",
                "primary_metric": spec.primary_metric,
                "metric_direction": spec.metric_direction,
                "relative_improvement": None,
                "decision_policy_version": spec.decision_policy_version,
            }
        value = metrics[spec.primary_metric]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            return {
                "status": "INVALID",
                "reason": f"{label.title()} primary metric is invalid.",
                "primary_metric": spec.primary_metric,
                "metric_direction": spec.metric_direction,
                "relative_improvement": None,
                "decision_policy_version": spec.decision_policy_version,
            }
        for key in ("prediction_count", "missing_prediction_count"):
            count = metrics[key]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                return {
                    "status": "INVALID",
                    "reason": f"{label.title()} {key} is invalid.",
                    "primary_metric": spec.primary_metric,
                    "metric_direction": spec.metric_direction,
                    "relative_improvement": None,
                    "decision_policy_version": spec.decision_policy_version,
                }

    baseline_error = float(baseline[spec.primary_metric])
    candidate_error = float(candidate[spec.primary_metric])
    relative_change = (
        0.0
        if baseline_error == 0 and candidate_error == 0
        else -1.0
        if baseline_error == 0
        else (baseline_error - candidate_error) / baseline_error
    )
    if baseline["missing_prediction_count"] or candidate["missing_prediction_count"]:
        status = "FAILED"
        reason = "A required prediction was missing."
    elif baseline["prediction_count"] != candidate["prediction_count"]:
        status = "INVALID"
        reason = "Baseline and candidate prediction counts differ."
    elif baseline["prediction_count"] == 0:
        status = "UNKNOWN"
        reason = "No valid predictions were available."
    elif baseline["prediction_count"] < spec.minimum_valid_samples:
        status = "INCONCLUSIVE"
        reason = "The valid paired sample count is below the protocol minimum."
    elif relative_change > 0 and relative_change >= spec.strong_improvement_threshold:
        status = "STRONGLY_IMPROVED"
        reason = "Candidate exceeded the predefined strong-improvement threshold."
    elif relative_change > 0 and relative_change >= spec.minimum_meaningful_improvement:
        status = "IMPROVED"
        reason = "Candidate reduced the primary error beyond the predefined threshold."
    elif relative_change < 0 and relative_change <= -spec.maximum_acceptable_regression:
        status = "REGRESSED"
        reason = "Candidate exceeded the predefined regression threshold."
    else:
        status = "NO_MEANINGFUL_CHANGE"
        reason = "Observed difference did not exceed a predefined decision threshold."
    return {
        "status": status,
        "reason": reason,
        "primary_metric": spec.primary_metric,
        "metric_direction": spec.metric_direction,
        "metric_unit": spec.metric_unit,
        "relative_improvement": relative_change,
        "decision_policy_version": spec.decision_policy_version,
    }


def evaluate_paired_predictions(
    spec: BenchmarkSpecification, cases: Sequence[dict]
) -> dict:
    """Validate paired scalar predictions and derive metrics from raw cases."""
    if not isinstance(cases, (list, tuple)):
        return {"status": "INVALID", "reason": "Cases must be a list or tuple."}
    if not cases:
        return {"status": "UNKNOWN", "reason": "No evaluation cases were provided."}

    seen_ids: set[str] = set()
    baseline_predictions: list[float] = []
    candidate_predictions: list[float] = []
    targets: list[float] = []
    raw_cases: list[dict] = []
    required_fields = {
        "case_id",
        "target",
        "baseline_prediction",
        "candidate_prediction",
        "unit",
    }
    for index, case in enumerate(cases):
        if not isinstance(case, dict) or not required_fields.issubset(case):
            return {
                "status": "INVALID",
                "reason": f"Case {index + 1} does not satisfy the paired-case schema.",
            }
        case_id = case["case_id"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_ids:
            return {
                "status": "INVALID",
                "reason": "Case IDs must be non-empty and unique.",
            }
        seen_ids.add(case_id)
        if case["unit"] != spec.metric_unit:
            return {
                "status": "INVALID",
                "reason": f"Case {case_id!r} uses a unit incompatible with the protocol.",
            }
        if case["target"] is None:
            return {
                "status": "UNKNOWN",
                "reason": f"Case {case_id!r} has no observed target.",
            }
        if case["baseline_prediction"] is None or case["candidate_prediction"] is None:
            return {
                "status": "FAILED",
                "reason": f"Case {case_id!r} is missing a required prediction.",
            }
        numeric_values = (
            case["target"],
            case["baseline_prediction"],
            case["candidate_prediction"],
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in numeric_values
        ):
            return {
                "status": "INVALID",
                "reason": f"Case {case_id!r} contains a non-finite or negative value.",
            }
        target = float(case["target"])
        baseline_prediction = float(case["baseline_prediction"])
        candidate_prediction = float(case["candidate_prediction"])
        targets.append(target)
        baseline_predictions.append(baseline_prediction)
        candidate_predictions.append(candidate_prediction)
        raw_cases.append(
            {
                "case_id": case_id,
                "target": target,
                "baseline_prediction": baseline_prediction,
                "candidate_prediction": candidate_prediction,
                "baseline_absolute_error": abs(baseline_prediction - target),
                "candidate_absolute_error": abs(candidate_prediction - target),
                "unit": spec.metric_unit,
            }
        )

    baseline_metrics = _metrics(
        baseline_predictions, targets, metric_unit=spec.metric_unit
    )
    candidate_metrics = _metrics(
        candidate_predictions, targets, metric_unit=spec.metric_unit
    )
    comparison = decide_comparison(baseline_metrics, candidate_metrics, spec)
    if (
        spec.expected_sample_count is not None
        and len(raw_cases) < spec.expected_sample_count
    ):
        comparison = {
            **comparison,
            "status": "INCONCLUSIVE",
            "reason": "The evaluation run did not contain the expected number of cases.",
        }
    elif spec.expected_sample_count is not None and len(raw_cases) > spec.expected_sample_count:
        return {
            "status": "INVALID",
            "reason": "The evaluation run exceeded the protocol's expected case count.",
        }
    return {
        "status": comparison["status"],
        "reason": comparison["reason"],
        "case_count": len(raw_cases),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "cases": raw_cases,
        "comparison": comparison,
        "evidence_status": (
            "EVALUATED"
            if comparison["status"] not in {"UNKNOWN", "INVALID", "FAILED"}
            else comparison["status"]
        ),
    }


class BenchmarkRegistry:
    """Registry of trusted in-process benchmark collectors using one evaluator."""

    def __init__(self) -> None:
        self._benchmarks: dict[
            str, tuple[BenchmarkSpecification, Callable[[], Sequence[dict]]]
        ] = {}

    def register(
        self,
        spec: BenchmarkSpecification,
        collector: Callable[[], Sequence[dict]],
    ) -> None:
        if spec.benchmark_id in self._benchmarks:
            raise ValueError(f"Benchmark {spec.benchmark_id!r} is already registered.")
        if not callable(collector):
            raise ValueError("Benchmark collector must be callable.")
        self._benchmarks[spec.benchmark_id] = (spec, collector)

    def run(self, benchmark_id: str) -> dict:
        try:
            spec, collector = self._benchmarks[benchmark_id]
        except KeyError as exc:
            raise ValueError(f"Unknown benchmark: {benchmark_id}") from exc
        return evaluate_paired_predictions(spec, collector())

    def specifications(self) -> list[dict]:
        return [
            {
                **spec.to_dict(),
                "specification_sha256": spec.specification_sha256,
            }
            for spec, _collector in self._benchmarks.values()
        ]


def _collect_cpu_prediction_cases(
    spec: BenchmarkSpecification = CPU_PREDICTOR_BENCHMARK,
) -> tuple[list[dict], list[dict], list[dict]]:
    baseline_sequences: list[dict] = []
    candidate_sequences: list[dict] = []
    paired_cases: list[dict] = []
    for index, sequence in enumerate(benchmark_dataset(spec), start=1):
        baseline_predictions, targets = _run_forecaster(
            sequence,
            spec.baseline_algorithm,
            spec.adaptation_observations_per_sequence,
        )
        candidate_predictions, candidate_targets = _run_forecaster(
            sequence,
            spec.candidate_algorithm,
            spec.adaptation_observations_per_sequence,
        )
        if targets != candidate_targets:
            raise ValueError("Candidate and baseline evaluation targets differ.")
        baseline_sequences.append(
            {
                "sequence": index,
                **_metrics(
                    baseline_predictions, targets, metric_unit=spec.metric_unit
                ),
            }
        )
        candidate_sequences.append(
            {
                "sequence": index,
                **_metrics(
                    candidate_predictions,
                    candidate_targets,
                    metric_unit=spec.metric_unit,
                ),
            }
        )
        for position, (baseline_prediction, candidate_prediction, target) in enumerate(
            zip(baseline_predictions, candidate_predictions, targets),
            start=spec.adaptation_observations_per_sequence + 1,
        ):
            paired_cases.append(
                {
                    "case_id": f"sequence-{index}-position-{position}",
                    "target": target,
                    "baseline_prediction": baseline_prediction,
                    "candidate_prediction": candidate_prediction,
                    "unit": spec.metric_unit,
                }
            )
    return baseline_sequences, candidate_sequences, paired_cases


BENCHMARK_REGISTRY = BenchmarkRegistry()
BENCHMARK_REGISTRY.register(
    CPU_PREDICTOR_BENCHMARK,
    lambda: _collect_cpu_prediction_cases()[2],
)


def run_registered_benchmark(benchmark_id: str) -> dict:
    """Run a registered trusted collector through the common evidence engine."""
    return BENCHMARK_REGISTRY.run(benchmark_id)


def run_cpu_predictor_benchmark(
    *,
    spec: BenchmarkSpecification = CPU_PREDICTOR_BENCHMARK,
    source_root: Path | None = None,
) -> dict:
    """Run the fixed synthetic CPU-predictor replay and return its evidence."""
    dataset = benchmark_dataset(spec)
    baseline_sequences, candidate_sequences, paired_cases = (
        _collect_cpu_prediction_cases(spec)
    )
    evaluated = evaluate_paired_predictions(spec, paired_cases)
    if evaluated["status"] in {"INVALID", "FAILED", "UNKNOWN"}:
        raise ValueError(
            f"CPU predictor benchmark evidence is {evaluated['status']}: "
            f"{evaluated['reason']}"
        )
    baseline = evaluated["baseline"]
    candidate = evaluated["candidate"]
    comparison = evaluated["comparison"]
    source_hashes = {}
    if source_root is not None:
        for relative in ("kai/predictive.py", "kai/benchmark.py"):
            path = source_root / relative
            source_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "run_id": str(uuid.uuid4()),
        "benchmark": {
            **spec.to_dict(),
            "specification_sha256": spec.specification_sha256,
            "dataset_sha256": hashlib.sha256(
                json.dumps(dataset, separators=(",", ":")).encode()
            ).hexdigest(),
        },
        "implementation": {
            "kai_version": __version__,
            "domain": CPU_BASELINE_DOMAIN,
            "iterations": CPU_BASELINE_ITERATIONS,
            "baseline_source_hashes": {
                key: value
                for key, value in source_hashes.items()
                if key == "kai/predictive.py"
            },
            "candidate_source_hashes": {
                key: value
                for key, value in source_hashes.items()
                if key == "kai/benchmark.py"
            },
        },
        "baseline": baseline,
        "candidate": candidate,
        "sequences": {
            "baseline": baseline_sequences,
            "candidate": candidate_sequences,
        },
        "case_results": evaluated["cases"],
        "comparison": comparison,
        "correctness": {
            "status": "PASSED",
            "requirements": list(spec.correctness_requirements),
            "evaluation_targets_hidden_until_prediction": True,
            "identical_targets": True,
        },
        "decision": comparison["status"],
        "evidence_status": "SYNTHETIC_REPLAY",
        "statistical_significance_claimed": False,
    }
