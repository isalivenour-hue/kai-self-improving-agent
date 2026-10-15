"""Small deterministic, model-independent prediction primitives."""

from __future__ import annotations

import hashlib
import json
import math
from typing import TypedDict

CPU_BASELINE_DOMAIN = "cpu_baseline_runtime"
CPU_BASELINE_ITERATIONS = 100_000
CPU_BASELINE_MODEL_TYPE = "running_mean"
CPU_BASELINE_UPDATE_RULE = "Welford online mean and sample variance"
CPU_BASELINE_TOLERANCE_SECONDS = 0.05
CALIBRATION_MINIMUM_SAMPLES = 20
CONFIDENCE_MINIMUM_PREDICTIONS = 5


class PredictionContext(TypedDict):
    domain: str
    task: str
    environment_profile: str
    workload_parameters: dict[str, int]
    environment_snapshot_id: int | None


class ResourceEstimate(TypedDict):
    cpu: str
    ram: str
    gpu: str
    storage: str
    duration: str


class PredictiveModel(TypedDict):
    model_id: str
    model_type: str
    domain: str
    context_key: str
    version: int
    status: str
    created_at: str
    updated_at: str
    parameters: dict
    confidence: dict
    performance: dict
    training_evidence_references: list[str]


class Experiment(TypedDict):
    experiment_id: int
    experience_id: int
    domain: str
    question: str
    hypothesis: str
    experiment_type: str
    resource_estimate: ResourceEstimate
    status: str
    uncertainty_before: dict
    uncertainty_after: dict | None
    information_gain: dict | None
    started_at: str
    completed_at: str | None
    output: dict | None


def cpu_baseline_context(
    environment_fingerprint: str,
    environment_snapshot_id: int | None = None,
) -> PredictionContext:
    return {
        "domain": CPU_BASELINE_DOMAIN,
        "task": "CPU_BASELINE",
        "environment_profile": environment_fingerprint,
        "workload_parameters": {"iterations": CPU_BASELINE_ITERATIONS},
        "environment_snapshot_id": environment_snapshot_id,
    }


def validate_cpu_baseline_context(context: dict) -> None:
    if not isinstance(context, dict):
        raise ValueError("CPU baseline context must be an object.")
    if set(context) != {
        "domain",
        "task",
        "environment_profile",
        "workload_parameters",
        "environment_snapshot_id",
    }:
        raise ValueError("CPU baseline context contains fields outside its allowlist.")
    if context.get("domain") != CPU_BASELINE_DOMAIN:
        raise ValueError("Only the allowlisted CPU baseline domain is supported.")
    if context.get("task") != "CPU_BASELINE":
        raise ValueError("CPU baseline context has an unsupported task.")
    if context.get("workload_parameters") != {
        "iterations": CPU_BASELINE_ITERATIONS
    }:
        raise ValueError("CPU baseline workload parameters are fixed and allowlisted.")
    profile = context.get("environment_profile")
    if not isinstance(profile, str) or not profile.strip():
        raise ValueError("CPU baseline context requires an environment profile.")
    snapshot_id = context.get("environment_snapshot_id")
    if snapshot_id is not None and (
        isinstance(snapshot_id, bool)
        or not isinstance(snapshot_id, int)
        or snapshot_id < 1
    ):
        raise ValueError("Environment snapshot ID must be a positive integer or null.")


def prediction_context_key(context: dict) -> str:
    relevant = {
        "domain": context["domain"],
        "task": context["task"],
        "environment_profile": context["environment_profile"],
        "workload_parameters": context["workload_parameters"],
    }
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def initial_parameters(measured_seconds: float) -> dict:
    _validate_runtime(measured_seconds)
    return {
        "observation_count": 1,
        "mean_seconds": measured_seconds,
        "sum_squared_deviations": 0.0,
    }


def update_running_mean(parameters: dict, measured_seconds: float) -> dict:
    _validate_runtime(measured_seconds)
    count = parameters["observation_count"] + 1
    delta = measured_seconds - parameters["mean_seconds"]
    mean = parameters["mean_seconds"] + delta / count
    delta_after = measured_seconds - mean
    sum_squared_deviations = (
        parameters["sum_squared_deviations"] + delta * delta_after
    )
    return {
        "observation_count": count,
        "mean_seconds": mean,
        "sum_squared_deviations": sum_squared_deviations,
    }


def uncertainty(parameters: dict) -> dict:
    count = parameters["observation_count"]
    if count < 2:
        return {
            "status": "UNKNOWN",
            "reason": "At least two observations are required to estimate sample spread.",
            "standard_error_seconds": None,
        }
    sample_variance = max(
        0.0, parameters["sum_squared_deviations"] / (count - 1)
    )
    sample_standard_deviation = math.sqrt(sample_variance)
    return {
        "status": "ESTIMATED",
        "method": "sample standard deviation divided by square root of sample count",
        "sample_standard_deviation_seconds": sample_standard_deviation,
        "standard_error_seconds": sample_standard_deviation / math.sqrt(count),
        "sample_count": count,
        "calibrated_interval": False,
    }


def resource_estimate() -> ResourceEstimate:
    return {
        "cpu": "low (100000 Python integer-loop iterations)",
        "ram": "low (bounded working set)",
        "gpu": "none",
        "storage": "low (bounded read-only sample of existing KAI database)",
        "duration": "short; wall-clock duration varies with system load",
    }


def predict_from_model(model: dict) -> dict:
    parameters = model["parameters"]
    return {
        "expected_outcome": parameters["mean_seconds"],
        "uncertainty": uncertainty(parameters),
        "confidence": model["confidence"].get("empirical_accuracy"),
        "confidence_semantics": (
            "Prior empirical hit rate for this context; not calibrated unless "
            "the separately reported sample-size requirement is met."
        ),
        "model_id": model["model_id"],
        "model_version": model["version"],
    }


def prediction_statistics(rows: list[dict]) -> dict:
    statuses = (
        "CONFIRMED",
        "PARTIALLY_CONFIRMED",
        "REFUTED",
        "UNKNOWN",
        "EXPIRED",
        "PENDING",
    )
    counts = {status: 0 for status in statuses}
    signed_errors = []
    absolute_errors = []
    confidence_results = []
    for item in rows:
        status = item["status"]
        if status not in counts:
            continue
        counts[status] += 1
        difference = item.get("difference")
        if isinstance(difference, dict):
            signed = difference.get("difference")
            magnitude = difference.get("error_magnitude")
            if (
                isinstance(signed, (int, float))
                and math.isfinite(signed)
                and isinstance(magnitude, (int, float))
                and math.isfinite(magnitude)
            ):
                signed_errors.append(float(signed))
                absolute_errors.append(float(magnitude))
        confidence = item.get("confidence")
        if status in {"CONFIRMED", "PARTIALLY_CONFIRMED", "REFUTED"} and (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and 0.0 <= confidence <= 1.0
        ):
            confidence_results.append(
                (float(confidence), status in {"CONFIRMED", "PARTIALLY_CONFIRMED"})
            )
    numeric_count = len(signed_errors)
    mean_signed = sum(signed_errors) / numeric_count if numeric_count else None
    mean_absolute = sum(absolute_errors) / numeric_count if numeric_count else None
    error_variance = None
    if numeric_count >= 2:
        error_variance = sum(
            (value - mean_signed) ** 2 for value in signed_errors
        ) / (numeric_count - 1)
    calibration_bins: dict[str, dict] = {}
    for confidence, hit in confidence_results:
        label = f"{min(int(confidence * 10), 9) / 10:.1f}-{min(int(confidence * 10), 9) / 10 + 0.1:.1f}"
        bucket = calibration_bins.setdefault(
            label, {"count": 0, "mean_confidence": 0.0, "observed_accuracy": 0.0}
        )
        bucket["count"] += 1
        bucket["mean_confidence"] += confidence
        bucket["observed_accuracy"] += int(hit)
    for bucket in calibration_bins.values():
        count = bucket["count"]
        bucket["mean_confidence"] /= count
        bucket["observed_accuracy"] /= count
    scored_count = sum(counts[key] for key in ("CONFIRMED", "PARTIALLY_CONFIRMED", "REFUTED"))
    accuracy = (
        (counts["CONFIRMED"] + counts["PARTIALLY_CONFIRMED"]) / scored_count
        if scored_count
        else None
    )
    return {
        "prediction_count": len(rows),
        "confirmed": counts["CONFIRMED"],
        "partially_confirmed": counts["PARTIALLY_CONFIRMED"],
        "refuted": counts["REFUTED"],
        "unknown": counts["UNKNOWN"],
        "expired": counts["EXPIRED"],
        "pending": counts["PENDING"],
        "numeric_error_count": numeric_count,
        "mean_absolute_error_seconds": mean_absolute,
        "mean_signed_error_seconds": mean_signed,
        "sample_error_variance_seconds_squared": error_variance,
        "empirical_accuracy": accuracy,
        "confidence_vs_observed_accuracy": {
            "status": (
                "ESTIMATED"
                if scored_count >= CALIBRATION_MINIMUM_SAMPLES
                else "INSUFFICIENT DATA"
            ),
            "minimum_sample_count": CALIBRATION_MINIMUM_SAMPLES,
            "scored_sample_count": scored_count,
            "bins": calibration_bins,
            "interpretation": (
                "Accuracy counts CONFIRMED and PARTIALLY_CONFIRMED as hits; "
                "this descriptive statistic is not proof of calibration."
            ),
        },
    }


def confidence_from_statistics(statistics: dict) -> dict:
    scored = statistics["confidence_vs_observed_accuracy"]["scored_sample_count"]
    if scored < CONFIDENCE_MINIMUM_PREDICTIONS:
        return {
            "status": "INSUFFICIENT DATA",
            "empirical_accuracy": None,
            "scored_prediction_count": scored,
            "minimum_prediction_count": CONFIDENCE_MINIMUM_PREDICTIONS,
            "calibrated": False,
        }
    return {
        "status": "EMPIRICAL",
        "empirical_accuracy": statistics["empirical_accuracy"],
        "scored_prediction_count": scored,
        "minimum_prediction_count": CONFIDENCE_MINIMUM_PREDICTIONS,
        "calibrated": False,
    }


def _validate_runtime(value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("CPU baseline runtime must be a finite positive number.")
