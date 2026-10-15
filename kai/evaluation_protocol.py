"""Predeclared repeated evaluation rules for bounded CPU predictor candidates."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Sequence


@dataclass(frozen=True)
class EvaluationProtocol:
    protocol_id: str
    version: int
    workload: dict
    sequence_count: int
    measurements_per_sequence: int
    warmup_measurements: int
    minimum_relative_change: float
    minimum_sequence_fraction: float
    metric: str
    direction: str
    comparison_method: str

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_id, str) or not self.protocol_id.strip():
            raise ValueError("Protocol ID must be non-empty.")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise ValueError("Protocol version must be a positive integer.")
        if not isinstance(self.workload, dict):
            raise ValueError("Protocol workload must be an object.")
        for label, value in (
            ("sequence count", self.sequence_count),
            ("measurements per sequence", self.measurements_per_sequence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Protocol {label} must be a positive integer.")
        if (
            isinstance(self.warmup_measurements, bool)
            or not isinstance(self.warmup_measurements, int)
            or self.warmup_measurements < 0
            or self.warmup_measurements >= self.measurements_per_sequence
        ):
            raise ValueError("Protocol warmup must be non-negative and shorter than a sequence.")
        if (
            isinstance(self.minimum_relative_change, bool)
            or not isinstance(self.minimum_relative_change, (int, float))
            or not math.isfinite(self.minimum_relative_change)
            or not 0 <= self.minimum_relative_change <= 1
        ):
            raise ValueError("Protocol relative-change threshold must be in [0, 1].")
        if (
            isinstance(self.minimum_sequence_fraction, bool)
            or not isinstance(self.minimum_sequence_fraction, (int, float))
            or not math.isfinite(self.minimum_sequence_fraction)
            or not 0 < self.minimum_sequence_fraction <= 1
        ):
            raise ValueError("Protocol sequence fraction must be in (0, 1].")
        if self.direction != "lower":
            raise ValueError("Only lower-is-better evaluation protocols are supported.")
        for label, value in (
            ("metric", self.metric),
            ("comparison method", self.comparison_method),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Protocol {label} must be non-empty.")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def definition_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


CPU_PREDICTOR_PROTOCOL = EvaluationProtocol(
    protocol_id="cpu-predictor-repeated-v1",
    version=1,
    workload={
        "procedure": "fixed 100000-iteration CPU baseline",
        "observations": "actual bounded CPU timing observations",
        "grouping": "consecutive measurements divided into fixed sequences",
    },
    sequence_count=3,
    measurements_per_sequence=10,
    warmup_measurements=2,
    minimum_relative_change=0.10,
    minimum_sequence_fraction=2 / 3,
    metric="prequential mean absolute prediction error in seconds",
    direction="lower",
    comparison_method=(
        "paired per-observation absolute errors, summarized per sequence and "
        "across sequences; directional consistency rule, not a significance test"
    ),
)


def _finite_nonnegative_sequences(sequences: Sequence[Sequence[float]]) -> list[list[float]]:
    normalized: list[list[float]] = []
    for sequence_index, sequence in enumerate(sequences):
        if not isinstance(sequence, (list, tuple)):
            raise ValueError(f"Sequence {sequence_index + 1} must be a list.")
        values: list[float] = []
        for value in sequence:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("Paired absolute errors must be finite and non-negative.")
            values.append(float(value))
        normalized.append(values)
    return normalized


def summarize_repeated_errors(
    baseline_errors: Sequence[Sequence[float]],
    candidate_errors: Sequence[Sequence[float]],
    protocol: EvaluationProtocol = CPU_PREDICTOR_PROTOCOL,
) -> dict:
    baseline = _finite_nonnegative_sequences(baseline_errors)
    candidate = _finite_nonnegative_sequences(candidate_errors)
    if len(baseline) != len(candidate):
        raise ValueError("Baseline and candidate sequence counts must be paired.")
    if len(baseline) != protocol.sequence_count:
        raise ValueError("Sequence count differs from the predeclared protocol.")
    if any(len(left) != len(right) for left, right in zip(baseline, candidate)):
        raise ValueError("Baseline and candidate paired sample counts must match.")
    expected_pairs = (
        protocol.measurements_per_sequence - protocol.warmup_measurements
    )
    if any(len(sequence) != expected_pairs for sequence in baseline):
        raise ValueError("Paired observations do not match the protocol window.")
    baseline_per_sequence = [math.fsum(values) / len(values) for values in baseline]
    candidate_per_sequence = [
        math.fsum(values) / len(values) for values in candidate
    ]
    baseline_mae = math.fsum(baseline_per_sequence) / len(baseline_per_sequence)
    candidate_mae = math.fsum(candidate_per_sequence) / len(candidate_per_sequence)
    relative_changes = [
        (baseline_value - candidate_value) / abs(baseline_value)
        if baseline_value
        else 0.0
        if not candidate_value
        else None
        for baseline_value, candidate_value in zip(
            baseline_per_sequence, candidate_per_sequence
        )
    ]
    threshold = protocol.minimum_relative_change
    improved_sequences = sum(
        change is not None and change >= threshold for change in relative_changes
    )
    regressed_sequences = sum(
        change is not None and change <= -threshold for change in relative_changes
    )
    required_directional_count = math.ceil(
        protocol.sequence_count * protocol.minimum_sequence_fraction
    )
    if (
        baseline_mae > candidate_mae
        and (baseline_mae - candidate_mae) / abs(baseline_mae) >= threshold
        and improved_sequences >= required_directional_count
    ):
        status = "IMPROVED"
    elif baseline_mae == 0 and candidate_mae > 0:
        status = "REGRESSED"
    elif (
        candidate_mae > baseline_mae
        and (candidate_mae - baseline_mae) / abs(baseline_mae) >= threshold
        and regressed_sequences >= required_directional_count
    ):
        status = "REGRESSED"
    else:
        status = "INCONCLUSIVE"
    flat_baseline = [value for sequence in baseline for value in sequence]
    flat_candidate = [value for sequence in candidate for value in sequence]
    return {
        "status": status,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.version,
        "protocol_sha256": protocol.definition_sha256,
        "sequence_count": len(baseline),
        "paired_forecast_count": sum(map(len, baseline)),
        "baseline_metrics": {
            "mae_seconds": baseline_mae,
            "sequence_mae_seconds": baseline_per_sequence,
            "mean_sequence_mae_seconds": baseline_mae,
            "median_sequence_mae_seconds": statistics.median(
                baseline_per_sequence
            ),
            "sequence_mae_sample_stdev_seconds": (
                statistics.stdev(baseline_per_sequence)
                if len(baseline_per_sequence) > 1
                else None
            ),
            "raw_paired_absolute_errors_seconds": baseline,
            "measurement_sample_count": len(flat_baseline),
        },
        "candidate_metrics": {
            "mae_seconds": candidate_mae,
            "sequence_mae_seconds": candidate_per_sequence,
            "mean_sequence_mae_seconds": candidate_mae,
            "median_sequence_mae_seconds": statistics.median(
                candidate_per_sequence
            ),
            "sequence_mae_sample_stdev_seconds": (
                statistics.stdev(candidate_per_sequence)
                if len(candidate_per_sequence) > 1
                else None
            ),
            "raw_paired_absolute_errors_seconds": candidate,
            "measurement_sample_count": len(flat_candidate),
        },
        "sequence_relative_improvement": relative_changes,
        "improved_sequence_count": improved_sequences,
        "regressed_sequence_count": regressed_sequences,
        "minimum_relative_change": threshold,
        "required_directional_sequence_count": required_directional_count,
        "uncertainty": {
            "baseline_sequence_mae_sample_stdev_seconds": (
                statistics.stdev(baseline_per_sequence)
            ),
            "candidate_sequence_mae_sample_stdev_seconds": (
                statistics.stdev(candidate_per_sequence)
            ),
            "statistical_significance_claimed": False,
        },
        "missing_measurements": 0,
    }


def validate_protocol_comparison(
    comparison: dict, protocol: EvaluationProtocol
) -> str:
    """Re-derive the decision from stored paired errors and immutable criteria."""
    summary = summarize_repeated_errors(
        comparison["baseline_metrics"]["raw_paired_absolute_errors_seconds"],
        comparison["candidate_metrics"]["raw_paired_absolute_errors_seconds"],
        protocol,
    )
    stored_protocol_status = comparison.get(
        "protocol_status", comparison.get("status")
    )
    if summary["status"] != stored_protocol_status:
        raise ValueError("Evaluation status conflicts with protocol measurements.")
    for key, value in summary.items():
        if key in {"baseline_metrics", "candidate_metrics"}:
            continue
        stored_value = (
            stored_protocol_status if key == "status" else comparison.get(key)
        )
        if stored_value != value:
            raise ValueError(f"Evaluation protocol field {key!r} is inconsistent.")
    for key in ("baseline_metrics", "candidate_metrics"):
        stored_metrics = comparison.get(key)
        derived_metrics = summary[key]
        if not isinstance(stored_metrics, dict):
            raise ValueError(f"Evaluation {key} must be an object.")
        for metric, value in derived_metrics.items():
            if stored_metrics.get(metric) != value:
                raise ValueError(f"Evaluation {key} metric {metric!r} conflicts with samples.")
    return summary["status"]
