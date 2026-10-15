"""Trusted, deterministic registry for explicitly allowlisted experiments."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from kai.environment_model import measure_baseline
from kai.errors import KaiError
from kai.predictive import (
    CPU_BASELINE_DOMAIN,
    CPU_BASELINE_ITERATIONS,
    cpu_baseline_context,
    validate_cpu_baseline_context,
)


class InvalidExperimentResultError(KaiError):
    """A trusted procedure returned malformed or incompatible measurement data."""


EXPERIMENT_STATUSES = ("DEFINED", "ENABLED", "DISABLED", "DEPRECATED")
EXECUTION_STATUSES = ("RUNNING", "SUCCEEDED", "FAILED", "ABORTED", "UNKNOWN")
MEASUREMENT_STATUSES = ("MEASURED", "UNKNOWN", "INVALID")
EVALUATION_STATUSES = ("NOT_EVALUATED", "EVALUATED", "UNKNOWN", "INVALID")
EVIDENCE_QUALITIES = ("NONE", "INSUFFICIENT", "PRELIMINARY", "ADEQUATE")
EVIDENCE_ADEQUATE_SAMPLE_COUNT = 20
CPU_BASELINE_EXPERIMENT_ID = "cpu-baseline"
CPU_BASELINE_EXPERIMENT_VERSION = 1
CPU_BASELINE_PROCEDURE_ID = "cpu-baseline-v1"


@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    domain: str
    name: str
    description: str
    version: int
    status: str
    procedure_id: str
    parameter_specification: dict
    context_specification: dict
    outcome_specification: dict

    def __post_init__(self) -> None:
        for field_name in (
            "experiment_id",
            "domain",
            "name",
            "description",
            "procedure_id",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Experiment {field_name} must be non-empty.")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
        ):
            raise ValueError("Experiment version must be a positive integer.")
        if self.status not in EXPERIMENT_STATUSES:
            raise ValueError(f"Unsupported experiment status: {self.status}")
        for field_name in (
            "parameter_specification",
            "context_specification",
            "outcome_specification",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, dict):
                raise ValueError(f"Experiment {field_name} must be an object.")
            try:
                json.dumps(value, sort_keys=True, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Experiment {field_name} must contain finite JSON data."
                ) from exc

    def to_dict(self) -> dict:
        return deepcopy(
            {
                "experiment_id": self.experiment_id,
                "domain": self.domain,
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "status": self.status,
                "procedure_id": self.procedure_id,
                "parameter_specification": self.parameter_specification,
                "context_specification": self.context_specification,
                "outcome_specification": self.outcome_specification,
            }
        )


@dataclass(frozen=True)
class ExperimentResult:
    outcome_value: float | None
    unit: str
    sample_count: int
    duration_seconds: float | None
    metadata: dict
    measurement_status: str
    raw_result: dict

    def to_dict(self) -> dict:
        return {
            "outcome_value": self.outcome_value,
            "unit": self.unit,
            "sample_count": self.sample_count,
            "duration_seconds": self.duration_seconds,
            "metadata": dict(self.metadata),
            "measurement_status": self.measurement_status,
            "raw_result": dict(self.raw_result),
        }


Procedure = Callable[[dict, Path], ExperimentResult | dict]


def validate_experiment_result(value: ExperimentResult | dict) -> ExperimentResult:
    if isinstance(value, ExperimentResult):
        result = value
    elif isinstance(value, dict):
        try:
            result = ExperimentResult(
                outcome_value=value.get("outcome_value"),
                unit=value["unit"],
                sample_count=value["sample_count"],
                duration_seconds=value.get("duration_seconds"),
                metadata=value.get("metadata", {}),
                measurement_status=value["measurement_status"],
                raw_result=value.get("raw_result", {}),
            )
        except (KeyError, TypeError) as exc:
            raise InvalidExperimentResultError(
                f"Malformed experiment result: {exc}"
            ) from exc
    else:
        raise InvalidExperimentResultError(
            "Experiment procedure returned an unsupported result type."
        )
    if result.measurement_status not in MEASUREMENT_STATUSES:
        raise InvalidExperimentResultError(
            "Experiment result has an unsupported measurement status."
        )
    if not isinstance(result.unit, str) or not result.unit.strip():
        raise InvalidExperimentResultError("Experiment result unit must be non-empty.")
    if (
        isinstance(result.sample_count, bool)
        or not isinstance(result.sample_count, int)
        or result.sample_count < 0
    ):
        raise InvalidExperimentResultError(
            "Experiment result sample count must be a non-negative integer."
        )
    if result.measurement_status == "MEASURED" and (
        result.outcome_value is None or result.sample_count < 1
    ):
        raise InvalidExperimentResultError(
            "A measured result requires a numeric value and at least one sample."
        )
    if result.measurement_status != "MEASURED" and (
        result.outcome_value is not None or result.sample_count != 0
    ):
        raise InvalidExperimentResultError(
            "An unknown or invalid result cannot claim an outcome or sample count."
        )
    for label, numeric in (
        ("outcome", result.outcome_value),
        ("duration", result.duration_seconds),
    ):
        if numeric is not None and (
            isinstance(numeric, bool)
            or not isinstance(numeric, (int, float))
            or not math.isfinite(numeric)
            or (label == "duration" and numeric < 0)
        ):
            raise InvalidExperimentResultError(
                f"Experiment {label} must be finite numeric data."
            )
    if not isinstance(result.metadata, dict) or not isinstance(result.raw_result, dict):
        raise InvalidExperimentResultError(
            "Experiment result metadata and raw result must be objects."
        )
    try:
        json.dumps(result.metadata, sort_keys=True, allow_nan=False)
        json.dumps(result.raw_result, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise InvalidExperimentResultError(
            f"Experiment result contains non-JSON data: {exc}"
        ) from exc
    return result


def classify_evidence_quality(
    *,
    sample_count: int,
    context_consistency: bool,
    measurement_status: str,
    prediction_available: bool,
    evaluation_available: bool,
) -> tuple[str, dict]:
    metadata = {
        "sample_count": sample_count,
        "context_consistency": context_consistency,
        "measurement_status": measurement_status,
        "prediction_available": prediction_available,
        "evaluation_available": evaluation_available,
        "adequate_sample_threshold": EVIDENCE_ADEQUATE_SAMPLE_COUNT,
        "threshold_interpretation": (
            "Engineering evidence tier aligned with the existing 20-prediction "
            "reporting threshold; not statistical significance or calibrated confidence."
        ),
    }
    if measurement_status != "MEASURED" or sample_count <= 0:
        return "NONE", metadata
    if not context_consistency or sample_count < 2:
        return "INSUFFICIENT", metadata
    if (
        sample_count < EVIDENCE_ADEQUATE_SAMPLE_COUNT
        or not prediction_available
        or not evaluation_available
    ):
        return "PRELIMINARY", metadata
    return "ADEQUATE", metadata


def _cpu_baseline_procedure(environment: dict, database_path: Path) -> ExperimentResult:
    raw = measure_baseline(environment, database_path)
    if not isinstance(raw, dict) or not isinstance(raw.get("cpu"), dict):
        raise InvalidExperimentResultError(
            "The CPU baseline procedure returned a malformed measurement report."
        )
    cpu = raw["cpu"]
    if cpu.get("status") != "measured":
        return ExperimentResult(
            outcome_value=None,
            unit="seconds",
            sample_count=0,
            duration_seconds=None,
            metadata={"measurement_reason": cpu.get("reason", "CPU measurement unavailable")},
            measurement_status="UNKNOWN",
            raw_result=raw,
        )
    return ExperimentResult(
        outcome_value=cpu.get("duration_seconds"),
        unit="seconds",
        sample_count=1,
        duration_seconds=cpu.get("duration_seconds"),
        metadata={
            "metric": cpu.get("metric"),
            "iterations": cpu.get("iterations"),
            "result_checksum": cpu.get("result_checksum"),
            "not_a_model_or_ML_benchmark": cpu.get("not_a_model_or_ML_benchmark"),
            "disk_read": raw.get("disk_read"),
        },
        measurement_status="MEASURED",
        raw_result={
            "captured_at": raw.get("captured_at"),
            "cpu": cpu,
            "disk_read": raw.get("disk_read"),
        },
    )


CPU_BASELINE_EXPERIMENT = ExperimentDefinition(
    experiment_id=CPU_BASELINE_EXPERIMENT_ID,
    domain="cpu_baseline_runtime",
    name="CPU baseline runtime",
    description=(
        "Measure elapsed wall-clock time for a fixed 100000-iteration Python "
        "integer loop; runtime varies with system conditions."
    ),
    version=CPU_BASELINE_EXPERIMENT_VERSION,
    status="ENABLED",
    procedure_id=CPU_BASELINE_PROCEDURE_ID,
    parameter_specification={
        "iterations": {
            "type": "integer",
            "value": CPU_BASELINE_ITERATIONS,
            "mutable": False,
        }
    },
    context_specification={
        "required": ["environment_profile"],
        "optional": ["environment_snapshot_id"],
        "excluded": [
            "hostname",
            "username",
            "ip_address",
            "filesystem_path",
            "serial_number",
        ],
    },
    outcome_specification={
        "type": "numeric",
        "unit": "seconds",
        "sample_count": 1,
        "measurement": "elapsed wall-clock duration",
        "scoring": "observed minus expected; relative error omitted when expected is zero",
    },
)


class ExperimentRegistry:
    def __init__(self, include_builtins: bool = True) -> None:
        self._definitions: dict[tuple[str, int], ExperimentDefinition] = {}
        self._procedures: dict[str, Procedure] = {}
        if include_builtins:
            self.register(CPU_BASELINE_EXPERIMENT, _cpu_baseline_procedure)

    def register(
        self, definition: ExperimentDefinition, procedure: Procedure | None = None
    ) -> None:
        if not isinstance(definition, ExperimentDefinition):
            raise KaiError("Only validated experiment definitions can be registered.")
        if procedure is not None and not callable(procedure):
            raise KaiError("Experiment procedure must be a trusted callable.")
        key = (definition.experiment_id, definition.version)
        if key in self._definitions:
            raise KaiError(
                f"Experiment {definition.experiment_id} version {definition.version} is already registered."
            )
        existing = self._procedures.get(definition.procedure_id)
        if procedure is not None and existing is not None and existing is not procedure:
            raise KaiError(
                f"Procedure {definition.procedure_id} is already bound to another implementation."
            )
        self._definitions[key] = deepcopy(definition)
        if procedure is not None:
            self._procedures[definition.procedure_id] = procedure

    def register_procedure(self, procedure_id: str, procedure: Procedure) -> None:
        if not isinstance(procedure_id, str) or not procedure_id.strip():
            raise KaiError("Trusted procedure identifier must be non-empty.")
        if not callable(procedure):
            raise KaiError("Experiment procedure must be a trusted callable.")
        if procedure_id in self._procedures:
            raise KaiError(f"Procedure {procedure_id} is already registered.")
        self._procedures[procedure_id] = procedure

    def get(
        self, experiment_id: str, version: int | None = None
    ) -> ExperimentDefinition:
        matches = [
            definition
            for (registered_id, _), definition in self._definitions.items()
            if registered_id == experiment_id
            and (version is None or definition.version == version)
        ]
        if not matches:
            raise KaiError(f"Experiment {experiment_id!r} is not registered.")
        return deepcopy(max(matches, key=lambda item: item.version))

    def list(self) -> list[ExperimentDefinition]:
        return sorted(
            (deepcopy(item) for item in self._definitions.values()),
            key=lambda item: (item.experiment_id, item.version),
        )

    def contains(self, experiment_id: str, version: int | None = None) -> bool:
        return any(
            definition.experiment_id == experiment_id
            and (version is None or definition.version == version)
            for definition in self._definitions.values()
        )

    def has_procedure(self, procedure_id: str) -> bool:
        return procedure_id in self._procedures

    def execute(
        self,
        experiment_id: str,
        parameters: dict,
        context: dict,
        environment: dict,
        database_path: Path,
        version: int | None = None,
    ) -> ExperimentResult:
        definition = self.get(experiment_id, version)
        if definition.status != "ENABLED":
            raise KaiError(
                f"Experiment {definition.experiment_id} is {definition.status.lower()}."
            )
        expected_parameters = {
            name: item["value"]
            for name, item in definition.parameter_specification.items()
            if isinstance(item, dict) and "value" in item
        }
        if parameters != expected_parameters:
            raise KaiError("Experiment parameters do not match the fixed specification.")
        allowed_context = {
            "domain",
            "task",
            "environment_profile",
            "environment_snapshot_id",
            "workload_parameters",
        }
        if set(context) - allowed_context:
            raise KaiError("Experiment context contains non-allowlisted fields.")
        if (
            context.get("environment_profile") is None
            or context.get("workload_parameters") != expected_parameters
        ):
            raise KaiError("Experiment context does not match its requirements.")
        try:
            validate_cpu_baseline_context(context)
        except ValueError as exc:
            raise KaiError(str(exc)) from exc
        if definition.domain != CPU_BASELINE_DOMAIN:
            raise KaiError(
                f"Experiment domain {definition.domain!r} has no trusted runner."
            )
        procedure = self._procedures.get(definition.procedure_id)
        if procedure is None:
            raise KaiError(
                f"Trusted procedure {definition.procedure_id!r} is not registered."
            )
        result = validate_experiment_result(procedure(environment, database_path))
        if result.unit != definition.outcome_specification.get("unit"):
            raise InvalidExperimentResultError(
                "Experiment result unit does not match its registered definition."
            )
        expected_samples = definition.outcome_specification.get("sample_count")
        if (
            result.measurement_status == "MEASURED"
            and expected_samples is not None
            and result.sample_count != expected_samples
        ):
            raise InvalidExperimentResultError(
                "Experiment result sample count does not match its registered definition."
            )
        return result


def cpu_baseline_execution_context(
    environment_profile: str, environment_snapshot_id: int | None
) -> dict:
    return cpu_baseline_context(environment_profile, environment_snapshot_id)


DEFAULT_EXPERIMENT_REGISTRY = ExperimentRegistry()
