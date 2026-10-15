"""Structured self-observation and evidence-based improvement evaluation."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

SELF_EPISTEMIC_STATES = ("OBSERVED", "INFERRED", "UNKNOWN")
OPPORTUNITY_STATUSES = (
    "OPEN",
    "INVESTIGATING",
    "PROPOSED",
    "EXPERIMENTING",
    "ACCEPTED",
    "REJECTED",
    "DEFERRED",
    "INVALIDATED",
)
IMPROVEMENT_EVALUATION_STATUSES = (
    "UNKNOWN",
    "FAILED",
    "REGRESSED",
    "INCONCLUSIVE",
    "IMPROVED",
    "STRONGLY_IMPROVED",
)
IMPROVEMENT_DECISIONS = (
    "REJECT",
    "DEFER",
    "REQUEST_MORE_EVIDENCE",
    "ACCEPT_CANDIDATE",
    "READY_FOR_DEPLOYMENT",
)
CPU_DEMO_MINIMUM_OBSERVATIONS = 22
CPU_DEMO_MAXIMUM_OBSERVATIONS = 22
IMPROVEMENT_EVIDENCE_MINIMUM = 20
CPU_DEMO_CANDIDATE_ID = "cpu-baseline-last-observation-v1"


def _validate_json_object(value: Any, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object.")
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain finite JSON data: {exc}") from exc
    return value


def validate_self_observation(value: dict) -> dict:
    required = {
        "subject",
        "observation_type",
        "observed_at",
        "value",
        "source",
        "epistemic_status",
        "evidence_status",
        "kai_version",
        "environment_context",
    }
    if not isinstance(value, dict) or required - set(value):
        raise ValueError("Self observation is missing required fields.")
    for key in ("subject", "observation_type", "observed_at", "source", "kai_version"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"Self observation {key} must be non-empty.")
    try:
        from datetime import datetime

        timestamp = value["observed_at"]
        if timestamp.endswith(("Z", "z")):
            timestamp = timestamp[:-1] + "+00:00"
        parsed = datetime.fromisoformat(timestamp)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("Self observation timestamp must be timezone-aware ISO-8601.") from exc
    if value["epistemic_status"] not in SELF_EPISTEMIC_STATES:
        raise ValueError("Unsupported self-observation epistemic status.")
    if value["evidence_status"] not in {
        "observed",
        "verified",
        "unverified",
        "contradicted",
        "unknown",
    }:
        raise ValueError("Unsupported self-observation evidence status.")
    confidence = value.get("confidence")
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("Self-observation confidence must be finite in [0, 1].")
    _validate_json_object(value["environment_context"], "Environment context")
    try:
        json.dumps(value["value"], sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Self-observation value must be finite JSON: {exc}") from exc
    return value


def inspect_source_tree(project_root: Path) -> dict:
    """Read and parse owned Python sources; never imports or executes them."""
    source_root = project_root / "kai"
    test_root = project_root / "tests"
    source_files = sorted(source_root.glob("*.py"))
    test_files = sorted(test_root.glob("test_*.py"))
    inventory: list[dict] = []
    module_roots: set[str] = set()
    syntax_failures: list[dict] = []
    test_count = 0
    class_count = 0
    function_count = 0
    complexity_points = 0
    source_digest = hashlib.sha256()
    for path in source_files + test_files:
        relative = path.relative_to(project_root).as_posix()
        content = path.read_bytes()
        source_digest.update(relative.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(content)
        try:
            tree = ast.parse(content, filename=relative)
        except (SyntaxError, UnicodeDecodeError) as exc:
            syntax_failures.append(
                {"path": relative, "error": str(exc), "status": "INVALID"}
            )
            continue
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        module_complexity = sum(
            1
            for node in ast.walk(tree)
            if isinstance(
                node,
                (
                    ast.If,
                    ast.IfExp,
                    ast.For,
                    ast.AsyncFor,
                    ast.While,
                    ast.ExceptHandler,
                    ast.Assert,
                    ast.comprehension,
                    ast.Match,
                    ast.match_case,
                ),
            )
        ) + sum(
            max(0, len(node.values) - 1)
            for node in ast.walk(tree)
            if isinstance(node, ast.BoolOp)
        )
        complexity_points += module_complexity
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        for node in imports:
            if isinstance(node, ast.Import):
                module_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif node.module:
                module_roots.add(node.module.split(".", 1)[0])
        test_count += sum(
            node.name.startswith("test_") for node in functions
        )
        class_count += len(classes)
        function_count += len(functions)
        inventory.append(
            {
                "path": relative,
                "class_names": sorted(node.name for node in classes),
                "function_count": len(functions),
                "decision_complexity_points": module_complexity,
                "test_function_count": sum(
                    node.name.startswith("test_") for node in functions
                ),
                "status": "OBSERVED",
            }
        )
    external_imports = sorted(
        root
        for root in module_roots
        if root not in sys.stdlib_module_names and root != "kai"
    )
    return {
        "source_root": "kai/",
        "test_root": "tests/",
        "source_file_count": len(source_files),
        "test_file_count": len(test_files),
        "module_inventory": inventory,
        "class_count": class_count,
        "function_count": function_count,
        "test_function_count": test_count,
        "decision_complexity_points": complexity_points,
        "complexity_interpretation": (
            "Static AST decision-node count only; not a calibrated maintainability metric."
        ),
        "syntax_failures": syntax_failures,
        "imports_outside_python_standard_library": external_imports,
        "repository_metadata": {
            "git_metadata_visible": (project_root / ".git").exists(),
            "status": (
                "OBSERVED" if (project_root / ".git").exists() else "UNKNOWN"
            ),
        },
        "source_snapshot_sha256": source_digest.hexdigest(),
        "source_digest_scope": "kai/*.py and tests/test_*.py source bytes",
        "source_inventory_method": "Python AST parse; source code is not executed.",
    }


def running_mean_forecast(values: list[float]) -> float:
    if not values:
        raise ValueError("A running-mean forecast requires prior observations.")
    return math.fsum(values) / len(values)


def last_observation_forecast(values: list[float]) -> float:
    if not values:
        raise ValueError("A last-observation forecast requires prior observations.")
    return values[-1]


def score_forecasters(values: list[float], minimum_observations: int = 22) -> dict:
    if len(values) < minimum_observations:
        return {
            "status": "INCONCLUSIVE",
            "reason": (
                f"At least {minimum_observations} real observations are required "
                "for 20 paired out-of-sample forecasts."
            ),
            "observation_count": len(values),
            "paired_forecast_count": max(0, len(values) - 2),
            "baseline": None,
            "candidate": None,
            "evidence_quality": "INSUFFICIENT",
        }
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
        for value in values
    ):
        raise ValueError("CPU timing observations must be finite positive numbers.")
    baseline_errors = []
    candidate_errors = []
    for index in range(2, len(values)):
        history = values[:index]
        observed = values[index]
        baseline_errors.append(abs(running_mean_forecast(history) - observed))
        candidate_errors.append(abs(last_observation_forecast(history) - observed))
    baseline_mae = math.fsum(baseline_errors) / len(baseline_errors)
    candidate_mae = math.fsum(candidate_errors) / len(candidate_errors)
    if math.isclose(candidate_mae, baseline_mae, rel_tol=1e-12, abs_tol=1e-15):
        status = "INCONCLUSIVE"
        reason = "Candidate and baseline mean absolute errors are indistinguishable."
    elif candidate_mae > baseline_mae:
        status = "REGRESSED"
        reason = "Candidate last-observation forecasts have greater mean absolute error."
    else:
        relative_gain = (baseline_mae - candidate_mae) / baseline_mae if baseline_mae else 0.0
        status = "STRONGLY_IMPROVED" if relative_gain >= 0.10 else "IMPROVED"
        reason = (
            "Candidate last-observation forecasts have lower mean absolute error "
            "on the measured held-out sequence."
        )
    return {
        "status": status,
        "reason": reason,
        "observation_count": len(values),
        "paired_forecast_count": len(baseline_errors),
        "baseline": {
            "algorithm": "running-mean-v1",
            "mean_absolute_error_seconds": baseline_mae,
            "absolute_errors_seconds": baseline_errors,
        },
        "candidate": {
            "algorithm": CPU_DEMO_CANDIDATE_ID,
            "mean_absolute_error_seconds": candidate_mae,
            "absolute_errors_seconds": candidate_errors,
        },
        "improvement_fraction": (
            (baseline_mae - candidate_mae) / baseline_mae if baseline_mae else None
        ),
        "evidence_quality": (
            "ADEQUATE"
            if len(baseline_errors) >= IMPROVEMENT_EVIDENCE_MINIMUM
            else "PRELIMINARY"
        ),
        "statistical_significance_claimed": False,
        "scoring_method": (
            "Prequential paired out-of-sample absolute error; first two samples "
            "are warm-up only; lower mean absolute error is better."
        ),
    }


def classify_candidate_comparison(
    *,
    baseline_metrics: dict,
    candidate_metrics: dict,
    primary_metric: str,
    metric_direction: str,
    evidence_quality: str,
    test_status: str,
    benchmark_status: str,
    regression_count: int,
    failure_count: int,
) -> tuple[str, bool | None]:
    if test_status == "FAILED" or benchmark_status == "FAILED" or failure_count:
        return "FAILED", None
    if test_status != "PASSED" or benchmark_status != "COMPLETED":
        return "UNKNOWN", None
    if primary_metric not in baseline_metrics or primary_metric not in candidate_metrics:
        return "UNKNOWN", None
    before = baseline_metrics[primary_metric]
    after = candidate_metrics[primary_metric]
    for name, value in (("baseline", before), ("candidate", after)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} metric {primary_metric!r} must be finite numeric data.")
    if metric_direction not in {"lower", "higher"}:
        raise ValueError("Candidate metric direction must be lower or higher.")
    if regression_count:
        return "REGRESSED", False
    if math.isclose(float(before), float(after), rel_tol=1e-12, abs_tol=1e-15):
        return "INCONCLUSIVE", None
    candidate_better = (
        after < before if metric_direction == "lower" else after > before
    )
    if not candidate_better:
        return "REGRESSED", False
    if evidence_quality != "ADEQUATE":
        return "INCONCLUSIVE", None
    relative_gain = (
        (before - after) / abs(before)
        if metric_direction == "lower" and before != 0
        else (after - before) / abs(before)
        if metric_direction == "higher" and before != 0
        else 0.0
    )
    if relative_gain >= 0.10:
        return "STRONGLY_IMPROVED", True
    return "IMPROVED", True


def validate_improvement_decision(
    decision: str,
    evaluation: dict | None,
    regression_count: int | None = None,
) -> None:
    if decision not in IMPROVEMENT_DECISIONS:
        raise ValueError(f"Unsupported improvement decision: {decision}")
    if decision in {"ACCEPT_CANDIDATE", "READY_FOR_DEPLOYMENT"}:
        if not evaluation or evaluation.get("status") not in {
            "IMPROVED",
            "STRONGLY_IMPROVED",
        }:
            raise ValueError(
                "A candidate can only be accepted after demonstrated improvement."
            )
        if evaluation.get("test_status") != "PASSED":
            raise ValueError("Candidate acceptance requires passing tests.")
        if regression_count != 0:
            raise ValueError("Candidate acceptance requires zero regressions.")
        if evaluation.get("evidence_quality") != "ADEQUATE":
            raise ValueError("Candidate acceptance requires adequate evidence.")
    if decision == "READY_FOR_DEPLOYMENT":
        raise ValueError(
            "Production deployment is not implemented in this milestone."
        )
