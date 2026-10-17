import json
import logging
import math
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kai.config import KaiConfig
from kai.database import SCHEMA_VERSION, Database
from kai.engineering import EngineeringCoordinator
from kai.errors import KaiError
from kai.experiment_registry import (
    CPU_BASELINE_EXPERIMENT_ID,
    DEFAULT_EXPERIMENT_REGISTRY,
    InvalidExperimentResultError,
    ExperimentRegistry,
)
from kai.local_llm import (
    LLMProvider,
    ModelGateway,
    TASK_CATEGORIES,
)
from kai.environment_model import (
    build_epistemic_claims,
    build_self_model,
    compare_environments,
    collect_environment_model,
    environment_fingerprint,
    measure_baseline,
)
from kai.discovery import capability_summary
from kai.candidate_runner import CandidateRunner
from kai.candidate_workspace import CandidateWorkspaceManager
from kai.benchmark import run_cpu_predictor_benchmark
from kai.evaluation_protocol import (
    CPU_PREDICTOR_PROTOCOL,
    summarize_repeated_errors,
)
from kai.predictive import (
    CPU_BASELINE_DOMAIN,
    cpu_baseline_context,
    predict_from_model,
    resource_estimate,
)
from kai.self_improvement import (
    CPU_DEMO_CANDIDATE_ID,
    CPU_DEMO_MAXIMUM_OBSERVATIONS,
    CPU_DEMO_MINIMUM_OBSERVATIONS,
    IMPROVEMENT_EVIDENCE_MINIMUM,
    inspect_source_tree,
    last_observation_forecast,
    running_mean_forecast,
    score_forecasters,
)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            },
            ensure_ascii=False,
        )


class KaiRuntime:
    def __init__(
        self,
        config: KaiConfig | None = None,
        experiment_registry: ExperimentRegistry | None = None,
        llm_provider: LLMProvider | None = None,
        configure_logging: bool = True,
    ):
        self.config = config or KaiConfig.from_environment()
        self.experiment_registry = experiment_registry or DEFAULT_EXPERIMENT_REGISTRY
        self._configured_llm_provider = llm_provider
        self.database = Database(self.config.database_path)
        self.logger = logging.getLogger(f"kai.{self.config.data_dir.resolve()}")
        self.logger.setLevel(logging.INFO)
        if configure_logging and not self.logger.handlers:
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(self.config.log_path, encoding="utf-8")
            handler.setFormatter(JsonLogFormatter())
            self.logger.addHandler(handler)
            self.logger.propagate = False

    def _llm_provider(self, *, read_only: bool = False) -> LLMProvider:
        if self._configured_llm_provider is not None:
            return self._configured_llm_provider
        available_memory = None
        try:
            available_memory = (
                int(os.sysconf("SC_AVPHYS_PAGES"))
                * int(os.sysconf("SC_PAGE_SIZE"))
            )
        except (AttributeError, OSError, ValueError):
            pass
        return ModelGateway(
            self.database,
            default_provider=self.config.llm_provider,
            default_model=self.config.llm_model,
            openai_enabled=self.config.openai_enabled,
            openai_allow_source=self.config.openai_allow_source,
            available_memory_bytes=available_memory,
            read_only=read_only,
            timeout_seconds=self.config.llm_timeout_seconds,
        )

    def local_llm_status(self) -> dict:
        return self._llm_provider().status()

    def run_cpu_predictor_benchmark(self) -> dict:
        """Run and persist one deterministic benchmark evidence record."""
        self.database.initialize()
        report = run_cpu_predictor_benchmark(source_root=Path(__file__).resolve().parent.parent)
        event_id = self.database.record_event(
            "BENCHMARK_COMPLETED",
            f"Benchmark {report['benchmark']['benchmark_id']} completed with {report['decision']}.",
            report,
            source="kai.benchmark",
            epistemic_status="OBSERVATION",
        )
        return {"event_id": event_id, **report}

    def generate_local_response(
        self,
        prompt: str,
        task_category: str = "general",
        model_override: str | None = None,
        local_only: bool = False,
    ) -> dict:
        provider = self._llm_provider()
        if isinstance(provider, ModelGateway):
            return provider.generate(
                prompt,
                task_category=task_category,
                explicit_target=model_override,
                local_only=local_only,
            )
        if local_only and not provider.status().get("local_only", False):
            raise KaiError("The injected model provider is not verified as local.")
        return provider.generate(prompt)

    def llm_providers(self) -> list[dict]:
        provider = self._llm_provider()
        if isinstance(provider, ModelGateway):
            return provider.providers()
        status = provider.status()
        return [{"provider": status.get("provider"), "status": status.get("status")}]

    def llm_models(self) -> list[dict]:
        provider = self._llm_provider()
        return provider.models() if isinstance(provider, ModelGateway) else []

    def llm_active(self) -> dict:
        provider = self._llm_provider()
        return provider.active() if isinstance(provider, ModelGateway) else {
            "status": provider.status(),
            "selection": {
                "provider": provider.status().get("provider"),
                "model_id": provider.status().get("model"),
            },
        }

    def llm_use(self, target_name: str) -> dict:
        provider = self._llm_provider()
        if not isinstance(provider, ModelGateway):
            raise KaiError("Model selection is unavailable for the injected provider.")
        target = provider.parse_target(target_name)
        status = provider.validate_selection(target)
        selection = self.database.set_llm_model_selection(
            target["provider"], target["model_id"]
        )
        return {"selection": selection, "status": status}

    def llm_test(self) -> dict:
        return self.generate_local_response(
            "Reply with the exact text: KAI model gateway connection test successful.",
            task_category="general",
        )

    def llm_history(self, limit: int = 50) -> list[dict]:
        return self.database.list_llm_requests(limit)

    def llm_routes(self) -> list[dict]:
        return self.database.list_llm_task_routes()

    def llm_route_set(self, task_category: str, target_name: str) -> dict:
        if task_category not in TASK_CATEGORIES:
            raise KaiError(f"Unsupported LLM task category: {task_category!r}.")
        provider = self._llm_provider()
        if not isinstance(provider, ModelGateway):
            raise KaiError("Task routing is unavailable for the injected provider.")
        target = provider.parse_target(target_name)
        provider.validate_selection(target, task_category=task_category)
        return self.database.set_llm_task_route(
            task_category, target["provider"], target["model_id"]
        )

    def llm_route_clear(self, task_category: str) -> dict:
        self.database.delete_llm_task_route(task_category)
        return {"task_category": task_category, "status": "CLEARED"}

    def llm_fallbacks(self) -> list[dict]:
        return self.database.list_llm_fallbacks()

    def llm_fallback_add(self, target_name: str) -> list[dict]:
        provider = self._llm_provider()
        if not isinstance(provider, ModelGateway):
            raise KaiError("Fallback configuration is unavailable for the injected provider.")
        target = provider.parse_target(target_name)
        provider.validate_selection(target)
        return self.database.add_llm_fallback(
            target["provider"], target["model_id"]
        )

    def llm_fallback_clear(self) -> dict:
        self.database.clear_llm_fallbacks()
        return {"status": "CLEARED", "fallbacks": []}

    def _engineering_coordinator(
        self, repair_attempts: int = 1, *, read_only: bool = False
    ) -> EngineeringCoordinator:
        if (
            isinstance(repair_attempts, bool)
            or not isinstance(repair_attempts, int)
            or repair_attempts < 0
            or repair_attempts > 2
        ):
            raise KaiError("Engineering repair attempts must be between zero and two.")
        if read_only:
            manager = CandidateWorkspaceManager(
                Path(__file__).resolve().parents[1],
                self.config.data_dir,
                self.database,
            )
            runner = CandidateRunner(self.database, manager)
            schema_version = SCHEMA_VERSION
        else:
            manager, runner = self._candidate_services()
            schema_version = self.database.schema_version()
        return EngineeringCoordinator(
            database=self.database,
            provider=self._llm_provider(read_only=read_only),
            workspace_manager=manager,
            runner=runner,
            source_root=Path(__file__).resolve().parents[1],
            kai_version=self.version,
            schema_version=schema_version,
            repair_attempts=repair_attempts,
        )

    def engineer_status(self) -> dict:
        return self._engineering_coordinator().status()

    def engineer_inspect(self) -> dict:
        return self._engineering_coordinator(read_only=True).inspect()

    def engineer_plan(self, objective: str) -> dict:
        return self._engineering_coordinator().plan(objective)

    def engineer_run(self, objective: str, repair_attempts: int = 1) -> dict:
        return self._engineering_coordinator(repair_attempts).run(objective)

    def list_engineering_tasks(self, limit: int = 50) -> list[dict]:
        self.database.initialize()
        return self.database.list_engineering_tasks(limit)

    def get_engineering_task(self, task_id: str) -> dict:
        self.database.initialize()
        return self.database.get_engineering_task(task_id)

    def candidate_diff(self, candidate_id: str) -> list[dict]:
        self.database.initialize()
        self.database.get_kai_candidate(candidate_id)
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT workspace_id FROM candidate_workspaces "
                "WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchone()
        if row is None:
            raise KaiError(f"Candidate {candidate_id!r} has no source workspace.")
        manager, _ = self._candidate_services()
        return manager.diff(row["workspace_id"])

    def _candidate_services(self) -> tuple[CandidateWorkspaceManager, CandidateRunner]:
        self.database.initialize()
        manager = CandidateWorkspaceManager(
            Path(__file__).resolve().parents[1],
            self.config.data_dir,
            self.database,
        )
        return manager, CandidateRunner(self.database, manager)

    def self_doctor(self) -> dict:
        manager, runner = self._candidate_services()
        sandbox_probe = runner.probe_sandbox()
        return {
            "candidate_development": {
                "workspace_root": str(manager.workspace_root),
                "workspace_root_outside_production_source": not manager._is_within(
                    manager.workspace_root.resolve(), manager.source_root
                ),
                "sandbox": sandbox_probe,
                "sandbox_binary_available": runner.sandbox_available(),
                "source_revision": manager._source_revision(),
                "evaluation_protocols": self.database.list_evaluation_protocols(),
                "workspaces": self.database.list_candidate_workspaces(limit=100),
                "execution_policy": {
                    "network": "disabled",
                    "workspace_source": "read-only during execution",
                    "automatic_code_generation": False,
                    "production_mutation": False,
                    "deployment": False,
                },
            },
            "state_database": {
                "schema_version": self.database.schema_version(),
                "integrity": self.database.integrity_check(),
            },
        }

    def create_candidate_workspace(self, candidate_id: str) -> dict:
        manager, _ = self._candidate_services()
        report = self._capture_model()
        record = manager.create(
            candidate_id=candidate_id,
            parent_kai_version=self.version,
            source_revision=None,
            schema_version=self.database.schema_version(),
            environment_fingerprint=report["environment_fingerprint"],
            configuration={
                "database_path": str(self.config.database_path),
                "data_dir": str(self.config.data_dir),
            },
        )
        validation = manager.validate(record["workspace_id"])
        if validation["status"] == "VALID":
            self.database.update_candidate_workspace_status(
                record["workspace_id"], "VALIDATED", validation
            )
        else:
            self.database.update_candidate_workspace_status(
                record["workspace_id"], "FAILED", validation
            )
        return {**record, "validation": validation}

    def preview_candidate_workspace(self, candidate_id: str) -> dict:
        manager, _ = self._candidate_services()
        return manager.preview(
            candidate_id=candidate_id,
            configuration={
                "database_path": str(self.config.database_path),
                "data_dir": str(self.config.data_dir),
            },
        )

    def list_candidate_workspaces(self, limit: int = 50) -> list[dict]:
        self.database.initialize()
        return self.database.list_candidate_workspaces(limit)

    def get_candidate_workspace(self, workspace_id: str) -> dict:
        self.database.initialize()
        return self.database.get_candidate_workspace(workspace_id)

    def validate_candidate_workspace(self, workspace_id: str) -> dict:
        manager, _ = self._candidate_services()
        validation = manager.validate(workspace_id)
        if validation["status"] == "VALID":
            self.database.update_candidate_workspace_status(
                workspace_id, "VALIDATED", validation
            )
        return validation

    def candidate_workspace_diff(self, workspace_id: str) -> list[dict]:
        manager, _ = self._candidate_services()
        return manager.diff(workspace_id)

    def edit_candidate_workspace(
        self,
        *,
        workspace_id: str,
        target_file: str,
        operation: str,
        content: str | None,
        reason: str,
        hypothesis_id: int,
    ) -> dict:
        manager, _ = self._candidate_services()
        return manager.apply_change(
            workspace_id=workspace_id,
            target_file=target_file,
            operation=operation,
            content=content,
            reason=reason,
            hypothesis_id=hypothesis_id,
        )

    def test_candidate_workspace(
        self, workspace_id: str, module: str | None = None
    ) -> dict:
        manager, runner = self._candidate_services()
        workspace = self.database.get_candidate_workspace(workspace_id)
        if workspace["status"] in {"PRESERVED", "CLEANED"}:
            raise KaiError("Preserved or cleaned candidate workspaces cannot be executed.")
        result = runner.run_tests(
            workspace["candidate_id"], workspace_id, module=module
        )
        if result.status == "SUCCESS":
            self.database.update_candidate_workspace_status(
                workspace_id, "TESTED", {"execution_id": result.execution_id}
            )
        elif result.status not in {"UNKNOWN"}:
            self.database.update_candidate_workspace_status(
                workspace_id, "FAILED", {"execution_id": result.execution_id}
            )
        return result.to_dict()

    def cleanup_candidate_workspace(
        self, workspace_id: str, preserve: bool = False
    ) -> dict:
        manager, _ = self._candidate_services()
        return manager.cleanup(workspace_id, preserve=preserve)

    def demonstrate_cpu_candidate_workspace(self) -> dict:
        """Run the predeclared repeated CPU protocol inside a disposable workspace."""
        report = self._capture_model()
        environment_context = {
            "environment_fingerprint": report["environment_fingerprint"],
            "environment_snapshot_id": report["snapshot_id"],
        }
        sequences: list[list[float]] = []
        observation_ids: list[int] = []
        failures: list[dict] = []
        total_measurements = (
            CPU_PREDICTOR_PROTOCOL.sequence_count
            * CPU_PREDICTOR_PROTOCOL.measurements_per_sequence
        )
        for trial in range(total_measurements):
            try:
                measurement = measure_baseline(
                    report["environment"], self.config.database_path
                )
            except (KaiError, OSError, ValueError) as exc:
                failures.append(
                    {"trial": trial + 1, "status": "FAILED", "reason": str(exc)}
                )
                unknown = self.database.record_self_observation(
                    {
                        "subject": "CPU repeated-protocol predictor",
                        "observation_type": "measurement.cpu_baseline_runtime",
                        "observed_at": datetime.now(timezone.utc).isoformat(),
                        "value": {"status": "FAILED", "trial": trial + 1},
                        "source": "Explicit isolated kai improvements demo-cpu measurement",
                        "epistemic_status": "UNKNOWN",
                        "evidence_status": "unknown",
                        "confidence": None,
                        "kai_version": self.version,
                        "environment_context": environment_context,
                    }
                )
                observation_ids.append(unknown["observation_id"])
                break
            cpu = measurement.get("cpu") if isinstance(measurement, dict) else None
            duration = cpu.get("duration_seconds") if isinstance(cpu, dict) else None
            if (
                not isinstance(cpu, dict)
                or cpu.get("status") != "measured"
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                failures.append(
                    {
                        "trial": trial + 1,
                        "status": "UNKNOWN",
                        "reason": (
                            cpu.get("reason", "CPU measurement unavailable")
                            if isinstance(cpu, dict)
                            else "Malformed CPU measurement report"
                        ),
                    }
                )
                unknown = self.database.record_self_observation(
                    {
                        "subject": "CPU repeated-protocol predictor",
                        "observation_type": "measurement.cpu_baseline_runtime",
                        "observed_at": measurement.get(
                            "captured_at", datetime.now(timezone.utc).isoformat()
                        ),
                        "value": {
                            "status": "UNKNOWN",
                            "trial": trial + 1,
                            "reason": failures[-1]["reason"],
                        },
                        "source": "Explicit isolated kai improvements demo-cpu measurement",
                        "epistemic_status": "UNKNOWN",
                        "evidence_status": "unknown",
                        "confidence": None,
                        "kai_version": self.version,
                        "environment_context": environment_context,
                    }
                )
                observation_ids.append(unknown["observation_id"])
                break
            duration = float(duration)
            if trial % CPU_PREDICTOR_PROTOCOL.measurements_per_sequence == 0:
                sequences.append([])
            sequences[-1].append(duration)
            observation = self.database.record_self_observation(
                {
                    "subject": "CPU repeated-protocol predictor",
                    "observation_type": "measurement.cpu_baseline_runtime",
                    "observed_at": measurement.get(
                        "captured_at", datetime.now(timezone.utc).isoformat()
                    ),
                    "value": {
                        "duration_seconds": duration,
                        "trial": trial + 1,
                        "protocol_id": CPU_PREDICTOR_PROTOCOL.protocol_id,
                        "measurement": cpu,
                    },
                    "source": "Explicit isolated kai improvements demo-cpu measurement",
                    "epistemic_status": "OBSERVED",
                    "evidence_status": "observed",
                    "confidence": 1.0,
                    "kai_version": self.version,
                    "environment_context": environment_context,
                }
            )
            observation_ids.append(observation["observation_id"])

        source_tree = inspect_source_tree(Path(__file__).resolve().parents[1])
        opportunity = self.database.create_improvement_opportunity(
            source_observation_ids=observation_ids,
            affected_subsystem="cpu_baseline_predictor",
            current_behavior="Predicts CPU workload duration with a running mean.",
            evidence={
                "source_observation_ids": observation_ids,
                "durations_seconds": [
                    value for sequence in sequences for value in sequence
                ],
                "measurement_failures": failures,
                "protocol_id": CPU_PREDICTOR_PROTOCOL.protocol_id,
                "environment_context": environment_context,
            },
            severity="MEDIUM",
            expected_impact="Reduce paired prequential CPU runtime prediction error.",
            confidence=None,
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "INVESTIGATING",
            "Evaluate a bounded predictor under a predeclared repeated protocol.",
            {"protocol_id": CPU_PREDICTOR_PROTOCOL.protocol_id},
        )
        hypothesis = self.database.create_improvement_hypothesis(
            opportunity_id=opportunity["opportunity_id"],
            observed_evidence={
                "source_observation_ids": observation_ids,
                "measurement_count": sum(map(len, sequences)),
            },
            interpretation=(
                "A last-observation predictor may reduce error for sufficiently "
                "stable CPU timing sequences."
            ),
            proposed_intervention=(
                "Compare a pure last-observation predictor only in a disposable "
                "source workspace."
            ),
            expected_outcome={
                "metric": CPU_PREDICTOR_PROTOCOL.metric,
                "direction": CPU_PREDICTOR_PROTOCOL.direction,
                "minimum_relative_change": (
                    CPU_PREDICTOR_PROTOCOL.minimum_relative_change
                ),
            },
            uncertainty={
                "classification": "UNKNOWN",
                "reasons": [
                    "CPU timings vary with load and scheduling.",
                    "The fixed repeated protocol is not a statistical significance test.",
                ],
            },
            negative_consequences={
                "possible_regression": "The latest observation may overreact to noise.",
                "production_change": False,
            },
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "PROPOSED",
            "Recorded an uncertain, falsifiable candidate hypothesis.",
            {"hypothesis_id": hypothesis["hypothesis_id"]},
        )
        proposal = self.database.create_improvement_proposal(
            hypothesis_id=hypothesis["hypothesis_id"],
            affected_component="kai.predictive.cpu_baseline",
            current_implementation="Running-mean CPU duration forecast.",
            proposed_implementation="Isolated last-observation forecast.",
            reason="Use paired samples and a predeclared repeated-sequence rule.",
            evidence={"source_observation_ids": observation_ids},
            expected_benefit="Lower prequential mean absolute error if supported.",
            expected_cost="One observation of state; sensitive to transient timing noise.",
            risk="The candidate may regress on unstable timing sequences.",
            dependencies=["Python standard library", "Linux bubblewrap"],
            test_requirements=["Fixed candidate predictor contract tests in sandbox"],
            benchmark_requirements=[
                "Three repeated 10-measurement CPU sequences with paired forecasts"
            ],
            rollback_strategy="Candidate remains isolated and is never deployed.",
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "EXPERIMENTING",
            "Created a non-deployable candidate and disposable workspace.",
            {"proposal_id": proposal["proposal_id"]},
        )
        candidate_id = str(uuid.uuid4())
        candidate = self.database.create_kai_candidate(
            candidate_id=candidate_id,
            proposal_id=proposal["proposal_id"],
            candidate_version=1,
            parent_kai_version=self.version,
            source_snapshot_ref=(
                "source-tree-sha256:" + source_tree["source_snapshot_sha256"]
            ),
            source_snapshot_sha256=source_tree["source_snapshot_sha256"],
            candidate_specification={
                "algorithm": "last observation",
                "protocol_id": CPU_PREDICTOR_PROTOCOL.protocol_id,
                "production_model_mutation": False,
            },
            isolation_metadata={
                "isolated": True,
                "mode": "read-only bubblewrap execution of disposable source snapshot",
                "production_files_modified": False,
                "production_model_mutation": False,
                "deployment_implemented": False,
            },
        )
        workspace_record = self.create_candidate_workspace(candidate_id)
        workspace_id = workspace_record["workspace_id"]
        manager, runner = self._candidate_services()
        module_source = (
            "def forecast(history):\n"
            "    if not history:\n"
            "        raise ValueError('history must not be empty')\n"
            "    return history[-1]\n"
        )
        test_source = (
            "import unittest\n"
            "from kai.candidate_algorithm import forecast\n"
            "\n"
            "class CandidateContractTests(unittest.TestCase):\n"
            "    def test_returns_latest_observation(self):\n"
            "        self.assertEqual(forecast([1.0, 3.0]), 3.0)\n"
            "\n"
            "    def test_rejects_empty_history(self):\n"
            "        with self.assertRaises(ValueError):\n"
            "            forecast([])\n"
        )
        manager.apply_change(
            workspace_id=workspace_id,
            target_file="kai/candidate_algorithm.py",
            operation="ADD",
            content=module_source,
            reason="Explicit bounded repeated-protocol CPU predictor demonstration.",
            hypothesis_id=hypothesis["hypothesis_id"],
        )
        manager.apply_change(
            workspace_id=workspace_id,
            target_file="tests/test_cpu_candidate_contract.py",
            operation="ADD",
            content=test_source,
            reason="Verify the isolated candidate predictor contract.",
            hypothesis_id=hypothesis["hypothesis_id"],
        )
        candidate_test = runner.run_tests(
            candidate_id, workspace_id, module="tests.test_cpu_candidate_contract"
        )
        if candidate_test.status == "SUCCESS":
            self.database.update_candidate_workspace_status(
                workspace_id, "TESTED", {"execution_id": candidate_test.execution_id}
            )
        elif candidate_test.status != "UNKNOWN":
            self.database.update_candidate_workspace_status(
                workspace_id, "FAILED", {"execution_id": candidate_test.execution_id}
            )
        if failures or len(observation_ids) != total_measurements:
            self.database.update_improvement_opportunity_status(
                opportunity["opportunity_id"],
                "DEFERRED",
                "The repeated protocol did not produce its complete measurement window.",
                {"measurement_failures": failures},
            )
            return {
                "status": "UNKNOWN",
                "reason": "The repeated protocol did not collect every predeclared measurement.",
                "measurement_failures": failures,
                "measurement_count": len(observation_ids),
                "required_measurement_count": total_measurements,
                "candidate_test": candidate_test.to_dict(),
                "candidate": candidate,
                "opportunity": opportunity,
                "hypothesis": hypothesis,
                "proposal": proposal,
                "workspace": self.database.get_candidate_workspace(workspace_id),
                "production_changed": False,
            }
        if candidate_test.status != "SUCCESS":
            self.database.update_improvement_opportunity_status(
                opportunity["opportunity_id"],
                "REJECTED" if candidate_test.status == "FAILED" else "DEFERRED",
                "Candidate tests did not establish a safe evaluation gate.",
                {"candidate_test": candidate_test.to_dict()},
            )
            return {
                "status": (
                    "UNKNOWN"
                    if candidate_test.status == "UNKNOWN"
                    else "FAILED"
                ),
                "reason": (
                    "Candidate testing did not pass; benchmark execution was not started."
                ),
                "candidate_test": candidate_test.to_dict(),
                "candidate": candidate,
                "opportunity": opportunity,
                "hypothesis": hypothesis,
                "proposal": proposal,
                "workspace": self.database.get_candidate_workspace(workspace_id),
                "production_changed": False,
            }

        protocol_execution, errors = runner.run_cpu_predictor_evaluation(
            candidate_id,
            workspace_id,
            sequences,
            warmup=CPU_PREDICTOR_PROTOCOL.warmup_measurements,
        )
        if errors is None:
            self.database.update_improvement_opportunity_status(
                opportunity["opportunity_id"],
                "REJECTED"
                if protocol_execution.status in {"FAILED", "INVALID", "CRASHED"}
                else "DEFERRED",
                "The sandboxed repeated evaluation did not produce valid metrics.",
                {"protocol_execution": protocol_execution.to_dict()},
            )
            return {
                "status": "UNKNOWN",
                "reason": protocol_execution.reason or "Isolated evaluation did not complete.",
                "protocol_execution": protocol_execution.to_dict(),
                "candidate_test": candidate_test.to_dict(),
                "candidate": candidate,
                "opportunity": opportunity,
                "hypothesis": hypothesis,
                "proposal": proposal,
                "workspace": self.database.get_candidate_workspace(workspace_id),
                "production_changed": False,
            }
        comparison = summarize_repeated_errors(
            errors["baseline_errors"],
            errors["candidate_errors"],
            CPU_PREDICTOR_PROTOCOL,
        )
        baseline_metrics = {
            "mae_seconds": comparison["baseline_metrics"]["mae_seconds"],
            "mean_absolute_error_seconds": comparison["baseline_metrics"]["mae_seconds"],
            "algorithm": "running-mean-v1",
        }
        candidate_metrics = {
            "mae_seconds": comparison["candidate_metrics"]["mae_seconds"],
            "mean_absolute_error_seconds": comparison["candidate_metrics"]["mae_seconds"],
            "algorithm": "last-observation-v1",
        }
        comparison.update(
            {
                "primary_metric": "mae_seconds",
                "metric_direction": "lower",
                "candidate_execution_id": protocol_execution.execution_id,
                "measurement_count": total_measurements,
                "statistical_significance_claimed": False,
            }
        )
        protocol_record = self.database.register_evaluation_protocol(
            CPU_PREDICTOR_PROTOCOL
        )
        baseline_snapshot = {
            "kai_version": self.version,
            "architecture_version": "0.007",
            "predictive_model_versions": [
                {
                    "model_id": item["model_id"],
                    "version": item["version"],
                    "domain": item["domain"],
                }
                for item in self.database.list_predictive_models()
            ],
            "experiment_definition_versions": [
                {
                    "experiment_id": item.experiment_id,
                    "version": item.version,
                    "procedure_id": item.procedure_id,
                }
                for item in self.experiment_registry.list()
            ],
            "configuration": {
                "protocol_id": protocol_record["protocol_id"],
                "protocol_version": protocol_record["version"],
                "protocol_sha256": protocol_record["definition_sha256"],
            },
            "environment_context": environment_context,
            "evaluation_dataset": {
                "observation_ids": observation_ids,
                "observation_count": len(observation_ids),
                "paired_forecast_count": comparison["paired_forecast_count"],
                "scoring": "Predeclared paired repeated-sequence absolute error.",
            },
            "metrics": baseline_metrics,
            "evaluation_protocol": protocol_record,
        }
        test_status = (
            "PASSED"
            if candidate_test.status == "SUCCESS"
            else "UNKNOWN"
            if candidate_test.status == "UNKNOWN"
            else "FAILED"
        )
        protocol_status = comparison["status"]
        status = (
            protocol_status
            if test_status == "PASSED"
            else "UNKNOWN"
            if test_status == "UNKNOWN"
            else "FAILED"
        )
        comparison["protocol_status"] = protocol_status
        comparison["status"] = status
        evaluation = self.database.record_candidate_evaluation(
            candidate_id=candidate_id,
            baseline=baseline_snapshot,
            evaluation_status=status,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            test_status=test_status,
            benchmark_status="COMPLETED",
            regression_count=int(status == "REGRESSED"),
            failure_count=int(test_status == "FAILED"),
            evidence_quality="ADEQUATE",
            evidence={
                "source_observation_ids": observation_ids,
                "source_execution_ids": [
                    candidate_test.execution_id,
                    protocol_execution.execution_id,
                ],
                "workspace_id": workspace_id,
                "protocol_id": protocol_record["protocol_id"],
                "protocol_definition_sha256": protocol_record["definition_sha256"],
            },
            hypothesis_satisfied=(
                True if status == "IMPROVED"
                else False if status == "REGRESSED"
                else None
            ),
        )
        decision_code = (
            "ACCEPT_CANDIDATE"
            if status == "IMPROVED" and test_status == "PASSED"
            else "REJECT"
            if status == "REGRESSED" or test_status == "FAILED"
            else "REQUEST_MORE_EVIDENCE"
        )
        decision = self.database.record_improvement_decision(
            candidate_id=candidate_id,
            evaluation_id=evaluation["evaluation_id"],
            decision=decision_code,
            reason=(
                "Predeclared repeated-sequence rule: "
                + status
                + "; production remains unchanged."
            ),
            evidence={
                "evaluation_id": evaluation["evaluation_id"],
                "workspace_id": workspace_id,
                "production_deployment": "NOT_AVAILABLE",
            },
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "ACCEPTED"
            if decision_code == "ACCEPT_CANDIDATE"
            else "REJECTED"
            if decision_code == "REJECT"
            else "DEFERRED",
            "Recorded the repeated-protocol decision; candidate remains undeployed.",
            {"decision_id": decision["decision_id"]},
        )
        return {
            "status": status,
            "workspace": self.database.get_candidate_workspace(workspace_id),
            "candidate_test": candidate_test.to_dict(),
            "protocol_execution": protocol_execution.to_dict(),
            "candidate": candidate,
            "opportunity": self.database.get_improvement_opportunity(
                opportunity["opportunity_id"]
            ),
            "hypothesis": hypothesis,
            "proposal": proposal,
            "comparison": comparison,
            "evaluation": evaluation,
            "decision": decision,
            "measurement_observation_ids": observation_ids,
            "production_changed": False,
        }

    def _capture_model(
        self, measure: bool = False, persist_self_model: bool = True
    ) -> dict:
        self.database.initialize()
        environment = collect_environment_model(self.config.data_dir)
        environment["kai_version"] = self.version
        captured_at = datetime.now(timezone.utc).isoformat()
        fingerprint = environment_fingerprint(environment)
        previous = self.database.latest_environment_snapshot()
        claims = build_epistemic_claims(environment, captured_at)
        if measure:
            measurements = measure_baseline(environment, self.config.database_path)
            claims.append(
                {
                    "claim": "A bounded CPU and read-only storage baseline measurement was performed.",
                    "classification": "OBSERVATION",
                    "source": "Explicit kai measure command",
                    "timestamp": measurements["captured_at"],
                    "confidence": 0.99,
                    "evidence": [
                        json.dumps(measurements["cpu"], sort_keys=True),
                        json.dumps(measurements["disk_read"], sort_keys=True),
                    ],
                    "verification_status": "observed",
                    "contradictory_evidence": [],
                }
            )
        elif previous and previous["environment_fingerprint"] == fingerprint:
            measurements = previous["measurements"]
        else:
            measurements = {}
        current = {
            "captured_at": captured_at,
            "environment_fingerprint": fingerprint,
            "environment": environment,
            "measurements": measurements,
        }
        changes = compare_environments(previous, current)
        if previous is not None:
            same_profile = changes["same_environment_profile"]
            claims.append(
                {
                    "claim": (
                        "This environment matches the previous hardware/architecture profile."
                        if same_profile
                        else "This environment has a new or substantially changed hardware/architecture profile."
                    ),
                    "classification": "INFERENCE",
                    "source": "Comparison of privacy-preserving environment profile fingerprints",
                    "timestamp": captured_at,
                    "confidence": 0.7,
                    "evidence": [
                        f"previous_fingerprint={previous['environment_fingerprint']}",
                        f"current_fingerprint={fingerprint}",
                        "A profile match does not prove it is the same physical host.",
                    ],
                    "verification_status": "observed",
                    "contradictory_evidence": [],
                }
            )
        saved = self.database.save_environment_snapshot(
            captured_at,
            fingerprint,
            environment,
            claims,
            measurements,
            changes,
        )
        identity = self.database.identity()
        capabilities = capability_summary(environment)
        llm_status = self.local_llm_status()
        capabilities["language_model"] = (
            "not configured"
            if llm_status["status"] == "DISABLED"
            else "configured"
        )
        capabilities["local_llm_runtime"] = llm_status["status"]
        goals = self.database.list_goals()
        self_model = build_self_model(
            identity,
            environment,
            fingerprint,
            capabilities,
            goals,
            claims,
            self.version,
            SCHEMA_VERSION,
        )
        self_model["environment_identity"]["same_as_previous_profile"] = (
            changes["same_environment_profile"]
        )
        self_model["language_model"] = llm_status
        if llm_status["status"] not in {"READY", "DISABLED"}:
            self_model["known_limitations"].append(
                "Configured local language model is not ready: "
                + str(llm_status.get("reason") or llm_status["status"])
            )
        self_model["software_capabilities"] = environment["software_capabilities"]
        self_model["experience_state"] = self.database.history_summary()
        self_model["candidate_development"] = {
            "evaluation_protocols": self.database.list_evaluation_protocols(),
            "workspaces": [
                {
                    "workspace_id": item["workspace_id"],
                    "candidate_id": item["candidate_id"],
                    "status": item["status"],
                    "source_tree_sha256": item["source_tree_sha256"],
                }
                for item in self.database.list_candidate_workspaces(limit=100)
            ],
            "sandbox_binary_available": CandidateRunner.sandbox_available(),
            "sandbox_execution_readiness": "UNKNOWN_NOT_PROBED",
            "automatic_source_generation": False,
            "production_source_mutation": False,
            "production_model_mutation": False,
            "deployment": False,
        }
        predictor_context = cpu_baseline_context(
            fingerprint,
            saved["snapshot_id"],
        )
        predictor = self.database.get_cpu_baseline_model(predictor_context)
        self_model["predictive_capabilities"] = {
            "models": [
                {
                    "model_id": item["model_id"],
                    "domain": item["domain"],
                    "model_type": item["model_type"],
                    "version": item["version"],
                    "status": item["status"],
                }
                for item in self.database.list_predictive_models()
                if item["status"] == "ACTIVE"
            ],
            "cpu_baseline": (
                {
                    **predict_from_model(predictor),
                    "performance": predictor["performance"],
                    "confidence_information": predictor["confidence"],
                }
                if predictor
                else {
                    "status": "INSUFFICIENT DATA",
                    "domain": CPU_BASELINE_DOMAIN,
                    "reason": "Run the explicit bounded CPU baseline experiment to seed a model.",
                }
            ),
        }
        if persist_self_model:
            self._persist_self_model(self_model, captured_at)
        return {
            "identity": identity,
            "version": self.version,
            "environment": environment,
            "environment_fingerprint": fingerprint,
            "snapshot_id": saved["snapshot_id"],
            "environment_event_id": saved["event_id"],
            "environment_change_event_id": saved["change_event_id"],
            "experience_id": saved["experience_id"],
            "claims": claims,
            "changes": changes,
            "capabilities": capabilities,
            "self_model": self_model,
            "measurements": measurements,
        }

    def _persist_self_model(self, self_model: dict, timestamp: str | None = None) -> None:
        self_model["experience_state"] = self.database.history_summary()
        observed_at = timestamp or datetime.now(timezone.utc).isoformat()
        source_tree = inspect_source_tree(Path(__file__).resolve().parents[1])
        models = self.database.list_predictive_models()
        definitions = [
            item.to_dict() for item in self.experiment_registry.list()
        ]
        failures = self.database.list_experiment_executions(limit=100)
        recent_execution_outcomes = {}
        for execution in failures:
            status = execution["execution_status"]
            recent_execution_outcomes[status] = (
                recent_execution_outcomes.get(status, 0) + 1
            )
        runtime_errors = self.database.recent_events(limit=100, event_type="ERROR")
        failure_modes = [
            {
                "mode": "bounded_cpu_measurement_failure",
                "occurrences": sum(
                    item["execution_status"] in {"FAILED", "UNKNOWN", "ABORTED"}
                    for item in failures
                ),
                "epistemic_status": "OBSERVED",
                "evidence_source": "Persisted experiment execution history",
            },
            {
                "mode": "recorded_runtime_error",
                "occurrences": len(runtime_errors),
                "epistemic_status": "OBSERVED",
                "evidence_source": "Persisted ERROR events",
            },
        ]
        improvement_metrics = self.database.improvement_metrics()
        candidates = self.database.list_kai_candidates(limit=100)
        decisions = self.database.list_improvement_decisions(limit=100)
        opportunities = self.database.list_improvement_opportunities(limit=100)
        prediction_status = self.database.prediction_report(limit=100)
        self_model.update(
            {
                "operational_self_model_version": "0.007",
                "software": {
                    "version": self.version,
                    "epistemic_status": "OBSERVED",
                    "source": "kai.__version__",
                },
                "architecture": {
                    "version": "0.007",
                    "pipeline": [
                        "OBSERVE",
                        "REPRESENT",
                        "VERIFY",
                        "MODEL",
                        "REASON",
                        "ACT",
                        "MEASURE",
                        "LEARN",
                        "REFLECT",
                        "EVALUATE_SELF",
                    ],
                    "schema_version": self.database.schema_version(),
                    "epistemic_status": "OBSERVED",
                },
                "self_epistemic_states": {
                    "observed": "Directly collected from KAI runtime state or source.",
                    "inferred": "Derived from linked observations and labeled as inference.",
                    "unknown": "Not established by current safe observation.",
                },
                "source_tree": source_tree,
                "test_suite": {
                    "status": (
                        "UNKNOWN"
                        if source_tree["syntax_failures"]
                        else "NOT_RUN_BY_SELF_OBSERVER"
                    ),
                    "execution_status": "UNKNOWN",
                    "discovered_test_function_count": source_tree[
                        "test_function_count"
                    ],
                    "execution_policy": (
                        "Self-observation parses test source but does not execute tests."
                    ),
                    "syntax_failures": source_tree["syntax_failures"],
                    "epistemic_status": (
                        "UNKNOWN"
                        if source_tree["syntax_failures"]
                        else "OBSERVED"
                    ),
                },
                "available_experiments": [
                    {
                        "experiment_id": item.experiment_id,
                        "version": item.version,
                        "procedure_id": item.procedure_id,
                        "status": item.status,
                        "procedure_registered": self.experiment_registry.has_procedure(
                            item.procedure_id
                        ),
                        "epistemic_status": "OBSERVED",
                    }
                    for item in self.experiment_registry.list()
                ],
                "current_predictive_models": [
                    {
                        "model_id": item["model_id"],
                        "domain": item["domain"],
                        "model_type": item["model_type"],
                        "version": item["version"],
                        "status": item["status"],
                        "epistemic_status": "OBSERVED",
                    }
                    for item in models
                ],
                "available_tools": {
                    "software_capabilities": self_model.get(
                        "software_capabilities", {}
                    ),
                    "not_discovered": self_model.get("not_discovered_tools", []),
                    "policy": "PATH visibility is reported; discovered tools are not executed.",
                    "epistemic_status": "OBSERVED",
                    "source": "Read-only software capability inventory",
                },
                "known_failure_modes": failure_modes,
                "recent_runtime_errors": runtime_errors,
                "known_weaknesses": [
                    {
                        "claim": "The adaptive predictor currently models only one fixed CPU timing workload.",
                        "epistemic_status": "OBSERVED",
                        "source": "Registered predictor and experiment definitions",
                    },
                    {
                        "claim": "Prediction performance beyond recorded contexts and sample history is unknown.",
                        "epistemic_status": "UNKNOWN",
                        "source": "Prediction evidence boundaries",
                    },
                ],
                "performance_metrics": {
                    "prediction": prediction_status.get("statistics", {}),
                    "recent_experiment_outcomes": {
                        "execution_count": len(failures),
                        "history_limit": 100,
                        "by_status": recent_execution_outcomes,
                    },
                    "improvement_process": improvement_metrics,
                    "epistemic_status": "OBSERVED",
                },
                "improvement_engine": {
                    "present": True,
                    "status": (
                        "audited disposable source workspaces and fixed sandbox "
                        "evaluation; no production self-modification"
                    ),
                    "autonomous": False,
                    "candidate_source_workspaces": True,
                    "sandbox_binary_available": CandidateRunner.sandbox_available(),
                    "sandbox_execution_readiness": "UNKNOWN_NOT_PROBED",
                    "automatic_candidate_generation": False,
                    "production_source_mutation": False,
                    "production_model_mutation": False,
                    "production_deployment_available": False,
                    "epistemic_status": "OBSERVED",
                },
                "improvement_opportunities": opportunities,
                "active_improvement_opportunities": [
                    item
                    for item in opportunities
                    if item["status"]
                    in {"OPEN", "INVESTIGATING", "PROPOSED", "EXPERIMENTING"}
                ],
                "attempted_improvements": candidates,
                "accepted_improvements": [
                    item for item in decisions if item["decision"] == "ACCEPT_CANDIDATE"
                ],
                "rejected_improvements": [
                    item for item in decisions if item["decision"] == "REJECT"
                ],
                "current_candidate_changes": [
                    item
                    for item in candidates
                    if item["evaluation_status"] in {"UNKNOWN", "INCONCLUSIVE"}
                ],
                "dependencies": {
                    "python_version": sys.version.split()[0],
                    "sqlite_version": sqlite3.sqlite_version,
                    "imports_outside_standard_library": source_tree[
                        "imports_outside_python_standard_library"
                    ],
                    "epistemic_status": "OBSERVED",
                },
                "configuration": {
                    "state_directory_configured": True,
                    "bounded_cpu_iterations": 100000,
                    "cpu_demo_measurements_per_explicit_run": (
                        CPU_PREDICTOR_PROTOCOL.sequence_count
                        * CPU_PREDICTOR_PROTOCOL.measurements_per_sequence
                    ),
                    "cpu_demo_paired_forecast_count": (
                        CPU_PREDICTOR_PROTOCOL.sequence_count
                        * (
                            CPU_PREDICTOR_PROTOCOL.measurements_per_sequence
                            - CPU_PREDICTOR_PROTOCOL.warmup_measurements
                        )
                    ),
                    "experiment_execution_requires_explicit_invocation": True,
                    "secrets_or_raw_paths_included": False,
                    "epistemic_status": "OBSERVED",
                },
                "repository_metadata": source_tree["repository_metadata"],
                "self_model_history_count": self.database.self_model_version_count() + 1,
            }
        )
        environment_context = {
            "environment_fingerprint": self_model.get("environment_fingerprint"),
            "schema_version": self.database.schema_version(),
        }
        observations = [
            {
                "subject": "KAI",
                "observation_type": "software.version",
                "value": self.version,
                "source": "kai.__version__",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
            },
            {
                "subject": "KAI",
                "observation_type": "architecture.version",
                "value": "0.007",
                "source": "KAI architecture metadata",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
            },
            {
                "subject": "KAI",
                "observation_type": "database.schema_version",
                "value": self.database.schema_version(),
                "source": "SQLite schema_info",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
            },
            {
                "subject": "KAI",
                "observation_type": "source.tree_inventory",
                "value": source_tree,
                "source": "Read-only Python AST inspection",
                "epistemic_status": (
                    "UNKNOWN" if source_tree["syntax_failures"] else "OBSERVED"
                ),
                "evidence_status": (
                    "unknown" if source_tree["syntax_failures"] else "observed"
                ),
                "confidence": None if source_tree["syntax_failures"] else 1.0,
            },
            {
                "subject": "KAI predictor",
                "observation_type": "predictive.models",
                "value": self_model["current_predictive_models"],
                "source": "Persisted predictive model registry",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
            },
            {
                "subject": "GPU",
                "observation_type": "capability.usability",
                "value": {
                    "state": "UNKNOWN",
                    "reason": "No GPU workload is executed by self-observation.",
                },
                "source": "Safe self-observation boundary",
                "epistemic_status": "UNKNOWN",
                "evidence_status": "unknown",
                "confidence": None,
            },
            {
                "subject": "KAI improvement engine",
                "observation_type": "capability.improvement",
                "value": self_model["improvement_engine"],
                "source": "KAI runtime and candidate ledger",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
            },
        ]
        for observation in observations:
            observation.update(
                {
                    "observed_at": observed_at,
                    "kai_version": self.version,
                    "environment_context": environment_context,
                }
            )
        persisted = self.database.save_self_model_version(
            identity_id=self_model["identity"]["identity_id"],
            kai_version=self.version,
            architecture_version="0.007",
            schema_version=self.database.schema_version(),
            environment_fingerprint=self_model.get("environment_fingerprint"),
            source_snapshot_sha256=source_tree["source_snapshot_sha256"],
            epistemic_summary={
                "observed": [
                    item["observation_type"]
                    for item in observations
                    if item["epistemic_status"] == "OBSERVED"
                ],
                "unknown": [
                    item["observation_type"]
                    for item in observations
                    if item["epistemic_status"] == "UNKNOWN"
                ],
            },
            model=self_model,
            observations=observations,
            timestamp=observed_at,
        )
        self_model["self_model_version_id"] = persisted["self_model_version_id"]
        self.database.set_working_memory(
            "self_model",
            self_model,
            "kai.runtime",
            observed_at,
        )

    def boot(self) -> dict:
        self.database.initialize()
        recovered_event_ids = self.database.recover_interrupted_operations()
        model = self._capture_model(persist_self_model=False)
        event_id = self.database.record_event(
            "SYSTEM_BOOT",
            "KAI booted and persisted an environment/self-model snapshot.",
            {
                "version": self.version,
                "snapshot_id": model["snapshot_id"],
                "environment_fingerprint": model["environment_fingerprint"],
                "changes": model["changes"],
                "environment_summary": {
                    "os": model["environment"]["operating_system"]["distribution"],
                    "architecture": model["environment"]["operating_system"]["architecture"],
                    "gpu_pci_status": model["environment"]["gpu"]["pci_enumeration"]["status"],
                },
            },
            source="kai.runtime",
            environment_snapshot_id=model["snapshot_id"],
            causal_parent_id=model["environment_event_id"],
        )
        self._persist_self_model(model["self_model"])
        self.logger.info("Boot completed; event_id=%s", event_id)
        model["event_id"] = event_id
        model["recovered_event_ids"] = recovered_event_ids
        return model

    @property
    def version(self) -> str:
        from kai import __version__

        return __version__

    def status(self) -> dict:
        model = self._capture_model()
        return {
            **model,
            "known_state": {
                "schema_version": self.database.schema_version(),
                "event_count": self.database.event_count(),
                "environment_snapshot_count": self.database.environment_snapshot_count(),
                "recent_events": self.database.recent_events(limit=10),
                "memory": self.database.memory_report(limit=5),
                "recent_predictions": self.database.list_predictions(limit=5),
                "predictive_model": self.database.prediction_report(limit=10),
                "epistemic_claims": model["claims"],
                "current_measurements": model["measurements"],
            },
            "uncertainties": model["self_model"]["uncertainties"],
            "current_limitations": model["self_model"]["known_limitations"],
        }

    def measure(self) -> dict:
        latest_snapshot = self.database.latest_environment_snapshot()
        started_event_id = self.database.record_event(
            "MEASUREMENT_STARTED",
            "The explicit safe CPU and read-only storage baseline started.",
            {"measurement_kind": "bounded_local_baseline"},
            source="kai.runtime",
            environment_snapshot_id=(
                latest_snapshot["snapshot_id"] if latest_snapshot else None
            ),
        )
        try:
            model = self._capture_model(measure=True, persist_self_model=False)
        except KaiError as exc:
            self.database.record_event(
                "MEASUREMENT_FAILED",
                "The explicit baseline measurement failed.",
                {"error": str(exc)},
                source="kai.runtime",
                causal_parent_id=started_event_id,
                epistemic_status="OBSERVATION",
            )
            raise
        event_id = self.database.record_event(
            "MEASUREMENT_COMPLETED",
            "An explicit safe CPU and read-only storage baseline completed.",
            {
                "snapshot_id": model["snapshot_id"],
                "measurements": model["measurements"],
            },
            source="kai.runtime",
            environment_snapshot_id=model["snapshot_id"],
            causal_parent_id=started_event_id,
        )
        experience_id = self.database.create_experience(
            {
                "environment_fingerprint": model["environment_fingerprint"],
                "kind": "baseline_measurement",
            },
            observation=model["measurements"],
            outcome={
                "cpu_measurement_status": model["measurements"]["cpu"]["status"],
                "disk_read_status": model["measurements"]["disk_read"]["status"],
            },
            causal_event_id=event_id,
            source_event_ids=[started_event_id, event_id],
        )
        self._persist_self_model(model["self_model"])
        result = {
            "snapshot_id": model["snapshot_id"],
            "environment_fingerprint": model["environment_fingerprint"],
            "measurements": model["measurements"],
            "event_id": event_id,
            "started_event_id": started_event_id,
            "experience_id": experience_id,
        }
        self.logger.info("Safe opt-in baseline measurement completed.")
        return result

    def predict_cpu_baseline(self) -> dict:
        report = self._capture_model()
        context = cpu_baseline_context(
            report["environment_fingerprint"],
            report["snapshot_id"],
        )
        model = self.database.get_cpu_baseline_model(context)
        if model is None:
            return {
                "status": "INSUFFICIENT DATA",
                "domain": CPU_BASELINE_DOMAIN,
                "reason": (
                    "No prior CPU baseline observation exists; KAI will not invent an estimate."
                ),
                "experiment_available": True,
                "resource_estimate": resource_estimate(),
            }
        prediction = self.database.create_cpu_baseline_forecast(context)
        return {
            "status": "PREDICTED",
            "prediction": prediction,
            "model": model,
            "statistics": self.database.prediction_statistics(model["model_id"]),
        }

    def list_experiment_definitions(self) -> list[dict]:
        definitions = []
        for definition in self.experiment_registry.list():
            stored = self.database.register_experiment_definition(definition.to_dict())
            definitions.append(stored)
        return definitions

    def show_experiment_definition(
        self, experiment_id: str, version: int | None = None
    ) -> dict:
        definition = self.experiment_registry.get(experiment_id, version)
        stored = self.database.register_experiment_definition(definition.to_dict())
        return {
            **stored,
            "registered": True,
            "executable": (
                stored["status"] == "ENABLED"
                and definition.procedure_id == "cpu-baseline-v1"
                and self.experiment_registry.has_procedure(definition.procedure_id)
            ),
        }

    def run_experiment(
        self, experiment_id: str, version: int | None = None
    ) -> dict:
        definition = self.experiment_registry.get(experiment_id, version)
        stored_definition = self.database.register_experiment_definition(
            definition.to_dict()
        )
        if stored_definition["status"] != "ENABLED":
            raise KaiError(
                f"Experiment {experiment_id!r} is {stored_definition['status'].lower()}."
            )
        if (
            definition.experiment_id != CPU_BASELINE_EXPERIMENT_ID
            or definition.procedure_id != "cpu-baseline-v1"
        ):
            raise KaiError(
                f"Experiment {experiment_id!r} has no implemented trusted runner."
            )
        if not self.experiment_registry.has_procedure(definition.procedure_id):
            raise KaiError(
                f"Experiment {experiment_id!r} has no registered trusted procedure."
            )
        report = self._capture_model(persist_self_model=False)
        context = cpu_baseline_context(
            report["environment_fingerprint"],
            report["snapshot_id"],
        )
        experiment = self.database.begin_cpu_baseline_experiment(context)
        try:
            parameters = {"iterations": 100_000}
            experiment_result = self.experiment_registry.execute(
                experiment_id,
                parameters,
                context,
                report["environment"],
                self.config.database_path,
                version=definition.version,
            )
            if experiment_result.measurement_status != "MEASURED":
                self.database.fail_cpu_baseline_experiment(
                    experiment["execution_id"],
                    "The registered procedure could not establish a measured outcome.",
                    execution_status=(
                        "UNKNOWN"
                        if experiment_result.measurement_status == "UNKNOWN"
                        else "FAILED"
                    ),
                    measurement_status=experiment_result.measurement_status,
                    evaluation_status=(
                        "INVALID"
                        if experiment_result.measurement_status == "INVALID"
                        else None
                    ),
                    raw_result=experiment_result.raw_result,
                )
                execution_record = self.database.get_experiment_execution(
                    experiment["execution_id"]
                )
                self._persist_self_model(report["self_model"])
                return {
                    "definition": stored_definition,
                    "experiment": experiment,
                    "execution": execution_record,
                    "result": experiment_result.to_dict(),
                    "evaluation_status": execution_record["evaluation"]["status"],
                    "model_update": None,
                    "next_prediction": None,
                }
            if experiment_result.outcome_value is None:
                raise KaiError(
                    "The registered CPU baseline result omitted its numeric outcome."
                )
            result = self.database.complete_cpu_baseline_experiment(
                experiment["execution_id"],
                experiment_result.outcome_value,
                experiment_result.raw_result.get("captured_at"),
                experiment_result=experiment_result,
            )
        except InvalidExperimentResultError as exc:
            self.database.fail_cpu_baseline_experiment(
                experiment["execution_id"],
                str(exc),
                measurement_status="INVALID",
                evaluation_status="INVALID",
            )
            self._persist_self_model(report["self_model"])
            raise
        except (KaiError, OSError, ValueError) as exc:
            self.database.fail_cpu_baseline_experiment(
                experiment["execution_id"], str(exc)
            )
            self._persist_self_model(report["self_model"])
            raise
        model = result["model"]
        report["self_model"]["predictive_capabilities"] = {
            "models": [
                {
                    "model_id": item["model_id"],
                    "domain": item["domain"],
                    "model_type": item["model_type"],
                    "version": item["version"],
                    "status": item["status"],
                }
                for item in self.database.list_predictive_models()
                if item["status"] == "ACTIVE"
            ],
            "cpu_baseline": {
                **predict_from_model(model),
                "performance": model["performance"],
                "confidence_information": model["confidence"],
            },
        }
        self._persist_self_model(report["self_model"])
        result["definition"] = stored_definition
        result["experiment"] = experiment
        result["measurement"] = experiment_result.to_dict()
        result["execution"] = self.database.get_experiment_execution(
            experiment["execution_id"]
        )
        self.logger.info(
            "Allowlisted CPU prediction experiment completed; experiment_id=%s",
            experiment["experiment_id"],
        )
        return result

    def run_cpu_baseline_experiment(self) -> dict:
        return self.run_experiment(CPU_BASELINE_EXPERIMENT_ID)

    def predictor_status(self) -> dict:
        return self.database.prediction_report(limit=20)

    def learning_history(self, limit: int = 20) -> dict:
        return {
            "records": self.database.list_learning_records(limit),
            "models": self.database.list_predictive_models(),
        }

    def self_status(self) -> dict:
        self.status()
        latest = self.database.get_self_model_version()
        if latest is None:
            raise KaiError("Self-model observation completed without a persisted version.")
        return latest

    def self_capabilities(self) -> dict:
        model = self.self_status()["model"]
        return {
            "available": model.get("available_capabilities", {}),
            "unavailable": model.get("unavailable_capabilities", []),
            "unknown": model.get("unknown_capabilities", []),
            "experiments": model.get("available_experiments", []),
            "epistemic_states": model.get("self_epistemic_states", {}),
        }

    def self_weaknesses(self) -> dict:
        model = self.self_status()["model"]
        return {
            "weaknesses": model.get("known_weaknesses", []),
            "limitations": model.get("known_limitations", []),
            "epistemic_status": "OBSERVED_AND_UNKNOWN_AS_LABELED",
        }

    def self_failures(self) -> dict:
        executions = self.database.list_experiment_executions(limit=100)
        failed = [
            item
            for item in executions
            if item["execution_status"] in {"FAILED", "UNKNOWN", "ABORTED"}
        ]
        return {
            "failures": failed,
            "failure_count": len(failed),
            "source": "Persisted experiment execution history",
        }

    def self_metrics(self) -> dict:
        prediction = self.database.prediction_report(limit=100)
        return {
            "prediction_performance": prediction.get("performance", {}),
            "improvement_process": self.database.improvement_metrics(),
            "self_model_history_count": self.database.self_model_version_count(),
            "current_predictive_models": self.database.list_predictive_models(),
        }

    def inspect_improvement(self, opportunity_id: int) -> dict:
        opportunity = self.database.get_improvement_opportunity(opportunity_id)
        hypotheses = [
            self.database.get_improvement_hypothesis(item)
            for item in opportunity["hypothesis_ids"]
        ]
        proposals = []
        candidates = []
        for hypothesis in hypotheses:
            with self.database.connection() as connection:
                rows = connection.execute(
                    "SELECT proposal_id FROM improvement_proposals "
                    "WHERE hypothesis_id = ? ORDER BY proposal_version",
                    (hypothesis["hypothesis_id"],),
                ).fetchall()
            for row in rows:
                proposal = self.database.get_improvement_proposal(row["proposal_id"])
                proposals.append(proposal)
                candidates.extend(
                    candidate
                    for candidate in self.database.list_kai_candidates(limit=100)
                    if candidate["proposal_id"] == proposal["proposal_id"]
                )
        return {
            "opportunity": opportunity,
            "hypotheses": hypotheses,
            "proposals": proposals,
            "candidates": candidates,
            "metrics": self.database.improvement_metrics(),
        }

    def inspect_candidate(self, candidate_id: str) -> dict:
        candidate = self.database.get_kai_candidate(candidate_id)
        proposal = self.database.get_improvement_proposal(candidate["proposal_id"])
        hypothesis = self.database.get_improvement_hypothesis(
            proposal["hypothesis_id"]
        )
        opportunity = self.database.get_improvement_opportunity(
            hypothesis["opportunity_id"]
        )
        evaluations = [
            self.database.get_improvement_evaluation(item)
            for item in candidate["evaluation_ids"]
        ]
        decisions = [
            self.database.get_improvement_decision(item)
            for item in candidate["decision_ids"]
        ]
        return {
            "candidate": candidate,
            "proposal": proposal,
            "hypothesis": hypothesis,
            "opportunity": opportunity,
            "evaluations": evaluations,
            "decisions": decisions,
        }

    def evaluate_candidate(self, candidate_id: str) -> dict:
        candidate = self.database.get_kai_candidate(candidate_id)
        if not candidate["evaluation_ids"]:
            raise KaiError(
                f"Candidate {candidate_id!r} has no persisted evaluation yet."
            )
        return self.inspect_candidate(candidate_id)

    @staticmethod
    def _cpu_candidate_contract_tests() -> dict:
        checks = {
            "running_mean_rejects_empty_history": False,
            "candidate_rejects_empty_history": False,
            "running_mean_exact_value": running_mean_forecast([1.0, 3.0]) == 2.0,
            "candidate_uses_latest_value": last_observation_forecast([1.0, 3.0]) == 3.0,
        }
        for name, function in (
            ("running_mean_rejects_empty_history", running_mean_forecast),
            ("candidate_rejects_empty_history", last_observation_forecast),
        ):
            try:
                function([])
            except ValueError:
                checks[name] = True
        return {
            "checks": checks,
            "status": "PASSED" if all(checks.values()) else "FAILED",
            "scope": "Deterministic algorithm contract checks; no source execution or mutation.",
        }

    def demonstrate_cpu_predictor_improvement(self) -> dict:
        report = self._capture_model()
        timestamp = datetime.now(timezone.utc).isoformat()
        source_tree = inspect_source_tree(Path(__file__).resolve().parents[1])
        environment_context = {
            "environment_fingerprint": report["environment_fingerprint"],
            "environment_snapshot_id": report["snapshot_id"],
        }
        observations = []
        durations = []
        measurement_failures = []
        for trial in range(CPU_DEMO_MAXIMUM_OBSERVATIONS):
            try:
                measurement = measure_baseline(
                    report["environment"], self.config.database_path
                )
            except (KaiError, OSError, ValueError) as exc:
                measurement_failures.append(
                    {"trial": trial + 1, "error": str(exc), "status": "FAILED"}
                )
                observations.append(
                    self.database.record_self_observation(
                        {
                            "subject": "CPU baseline predictor",
                            "observation_type": "measurement.cpu_baseline_runtime",
                            "observed_at": timestamp,
                            "value": {
                                "status": "FAILED",
                                "trial": trial + 1,
                                "error": str(exc),
                            },
                            "source": "Explicit bounded kai improvements demo-cpu measurement",
                            "epistemic_status": "UNKNOWN",
                            "evidence_status": "unknown",
                            "confidence": None,
                            "kai_version": self.version,
                            "environment_context": environment_context,
                        }
                    )
                )
                break
            cpu = measurement.get("cpu") if isinstance(measurement, dict) else None
            duration = cpu.get("duration_seconds") if isinstance(cpu, dict) else None
            observed_at = measurement.get("captured_at", timestamp)
            if (
                not isinstance(cpu, dict)
                or cpu.get("status") != "measured"
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration <= 0
            ):
                observation = {
                    "subject": "CPU baseline predictor",
                    "observation_type": "measurement.cpu_baseline_runtime",
                    "observed_at": observed_at,
                    "value": {
                        "status": "UNKNOWN",
                        "trial": trial + 1,
                        "measurement": measurement,
                    },
                    "source": "Explicit bounded kai improvements demo-cpu measurement",
                    "epistemic_status": "UNKNOWN",
                    "evidence_status": "unknown",
                    "confidence": None,
                    "kai_version": self.version,
                    "environment_context": environment_context,
                }
                observations.append(self.database.record_self_observation(observation))
                measurement_failures.append(
                    {
                        "trial": trial + 1,
                        "reason": (
                            cpu.get("reason", "CPU runtime measurement unavailable")
                            if isinstance(cpu, dict)
                            else "Malformed CPU measurement report"
                        ),
                        "status": "UNKNOWN",
                    }
                )
                break
            durations.append(float(duration))
            observations.append(
                self.database.record_self_observation(
                    {
                        "subject": "CPU baseline predictor",
                        "observation_type": "measurement.cpu_baseline_runtime",
                        "observed_at": observed_at,
                        "value": {
                            "duration_seconds": float(duration),
                            "trial": trial + 1,
                            "measurement": cpu,
                        },
                        "source": "Explicit bounded kai improvements demo-cpu measurement",
                        "epistemic_status": "OBSERVED",
                        "evidence_status": "observed",
                        "confidence": 1.0,
                        "kai_version": self.version,
                        "environment_context": environment_context,
                    }
                )
            )

        evidence_observation_ids = [
            item["observation_id"] for item in observations
        ]
        opportunity = self.database.create_improvement_opportunity(
            source_observation_ids=evidence_observation_ids,
            affected_subsystem="cpu_baseline_predictor",
            current_behavior="Predicts the next runtime using the Welford running mean.",
            evidence={
                "observation_ids": evidence_observation_ids,
                "measured_runtime_seconds": durations,
                "measurement_failures": measurement_failures,
                "environment_context": environment_context,
                "source_digest": source_tree["source_snapshot_sha256"],
            },
            severity="MEDIUM",
            expected_impact="Reduce prequential CPU-baseline runtime mean absolute error.",
            confidence=None,
            timestamp=timestamp,
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "INVESTIGATING",
            "Compare the existing running-mean rule with a fixed isolated candidate.",
            {"observation_ids": evidence_observation_ids},
            timestamp,
        )
        hypothesis = self.database.create_improvement_hypothesis(
            opportunity_id=opportunity["opportunity_id"],
            observed_evidence={
                "observation_ids": evidence_observation_ids,
                "durations_seconds": durations,
            },
            interpretation=(
                "The workload timing stream may be sufficiently stable for a "
                "last-observation rule to reduce short-horizon absolute error."
            ),
            proposed_intervention=(
                "Evaluate the last-observation predictor only as an isolated candidate; "
                "do not change the production running-mean model."
            ),
            expected_outcome={
                "metric": "prequential mean absolute error in seconds",
                "direction": "lower is better",
                "minimum_paired_forecasts": IMPROVEMENT_EVIDENCE_MINIMUM,
            },
            uncertainty={
                "classification": "UNKNOWN",
                "reasons": [
                    "Wall-clock timings vary with scheduling and system load.",
                    "A short observation sequence may not generalize.",
                ],
            },
            negative_consequences={
                "possible_regression": (
                    "The last sample may overreact to transient timing noise."
                ),
                "resource_cost": (
                    "The explicit demonstration runs a fixed 100000-iteration CPU "
                    "measurement 22 times."
                ),
            },
            timestamp=timestamp,
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "PROPOSED",
            "Recorded an explicitly uncertain testable hypothesis.",
            {"hypothesis_id": hypothesis["hypothesis_id"]},
            timestamp,
        )
        proposal = self.database.create_improvement_proposal(
            hypothesis_id=hypothesis["hypothesis_id"],
            affected_component="kai.predictive.cpu_baseline",
            current_implementation="Welford running-mean forecast (production unchanged).",
            proposed_implementation=(
                "Last-observation forecast, implemented as a pure isolated candidate rule."
            ),
            reason="Test the hypothesis against paired measurements from the same workload.",
            evidence={
                "observation_ids": evidence_observation_ids,
                "measurement_failures": measurement_failures,
            },
            expected_benefit="Lower out-of-sample mean absolute runtime error if supported by measurements.",
            expected_cost="One remembered value; actual prediction quality may regress.",
            risk="Transient timing noise can make the latest observation unrepresentative.",
            dependencies=["Python standard library", "existing fixed CPU baseline"],
            test_requirements=[
                "Candidate deterministic contract checks",
                "Full project regression suite remains an external release gate",
            ],
            benchmark_requirements=[
                "Paired prequential absolute error on real bounded CPU measurements",
                "Record timing variability and measured sample count",
            ],
            rollback_strategy=(
                "No production change is applied; discard the candidate decision "
                "and retain the running-mean model."
            ),
            timestamp=timestamp,
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            "EXPERIMENTING",
            "Created an isolated candidate; production state remains unchanged.",
            {"proposal_id": proposal["proposal_id"]},
            timestamp,
        )
        candidate_id = str(uuid.uuid4())
        candidate = self.database.create_kai_candidate(
            candidate_id=candidate_id,
            proposal_id=proposal["proposal_id"],
            candidate_version=1,
            parent_kai_version=self.version,
            source_snapshot_ref="source-tree-sha256:" + source_tree["source_snapshot_sha256"],
            source_snapshot_sha256=source_tree["source_snapshot_sha256"],
            candidate_specification={
                "candidate_id": CPU_DEMO_CANDIDATE_ID,
                "algorithm": "last observation",
                "production_model_mutation": False,
            },
            isolation_metadata={
                "isolated": True,
                "mode": "persisted algorithm candidate; no code checkout or file writes",
                "production_files_modified": False,
                "production_model_mutation": False,
                "deployment_implemented": False,
            },
            timestamp=timestamp,
        )
        contract_tests = self._cpu_candidate_contract_tests()
        score = score_forecasters(
            durations,
            minimum_observations=CPU_DEMO_MINIMUM_OBSERVATIONS,
        ) if not measurement_failures else {
            "status": (
                "FAILED"
                if measurement_failures[0]["status"] == "FAILED"
                else "UNKNOWN"
            ),
            "reason": "The bounded measurement sequence did not complete.",
            "observation_count": len(durations),
            "paired_forecast_count": max(0, len(durations) - 2),
            "baseline": None,
            "candidate": None,
            "evidence_quality": "INSUFFICIENT",
        }
        evaluation_status = score["status"]
        if contract_tests["status"] == "FAILED":
            evaluation_status = "FAILED"
        regression_count = int(evaluation_status == "REGRESSED")
        failure_count = sum(
            item["status"] == "FAILED" for item in measurement_failures
        ) + int(contract_tests["status"] == "FAILED")
        test_status = contract_tests["status"]
        benchmark_status = (
            "FAILED"
            if measurement_failures and measurement_failures[0]["status"] == "FAILED"
            else "COMPLETED"
            if len(durations) == CPU_DEMO_MAXIMUM_OBSERVATIONS
            else "NOT_RUN"
        )
        baseline_metrics = score["baseline"] or {
            "algorithm": "running-mean-v1",
            "mean_absolute_error_seconds": None,
        }
        candidate_metrics = score["candidate"] or {
            "algorithm": CPU_DEMO_CANDIDATE_ID,
            "mean_absolute_error_seconds": None,
        }
        comparison = {
            **score,
            "resource_usage": {
                "cpu_measurement_iterations_per_sample": 100000,
                "sample_count": len(durations),
                "candidate_algorithm_overhead": "UNKNOWN; not independently benchmarked",
            },
            "reliability": {
                "measurement_failure_count": len(measurement_failures),
                "candidate_contract_test_status": contract_tests["status"],
            },
            "regression_count": regression_count,
            "failure_count": failure_count,
            "regression_policy": "Any recorded regression prevents candidate acceptance.",
        }
        comparison["status"] = evaluation_status
        comparison["primary_metric"] = "mean_absolute_error_seconds"
        comparison["metric_direction"] = "lower"
        paired_count = comparison.get("paired_forecast_count", 0)
        observed_count = sum(
            item["epistemic_status"] == "OBSERVED" for item in observations
        )
        evidence_quality = (
            "NONE"
            if observed_count == 0
            else "INSUFFICIENT"
            if observed_count < 2 or paired_count < 2
            else "PRELIMINARY"
            if paired_count < IMPROVEMENT_EVIDENCE_MINIMUM
            else "ADEQUATE"
            if observed_count >= paired_count
            else "INSUFFICIENT"
        )
        comparison["evidence_quality"] = evidence_quality
        baseline_snapshot = {
            "kai_version": self.version,
            "architecture_version": "0.007",
            "predictive_model_versions": [
                {
                    "model_id": model["model_id"],
                    "version": model["version"],
                    "domain": model["domain"],
                }
                for model in self.database.list_predictive_models()
            ],
            "experiment_definition_versions": [
                {
                    "experiment_id": definition.experiment_id,
                    "version": definition.version,
                    "procedure_id": definition.procedure_id,
                }
                for definition in self.experiment_registry.list()
            ],
            "configuration": {
                "cpu_iterations_per_sample": 100000,
                "measurement_sample_count": len(durations),
            },
            "environment_context": environment_context,
            "evaluation_dataset": {
                "observation_ids": evidence_observation_ids,
                "observation_count": len(durations),
                "paired_forecast_count": paired_count,
                "scoring": "Prequential paired out-of-sample absolute error.",
            },
            "metrics": baseline_metrics,
        }
        hypothesis_satisfied = (
            True if evaluation_status in {"IMPROVED", "STRONGLY_IMPROVED"}
            else False if evaluation_status == "REGRESSED"
            else None
        )
        evaluation = self.database.record_candidate_evaluation(
            candidate_id=candidate_id,
            baseline=baseline_snapshot,
            evaluation_status=evaluation_status,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            test_status=test_status,
            benchmark_status=benchmark_status,
            regression_count=regression_count,
            failure_count=failure_count,
            evidence_quality=evidence_quality,
            evidence={
                "source_observation_ids": evidence_observation_ids,
                "source_execution_ids": [],
                "measurement_failures": measurement_failures,
                "candidate_contract_tests": contract_tests,
                "source_tree_sha256": source_tree["source_snapshot_sha256"],
                "full_project_test_suite": (
                    "Not executed by kai CLI; it is a separate release gate."
                ),
            },
            hypothesis_satisfied=hypothesis_satisfied,
            timestamp=timestamp,
        )
        if evaluation_status in {"IMPROVED", "STRONGLY_IMPROVED"}:
            decision_code = "ACCEPT_CANDIDATE"
            decision_reason = (
                "Candidate improved the declared paired metric with adequate evidence; "
                "production deployment remains unavailable and unchanged."
            )
            opportunity_status = "ACCEPTED"
        elif evaluation_status in {"REGRESSED", "FAILED"}:
            decision_code = "REJECT"
            decision_reason = score["reason"]
            opportunity_status = "REJECTED"
        else:
            decision_code = "REQUEST_MORE_EVIDENCE"
            decision_reason = score["reason"]
            opportunity_status = "DEFERRED"
        decision = self.database.record_improvement_decision(
            candidate_id=candidate_id,
            evaluation_id=evaluation["evaluation_id"],
            decision=decision_code,
            reason=decision_reason,
            evidence={
                "evaluation_id": evaluation["evaluation_id"],
                "evidence_quality": evaluation["evidence_quality"],
                "production_deployment": "NOT_AVAILABLE",
            },
            timestamp=timestamp,
        )
        self.database.update_improvement_opportunity_status(
            opportunity["opportunity_id"],
            opportunity_status,
            decision_reason,
            {"decision_id": decision["decision_id"]},
            timestamp,
        )
        return {
            "opportunity": self.database.get_improvement_opportunity(
                opportunity["opportunity_id"]
            ),
            "hypothesis": hypothesis,
            "proposal": proposal,
            "candidate": self.database.get_kai_candidate(candidate_id),
            "evaluation": evaluation,
            "decision": decision,
            "measurement_observations": observations,
            "production_changed": False,
        }

    def benchmark(self, rounds: int = 10) -> dict:
        started = time.perf_counter()
        self.database.initialize()
        initialization_ms = (time.perf_counter() - started) * 1000
        write_started = time.perf_counter()
        for index in range(rounds):
            self.database.record_event(
                "benchmark",
                "SQLite event persistence benchmark sample.",
                {"sample": index + 1},
            )
        write_ms = (time.perf_counter() - write_started) * 1000 / rounds
        read_started = time.perf_counter()
        self.database.recent_events(limit=rounds)
        read_ms = (time.perf_counter() - read_started) * 1000
        result = {
            "rounds": rounds,
            "database_initialize_ms": round(initialization_ms, 3),
            "event_write_mean_ms": round(write_ms, 3),
            "recent_events_read_ms": round(read_ms, 3),
            "measurement_note": (
                "Local smoke benchmark only; values depend on current machine load and "
                "are not a comparative performance claim."
            ),
        }
        self.logger.info("Benchmark completed: %s", json.dumps(result, sort_keys=True))
        return result
