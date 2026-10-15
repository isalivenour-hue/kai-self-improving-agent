"""Bounded engineering tasks using KAI's audited disposable candidate pipeline."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from kai.errors import KaiError
from kai.local_llm import LLMProvider
from kai.self_improvement import inspect_source_tree

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "target_file": {"type": "string"},
        "test_module": {"type": "string"},
        "reason": {"type": "string"},
        "acceptance_criteria": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "risk": {"type": "string"},
    },
    "required": [
        "summary",
        "target_file",
        "test_module",
        "reason",
        "acceptance_criteria",
        "risk",
    ],
    "additionalProperties": False,
}
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5,
        },
        "risk_assessment": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "UNKNOWN"],
        },
        "recommendation": {
            "type": "string",
            "enum": ["PROCEED_TO_TESTS", "REVISE", "UNKNOWN"],
        },
    },
    "required": ["findings", "risk_assessment", "recommendation"],
    "additionalProperties": False,
}


class EngineeringCoordinator:
    def __init__(
        self,
        *,
        database,
        provider: LLMProvider,
        workspace_manager,
        runner,
        source_root: Path,
        kai_version: str,
        schema_version: int,
        repair_attempts: int = 1,
    ):
        self.database = database
        self.provider = provider
        self.workspace_manager = workspace_manager
        self.runner = runner
        self.source_root = source_root.resolve()
        self.kai_version = kai_version
        self.schema_version = schema_version
        self.repair_attempts = repair_attempts

    def status(self) -> dict:
        provider = self.provider.status()
        return {
            "provider": provider,
            "sandbox": self.runner.probe_sandbox(),
            "task_count": self.database.engineering_task_count(),
            "safety": {
                "candidate_changes_only": True,
                "network_during_candidate_execution": "disabled",
                "automatic_deployment": False,
                "repair_attempt_limit": self.repair_attempts,
            },
        }

    def inspect(self) -> dict:
        inventory = inspect_source_tree(self.source_root)
        provider = self.provider.status()
        environment = self._environment_summary(provider)
        return {
            "source_inventory": inventory,
            "candidate_source_paths": [
                item["path"]
                for item in inventory["module_inventory"]
            ],
            "environment": environment,
            "provider": provider,
            "sandbox": self.runner.probe_sandbox(),
            "inspection_is_read_only": True,
        }

    def plan(self, objective: str) -> dict:
        task = self._new_task(objective)
        try:
            inspection = self.inspect()
        except KaiError as exc:
            return self._finish(
                task["task_id"],
                status="BLOCKED",
                stage="REPOSITORY_INSPECTION",
                result={"decision": "UNKNOWN", "reason": str(exc)},
            )
        self.database.record_engineering_task_event(
            task["task_id"],
            stage="REPOSITORY_INSPECTION",
            status="REPOSITORY_INSPECTION",
            details={
                "source_snapshot_sha256": inspection["source_inventory"]["source_snapshot_sha256"],
                "provider": inspection["provider"],
            },
            updates={"inspection": inspection, "provider": inspection["provider"]},
        )
        provider_status = inspection["provider"]
        if provider_status.get("status") != "READY":
            return self._block_for_provider(task["task_id"], provider_status)
        try:
            plan = self._request_plan(objective, inspection)
            self.database.record_engineering_task_event(
                task["task_id"],
                stage="PLAN",
                status="PLAN",
                details={"plan": plan},
                updates={"plan": plan},
            )
            return self.database.get_engineering_task(task["task_id"])
        except KaiError as exc:
            return self._finish(
                task["task_id"],
                status="FAILED",
                stage="PLAN",
                result={"decision": "INCONCLUSIVE", "error": str(exc)},
            )

    def run(self, objective: str) -> dict:
        task = self._new_task(objective)
        task_id = task["task_id"]
        try:
            inspection = self.inspect()
            self.database.record_engineering_task_event(
                task_id,
                stage="REPOSITORY_INSPECTION",
                status="REPOSITORY_INSPECTION",
                details={"source_snapshot_sha256": inspection["source_inventory"]["source_snapshot_sha256"]},
                updates={"inspection": inspection, "provider": inspection["provider"]},
            )
            provider_status = inspection["provider"]
            if provider_status.get("status") != "READY":
                return self._block_for_provider(task_id, provider_status)

            plan = self._request_plan(objective, inspection)
            self.database.record_engineering_task_event(
                task_id,
                stage="PLAN",
                status="PLAN",
                details={"plan": plan},
                updates={"plan": plan},
            )
            self.database.record_engineering_task_event(
                task_id,
                stage="PROPOSED_CHANGES",
                status="PROPOSED_CHANGES",
                details={"target_file": plan["target_file"], "reason": plan["reason"]},
            )

            lineage = self._create_candidate_lineage(
                task_id, objective, plan, inspection
            )
            baseline_id = lineage["baseline_candidate_id"]
            candidate_id = lineage["candidate_id"]
            baseline_workspace = self.workspace_manager.create(
                candidate_id=baseline_id,
                parent_kai_version=self.kai_version,
                source_revision=None,
                schema_version=self.schema_version,
                environment_fingerprint=None,
                configuration={"provider": provider_status["provider"], "model": provider_status["model"]},
            )
            candidate_workspace = self.workspace_manager.create(
                candidate_id=candidate_id,
                parent_kai_version=self.kai_version,
                source_revision=None,
                schema_version=self.schema_version,
                environment_fingerprint=None,
                configuration={"provider": provider_status["provider"], "model": provider_status["model"]},
            )
            self.database.record_engineering_task_event(
                task_id,
                stage="CANDIDATE_WORKSPACE",
                status="CANDIDATE_WORKSPACE",
                details={
                    "baseline_candidate_id": baseline_id,
                    "baseline_workspace_id": baseline_workspace["workspace_id"],
                    "candidate_id": candidate_id,
                    "workspace_id": candidate_workspace["workspace_id"],
                },
                updates={
                    "candidate_id": candidate_id,
                    "workspace_id": candidate_workspace["workspace_id"],
                },
            )

            sandbox = self.runner.probe_sandbox()
            sandbox_ready = sandbox.get("probe_status") == "READY"
            baseline_result = None
            if sandbox_ready:
                baseline_result = self.runner.run_tests(
                    baseline_id,
                    baseline_workspace["workspace_id"],
                    module=plan["test_module"],
                )
                self.database.record_engineering_task_event(
                    task_id,
                    stage="TESTING",
                    status="TESTING",
                    details={"role": "baseline", "execution": self._execution_report(baseline_result)},
                )

            patch = self._request_patch(
                task_id=task_id,
                objective=objective,
                plan=plan,
                workspace_id=candidate_workspace["workspace_id"],
            )
            modification = self.workspace_manager.apply_change(
                workspace_id=candidate_workspace["workspace_id"],
                target_file=patch["target_file"],
                operation=patch["operation"],
                content=patch["content"],
                reason=patch["reason"],
                hypothesis_id=lineage["hypothesis_id"],
            )
            self.database.record_engineering_task_event(
                task_id,
                stage="PATCH_APPLICATION",
                status="PATCH_APPLICATION",
                details={
                    "modification_id": modification["modification_id"],
                    "target_file": modification["target_file"],
                    "original_sha256": modification["original_sha256"],
                    "resulting_sha256": modification["resulting_sha256"],
                    "operation": modification["operation"],
                    "model_provider": patch["provider"],
                    "model_id": patch["model"],
                    "model_request_id": patch["request_id"],
                },
            )
            validation = self.workspace_manager.validate(
                candidate_workspace["workspace_id"]
            )
            self.database.record_engineering_task_event(
                task_id,
                stage="STATIC_VALIDATION",
                status="STATIC_VALIDATION" if validation["status"] == "VALID" else "FAILED",
                details=validation,
            )
            if validation["status"] != "VALID":
                return self._evaluate_and_finish(
                    task_id=task_id,
                    objective=objective,
                    plan=plan,
                    lineage=lineage,
                    baseline_result=baseline_result,
                    candidate_result=None,
                    sandbox=sandbox,
                    patch=patch,
                    decision_status="FAILED",
                    error="Candidate source failed static validation.",
                )

            review = self._review_candidate(
                objective=objective,
                plan=plan,
                workspace_id=candidate_workspace["workspace_id"],
                target_file=patch["target_file"],
            )
            self.database.record_engineering_task_event(
                task_id,
                stage="MODEL_REVIEW",
                status="PATCH_APPLICATION",
                details={"review": review},
            )

            if not sandbox_ready:
                return self._evaluate_and_finish(
                    task_id=task_id,
                    objective=objective,
                    plan=plan,
                    lineage=lineage,
                    baseline_result=None,
                    candidate_result=None,
                    sandbox=sandbox,
                    patch=patch,
                    decision_status="UNKNOWN",
                    error=sandbox.get("reason") or "Sandbox is unavailable.",
                )

            candidate_result = self.runner.run_tests(
                candidate_id,
                candidate_workspace["workspace_id"],
                module=plan["test_module"],
            )
            self.database.record_engineering_task_event(
                task_id,
                stage="TESTING",
                status="TESTING",
                details={"role": "candidate", "execution": self._execution_report(candidate_result)},
            )
            attempts = 0
            while (
                candidate_result.status != "SUCCESS"
                and baseline_result is not None
                and baseline_result.status == "SUCCESS"
                and attempts < self.repair_attempts
            ):
                attempts += 1
                self.database.record_engineering_task_event(
                    task_id,
                    stage="PATCH_APPLICATION",
                    status="PATCH_APPLICATION",
                    details={"repair_attempt": attempts, "status": "REQUESTED"},
                    updates={"repair_attempts": attempts},
                )
                patch = self._request_patch(
                    task_id=task_id,
                    objective=objective,
                    plan=plan,
                    workspace_id=candidate_workspace["workspace_id"],
                    failure=self._execution_report(candidate_result),
                    repair_attempt=attempts,
                )
                repair_modification = self.workspace_manager.apply_change(
                    workspace_id=candidate_workspace["workspace_id"],
                    target_file=patch["target_file"],
                    operation=patch["operation"],
                    content=patch["content"],
                    reason=patch["reason"],
                    hypothesis_id=lineage["hypothesis_id"],
                )
                self.database.record_engineering_task_event(
                    task_id,
                    stage="PATCH_APPLICATION",
                    status="PATCH_APPLICATION",
                    details={
                        "repair_attempt": attempts,
                        "modification_id": repair_modification["modification_id"],
                        "target_file": repair_modification["target_file"],
                        "model_provider": patch["provider"],
                        "model_id": patch["model"],
                        "model_request_id": patch["request_id"],
                        "routing": patch.get("routing"),
                    },
                )
                validation = self.workspace_manager.validate(
                    candidate_workspace["workspace_id"]
                )
                self.database.record_engineering_task_event(
                    task_id,
                    stage="STATIC_VALIDATION",
                    status="STATIC_VALIDATION" if validation["status"] == "VALID" else "FAILED",
                    details={"repair_attempt": attempts, **validation},
                )
                if validation["status"] != "VALID":
                    break
                repair_review = self._review_candidate(
                    objective=objective,
                    plan=plan,
                    workspace_id=candidate_workspace["workspace_id"],
                    target_file=patch["target_file"],
                )
                self.database.record_engineering_task_event(
                    task_id,
                    stage="MODEL_REVIEW",
                    status="PATCH_APPLICATION",
                    details={"repair_attempt": attempts, "review": repair_review},
                )
                candidate_result = self.runner.run_tests(
                    candidate_id,
                    candidate_workspace["workspace_id"],
                    module=plan["test_module"],
                )
                self.database.record_engineering_task_event(
                    task_id,
                    stage="TESTING",
                    status="TESTING",
                    details={
                        "role": "candidate",
                        "repair_attempt": attempts,
                        "execution": self._execution_report(candidate_result),
                    },
                )

            return self._evaluate_and_finish(
                task_id=task_id,
                objective=objective,
                plan=plan,
                lineage=lineage,
                baseline_result=baseline_result,
                candidate_result=candidate_result,
                sandbox=sandbox,
                patch=patch,
                decision_status=None,
            )
        except (KaiError, OSError, ValueError, TypeError) as exc:
            return self._finish(
                task_id,
                status="FAILED",
                stage="FAILED",
                result={
                    "decision": "INCONCLUSIVE",
                    "error": str(exc),
                    "candidate_id": self.database.get_engineering_task(task_id)["candidate_id"],
                    "workspace_id": self.database.get_engineering_task(task_id)["workspace_id"],
                    "production_source_modified": False,
                },
            )

    def _new_task(self, objective: str) -> dict:
        task_id = str(uuid.uuid4())
        return self.database.create_engineering_task(
            task_id=task_id,
            objective=objective,
            provider={
                "provider": "not checked",
                "model": None,
                "status": "NOT_CHECKED",
                "local_only": True,
            },
            max_repair_attempts=self.repair_attempts,
        )

    def _block_for_provider(self, task_id: str, provider_status: dict) -> dict:
        return self._finish(
            task_id,
            status="BLOCKED",
            stage="PLAN",
            result={
                "decision": "UNKNOWN",
                "reason": provider_status.get("reason")
                or f"Configured local model is not ready ({provider_status.get('status')}).",
                "provider_status": provider_status,
                "production_source_modified": False,
            },
        )

    def _finish(self, task_id: str, *, status: str, stage: str, result: dict) -> dict:
        return self.database.record_engineering_task_event(
            task_id,
            stage=stage,
            status=status,
            details=result,
            updates={"result": result},
        )

    def _request_plan(self, objective: str, inspection: dict) -> dict:
        inventory = inspection["source_inventory"]
        prompt = (
            "You are planning one small, reversible KAI engineering change. "
            "Return only the required JSON object. Choose exactly one Python source "
            "file under kai/ or tests/ and one existing tests.test_<name> module. "
            "Do not request shell commands, dependencies, credentials, network, or "
            "production changes. Do not claim that tests or benchmarks have run.\n"
            f"Objective:\n{objective}\n"
            "Repository inventory:\n"
            + json.dumps(
                {
                    "source_paths": [
                        item["path"] for item in inventory["module_inventory"]
                    ][:100],
                    "file_count": inventory["source_file_count"]
                    + inventory["test_file_count"],
                    "source_digest": inventory["source_snapshot_sha256"],
                    "python_version": inspection["environment"]["python_version"],
                    "memory_available_bytes": inspection["environment"][
                        "memory_available_bytes"
                    ],
                },
                separators=(",", ":"),
            )
        )
        result = self._generate_json(prompt, PLAN_SCHEMA, "planning")
        self._validate_provider_result(result)
        plan = result.get("structured")
        if not isinstance(plan, dict):
            raise KaiError("Local model returned no structured engineering plan.")
        required = set(PLAN_SCHEMA["required"])
        if set(plan) != required:
            raise KaiError("Local model plan fields do not match the engineering schema.")
        for key in ("summary", "target_file", "test_module", "reason", "risk"):
            if not isinstance(plan[key], str) or not plan[key].strip():
                raise KaiError(f"Engineering plan {key} must be non-empty text.")
        if not re.fullmatch(r"tests\.test_[A-Za-z0-9_]+", plan["test_module"]):
            raise KaiError("Engineering plan must select one existing unittest module.")
        test_path = self.source_root / (plan["test_module"].replace(".", "/") + ".py")
        if not test_path.is_file():
            raise KaiError("Engineering plan selected a test module that does not exist.")
        self.workspace_manager._target_path(
            self.source_root, plan["target_file"]
        )
        criteria = plan["acceptance_criteria"]
        if (
            not isinstance(criteria, list)
            or not 1 <= len(criteria) <= 4
            or any(not isinstance(item, str) or not item.strip() for item in criteria)
        ):
            raise KaiError("Engineering plan acceptance criteria are invalid.")
        return {
            **plan,
            "provider": result["provider"],
            "model": result["model"],
            "local_only": result["local_only"],
            "model_request_id": result["request_id"],
            "actual_model": result.get("actual_model"),
            "routing": result.get("routing"),
        }

    def _request_patch(
        self,
        *,
        task_id: str,
        objective: str,
        plan: dict,
        workspace_id: str,
        failure: dict | None = None,
        repair_attempt: int = 0,
    ) -> dict:
        target_file = plan["target_file"]
        root = self.workspace_manager.path_for(workspace_id)
        target = self.workspace_manager._target_path(root, target_file)
        current = target.read_text(encoding="utf-8") if target.exists() else ""
        if len(current) > 4_000:
            raise KaiError(
                f"Selected source {target_file} exceeds the bounded 4000-character model context."
            )
        prompt = (
            "Propose exactly one complete-file operation as JSON. Never emit shell "
            "commands, unified-diff text, credentials, network instructions, or "
            "changes to any path other than the single allowed target. "
            "Keep the change minimal, syntactically valid, and testable.\n"
            f"Objective: {objective}\n"
            f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Allowed target: {target_file}\n"
            f"Current source (empty means ADD):\n{current}\n"
        )
        if failure is not None:
            prompt += (
                f"Repair attempt {repair_attempt}; budget is finite. "
                "Diagnose only this candidate test failure and return a corrected "
                "whole-file operation for the same target.\n"
                f"Failure evidence:\n{json.dumps(failure, ensure_ascii=False)[:2_500]}\n"
            )
        schema = {
            "type": "object",
            "properties": {
                "target_file": {"const": target_file},
                "operation": {"type": "string", "enum": ["ADD", "MODIFY", "DELETE"]},
                "content": {"type": ["string", "null"]},
                "reason": {"type": "string"},
            },
            "required": ["target_file", "operation", "content", "reason"],
            "additionalProperties": False,
        }
        result = self._generate_json(prompt, schema, "coding")
        self._validate_provider_result(result)
        patch = result.get("structured")
        if not isinstance(patch, dict) or set(patch) != {
            "target_file",
            "operation",
            "content",
            "reason",
        }:
            raise KaiError("Local model returned a malformed candidate operation.")
        if patch["target_file"] != target_file:
            raise KaiError("Local model candidate operation escaped its planned target.")
        if patch["operation"] not in {"ADD", "MODIFY", "DELETE"}:
            raise KaiError("Local model candidate operation is unsupported.")
        if not isinstance(patch["reason"], str) or not patch["reason"].strip():
            raise KaiError("Candidate operation requires a non-empty reason.")
        if patch["operation"] == "DELETE":
            if patch["content"] is not None:
                raise KaiError("DELETE candidate operations must not include content.")
        elif not isinstance(patch["content"], str):
            raise KaiError("ADD and MODIFY candidate operations require source text.")
        return {
            **patch,
            "provider": result["provider"],
            "model": result["model"],
            "request_id": result["request_id"],
            "actual_model": result.get("actual_model"),
            "routing": result.get("routing"),
            "task_id": task_id,
            "repair_attempt": repair_attempt,
        }

    @staticmethod
    def _validate_provider_result(result: Any) -> None:
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("provider"), str)
            or not result["provider"].strip()
            or not isinstance(result.get("model"), str)
            or not result["model"].strip()
            or result.get("done") is not True
            or not isinstance(result.get("structured"), dict)
            or not isinstance(result.get("local_only"), bool)
        ):
            raise KaiError(
                "Configured model provider returned an invalid structured response."
            )

    def _generate_json(self, prompt: str, schema: dict, task_category: str) -> dict:
        routed = getattr(self.provider, "generate_json_for_task", None)
        if callable(routed):
            return routed(prompt, schema, task_category)
        return self.provider.generate_json(prompt, schema)

    def _review_candidate(
        self,
        *,
        objective: str,
        plan: dict,
        workspace_id: str,
        target_file: str,
    ) -> dict:
        workspace = self.workspace_manager.path_for(workspace_id)
        target = self.workspace_manager._target_path(workspace, target_file)
        content = target.read_text(encoding="utf-8") if target.exists() else ""
        if len(content) > 4_000:
            raise KaiError("Candidate source exceeds the bounded review context.")
        prompt = (
            "Review this disposable candidate change for likely correctness risks. "
            "Return only the requested JSON. This review is advisory: do not claim "
            "tests passed, do not approve deployment, and do not request changes "
            "outside the shown target. Automated tests and evaluation remain "
            "mandatory.\n"
            f"Objective: {objective}\n"
            f"Plan: {json.dumps(plan, ensure_ascii=False)}\n"
            f"Target: {target_file}\n"
            f"Candidate source:\n{content}\n"
        )
        response = self._generate_json(prompt, REVIEW_SCHEMA, "code_review")
        self._validate_provider_result(response)
        review = response["structured"]
        if set(review) != set(REVIEW_SCHEMA["required"]):
            raise KaiError("Model review fields do not match the review schema.")
        if (
            not isinstance(review["findings"], list)
            or len(review["findings"]) > 5
            or any(
                not isinstance(item, str) or not item.strip()
                for item in review["findings"]
            )
        ):
            raise KaiError("Model review findings are invalid.")
        if review["risk_assessment"] not in {"LOW", "MEDIUM", "HIGH", "UNKNOWN"}:
            raise KaiError("Model review risk assessment is invalid.")
        if review["recommendation"] not in {
            "PROCEED_TO_TESTS",
            "REVISE",
            "UNKNOWN",
        }:
            raise KaiError("Model review recommendation is invalid.")
        return {
            **review,
            "advisory_only": True,
            "provider": response["provider"],
            "model": response["model"],
            "model_request_id": response["request_id"],
            "actual_model": response.get("actual_model"),
            "routing": response.get("routing"),
        }

    def _create_candidate_lineage(
        self, task_id: str, objective: str, plan: dict, inspection: dict
    ) -> dict:
        task = self.database.get_engineering_task(task_id)
        observation = self.database.record_self_observation(
            {
                "subject": "Engineering objective",
                "observation_type": "engineering.user_objective",
                "observed_at": task["created_at"],
                "value": {
                    "task_id": task_id,
                    "objective": objective,
                    "source_snapshot_sha256": inspection["source_inventory"][
                        "source_snapshot_sha256"
                    ],
                },
                "source": "Explicit user request to run KAI engineering task",
                "epistemic_status": "OBSERVED",
                "evidence_status": "observed",
                "confidence": 1.0,
                "kai_version": self.kai_version,
                "environment_context": {},
            }
        )
        opportunity = self.database.create_improvement_opportunity(
            source_observation_ids=[observation["observation_id"]],
            affected_subsystem=plan["target_file"],
            current_behavior="Current source snapshot is unchanged before this task.",
            evidence={
                "task_id": task_id,
                "source_snapshot_sha256": inspection["source_inventory"][
                    "source_snapshot_sha256"
                ],
            },
            severity="MEDIUM",
            expected_impact=plan["summary"],
            confidence=None,
        )
        hypothesis = self.database.create_improvement_hypothesis(
            opportunity_id=opportunity["opportunity_id"],
            observed_evidence={
                "observation_id": observation["observation_id"],
                "acceptance_criteria": plan["acceptance_criteria"],
            },
            interpretation=plan["summary"],
            proposed_intervention=plan["reason"],
            expected_outcome={"criteria": plan["acceptance_criteria"]},
            uncertainty={"status": "UNKNOWN", "risk": plan["risk"]},
            negative_consequences={"status": "UNKNOWN"},
        )
        proposal = self.database.create_improvement_proposal(
            hypothesis_id=hypothesis["hypothesis_id"],
            affected_component=plan["target_file"],
            current_implementation="Unmodified production source snapshot",
            proposed_implementation=plan["summary"],
            reason=plan["reason"],
            evidence={"task_id": task_id, "observation_id": observation["observation_id"]},
            expected_benefit=plan["summary"],
            expected_cost="Local LLM inference and bounded sandbox test execution.",
            risk=plan["risk"],
            dependencies=["Python standard library", "local Ollama service"],
            test_requirements=[plan["test_module"]],
            benchmark_requirements=["Not defined; evaluation remains incomplete."],
            rollback_strategy="Discard or preserve the isolated candidate workspace; production is never modified.",
        )
        digest = inspection["source_inventory"]["source_snapshot_sha256"]
        baseline_candidate_id = f"engineer-baseline-{task_id}"
        candidate_id = f"engineer-candidate-{task_id}"
        self.database.create_kai_candidate(
            candidate_id=baseline_candidate_id,
            proposal_id=proposal["proposal_id"],
            candidate_version=1,
            parent_kai_version=self.kai_version,
            source_snapshot_ref=f"engineering-task:{task_id}:baseline",
            source_snapshot_sha256=digest,
            candidate_specification={"role": "unchanged baseline", "task_id": task_id},
            isolation_metadata={
                "isolated": True,
                "production_files_modified": False,
                "production_model_mutation": False,
            },
        )
        self.database.create_kai_candidate(
            candidate_id=candidate_id,
            proposal_id=proposal["proposal_id"],
            candidate_version=2,
            parent_candidate_id=baseline_candidate_id,
            parent_kai_version=self.kai_version,
            source_snapshot_ref=f"engineering-task:{task_id}:candidate",
            source_snapshot_sha256=digest,
            candidate_specification={
                "role": "proposed engineering candidate",
                "task_id": task_id,
                "plan": plan,
            },
            isolation_metadata={
                "isolated": True,
                "production_files_modified": False,
                "production_model_mutation": False,
            },
        )
        return {
            "observation_id": observation["observation_id"],
            "opportunity_id": opportunity["opportunity_id"],
            "hypothesis_id": hypothesis["hypothesis_id"],
            "proposal_id": proposal["proposal_id"],
            "baseline_candidate_id": baseline_candidate_id,
            "candidate_id": candidate_id,
        }

    def _evaluate_and_finish(
        self,
        *,
        task_id: str,
        objective: str,
        plan: dict,
        lineage: dict,
        baseline_result,
        candidate_result,
        sandbox: dict,
        patch: dict,
        decision_status: str | None,
        error: str | None = None,
    ) -> dict:
        baseline_report = (
            self._execution_report(baseline_result) if baseline_result else None
        )
        candidate_report = (
            self._execution_report(candidate_result) if candidate_result else None
        )
        if decision_status is None:
            if (
                baseline_result is None
                or baseline_result.status != "SUCCESS"
                or candidate_result is None
                or candidate_result.status == "UNKNOWN"
            ):
                decision_status = "UNKNOWN"
            elif candidate_result.status != "SUCCESS":
                decision_status = "FAILED"
            else:
                decision_status = "INCONCLUSIVE"

        if candidate_result is None:
            test_status = "NOT_RUN"
            failure_count = 0
        elif candidate_result.status == "SUCCESS":
            test_status = "PASSED"
            failure_count = 0
        elif candidate_result.status == "UNKNOWN":
            test_status = "NOT_RUN"
            failure_count = 0
        else:
            test_status = "FAILED"
            failure_count = 1
        benchmark_status = "NOT_RUN"
        baseline_metric = int(bool(baseline_result and baseline_result.status == "SUCCESS"))
        candidate_metric = int(test_status == "PASSED")
        evaluation_status = (
            "FAILED"
            if test_status == "FAILED"
            else "UNKNOWN"
        )
        baseline_metrics = {"selected_test_passed": baseline_metric}
        candidate_metrics = {"selected_test_passed": candidate_metric}
        comparison = {
            "primary_metric": "selected_test_passed",
            "metric_direction": "higher",
            "sample_count": 0,
            "status": evaluation_status,
            "note": "A single targeted test run is a compatibility check, not improvement evidence.",
        }
        task = self.database.get_engineering_task(task_id)
        baseline_snapshot = {
            "kai_version": self.kai_version,
            "architecture_version": str(self.schema_version),
            "predictive_model_versions": [],
            "experiment_definition_versions": [],
            "configuration": {
                "provider": task["provider"],
                "max_repair_attempts": task["max_repair_attempts"],
            },
            "environment_context": task["inspection"].get("environment", {}),
            "evaluation_dataset": {
                "observation_ids": [lineage["observation_id"]],
                "observation_count": 1,
                "paired_forecast_count": 0,
                "target_test_module": plan["test_module"],
            },
            "metrics": baseline_metrics,
        }
        evidence = {
            "source_observation_ids": [lineage["observation_id"]],
            "baseline_execution": baseline_report,
            "candidate_execution": candidate_report,
            "sandbox": sandbox,
            "patch": patch,
            "error": error,
            "production_source_modified": False,
            "production_database_used_for_candidate": False,
        }
        evaluation = self.database.record_candidate_evaluation(
            candidate_id=lineage["candidate_id"],
            baseline=baseline_snapshot,
            evaluation_status=evaluation_status,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            comparison=comparison,
            test_status=test_status,
            benchmark_status=benchmark_status,
            regression_count=(
                1
                if baseline_metric and test_status == "FAILED"
                else 0
            ),
            failure_count=failure_count,
            evidence_quality="INSUFFICIENT",
            evidence=evidence,
            hypothesis_satisfied=None,
        )
        decision = (
            "REJECT"
            if decision_status in {"FAILED", "REJECTED"}
            else "REQUEST_MORE_EVIDENCE"
        )
        persisted_decision = self.database.record_improvement_decision(
            candidate_id=lineage["candidate_id"],
            evaluation_id=evaluation["evaluation_id"],
            decision=decision,
            reason=(
                "Candidate test failure; never promote a failing isolated change."
                if decision == "REJECT"
                else "No benchmark or adequate comparative evidence; candidate is not accepted."
            ),
            evidence={"task_id": task_id, "decision_status": decision_status},
        )
        result = {
            "decision": decision_status,
            "evaluation_status": evaluation["evaluation_status"],
            "evaluation_id": evaluation["evaluation_id"],
            "persisted_decision": persisted_decision,
            "baseline_execution": baseline_report,
            "candidate_execution": candidate_report,
            "sandbox": sandbox,
            "repair_attempts": task["repair_attempts"],
            "maximum_repair_attempts": task["max_repair_attempts"],
            "benchmark": {"status": "UNKNOWN", "reason": "No objective-specific benchmark is registered."},
            "model_assessment": "NOT_USED_FOR_EVALUATION",
            "production_source_modified": False,
            "production_database_used_for_candidate": False,
            "error": error,
        }
        final_status = {
            "FAILED": "REJECTED",
            "REJECTED": "REJECTED",
            "UNKNOWN": "UNKNOWN",
            "INCONCLUSIVE": "INCONCLUSIVE",
        }.get(decision_status, "INCONCLUSIVE")
        self.database.record_engineering_task_event(
            task_id,
            stage="BENCHMARKING",
            status="UNKNOWN",
            details=result["benchmark"],
        )
        return self._finish(
            task_id,
            status=final_status,
            stage="EVALUATION",
            result={
                **result,
                "lineage": lineage,
                "objective": objective,
                "plan": plan,
            },
        )

    @staticmethod
    def _execution_report(execution) -> dict:
        result = execution.to_dict()
        result["stdout"] = result["stdout"][-4_000:]
        result["stderr"] = result["stderr"][-4_000:]
        return result

    def _environment_summary(self, llm_status: dict) -> dict:
        from kai.discovery import capability_summary
        from kai.environment_model import collect_environment_model

        environment = collect_environment_model(self.source_root)
        capabilities = capability_summary(environment)
        capabilities["language_model"] = (
            "not configured"
            if llm_status.get("status") == "DISABLED"
            else "configured"
        )
        capabilities["local_llm_runtime"] = llm_status.get("status", "unknown")
        return {
            "python_version": environment["python"],
            "cpu": environment["cpu"],
            "memory_total_bytes": environment["memory"]["total_bytes"],
            "memory_available_bytes": environment["memory"]["available_bytes"],
            "gpu": environment["gpu"],
            "capabilities": capabilities,
        }
