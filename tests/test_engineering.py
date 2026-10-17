import hashlib
import io
import json
import os
import sqlite3
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from kai.candidate_runner import CandidateExecutionResult
from kai.candidate_workspace import CandidateWorkspaceManager
from kai.database import Database
from kai.errors import KaiError
from kai.engineering import EngineeringCoordinator, _eligible_plan_inventory


class FakeProvider:
    def __init__(self, *, ready=True, plan=None, patches=None):
        self.ready = ready
        self.plan = plan or {
            "summary": "Add a small tested helper.",
            "target_file": "kai/simple.py",
            "test_module": "tests.test_simple",
            "reason": "Implement the requested bounded change.",
            "acceptance_criteria": ["The selected test module passes."],
            "risk": "A local helper may not satisfy broader requirements.",
        }
        self.patches = list(
            patches
            or [
                {
                    "target_file": "kai/simple.py",
                    "operation": "MODIFY",
                    "content": '"""Candidate change."""\n\ndef answer():\n    return 1\n',
                    "reason": "Keep behavior stable while adding a module description.",
                }
            ]
        )
        self.review = {
            "findings": [],
            "risk_assessment": "LOW",
            "recommendation": "PROCEED_TO_TESTS",
        }

    def status(self):
        return {
            "provider": "ollama",
            "model": "qwen2.5:0.5b",
            "status": "READY" if self.ready else "RUNTIME_UNAVAILABLE",
            "local_only": True,
        }

    def generate(self, prompt):
        return {"provider": "ollama", "model": "qwen2.5:0.5b", "response": "ok"}

    def generate_json(self, prompt, schema):
        if "target_file" in schema["properties"] and "const" not in schema["properties"]["target_file"]:
            value = self.plan
        elif "findings" in schema["properties"]:
            value = self.review
        else:
            value = self.patches.pop(0)
        return {
            "provider": "ollama",
            "model": "qwen2.5:0.5b",
            "request_id": str(uuid.uuid4()),
            "actual_model": {"provider": "ollama", "model_id": "qwen2.5:0.5b"},
            "routing": {"used_fallback": False, "fallback_from": []},
            "done": True,
            "local_only": True,
            "structured": value,
        }


class RoutedFakeProvider(FakeProvider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.task_categories = []

    def generate_json_for_task(self, prompt, schema, task_category):
        self.task_categories.append(task_category)
        return self.generate_json(prompt, schema)


class FakeRunner:
    def __init__(self, statuses=None):
        self.statuses = list(statuses or ["SUCCESS", "SUCCESS"])
        self.calls = []

    @staticmethod
    def probe_sandbox():
        return {"binary_status": "AVAILABLE", "probe_status": "READY"}

    def run_tests(self, candidate_id, workspace_id, module=None):
        self.calls.append((candidate_id, workspace_id, module))
        status = self.statuses.pop(0)
        return CandidateExecutionResult(
            execution_id=str(uuid.uuid4()),
            candidate_id=candidate_id,
            workspace_id=workspace_id,
            procedure_id=f"unittest:{module}",
            status=status,
            started_at="2026-10-15T00:00:00+00:00",
            completed_at="2026-10-15T00:00:01+00:00",
            duration_seconds=1.0,
            exit_code=0 if status == "SUCCESS" else 1,
            stdout="Ran 1 test in 0.001s\nOK\n" if status == "SUCCESS" else "",
            stderr="" if status == "SUCCESS" else "assertion failed",
            environment={"sandbox": "mocked"},
            reason=None if status == "SUCCESS" else "mocked test failure",
        )


class EngineeringDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Database(Path(self.temporary_directory.name) / "kai.sqlite3")
        self.database.initialize()

    def test_task_stages_persist_append_only_and_repair_budget_is_enforced(self):
        task_id = str(uuid.uuid4())
        self.database.create_engineering_task(
            task_id=task_id,
            objective="Make a small tested change.",
            provider={"provider": "ollama", "model": "qwen2.5:0.5b"},
            max_repair_attempts=1,
        )
        self.database.record_engineering_task_event(
            task_id,
            stage="PLAN",
            status="PLAN",
            details={"summary": "A small plan."},
            updates={"plan": {"summary": "A small plan."}},
        )
        self.database.record_engineering_task_event(
            task_id,
            stage="PATCH_APPLICATION",
            status="PATCH_APPLICATION",
            details={"attempt": 1},
            updates={"repair_attempts": 1},
        )
        with self.assertRaisesRegex(KaiError, "cannot decrease or exceed"):
            self.database.record_engineering_task_event(
                task_id,
                stage="PATCH_APPLICATION",
                status="PATCH_APPLICATION",
                details={"attempt": 2},
                updates={"repair_attempts": 2},
            )
        task = self.database.get_engineering_task(task_id)
        self.assertEqual(task["repair_attempts"], 1)
        self.assertEqual([event["stage"] for event in task["events"]], [
            "OBJECTIVE", "PLAN", "PATCH_APPLICATION"
        ])
        with self.database.connection() as connection:
            with self.assertRaisesRegex(sqlite3.IntegrityError, "history is immutable"):
                connection.execute(
                    "UPDATE engineering_task_events SET status = 'FAILED' "
                    "WHERE task_id = ? AND task_event_id = 1",
                    (task_id,),
                )

    def test_schema_v7_migration_preserves_identity_and_adds_task_history(self):
        path = Path(self.temporary_directory.name) / "legacy.sqlite3"
        legacy = Database(path)
        with legacy.connection() as connection:
            connection.execute(
                "CREATE TABLE schema_info "
                "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "version INTEGER NOT NULL)"
            )
            connection.execute("INSERT INTO schema_info VALUES (1, 0)")
            for version in range(1, 8):
                getattr(Database, f"_migrate_to_v{version}")(connection)
            connection.commit()
        identity = legacy.identity()["identity_id"]
        legacy.initialize()
        task_id = str(uuid.uuid4())
        task = legacy.create_engineering_task(
            task_id=task_id,
            objective="Check v7 upgrade.",
            provider={"provider": "disabled"},
        )
        self.assertEqual(legacy.schema_version(), 10)
        self.assertEqual(legacy.identity()["identity_id"], identity)
        self.assertEqual(task["status"], "OBJECTIVE")
        self.assertEqual(len(task["events"]), 1)


class EngineeringCoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.source_root = self.root / "source"
        (self.source_root / "kai").mkdir(parents=True)
        (self.source_root / "tests").mkdir()
        (self.source_root / "kai" / "__init__.py").write_text("", encoding="utf-8")
        (self.source_root / "kai" / "cli.py").write_text(
            "def main():\n    return 0\n", encoding="utf-8"
        )
        self.target = self.source_root / "kai" / "simple.py"
        self.target.write_text("def answer():\n    return 1\n", encoding="utf-8")
        (self.source_root / "tests" / "test_simple.py").write_text(
            "import unittest\n", encoding="utf-8"
        )
        self.database = Database(self.root / "state" / "kai.sqlite3")
        self.database.initialize()
        self.manager = CandidateWorkspaceManager(
            self.source_root, self.root / "state", self.database
        )

    def coordinator(self, provider, runner, repair_attempts=1):
        return EngineeringCoordinator(
            database=self.database,
            provider=provider,
            workspace_manager=self.manager,
            runner=runner,
            source_root=self.source_root,
            kai_version="test-version",
            schema_version=self.database.schema_version(),
            repair_attempts=repair_attempts,
        )

    def test_verified_plan_inventory_is_deterministic_and_excludes_protected_files(self):
        inventory = self.coordinator(RoutedFakeProvider(), FakeRunner()).inspect()["source_inventory"]
        first = _eligible_plan_inventory(self.source_root, inventory)
        second = _eligible_plan_inventory(self.source_root, inventory)
        self.assertEqual(first, second)
        self.assertTrue(first["options"])
        self.assertEqual(first["inventory_sha256"], _eligible_plan_inventory(self.source_root, inventory)["inventory_sha256"])
        self.assertNotIn("kai/database.py", [item["target_file"] for item in first["options"]])
        self.assertNotIn("kai/local_llm.py", [item["target_file"] for item in first["options"]])

    def test_mocked_end_to_end_candidate_never_modifies_production(self):
        original = self.target.read_bytes()
        runner = FakeRunner()
        provider = RoutedFakeProvider()
        result = self.coordinator(provider, runner).run(
            "Add a docstring to the helper without changing its behavior."
        )
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["result"]["decision"], "INCONCLUSIVE")
        self.assertEqual(result["result"]["persisted_decision"]["decision"], "REQUEST_MORE_EVIDENCE")
        self.assertEqual(result["result"]["evaluation_status"], "UNKNOWN")
        self.assertFalse(result["result"]["production_source_modified"])
        reviews = [
            event for event in result["events"] if event["stage"] == "MODEL_REVIEW"
        ]
        self.assertEqual(len(reviews), 1)
        self.assertTrue(reviews[0]["details"]["review"]["advisory_only"])
        self.assertEqual(
            reviews[0]["details"]["review"]["model"], "qwen2.5:0.5b"
        )
        self.assertTrue(reviews[0]["details"]["review"]["model_request_id"])
        self.assertEqual(
            provider.task_categories, ["planning", "coding", "code_review"]
        )
        patch_events = [
            event
            for event in result["events"]
            if event["stage"] == "PATCH_APPLICATION"
            and event["details"].get("operation")
        ]
        self.assertTrue(patch_events[0]["details"]["model_request_id"])
        self.assertEqual(self.target.read_bytes(), original)
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[0][2], "tests.test_simple")
        self.assertEqual(runner.calls[1][2], "tests.test_simple")
        workspace = self.database.get_candidate_workspace(result["workspace_id"])
        self.assertEqual(workspace["status"], "ACTIVE")
        with self.database.connection() as connection:
            task_count = connection.execute(
                "SELECT COUNT(*) FROM engineering_tasks"
            ).fetchone()[0]
        self.assertEqual(task_count, 1)

    def test_validated_plan_has_contract_version(self):
        result = self.coordinator(RoutedFakeProvider(), FakeRunner()).plan(
            "Add a docstring to the helper without changing its behavior."
        )
        self.assertEqual(result["status"], "PLAN")
        self.assertEqual(result["plan"]["contract_version"], 1)

    def test_invalid_model_plan_persists_typed_diagnostic(self):
        provider = RoutedFakeProvider(
            plan={
                "summary": "Bad plan",
                "target_file": "kai/simple.py",
                "test_module": "tests.test_missing",
                "reason": "The selected module does not exist.",
                "acceptance_criteria": ["Pass."],
                "risk": "Unknown.",
            }
        )
        result = self.coordinator(provider, FakeRunner()).plan("Make a bounded change.")
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["result"]["diagnostic"]["stage"], "PLAN")
        self.assertEqual(result["result"]["diagnostic"]["failure_category"], "invalid_reference")
        self.assertFalse(result["result"]["diagnostic"]["retry_eligible"])

    def test_predictor_objective_cannot_drift_to_another_verified_pair(self):
        provider = RoutedFakeProvider(
            plan={
                "summary": "Change the predictor.",
                "target_file": "kai/simple.py",
                "test_module": "tests.test_simple",
                "reason": "Wrong verified pair.",
                "acceptance_criteria": ["Pass."],
                "risk": "Unknown.",
            }
        )
        result = self.coordinator(provider, FakeRunner()).plan(
            "Make one small improvement to the CPU predictor."
        )
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("predictor objectives", result["result"]["error"].lower())

    def test_unavailable_local_provider_persists_blocked_task_without_candidate(self):
        result = self.coordinator(
            FakeProvider(ready=False), FakeRunner()
        ).run("Do not proceed without a local model.")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIsNone(result["candidate_id"])
        self.assertEqual(result["result"]["decision"], "UNKNOWN")
        self.assertEqual(result["result"]["production_source_modified"], False)

    def test_invalid_target_plan_is_rejected_before_candidate_creation(self):
        provider = FakeProvider(
            plan={
                "summary": "Escape workspace.",
                "target_file": "../README.md",
                "test_module": "tests.test_simple",
                "reason": "Invalid path.",
                "acceptance_criteria": ["Never escape."],
                "risk": "Traversal.",
            }
        )
        result = self.coordinator(provider, FakeRunner()).plan("Reject traversal.")
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("traversal", result["result"]["error"].lower())
        self.assertIsNone(result["candidate_id"])

    def test_one_failed_candidate_test_triggers_only_one_bounded_repair(self):
        patched = {
            "target_file": "kai/simple.py",
            "operation": "MODIFY",
            "content": '"""First candidate."""\n\ndef answer():\n    return 1\n',
            "reason": "First attempt.",
        }
        repaired = {
            **patched,
            "content": '"""Repaired candidate."""\n\ndef answer():\n    return 1\n',
            "reason": "Bounded correction after a failing candidate test.",
        }
        runner = FakeRunner(["SUCCESS", "FAILED", "SUCCESS"])
        result = self.coordinator(
            FakeProvider(patches=[patched, repaired]), runner, repair_attempts=1
        ).run("Add a docstring and repair one test failure.")
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["repair_attempts"], 1)
        self.assertEqual(len(runner.calls), 3)
        self.assertEqual(result["result"]["benchmark"]["status"], "UNKNOWN")
        self.assertEqual(
            [event["stage"] for event in result["events"]].count("PATCH_APPLICATION"),
            3,
        )
        self.assertEqual(
            [event["stage"] for event in result["events"]].count("MODEL_REVIEW"),
            2,
        )

    def test_production_tree_digest_unchanged_by_mocked_task(self):
        before = hashlib.sha256(self.target.read_bytes()).hexdigest()
        self.coordinator(FakeProvider(), FakeRunner()).run("Make a local candidate.")
        after = hashlib.sha256(self.target.read_bytes()).hexdigest()
        self.assertEqual(before, after)

    def test_structured_generation_requires_complete_local_ollama_response(self):
        with self.assertRaisesRegex(KaiError, "invalid structured response"):
            self.coordinator(FakeProvider(), FakeRunner())._validate_provider_result(
                {
                    "provider": "ollama",
                    "model": "model",
                    "done": True,
                    "structured": {},
                }
            )

    def test_provider_contract_accepts_future_gateway_adapters(self):
        self.coordinator(FakeProvider(), FakeRunner())._validate_provider_result(
            {
                "provider": "future-local-provider",
                "model": "future-model",
                "done": True,
                "local_only": True,
                "structured": {},
            }
        )
        with self.assertRaisesRegex(KaiError, "invalid structured response"):
            self.coordinator(FakeProvider(), FakeRunner())._validate_provider_result(
                {
                    "provider": "ollama",
                    "model": "qwen2.5:0.5b",
                    "local_only": True,
                    "structured": {},
                }
            )

    def test_engineer_inspect_does_not_create_state_or_log_files(self):
        from kai.cli import main

        state_dir = self.root / "inspect-state"
        inventory = {
            "module_inventory": [],
            "source_file_count": 0,
            "test_file_count": 0,
            "source_snapshot_sha256": "digest",
        }
        with patch.dict(
            os.environ,
            {
                "KAI_HOME": str(state_dir),
                "KAI_LLM_PROVIDER": "ollama",
                "KAI_LLM_MODEL": "qwen2.5:0.5b",
            },
        ), patch(
            "kai.engineering.inspect_source_tree", return_value=inventory
        ), patch.object(
            EngineeringCoordinator, "_environment_summary", return_value={}
        ), patch(
            "kai.local_llm.LocalOllamaProvider.status",
            return_value={
                "provider": "ollama",
                "model": "qwen2.5:0.5b",
                "status": "RUNTIME_UNAVAILABLE",
                "local_only": True,
            },
        ), patch(
            "kai.candidate_runner.CandidateRunner.probe_sandbox",
            return_value={"probe_status": "READY"},
        ), patch("sys.stdout", new_callable=io.StringIO) as stdout:
            self.assertEqual(main(["engineer", "inspect"]), 0)

        report = json.loads(stdout.getvalue())
        self.assertTrue(report["inspection_is_read_only"])
        self.assertFalse(state_dir.exists())


if __name__ == "__main__":
    unittest.main()
