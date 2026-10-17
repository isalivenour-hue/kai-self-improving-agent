import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kai.database import Database
from kai.config import KaiConfig
from kai.errors import KaiError
from kai.local_llm import (
    LocalOllamaProvider,
    ModelGateway,
    OpenAIProvider,
    PROVIDER_CAPABILITIES,
)


class FakeModels:
    def __init__(self, entries):
        self.entries = entries
        self.calls = 0

    def list(self):
        self.calls += 1
        return SimpleNamespace(data=self.entries)


class FakeResponses:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class FakeProvider:
    def __init__(self, *, error=None, text="result"):
        self.error = error
        self.text = text

    def status(self):
        return {
            "provider": "ollama",
            "model": "fake",
            "status": "READY",
            "local_only": True,
        }

    def generate(self, prompt):
        if self.error:
            raise self.error
        return {
            "provider": "ollama",
            "model": "fake",
            "response": self.text,
            "done": True,
            "local_only": True,
            "token_usage": {"input_tokens": 3, "output_tokens": 2},
        }

    def generate_json(self, prompt, schema):
        if self.error:
            raise self.error
        return {
            "provider": "ollama",
            "model": "fake",
            "response": "{}",
            "structured": {},
            "done": True,
            "local_only": True,
        }


class ModelGatewayTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Database(Path(self.temporary_directory.name) / "kai.sqlite3")
        self.database.initialize()
        self.gateway = ModelGateway(
            self.database,
            default_provider="ollama",
            default_model="qwen2.5:0.5b",
            available_memory_bytes=1024,
        )

    def test_schema_v8_upgrade_preserves_llm_history_and_adds_latest_index(self):
        path = Path(self.temporary_directory.name) / "schema-v8.sqlite3"
        legacy = Database(path)
        with legacy.connection() as connection:
            connection.execute(
                "CREATE TABLE schema_info "
                "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "version INTEGER NOT NULL)"
            )
            connection.execute("INSERT INTO schema_info VALUES (1, 0)")
            for version in range(1, 9):
                getattr(Database, f"_migrate_to_v{version}")(connection)
            Database._migrate_to_v9(connection)
            connection.commit()
        legacy.record_llm_request(
                {
                    "request_id": "preserve-across-v10",
                    "started_at": "2026-10-17T00:00:00+00:00",
                    "completed_at": "2026-10-17T00:00:01+00:00",
                    "duration_ms": 1000,
                    "task_category": "general",
                    "provider": "ollama",
                    "model_id": "qwen2.5:0.5b",
                    "outcome": "SUCCESS",
                    "token_usage": {},
                    "error_category": None,
                    "fallback_from": [],
                }
        )
        legacy.initialize()
        self.assertEqual(legacy.schema_version(), 10)
        self.assertIsNone(legacy.get_llm_model_selection())
        self.assertEqual(legacy.list_llm_task_routes(), [])
        self.assertEqual(legacy.list_llm_fallbacks(), [])
        self.assertEqual(
            legacy.latest_llm_request("ollama", "qwen2.5:0.5b")["request_id"],
            "preserve-across-v10",
        )
        with legacy.connection() as connection:
                indexes = {
                    row["name"]
                    for row in connection.execute(
                        "PRAGMA index_list(llm_request_history)"
                    ).fetchall()
                }
                query_plan = connection.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM llm_request_history "
                    "WHERE provider = ? AND model_id = ? "
                    "ORDER BY completed_at DESC, request_id LIMIT 1",
                    ("ollama", "qwen2.5:0.5b"),
                ).fetchall()
        self.assertIn("llm_request_history_model_latest_idx", indexes)
        self.assertIn(
                "llm_request_history_model_latest_idx",
                " ".join(row["detail"] for row in query_plan),
        )

    def test_ollama_model_discovery_preserves_installed_size_and_capabilities(self):
        from tests.test_local_llm import FakeConnection, FakeResponse

        connection = FakeConnection(
            FakeResponse(
                {
                    "models": [
                        {
                            "name": "qwen2.5:3b",
                            "model": "qwen2.5:3b",
                            "size": 2048,
                            "digest": "digest",
                        }
                    ]
                }
            )
        )
        with patch("kai.local_llm.HTTPConnection", return_value=connection):
            models = LocalOllamaProvider("qwen2.5:3b").list_models()
        self.assertEqual(models[0]["status"], "INSTALLED")
        self.assertEqual(models[0]["size_bytes"], 2048)
        self.assertFalse(models[0]["capabilities"]["tools"])
        self.assertFalse(models[0]["capabilities"]["cancellation"])

    def test_model_registry_warns_when_installed_weight_is_large_for_free_ram(self):
        from tests.test_local_llm import FakeConnection, FakeResponse

        connections = [
            FakeConnection(
                FakeResponse(
                    {
                        "models": [
                            {"name": "qwen2.5:0.5b", "size": 2048},
                        ]
                    }
                )
            ),
        ]
        with patch("kai.local_llm.HTTPConnection", side_effect=connections):
            models = self.gateway.models()
        selected = next(item for item in models if item["provider"] == "ollama")
        self.assertEqual(selected["availability_status"], "INSTALLED")
        self.assertTrue(selected["configured"])
        self.assertIn("inference memory requirements are unknown", selected["resource_warning"])

    def test_selection_is_persisted_only_after_model_validation(self):
        with patch.object(
            self.gateway,
            "provider",
            return_value=FakeProvider(),
        ):
            saved = self.database.set_llm_model_selection("ollama", "qwen2.5:1.5b")
            self.assertEqual(saved["model_id"], "qwen2.5:1.5b")
            self.assertEqual(
                self.gateway.selection(read_only=True)["model_id"], "qwen2.5:1.5b"
            )

            with patch.object(
                self.gateway,
                "provider",
                return_value=SimpleNamespace(
                    status=lambda: {
                        "status": "MODEL_NOT_INSTALLED",
                        "provider": "ollama",
                        "model": "missing",
                    }
                ),
            ):
                with self.assertRaisesRegex(KaiError, "MODEL_NOT_INSTALLED"):
                    self.gateway.validate_selection(
                        {"provider": "ollama", "model_id": "missing"}
                    )
            self.assertEqual(
                self.database.get_llm_model_selection()["model_id"], "qwen2.5:1.5b"
            )

    def test_task_route_overrides_active_model_without_mutating_selection(self):
        self.database.set_llm_model_selection("ollama", "qwen2.5:0.5b")
        self.database.set_llm_task_route("coding", "ollama", "llama3.2:latest")
        self.assertEqual(
            self.gateway.selection("coding")["model_id"], "llama3.2:latest"
        )
        self.assertEqual(
            self.gateway.selection("general")["model_id"], "qwen2.5:0.5b"
        )

    def test_fallback_is_bounded_audited_and_never_persists_prompt_or_response(self):
        self.database.add_llm_fallback("ollama", "qwen2.5:1.5b")
        failing = FakeProvider(error=KaiError("Ollama service unavailable"))
        succeeding = FakeProvider(text="secret response text")

        def provider(target):
            return failing if target["model_id"] == "qwen2.5:0.5b" else succeeding

        with patch.object(self.gateway, "provider", side_effect=provider):
            result = self.gateway.generate("private prompt", task_category="general")

        self.assertEqual(result["response"], "secret response text")
        self.assertTrue(result["routing"]["used_fallback"])
        self.assertEqual(result["actual_model"]["model_id"], "qwen2.5:1.5b")
        self.assertEqual(result["routing"]["fallback_from"][0]["reason"], "unavailable")
        history = self.database.list_llm_requests()
        self.assertEqual([item["outcome"] for item in history], ["SUCCESS", "FAILED"])
        self.assertEqual(history[0]["token_usage"]["input_tokens"], 3)
        serialized = json.dumps(history)
        self.assertNotIn("private prompt", serialized)
        self.assertNotIn("secret response text", serialized)

    def test_disabled_provider_never_tries_configured_fallbacks(self):
        self.database.add_llm_fallback("ollama", "qwen2.5:1.5b")
        disabled = ModelGateway(
            self.database,
            default_provider="disabled",
            default_model="disabled",
        )
        with patch.object(disabled, "provider") as provider_factory:
            with self.assertRaisesRegex(KaiError, "disabled by the active configuration"):
                disabled.generate("must not run")
        provider_factory.assert_not_called()
        self.assertEqual(self.database.list_llm_requests(), [])

    def test_local_only_blocks_cloud_selection_and_cloud_fallback(self):
        self.database.set_llm_model_selection("ollama", "qwen2.5:0.5b")
        self.database.add_llm_fallback("ollama", "qwen2.5:1.5b")
        with patch.object(self.gateway, "provider", return_value=FakeProvider()):
            result = self.gateway.generate(
                "safe prompt", task_category="general", local_only=True
            )
        self.assertTrue(result["actual_model"]["provider"] == "ollama")

        self.database.set_llm_model_selection("openai", "configured-model")
        with self.assertRaisesRegex(KaiError, "local-only routing"):
            self.gateway.generate("source", task_category="coding", local_only=True)

    def test_cloud_and_source_code_authorization_are_separate(self):
        self.gateway.openai_enabled = True
        self.gateway.openai_allow_source = False
        with self.assertRaisesRegex(KaiError, "source-code tasks"):
            self.gateway.validate_selection(
                {"provider": "openai", "model_id": "selected-model"},
                task_category="coding",
            )
        with self.assertRaisesRegex(KaiError, "OpenAI is disabled|Cloud routing"):
            ModelGateway(
                self.database,
                openai_enabled=False,
            ).generate(
                "do not send",
                explicit_target="openai/some-model",
            )

    def test_active_status_is_read_only_when_database_does_not_exist(self):
        path = Path(self.temporary_directory.name) / "not-created" / "kai.sqlite3"
        read_only = ModelGateway(Database(path), read_only=True)
        with patch(
            "kai.local_llm.LocalOllamaProvider.status",
            return_value={
                "provider": "ollama",
                "model": "qwen2.5:0.5b",
                "status": "RUNTIME_UNAVAILABLE",
                "local_only": True,
            },
        ):
            status = read_only.status()
        self.assertEqual(status["selection"]["model_id"], "qwen2.5:0.5b")
        self.assertFalse(path.parent.exists())

    def test_openai_configuration_requires_separate_source_sharing_flag(self):
        with patch.dict(
            "os.environ",
            {
                "KAI_HOME": self.temporary_directory.name,
                "KAI_OPENAI_ENABLED": "true",
                "KAI_OPENAI_ALLOW_SOURCE": "false",
                "OPENAI_API_KEY": "test-secret",
            },
            clear=True,
        ):
            config = KaiConfig.from_environment()
        self.assertTrue(config.openai_enabled)
        self.assertFalse(config.openai_allow_source)
        self.assertEqual(config.llm_provider, "ollama")

    def test_openai_is_inert_without_explicit_enablement_or_key(self):
        provider = OpenAIProvider(
            "user-selected-id", enabled=False, api_key=None
        )
        with patch("builtins.__import__", side_effect=AssertionError("unexpected import")):
            status = provider.status()
        self.assertEqual(status["status"], "DISABLED")
        self.assertFalse(status["local_only"])

        missing_key = OpenAIProvider("user-selected-id", enabled=True, api_key="")
        self.assertEqual(missing_key.status()["status"], "MISSING_API_KEY")

    def test_openai_reports_missing_optional_sdk_without_network_or_install(self):
        provider = OpenAIProvider(
            "user-selected-id", enabled=True, api_key="test-secret"
        )
        with patch.dict("sys.modules", {"openai": None}):
            status = provider.status()
        self.assertEqual(status["status"], "SDK_UNAVAILABLE")
        self.assertNotIn("test-secret", json.dumps(status))

    def test_openai_model_listing_and_generation_report_usage(self):
        models = FakeModels([SimpleNamespace(id="selected-model")])
        responses = FakeResponses(
            result=SimpleNamespace(
                output_text="hello",
                usage=SimpleNamespace(
                    input_tokens=7, output_tokens=4, total_tokens=11
                ),
            )
        )
        client = SimpleNamespace(
            models=models,
            responses=responses,
        )
        provider = OpenAIProvider(
            "selected-model",
            enabled=True,
            api_key="never-persist",
            client=client,
        )
        self.assertEqual(provider.status()["status"], "CONFIGURED")
        self.assertEqual(models.calls, 0)
        self.assertEqual(provider.list_models()[0]["model_id"], "selected-model")
        self.assertEqual(models.calls, 1)
        result = provider.generate("hello")
        self.assertEqual(result["token_usage"]["total_tokens"], 11)
        self.assertFalse(result["local_only"])
        self.assertEqual(
            responses.calls[0]["max_output_tokens"], 256
        )
        self.assertNotIn("never-persist", json.dumps(result))

    def test_openai_model_validation_is_explicit_and_checks_model_access(self):
        client = SimpleNamespace(
            models=FakeModels([SimpleNamespace(id="available-model")]),
            responses=FakeResponses(),
        )
        gateway = ModelGateway(
            self.database,
            openai_enabled=True,
        )
        with patch.object(
            gateway,
            "provider",
            return_value=OpenAIProvider(
                "available-model",
                enabled=True,
                api_key="test-secret",
                client=client,
            ),
        ):
            status = gateway.validate_selection(
                {"provider": "openai", "model_id": "available-model"}
            )
        self.assertEqual(status["status"], "READY")
        self.assertEqual(client.models.calls, 1)

    def test_openai_structured_generation_uses_strict_schema_and_parses_output(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        responses = FakeResponses(
            result=SimpleNamespace(
                output_text='{"answer":"ok"}',
                usage=SimpleNamespace(input_tokens=5, output_tokens=3, total_tokens=8),
            )
        )
        client = SimpleNamespace(
            models=FakeModels([SimpleNamespace(id="selected-model")]),
            responses=responses,
        )
        provider = OpenAIProvider(
            "selected-model", enabled=True, api_key="unused", client=client
        )
        result = provider.generate_json("structured please", schema)
        self.assertEqual(result["structured"], {"answer": "ok"})
        self.assertEqual(
            responses.calls[0]["text"]["format"]["type"], "json_schema"
        )
        self.assertTrue(responses.calls[0]["text"]["format"]["strict"])

    def test_openai_api_errors_are_redacted_and_timeout_classified(self):
        class APITimeoutError(Exception):
            pass

        responses = FakeResponses(error=APITimeoutError("private key=value"))
        client = SimpleNamespace(
            models=FakeModels([SimpleNamespace(id="selected-model")]),
            responses=responses,
        )
        provider = OpenAIProvider(
            "selected-model",
            enabled=True,
            api_key="test-secret",
            client=client,
        )
        with self.assertRaisesRegex(KaiError, "timed out") as raised:
            provider.generate("hello")
        self.assertNotIn("private key", str(raised.exception))
        self.assertNotIn("test-secret", str(raised.exception))

    def test_capabilities_do_not_claim_tools_streaming_or_cancellation(self):
        for provider in ("ollama", "openai"):
            self.assertTrue(PROVIDER_CAPABILITIES[provider]["generate_text"])
            self.assertFalse(PROVIDER_CAPABILITIES[provider]["tools"])
            self.assertFalse(PROVIDER_CAPABILITIES[provider]["streaming"])
            self.assertFalse(PROVIDER_CAPABILITIES[provider]["cancellation"])
            self.assertEqual(
                PROVIDER_CAPABILITIES[provider][
                    "structured_output_model_compatibility"
                ],
                "UNKNOWN",
            )

    def test_cli_exposes_discovery_selection_routing_fallback_and_history(self):
        import io

        from kai.cli import main

        runtime = unittest.mock.MagicMock()
        runtime.llm_providers.return_value = [{"provider": "ollama"}]
        runtime.llm_models.return_value = [{"model_id": "qwen2.5:0.5b"}]
        runtime.llm_active.return_value = {"selection": {"provider": "ollama"}}
        runtime.llm_use.return_value = {"selection": {"provider": "ollama"}}
        runtime.llm_routes.return_value = []
        runtime.llm_route_set.return_value = {"task_category": "coding"}
        runtime.llm_fallback_add.return_value = [{"position": 0}]
        runtime.llm_fallbacks.return_value = [{"position": 0}]
        runtime.llm_history.return_value = []
        with patch("kai.cli.KaiRuntime", return_value=runtime):
            cases = [
                (["llm", "providers"], runtime.llm_providers),
                (["llm", "models"], runtime.llm_models),
                (["llm", "active"], runtime.llm_active),
                (["llm", "use", "ollama/qwen2.5:0.5b"], runtime.llm_use),
                (["llm", "route", "set", "coding", "ollama/qwen2.5:0.5b"], runtime.llm_route_set),
                (["llm", "fallback", "add", "ollama/qwen2.5:1.5b"], runtime.llm_fallback_add),
                (["llm", "history", "--limit", "4"], runtime.llm_history),
            ]
            for args, method in cases:
                with self.subTest(args=args), patch(
                    "sys.stdout", new_callable=io.StringIO
                ) as stdout:
                    self.assertEqual(main(args), 0)
                    json.loads(stdout.getvalue())
                method.assert_called()


if __name__ == "__main__":
    unittest.main()
