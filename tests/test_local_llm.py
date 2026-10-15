import io
import json
import unittest
from unittest.mock import MagicMock, patch

from kai.errors import KaiError
from kai.local_llm import (
    LOCAL_MODEL,
    MAX_PROMPT_CHARACTERS,
    LocalOllamaProvider,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.body = (
            payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
        )
        self.status = status

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]


class FakeConnection:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.request_args = None
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        self.request_args = (method, path, body, headers)
        if self.error:
            raise self.error

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


class LocalOllamaProviderTests(unittest.TestCase):
    def test_status_reports_ready_only_when_configured_model_is_installed(self):
        connections = [
            FakeConnection(
                FakeResponse(
                    {
                        "models": [
                            {"name": LOCAL_MODEL, "size": 123, "digest": "abc"},
                            {"name": "other:latest"},
                        ]
                    }
                )
            ),
            FakeConnection(FakeResponse({"version": "0.6.0"})),
        ]
        with patch(
            "kai.local_llm.HTTPConnection", side_effect=connections
        ) as connect:
            status = LocalOllamaProvider().status()

        self.assertEqual(status["status"], "READY")
        self.assertEqual(status["available_models"][0]["size"], 123)
        self.assertEqual(status["runtime_version"], "0.6.0")
        self.assertTrue(status["local_only"])
        self.assertEqual(connect.call_count, 2)
        connect.assert_any_call("127.0.0.1", 11434, timeout=5)
        self.assertEqual(connections[0].request_args[:2], ("GET", "/api/tags"))
        self.assertTrue(all(connection.closed for connection in connections))

    def test_status_reports_missing_model(self):
        with patch(
            "kai.local_llm.HTTPConnection",
            return_value=FakeConnection(FakeResponse({"models": []})),
        ):
            status = LocalOllamaProvider().status()

        self.assertEqual(status["status"], "MODEL_NOT_INSTALLED")

    def test_model_discovery_rejects_malformed_model_entries(self):
        with patch(
            "kai.local_llm.HTTPConnection",
            return_value=FakeConnection(FakeResponse({"models": [{"size": 123}]})),
        ):
            with self.assertRaisesRegex(KaiError, "without a valid name"):
                LocalOllamaProvider().list_models()

    def test_status_reports_unavailable_service(self):
        with patch(
            "kai.local_llm.HTTPConnection",
            return_value=FakeConnection(error=OSError("connection refused")),
        ):
            status = LocalOllamaProvider().status()

        self.assertEqual(status["status"], "RUNTIME_UNAVAILABLE")
        self.assertIn("Could not contact", status["reason"])

    def test_generation_uses_fixed_local_model_and_bounded_non_streaming_output(self):
        connections = [
            FakeConnection(FakeResponse({"models": [{"name": LOCAL_MODEL}]})),
            FakeConnection(FakeResponse({"version": "0.6.0"})),
            FakeConnection(FakeResponse({"response": "Hello from KAI.", "done": True})),
        ]
        with patch(
            "kai.local_llm.HTTPConnection",
            side_effect=connections,
        ):
            result = LocalOllamaProvider().generate("Say hello.")

        generate_request = connections[2].request_args
        request_payload = json.loads(generate_request[2])
        self.assertEqual(generate_request[:2], ("POST", "/api/generate"))
        self.assertEqual(request_payload["model"], LOCAL_MODEL)
        self.assertEqual(request_payload["prompt"], "Say hello.")
        self.assertFalse(request_payload["stream"])
        self.assertEqual(request_payload["options"]["num_predict"], 256)
        self.assertEqual(result["response"], "Hello from KAI.")
        self.assertTrue(result["done"])
        self.assertTrue(result["local_only"])

    def test_generation_fails_explicitly_when_model_is_not_installed(self):
        with patch(
            "kai.local_llm.HTTPConnection",
            return_value=FakeConnection(FakeResponse({"models": []})),
        ) as connect:
            with self.assertRaisesRegex(KaiError, "ollama pull"):
                LocalOllamaProvider().generate("Hello")

        self.assertEqual(connect.call_count, 2)

    def test_generation_rejects_empty_or_oversized_prompts_before_network_access(self):
        with patch("kai.local_llm.HTTPConnection") as connect:
            with self.assertRaisesRegex(KaiError, "non-empty"):
                LocalOllamaProvider().generate("  ")
            with self.assertRaisesRegex(KaiError, "character limit"):
                LocalOllamaProvider().generate("x" * (MAX_PROMPT_CHARACTERS + 1))

        connect.assert_not_called()

    def test_api_errors_and_invalid_responses_are_explicit(self):
        with patch(
            "kai.local_llm.HTTPConnection",
            return_value=FakeConnection(FakeResponse(b"missing", status=404)),
        ):
            with self.assertRaisesRegex(KaiError, "HTTP 404"):
                LocalOllamaProvider().status()

        with patch(
            "kai.local_llm.HTTPConnection",
            return_value=FakeConnection(FakeResponse(b"not-json")),
        ):
            with self.assertRaisesRegex(KaiError, "invalid JSON"):
                LocalOllamaProvider().status()

    def test_structured_output_uses_provider_schema_and_rejects_malformed_json(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        connections = [
            FakeConnection(FakeResponse({"models": [{"name": LOCAL_MODEL}]})),
            FakeConnection(FakeResponse({"version": "0.6.0"})),
            FakeConnection(
                FakeResponse({"response": '{"answer":"ok"}', "done": True})
            ),
        ]
        with patch("kai.local_llm.HTTPConnection", side_effect=connections):
            result = LocalOllamaProvider().generate_json("Answer in JSON", schema)
        payload = json.loads(connections[2].request_args[2])
        self.assertEqual(payload["format"], schema)
        self.assertEqual(payload["options"]["num_predict"], 2048)
        self.assertEqual(result["structured"], {"answer": "ok"})

        connections = [
            FakeConnection(FakeResponse({"models": [{"name": LOCAL_MODEL}]})),
            FakeConnection(FakeResponse({"version": "0.6.0"})),
            FakeConnection(FakeResponse({"response": "not json", "done": True})),
        ]
        with patch("kai.local_llm.HTTPConnection", side_effect=connections):
            with self.assertRaisesRegex(KaiError, "malformed structured JSON"):
                LocalOllamaProvider().generate_json("Answer in JSON", schema)

    def test_status_timeout_is_reported_as_runtime_unavailable(self):
        with patch(
            "kai.local_llm.HTTPConnection",
            return_value=FakeConnection(error=TimeoutError("timed out")),
        ):
            status = LocalOllamaProvider().status()

        self.assertEqual(status["status"], "RUNTIME_UNAVAILABLE")
        self.assertIn("timed out", status["reason"])

    def test_cli_dispatches_local_model_status_and_generation(self):
        from kai.cli import main

        runtime = MagicMock()
        runtime.local_llm_status.return_value = {"status": "READY"}
        runtime.generate_local_response.return_value = {"response": "local"}
        with patch("kai.cli.KaiRuntime", return_value=runtime):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(main(["llm", "status"]), 0)
                self.assertEqual(json.loads(stdout.getvalue())["status"], "READY")
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                self.assertEqual(main(["llm", "generate", "hello"]), 0)
                self.assertEqual(json.loads(stdout.getvalue())["response"], "local")

        self.assertEqual(runtime.database.initialize.call_count, 2)
        runtime.generate_local_response.assert_called_once_with(
            "hello", task_category="general", model_override=None, local_only=False
        )


if __name__ == "__main__":
    unittest.main()
