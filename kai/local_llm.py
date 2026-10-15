"""Opt-in local text generation through the loopback Ollama API."""

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from http.client import HTTPConnection, HTTPException
from typing import Any, Protocol

from kai.errors import KaiError

OLLAMA_API_BASE = "http://127.0.0.1:11434"
LOCAL_MODEL = "qwen2.5:0.5b"
MAX_PROMPT_CHARACTERS = 8_000
MAX_OUTPUT_TOKENS = 256
MAX_RESPONSE_BYTES = 1_000_000
REQUEST_TIMEOUT_SECONDS = 120
STATUS_TIMEOUT_SECONDS = 5
_LOCAL_INFERENCE_LOCK = threading.Lock()
TASK_CATEGORIES = (
    "general",
    "planning",
    "coding",
    "code_review",
    "summarization",
    "classification",
    "reasoning",
    "structured_output",
)
PROVIDER_CAPABILITIES = {
    "ollama": {
        "generate_text": True,
        "structured_output": True,
        "structured_output_model_compatibility": "UNKNOWN",
        "token_usage": True,
        "streaming": False,
        "tools": False,
        "cancellation": False,
        "request_output_limits_tokens": {
            "text": MAX_OUTPUT_TOKENS,
            "structured_output": 2_048,
        },
    },
    "openai": {
        "generate_text": True,
        "structured_output": True,
        "structured_output_model_compatibility": "UNKNOWN",
        "token_usage": True,
        "streaming": False,
        "tools": False,
        "cancellation": False,
        "request_output_limits_tokens": {
            "text": MAX_OUTPUT_TOKENS,
            "structured_output": 2_048,
        },
    },
}
PROVIDER_CLASSIFICATIONS = {"ollama": "local", "openai": "cloud"}


class _OllamaConnectionError(KaiError):
    pass


class LLMProvider(Protocol):
    def status(self) -> dict: ...

    def list_models(self) -> list[dict]: ...

    def capabilities(self) -> dict: ...

    def generate(self, prompt: str) -> dict: ...

    def generate_json(self, prompt: str, schema: dict) -> dict: ...


class LocalOllamaProvider:
    def __init__(
        self, model: str = LOCAL_MODEL, timeout_seconds: int = REQUEST_TIMEOUT_SECONDS
    ):
        if not isinstance(model, str) or not model.strip():
            raise KaiError("Configured local model name must be non-empty.")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 600
        ):
            raise KaiError("Model request timeout must be between 1 and 600 seconds.")
        self.model = model.strip()
        self.timeout_seconds = timeout_seconds

    def status(self) -> dict:
        try:
            available_models = self.list_models()
        except _OllamaConnectionError as exc:
            return {
                "provider": "ollama",
                "endpoint": OLLAMA_API_BASE,
                "model": self.model,
                "status": "RUNTIME_UNAVAILABLE",
                "reason": str(exc),
                "local_only": True,
            }

        model_names = [item["model_id"] for item in available_models]
        available = self.model in model_names
        try:
            version_payload = self._request_json(
                "GET", "/api/version", timeout_seconds=STATUS_TIMEOUT_SECONDS
            )
            version = version_payload.get("version")
        except _OllamaConnectionError:
            version = None
        return {
            "provider": "ollama",
            "endpoint": OLLAMA_API_BASE,
            "model": self.model,
            "runtime_version": version,
            "status": "READY" if available else "MODEL_NOT_INSTALLED",
            "available_models": available_models,
            "local_only": True,
            "capabilities": self.capabilities(),
        }

    def generate(self, prompt: str) -> dict:
        return self._generate(prompt, MAX_OUTPUT_TOKENS)

    def generate_json(self, prompt: str, schema: dict) -> dict:
        if not isinstance(schema, dict):
            raise KaiError("Structured output schema must be a JSON object.")
        try:
            json.dumps(schema, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Structured output schema must be finite JSON: {exc}") from exc
        result = self._generate(prompt, 2_048, output_format=schema)
        if not result["done"]:
            raise KaiError("Ollama structured response did not complete.")
        try:
            decoded = json.loads(result["response"])
        except json.JSONDecodeError as exc:
            raise KaiError("Ollama returned malformed structured JSON.") from exc
        if not isinstance(decoded, dict):
            raise KaiError("Ollama structured output must be a JSON object.")
        return {
            **result,
            "structured": decoded,
        }

    def _generate(
        self, prompt: str, maximum_tokens: int, output_format: dict | None = None
    ) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise KaiError("A non-empty prompt is required.")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise KaiError(
                f"Prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit."
            )

        provider_status = self.status()
        if provider_status["status"] == "RUNTIME_UNAVAILABLE":
            raise KaiError(
                "The local Ollama service is unavailable. Start Ollama, then retry "
                "'kai llm status'."
            )
        if provider_status["status"] == "MODEL_NOT_INSTALLED":
            raise KaiError(
                f"Model {self.model} is not installed. Pull it with "
                f"'ollama pull {self.model}', then retry."
            )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": 0,
            "options": {
                "temperature": 0.2,
                "num_predict": maximum_tokens,
            },
        }
        if output_format is not None:
            payload["format"] = output_format
        result = self._request_json(
            "POST",
            "/api/generate",
            payload,
            timeout_seconds=self.timeout_seconds,
        )
        response = result.get("response")
        if not isinstance(response, str):
            raise KaiError("Ollama returned a response without generated text.")
        return {
            "provider": "ollama",
            "model": self.model,
            "response": response,
            "done": result.get("done") is True,
            "local_only": True,
            "token_usage": {
                key: result[key]
                for key in ("prompt_eval_count", "eval_count")
                if isinstance(result.get(key), int) and result[key] >= 0
            },
        }

    def list_models(self) -> list[dict]:
        payload = self._request_json(
            "GET", "/api/tags", timeout_seconds=STATUS_TIMEOUT_SECONDS
        )
        models = payload.get("models")
        if not isinstance(models, list):
            raise KaiError("Ollama returned an invalid model list.")
        discovered = []
        for item in models:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"].strip()
            ):
                raise KaiError("Ollama returned a model without a valid name.")
            model_id = item["name"].strip()
            discovered.append(
                {
                    "provider": "ollama",
                    "model_id": model_id,
                    "display_name": item.get("model") or model_id,
                    "name": model_id,
                    "size": item.get("size"),
                    "digest": item.get("digest"),
                    "modified_at": item.get("modified_at"),
                    "classification": "local",
                    "status": "INSTALLED",
                    "installed": True,
                    "available": True,
                    "size_bytes": item.get("size")
                    if isinstance(item.get("size"), int)
                    else None,
                    "context_window": None,
                    "maximum_output_tokens": None,
                    "resource_requirements": {
                        "installed_model_size_bytes": item.get("size")
                        if isinstance(item.get("size"), int)
                        else None,
                        "inference_memory_bytes": None,
                    },
                    "capabilities": self.capabilities(),
                }
            )
        return discovered

    @staticmethod
    def capabilities() -> dict:
        return dict(PROVIDER_CAPABILITIES["ollama"])

    @staticmethod
    def _request_json(
        method: str,
        path: str,
        payload: dict | None = None,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ) -> dict:
        data = (
            json.dumps(payload).encode("utf-8")
            if payload is not None
            else None
        )
        connection = HTTPConnection("127.0.0.1", 11434, timeout=timeout_seconds)
        try:
            headers = {"Content-Type": "application/json"} if data is not None else {}
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            body = response.read(MAX_RESPONSE_BYTES + 1)
        except (HTTPException, OSError, TimeoutError) as exc:
            raise _OllamaConnectionError(
                f"Could not contact the local Ollama API: {exc}"
            ) from exc
        finally:
            connection.close()

        if len(body) > MAX_RESPONSE_BYTES:
            raise KaiError("Ollama API response exceeded the size limit.")
        if response.status >= 400:
            detail = body[:2_000].decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise KaiError(
                f"Ollama API returned HTTP {response.status}{suffix}"
            )
        try:
            result = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise KaiError("Ollama API returned invalid JSON.") from exc
        if not isinstance(result, dict):
            raise KaiError("Ollama API returned an invalid response.")
        return result


class OpenAIProvider:
    """Optional OpenAI adapter; it is inert unless explicitly enabled with a key."""

    def __init__(
        self,
        model: str,
        *,
        enabled: bool | None = None,
        api_key: str | None = None,
        client: Any | None = None,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ):
        self.model = model
        self.enabled = (
            os.environ.get("KAI_OPENAI_ENABLED", "").strip().lower()
            in {"1", "true", "yes"}
            if enabled is None
            else enabled
        )
        self.api_key = os.environ.get("OPENAI_API_KEY") if api_key is None else api_key
        self._client = client
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 600
        ):
            raise KaiError("Model request timeout must be between 1 and 600 seconds.")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def capabilities() -> dict:
        return dict(PROVIDER_CAPABILITIES["openai"])

    def _client_or_raise(self):
        if not self.enabled:
            raise KaiError(
                "OpenAI is disabled. Set KAI_OPENAI_ENABLED=true and configure "
                "OPENAI_API_KEY to explicitly enable it."
            )
        if not self.api_key:
            raise KaiError("OPENAI_API_KEY is not configured.")
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise KaiError(
                "OpenAI support requires the optional 'openai' package; install it "
                "in this environment to enable this provider."
            ) from exc
        self._client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.openai.com/v1",
            timeout=self.timeout_seconds,
            max_retries=0,
        )
        return self._client

    def list_models(self) -> list[dict]:
        client = self._client_or_raise()
        try:
            response = client.models.list()
        except Exception as exc:
            raise _openai_error(exc, "model discovery") from exc
        models = getattr(response, "data", None)
        if not isinstance(models, (list, tuple)):
            raise KaiError("OpenAI returned an invalid model list.")
        return [
            {
                "provider": "openai",
                "model_id": model_id,
                "display_name": model_id,
                "classification": "cloud",
                "status": "AVAILABLE",
                "installed": None,
                "available": True,
                "size_bytes": None,
                "context_window": None,
                "maximum_output_tokens": None,
                "resource_requirements": None,
                "capabilities": self.capabilities(),
            }
            for item in models
            if isinstance((model_id := _value(item, "id")), str) and model_id
        ]

    def status(self) -> dict:
        if not self.enabled:
            return {
                "provider": "openai",
                "model": self.model,
                "classification": "cloud",
                "status": "DISABLED",
                "reason": "Explicit enablement is required; no API request was made.",
                "local_only": False,
                "capabilities": self.capabilities(),
            }
        if not self.api_key:
            return {
                "provider": "openai",
                "model": self.model,
                "classification": "cloud",
                "status": "MISSING_API_KEY",
                "reason": "Set OPENAI_API_KEY in the process environment.",
                "local_only": False,
                "capabilities": self.capabilities(),
            }
        try:
            self._client_or_raise()
        except KaiError as exc:
            text = str(exc).lower()
            status = (
                "SDK_UNAVAILABLE"
                if "optional 'openai' package" in text
                else "UNAVAILABLE"
            )
            return {
                "provider": "openai",
                "model": self.model,
                "classification": "cloud",
                "status": status,
                "reason": str(exc),
                "local_only": False,
                "capabilities": self.capabilities(),
            }
        return {
            "provider": "openai",
            "model": self.model,
            "classification": "cloud",
            "status": "CONFIGURED",
            "reason": (
                "Credentials and SDK are configured; model access is verified only "
                "by an explicit model discovery or selection command."
            ),
            "local_only": False,
            "capabilities": self.capabilities(),
        }

    def generate(self, prompt: str) -> dict:
        if not isinstance(prompt, str) or not prompt.strip():
            raise KaiError("A non-empty prompt is required.")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise KaiError(
                f"Prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit."
            )
        client = self._client_or_raise()
        try:
            response = client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
        except Exception as exc:
            raise _openai_error(exc, "text generation") from exc
        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            raise KaiError("OpenAI returned no text output.")
        return {
            "provider": "openai",
            "model": self.model,
            "response": text,
            "done": True,
            "local_only": False,
            "token_usage": _openai_usage(getattr(response, "usage", None)),
        }

    def generate_json(self, prompt: str, schema: dict) -> dict:
        if not isinstance(schema, dict):
            raise KaiError("Structured output schema must be a JSON object.")
        try:
            json.dumps(schema, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Structured output schema must be finite JSON: {exc}") from exc
        if not isinstance(prompt, str) or not prompt.strip():
            raise KaiError("A non-empty prompt is required.")
        if len(prompt) > MAX_PROMPT_CHARACTERS:
            raise KaiError(
                f"Prompt exceeds the {MAX_PROMPT_CHARACTERS}-character limit."
            )
        client = self._client_or_raise()
        try:
            response = client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=2_048,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "kai_structured_response",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            raise _openai_error(exc, "structured generation") from exc
        text = getattr(response, "output_text", None)
        if not isinstance(text, str):
            raise KaiError("OpenAI returned no structured text output.")
        try:
            structured = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KaiError("OpenAI returned malformed structured JSON.") from exc
        if not isinstance(structured, dict):
            raise KaiError("OpenAI structured output must be a JSON object.")
        return {
            "provider": "openai",
            "model": self.model,
            "response": text,
            "structured": structured,
            "done": True,
            "local_only": False,
            "token_usage": _openai_usage(getattr(response, "usage", None)),
        }


def _value(item: Any, key: str) -> Any:
    return item.get(key) if isinstance(item, dict) else getattr(item, key, None)


def _openai_usage(usage: Any) -> dict:
    if usage is None:
        return {}
    values = {
        "input_tokens": _value(usage, "input_tokens"),
        "output_tokens": _value(usage, "output_tokens"),
        "total_tokens": _value(usage, "total_tokens"),
    }
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
    }


def _openai_error(exc: Exception, operation: str) -> KaiError:
    name = type(exc).__name__
    if "timeout" in name.lower():
        return KaiError(f"OpenAI {operation} timed out ({name}).")
    return KaiError(f"OpenAI {operation} failed ({name}).")


class ModelGateway:
    """One selected-model interface for Ollama and explicitly enabled OpenAI."""

    def __init__(
        self,
        database,
        *,
        default_provider: str = "ollama",
        default_model: str = LOCAL_MODEL,
        openai_enabled: bool = False,
        openai_allow_source: bool = False,
        available_memory_bytes: int | None = None,
        read_only: bool = False,
        timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    ):
        self.database = database
        self.default_selection = {
            "provider": default_provider,
            "model_id": default_model,
        }
        self.openai_enabled = openai_enabled
        self.openai_allow_source = openai_allow_source
        self.available_memory_bytes = available_memory_bytes
        self.read_only = read_only
        self.timeout_seconds = timeout_seconds

    def selection(
        self,
        task_category: str = "general",
        *,
        explicit_target: str | None = None,
        read_only: bool = False,
    ) -> dict:
        if task_category not in TASK_CATEGORIES:
            raise KaiError(f"Unsupported LLM task category: {task_category!r}.")
        if explicit_target is not None:
            return self.parse_target(explicit_target)
        route = None
        if not read_only:
            route = self.database.get_llm_task_route(task_category)
        if route is not None:
            return {"provider": route["provider"], "model_id": route["model_id"]}
        selected = self.database.get_llm_model_selection(read_only=read_only)
        return selected or dict(self.default_selection)

    @staticmethod
    def parse_target(target: str) -> dict:
        if not isinstance(target, str) or "/" not in target:
            raise KaiError("Model target must use provider/model-id syntax.")
        provider, model_id = target.split("/", 1)
        if provider not in PROVIDER_CLASSIFICATIONS or not model_id.strip():
            raise KaiError(f"Unsupported model target: {target!r}.")
        if len(model_id) > 200 or any(ord(char) < 32 for char in model_id):
            raise KaiError("Model ID is invalid or exceeds the length limit.")
        return {"provider": provider, "model_id": model_id.strip()}

    def provider(self, target: dict):
        provider = target["provider"]
        model = target["model_id"]
        if provider == "ollama":
            return LocalOllamaProvider(model, timeout_seconds=self.timeout_seconds)
        if provider == "openai":
            return OpenAIProvider(
                model,
                enabled=self.openai_enabled,
                timeout_seconds=self.timeout_seconds,
            )
        if provider == "disabled":
            return DisabledLLMProvider(model)
        raise KaiError(f"Unsupported model provider: {provider!r}.")

    def _check_cloud_policy(self, target: dict, task_category: str) -> None:
        if PROVIDER_CLASSIFICATIONS.get(target["provider"]) != "cloud":
            return
        if target["provider"] != "openai":
            raise KaiError(
                f"Cloud provider {target['provider']!r} is not configured."
            )
        if not self.openai_enabled:
            raise KaiError(
                "Cloud routing is disabled. Explicitly set KAI_OPENAI_ENABLED=true "
                "and configure OPENAI_API_KEY before selecting OpenAI."
            )
        if task_category in {"coding", "code_review"} and not self.openai_allow_source:
            raise KaiError(
                "Cloud routing for source-code tasks is disabled. Explicitly set "
                "KAI_OPENAI_ALLOW_SOURCE=true to authorize sending source context."
            )

    def _status(self, target: dict) -> dict:
        provider = self.provider(target)
        result = provider.status()
        result["selected"] = True
        result["configuration_status"] = (
            "UNCONFIGURED"
            if result["status"] in {"DISABLED", "MISSING_API_KEY"}
            else "CONFIGURED"
        )
        models = result.get("available_models", [])
        selected_model = next(
            (
                item
                for item in models
                if item.get("model_id") == target["model_id"]
                or item.get("name") == target["model_id"]
            ),
            None,
        )
        if (
            target["provider"] == "ollama"
            and selected_model is not None
            and isinstance(selected_model.get("size_bytes"), int)
            and self.available_memory_bytes is not None
            and selected_model["size_bytes"] >= self.available_memory_bytes
        ):
            result["resource_warning"] = (
                "Installed weight size is near/exceeds available RAM; actual "
                "inference memory requirements are unknown."
            )
        return result

    def status(self) -> dict:
        selection = self.selection(read_only=True)
        status = self._status(selection)
        return {
            **status,
            "selection": selection,
            "task_routes": (
                [] if self.read_only else self.database.list_llm_task_routes()
            ),
            "fallbacks": (
                [] if self.read_only else self.database.list_llm_fallbacks()
            ),
            "resources": self._resource_report(),
        }

    def providers(self) -> list[dict]:
        current = self.selection(read_only=True)
        providers = []
        for provider_name in ("ollama", "openai"):
            target = {
                "provider": provider_name,
                "model_id": (
                    current["model_id"]
                    if current["provider"] == provider_name
                    else (LOCAL_MODEL if provider_name == "ollama" else "not-selected")
                ),
            }
            status = self._status(target)
            providers.append(
                {
                    "provider": provider_name,
                    "classification": "local" if provider_name == "ollama" else "cloud",
                    "active": current["provider"] == provider_name,
                    "status": status["status"],
                    "configuration_status": status["configuration_status"],
                    "capabilities": PROVIDER_CAPABILITIES[provider_name],
                    "local_only": provider_name == "ollama",
                }
            )
        providers.append(
            {
                "provider": "disabled",
                "classification": "none",
                "active": current["provider"] == "disabled",
                "status": "DISABLED",
                "configuration_status": "AVAILABLE",
                "capabilities": {},
                "local_only": True,
            }
        )
        return providers

    def models(self) -> list[dict]:
        current = self.selection(read_only=True)
        models: list[dict] = []
        ollama = LocalOllamaProvider(current["model_id"] if current["provider"] == "ollama" else LOCAL_MODEL)
        try:
            models.extend(ollama.list_models())
        except _OllamaConnectionError:
            models.append(
                self._unavailable_entry(
                    current
                    if current["provider"] == "ollama"
                    else {"provider": "ollama", "model_id": "unknown"},
                    "RUNTIME_UNAVAILABLE",
                    "local",
                )
            )
        except KaiError:
            raise
        if self.openai_enabled:
            openai = OpenAIProvider(
                current["model_id"] if current["provider"] == "openai" else "not-selected",
                enabled=True,
            )
            try:
                models.extend(openai.list_models())
            except KaiError as exc:
                message = str(exc).lower()
                status = (
                    "SDK_UNAVAILABLE"
                    if "optional 'openai' package" in message
                    else "MISSING_API_KEY"
                    if "openai_api_key" in message
                    else "UNAVAILABLE"
                )
                models.append(
                    self._unavailable_entry(
                        current
                        if current["provider"] == "openai"
                        else {"provider": "openai", "model_id": "unknown"},
                        status,
                        "cloud",
                    )
                )
        elif not any(item["provider"] == "openai" for item in models):
            models.append(
                self._unavailable_entry(
                    current
                    if current["provider"] == "openai"
                    else {"provider": "openai", "model_id": "not-selected"},
                    "DISABLED",
                    "cloud",
                )
            )
        for model in models:
            model["configured"] = (
                model["provider"] == current["provider"]
                and model["model_id"] == current["model_id"]
            )
            if model.get("available") is not False:
                model["availability_status"] = (
                    "INSTALLED" if model["provider"] == "ollama" else "AVAILABLE"
                )
            model["configuration_status"] = (
                "CONFIGURED" if model["configured"] else "DISCOVERED"
            )
            model["last_health_check_at"] = _utc_now()
            size = model.get("size_bytes")
            if (
                model["provider"] == "ollama"
                and isinstance(size, int)
                and self.available_memory_bytes is not None
            ):
                model["resource_warning"] = (
                    "Installed weight size is near/exceeds currently available RAM; "
                    "actual inference memory requirements are unknown."
                    if size >= self.available_memory_bytes
                    else None
                )
            else:
                model["resource_warning"] = None
        return models

    @staticmethod
    def _unavailable_entry(target: dict, status: str, classification: str) -> dict:
        return {
            "provider": target["provider"],
            "model_id": target["model_id"],
            "display_name": target["model_id"],
            "classification": classification,
            "status": status,
            "availability_status": status,
            "installed": None,
            "available": False,
            "configured": True,
            "resource_requirements": None,
            "resource_warning": None,
            "capabilities": PROVIDER_CAPABILITIES.get(target["provider"], {}),
            "configuration_status": "CONFIGURED",
            "last_health_check_at": _utc_now(),
        }

    def _resource_report(self) -> dict:
        available = self.available_memory_bytes
        try:
            cpu_count = os.cpu_count()
        except OSError:
            cpu_count = None
        return {
            "available_memory_bytes": available,
            "logical_cpu_count": cpu_count,
            "gpu_inference": "UNKNOWN; no inference-runtime capability probe exists",
            "concurrent_local_models": 1,
            "resource_requirements": "Model-specific memory requirements are unknown.",
        }

    def active(self, *, read_only: bool = False) -> dict:
        selection = self.selection(read_only=read_only)
        status = self._status(selection)
        persisted = self.database.get_llm_model_selection(read_only=read_only)
        routes = [] if read_only else self.database.list_llm_task_routes()
        fallbacks = [] if read_only else self.database.list_llm_fallbacks()
        return {
            "selection": selection,
            "selection_source": "persistent" if persisted else "environment_default",
            "status": status,
            "task_routes": routes,
            "fallbacks": fallbacks,
            "cloud_authorization": {
                "openai_enabled": self.openai_enabled,
                "source_code_authorized": self.openai_allow_source,
            },
        }

    def validate_selection(
        self, target: dict, *, task_category: str = "general"
    ) -> dict:
        if target["provider"] == "disabled":
            raise KaiError("The disabled provider cannot be selected as a model.")
        self._check_cloud_policy(target, task_category)
        status = self._status(target)
        if target["provider"] == "openai" and status["status"] == "CONFIGURED":
            models = self.provider(target).list_models()
            status["available_models"] = models
            status["status"] = (
                "READY"
                if any(item["model_id"] == target["model_id"] for item in models)
                else "MODEL_UNAVAILABLE"
            )
        if status["status"] != "READY":
            raise KaiError(
                f"Cannot select {target['provider']}/{target['model_id']}: "
                f"status is {status['status']}. {status.get('reason') or ''}".strip()
            )
        return status

    def generate(
        self,
        prompt: str,
        *,
        task_category: str = "general",
        explicit_target: str | None = None,
        local_only: bool = False,
    ) -> dict:
        return self._generate(
            prompt,
            task_category=task_category,
            explicit_target=explicit_target,
            local_only=local_only,
            schema=None,
        )

    def generate_json(
        self,
        prompt: str,
        schema: dict,
        *,
        task_category: str = "structured_output",
        explicit_target: str | None = None,
        local_only: bool = False,
    ) -> dict:
        return self._generate(
            prompt,
            task_category=task_category,
            explicit_target=explicit_target,
            local_only=local_only,
            schema=schema,
        )

    def generate_json_for_task(
        self, prompt: str, schema: dict, task_category: str
    ):
        return self.generate_json(
            prompt, schema, task_category=task_category
        )

    def _generate(
        self,
        prompt: str,
        *,
        task_category: str,
        explicit_target: str | None,
        local_only: bool,
        schema: dict | None,
    ) -> dict:
        if task_category not in TASK_CATEGORIES:
            raise KaiError(f"Unsupported LLM task category: {task_category!r}.")
        requested = self.selection(
            task_category, explicit_target=explicit_target
        )
        if requested["provider"] == "disabled":
            raise KaiError("Model access is disabled by the active configuration.")
        targets = [requested]
        if explicit_target is None:
            seen = {(requested["provider"], requested["model_id"])}
            for item in self.database.list_llm_fallbacks():
                identity = (item["provider"], item["model_id"])
                if identity not in seen:
                    targets.append(
                        {"provider": item["provider"], "model_id": item["model_id"]}
                    )
                    seen.add(identity)
        if local_only:
            if PROVIDER_CLASSIFICATIONS[requested["provider"]] != "local":
                raise KaiError(
                    "The selected model is cloud-hosted, but this request requires "
                    "local-only routing."
                )
            targets = [
                target
                for target in targets
                if PROVIDER_CLASSIFICATIONS[target["provider"]] == "local"
            ]
        self._check_cloud_policy(requested, task_category)
        fallback_from: list[dict] = []
        last_category = None
        for target in targets:
            started = time.monotonic()
            started_at = _utc_now()
            request_id = str(uuid.uuid4())
            try:
                self._check_cloud_policy(target, task_category)
                provider = self.provider(target)
                if PROVIDER_CLASSIFICATIONS[target["provider"]] == "local":
                    with _local_inference_lock(
                        self.database.path.parent / ".llm-inference.lock"
                    ):
                        response = (
                            provider.generate(prompt)
                            if schema is None
                            else provider.generate_json(prompt, schema)
                        )
                else:
                    response = (
                        provider.generate(prompt)
                        if schema is None
                        else provider.generate_json(prompt, schema)
                    )
                completed_at = _utc_now()
                duration_ms = max(0, int((time.monotonic() - started) * 1000))
                usage = response.get("token_usage", {})
                self.database.record_llm_request(
                    {
                        "request_id": request_id,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_ms": duration_ms,
                        "task_category": task_category,
                        "provider": target["provider"],
                        "model_id": target["model_id"],
                        "outcome": "SUCCESS",
                        "token_usage": usage,
                        "fallback_from": fallback_from,
                    }
                )
                return {
                    **response,
                    "request_id": request_id,
                    "task_category": task_category,
                    "requested_model": requested,
                    "actual_model": target,
                    "routing": {
                        "used_fallback": bool(fallback_from),
                        "fallback_from": fallback_from,
                    },
                }
            except KaiError as exc:
                completed_at = _utc_now()
                duration_ms = max(0, int((time.monotonic() - started) * 1000))
                last_category = _error_category(exc)
                self.database.record_llm_request(
                    {
                        "request_id": request_id,
                        "started_at": started_at,
                        "completed_at": completed_at,
                        "duration_ms": duration_ms,
                        "task_category": task_category,
                        "provider": target["provider"],
                        "model_id": target["model_id"],
                        "outcome": "FAILED",
                        "token_usage": {},
                        "error_category": last_category,
                        "fallback_from": fallback_from,
                    }
                )
                fallback_from.append(
                    {
                        "provider": target["provider"],
                        "model_id": target["model_id"],
                        "reason": last_category,
                    }
                )
                if explicit_target is not None or target == targets[-1]:
                    raise
        raise KaiError(
            "All configured model attempts failed. Last failure category: "
            f"{last_category or 'provider_unavailable'}. Review `kai llm history` "
            "and provider status; no unconfigured provider was contacted."
        )


def _error_category(error: KaiError) -> str:
    message = str(error).lower()
    if "timed out" in message or "timeout" in message:
        return "timeout"
    if "not installed" in message or "unavailable" in message:
        return "unavailable"
    if "disabled" in message or "not configured" in message or "missing" in message:
        return "configuration"
    if "invalid" in message or "malformed" in message:
        return "invalid_response"
    return "provider_error"


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _local_inference_lock(lock_path):
    with _LOCAL_INFERENCE_LOCK:
        if os.name != "posix":
            yield
            return
        descriptor = None
        try:
            import fcntl

            lock_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            raise KaiError(f"Cannot acquire local inference lock: {exc}") from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


class DisabledLLMProvider:
    def __init__(self, model: str):
        self.model = model

    def status(self) -> dict:
        return {
            "provider": "disabled",
            "model": self.model,
            "status": "DISABLED",
            "local_only": True,
            "capabilities": {},
        }

    def list_models(self) -> list[dict]:
        return []

    @staticmethod
    def capabilities() -> dict:
        return {}

    def generate(self, prompt: str) -> dict:
        raise KaiError("Local language model provider is disabled by configuration.")

    def generate_json(self, prompt: str, schema: dict) -> dict:
        raise KaiError("Local language model provider is disabled by configuration.")
