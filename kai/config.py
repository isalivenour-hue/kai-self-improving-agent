import os
from dataclasses import dataclass
from pathlib import Path

from kai.errors import KaiError


@dataclass(frozen=True)
class KaiConfig:
    data_dir: Path
    llm_provider: str = "ollama"
    llm_model: str = "qwen2.5:0.5b"
    openai_enabled: bool = False
    openai_allow_source: bool = False
    llm_timeout_seconds: int = 120

    @classmethod
    def from_environment(cls) -> "KaiConfig":
        configured_home = os.environ.get("KAI_HOME")
        data_dir = (
            Path(configured_home).expanduser()
            if configured_home
            else Path.home() / ".local" / "share" / "kai"
        )
        provider = os.environ.get("KAI_LLM_PROVIDER", "ollama").strip().lower()
        model = os.environ.get("KAI_LLM_MODEL", "qwen2.5:0.5b").strip()
        if provider not in {"ollama", "openai", "disabled"}:
            raise KaiError(
                "KAI_LLM_PROVIDER must be 'ollama', 'openai', or 'disabled'."
            )
        if not model:
            raise KaiError("KAI_LLM_MODEL must be non-empty.")
        openai_enabled = os.environ.get("KAI_OPENAI_ENABLED", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        openai_allow_source = (
            os.environ.get("KAI_OPENAI_ALLOW_SOURCE", "").strip().lower()
            in {"1", "true", "yes"}
        )
        timeout_value = os.environ.get("KAI_LLM_TIMEOUT_SECONDS", "120").strip()
        try:
            timeout = int(timeout_value)
        except ValueError as exc:
            raise KaiError("KAI_LLM_TIMEOUT_SECONDS must be an integer.") from exc
        if not 1 <= timeout <= 600:
            raise KaiError("KAI_LLM_TIMEOUT_SECONDS must be between 1 and 600.")
        return cls(
            data_dir=data_dir,
            llm_provider=provider,
            llm_model=model,
            openai_enabled=openai_enabled,
            openai_allow_source=openai_allow_source,
            llm_timeout_seconds=timeout,
        )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "kai.sqlite3"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "kai.log"
