"""Managed filesystem snapshots and explicit, audited candidate source edits."""

from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Protocol

from kai.errors import KaiError

WORKSPACE_VERSION = "0.005.1"
MAX_WORKSPACE_BYTES = 32 * 1024 * 1024
MAX_SOURCE_FILE_BYTES = 1024 * 1024
ALLOWED_SOURCE_ROOTS = ("kai", "tests")
PROTECTED_CANDIDATE_FILES = {
    "kai/candidate_workspace.py",
    "kai/candidate_runner.py",
    "kai/evaluation_protocol.py",
}
_SECRET_NAME_PARTS = ("secret", "credential", "token", "password", "id_rsa", "id_ed25519")
_SECRET_VALUE = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"]{16,}['\"]"
)


class CandidateWorkspaceError(KaiError):
    pass


class ImprovementGenerator(Protocol):
    def propose_change(
        self,
        self_model: dict,
        opportunity: dict,
        hypothesis: dict,
        proposal: dict,
        candidate_context: dict,
    ) -> dict:
        """Return an explicit proposed file operation; never writes source."""


@dataclass(frozen=True)
class CandidateModification:
    modification_id: str
    candidate_id: str
    workspace_id: str
    target_file: str
    operation: str
    original_sha256: str | None
    resulting_sha256: str | None
    patch: str
    reason: str
    hypothesis_id: int
    validation_status: str
    created_at: str


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_json(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class CandidateWorkspaceManager:
    """Create and operate on snapshots confined to KAI-owned state storage."""

    def __init__(self, source_root: Path, state_dir: Path, database):
        self.source_root = source_root.resolve(strict=True)
        self.state_dir = state_dir.resolve()
        self.workspace_root = self.state_dir / "candidate-workspaces"
        self.database = database
        if self._is_within(self.workspace_root.resolve(), self.source_root):
            raise CandidateWorkspaceError(
                "Candidate workspaces must be outside the production source tree."
            )

    @staticmethod
    def _is_within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _ensure_workspace_root(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.workspace_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.workspace_root.is_symlink():
            raise CandidateWorkspaceError("Candidate workspace root cannot be a symlink.")
        resolved_root = self.workspace_root.resolve(strict=True)
        if self._is_within(resolved_root, self.source_root):
            raise CandidateWorkspaceError(
                "Candidate workspace root resolves inside production source."
            )
        os.chmod(self.workspace_root, 0o700)

    def _source_revision(self) -> str | None:
        if not (self.source_root / ".git").exists():
            return None
        git = shutil.which("git")
        if not git:
            return None
        try:
            result = subprocess.run(
                [git, "-C", str(self.source_root), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
                env={"PATH": os.path.dirname(git), "HOME": str(self.state_dir)},
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        revision = result.stdout.strip()
        return revision if result.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40,64}", revision) else None

    def _copy_snapshot(self, destination: Path) -> str:
        digest = hashlib.sha256()
        for name in ALLOWED_SOURCE_ROOTS:
            source = self.source_root / name
            if not source.is_dir() or source.is_symlink():
                raise CandidateWorkspaceError(f"Required source directory {name!r} is missing or unsafe.")
            target = destination / name
            target.mkdir(mode=0o700)
            for path in sorted(source.rglob("*")):
                relative = path.relative_to(source)
                if any(part in {".git", "__pycache__", ".venv", "venv"} for part in relative.parts):
                    continue
                target_path = target / relative
                if path.is_symlink():
                    raise CandidateWorkspaceError(
                        f"Source snapshot contains a symbolic link: {name}/{relative.as_posix()}"
                    )
                if path.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True, mode=0o700)
                elif path.is_file():
                    data = path.read_bytes()
                    if len(data) > MAX_SOURCE_FILE_BYTES:
                        raise CandidateWorkspaceError(
                            f"Source file exceeds the snapshot limit: {name}/{relative.as_posix()}"
                        )
                    target_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    target_path.write_bytes(data)
                    os.chmod(target_path, 0o600)
                    relative_name = f"{name}/{relative.as_posix()}"
                    digest.update(relative_name.encode())
                    digest.update(b"\0")
                    digest.update(data)
        if not (destination / "kai" / "__init__.py").is_file():
            raise CandidateWorkspaceError("Snapshot is missing kai/__init__.py.")
        if not (destination / "tests").is_dir():
            raise CandidateWorkspaceError("Snapshot is missing the test tree.")
        return digest.hexdigest()

    def create(
        self,
        *,
        candidate_id: str,
        parent_kai_version: str,
        source_revision: str | None,
        schema_version: int,
        environment_fingerprint: str | None,
        configuration: dict,
    ) -> dict:
        self._ensure_workspace_root()
        workspace_id = str(uuid.uuid4())
        path = self.workspace_root / workspace_id
        path.mkdir(mode=0o700)
        try:
            source_hash = self._copy_snapshot(path)
            source_size = sum(
                item.stat().st_size for item in path.rglob("*") if item.is_file()
            )
            if source_size > MAX_WORKSPACE_BYTES:
                raise CandidateWorkspaceError("Source snapshot exceeds workspace size limit.")
            safe_configuration = {
                key: value
                for key, value in configuration.items()
                if not any(part in key.lower() for part in _SECRET_NAME_PARTS)
            }
            config_digest = _sha256(_safe_json(safe_configuration).encode())
            git_revision = source_revision or self._source_revision()
            metadata = {
                "workspace_version": WORKSPACE_VERSION,
                "snapshot_backend": (
                    "git-revision+filesystem-snapshot"
                    if git_revision
                    else "filesystem-snapshot"
                ),
                "source_revision": git_revision,
                "source_revision_status": "OBSERVED" if git_revision else "UNKNOWN",
                "source_tree_sha256": source_hash,
                "configuration_sha256": config_digest,
                "database_schema_version": schema_version,
                "python_version": sys.version,
                "python_executable": Path(sys.executable).name,
                "operating_system": platform.platform(),
                "environment_fingerprint": environment_fingerprint,
                "source_bytes": source_size,
                "network_policy": "disabled by bubblewrap namespace",
                "candidate_permissions": "workspace owned by current user; source files read-only during runner execution",
            }
            record = self.database.create_candidate_workspace(
                workspace_id=workspace_id,
                candidate_id=candidate_id,
                relative_path=workspace_id,
                source_revision=git_revision,
                source_tree_sha256=source_hash,
                configuration_sha256=config_digest,
                schema_version=schema_version,
                python_version=sys.version,
                operating_system=platform.platform(),
                environment_fingerprint=environment_fingerprint,
                metadata=metadata,
            )
            return {**record, "path": str(path)}
        except Exception:
            if path.exists() and not path.is_symlink():
                shutil.rmtree(path)
            raise

    def preview(
        self, *, candidate_id: str, configuration: dict
    ) -> dict:
        self.database.get_kai_candidate(candidate_id)
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        safe_configuration = {
            key: value
            for key, value in configuration.items()
            if not any(part in key.lower() for part in _SECRET_NAME_PARTS)
        }
        config_digest = _sha256(_safe_json(safe_configuration).encode())
        with tempfile.TemporaryDirectory(
            prefix=".kai-workspace-preview-", dir=self.state_dir
        ) as temporary_path:
            preview_root = Path(temporary_path)
            source_hash = self._copy_snapshot(preview_root)
            source_size = sum(
                item.stat().st_size
                for item in preview_root.rglob("*")
                if item.is_file()
            )
            if source_size > MAX_WORKSPACE_BYTES:
                raise CandidateWorkspaceError(
                    "Source snapshot exceeds workspace size limit."
                )
        git_revision = self._source_revision()
        return {
            "candidate_id": candidate_id,
            "status": "READY",
            "workspace_created": False,
            "source_tree_sha256": source_hash,
            "configuration_sha256": config_digest,
            "source_revision": git_revision,
            "source_revision_status": "OBSERVED" if git_revision else "UNKNOWN",
            "source_bytes": source_size,
            "workspace_version": WORKSPACE_VERSION,
        }

    def path_for(self, workspace_id: str) -> Path:
        workspace = self.database.get_candidate_workspace(workspace_id)
        candidate_path = (self.workspace_root / workspace["relative_path"]).resolve(strict=True)
        root = self.workspace_root.resolve(strict=True)
        if not self._is_within(candidate_path, root) or candidate_path == root:
            raise CandidateWorkspaceError("Workspace path escaped its managed root.")
        if candidate_path.is_symlink() or not candidate_path.is_dir():
            raise CandidateWorkspaceError("Candidate workspace is missing or unsafe.")
        return candidate_path

    @staticmethod
    def _target_path(root: Path, relative_name: str) -> Path:
        if (
            not isinstance(relative_name, str)
            or not relative_name
            or "\\" in relative_name
            or "\x00" in relative_name
        ):
            raise CandidateWorkspaceError("Candidate target must be a safe relative path.")
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise CandidateWorkspaceError("Candidate target cannot contain traversal or absolute paths.")
        if relative.parts[0] not in ALLOWED_SOURCE_ROOTS or relative.suffix != ".py":
            raise CandidateWorkspaceError("Only Python files under kai/ or tests/ may be modified.")
        if relative_name in PROTECTED_CANDIDATE_FILES:
            raise CandidateWorkspaceError("Candidate development infrastructure is protected.")
        if any(part.startswith(".") or any(secret in part.lower() for secret in _SECRET_NAME_PARTS)
               for part in relative.parts):
            raise CandidateWorkspaceError("Protected or secret-like paths cannot be modified.")
        target = root.joinpath(*relative.parts)
        current = root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise CandidateWorkspaceError("Candidate path cannot traverse a symbolic link.")
        resolved_parent = target.parent.resolve()
        if not CandidateWorkspaceManager._is_within(resolved_parent, root.resolve()):
            raise CandidateWorkspaceError("Candidate target escapes its workspace.")
        return target

    @staticmethod
    def _workspace_source_digest(root: Path) -> str:
        digest = hashlib.sha256()
        for source_root in ALLOWED_SOURCE_ROOTS:
            for path in sorted((root / source_root).rglob("*.py")):
                if path.is_symlink():
                    raise CandidateWorkspaceError("Candidate source may not contain symlinks.")
                relative = path.relative_to(root).as_posix()
                data = path.read_bytes()
                digest.update(relative.encode())
                digest.update(b"\0")
                digest.update(data)
        return digest.hexdigest()

    def validate(self, workspace_id: str) -> dict:
        root = self.path_for(workspace_id)
        errors = []
        required = ("kai/__init__.py", "kai/cli.py", "tests")
        for name in required:
            if not (root / name).exists():
                errors.append(f"Required candidate path is missing: {name}")
        total = 0
        python_files = []
        for name in ALLOWED_SOURCE_ROOTS:
            for path in (root / name).rglob("*"):
                if path.is_symlink():
                    errors.append(f"Symbolic link is not allowed: {path.relative_to(root)}")
                    continue
                if path.is_file():
                    total += path.stat().st_size
                    if path.suffix == ".py":
                        python_files.append(path)
        if total > MAX_WORKSPACE_BYTES:
            errors.append("Candidate exceeds the workspace byte limit.")
        for path in python_files:
            try:
                source = path.read_text(encoding="utf-8")
                ast.parse(source, filename=path.relative_to(root).as_posix())
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                errors.append(f"Invalid Python source {path.relative_to(root)}: {exc}")
        status = "VALID" if not errors else "INVALID"
        return {
            "workspace_id": workspace_id,
            "status": status,
            "errors": errors,
            "python_file_count": len(python_files),
            "source_tree_sha256": self._workspace_source_digest(root) if not errors else None,
            "source_bytes": total,
        }

    def apply_change(
        self,
        *,
        workspace_id: str,
        target_file: str,
        operation: str,
        content: str | None,
        reason: str,
        hypothesis_id: int,
    ) -> dict:
        if operation not in {"ADD", "MODIFY", "DELETE"}:
            raise CandidateWorkspaceError("Modification operation must be ADD, MODIFY, or DELETE.")
        if not isinstance(reason, str) or not reason.strip():
            raise CandidateWorkspaceError("Modification requires a reason.")
        if isinstance(hypothesis_id, bool) or not isinstance(hypothesis_id, int) or hypothesis_id < 1:
            raise CandidateWorkspaceError("Modification requires a valid hypothesis ID.")
        root = self.path_for(workspace_id)
        workspace = self.database.get_candidate_workspace(workspace_id)
        if workspace["status"] in {"PRESERVED", "CLEANED"}:
            raise CandidateWorkspaceError(
                "Preserved or cleaned candidate workspaces are immutable."
            )
        target = self._target_path(root, target_file)
        original = target.read_bytes() if target.exists() else None
        if operation == "ADD" and original is not None:
            raise CandidateWorkspaceError("ADD target already exists.")
        if operation in {"MODIFY", "DELETE"} and original is None:
            raise CandidateWorkspaceError(f"{operation} target does not exist.")
        if operation == "DELETE":
            resulting = None
        else:
            if not isinstance(content, str):
                raise CandidateWorkspaceError(f"{operation} requires UTF-8 Python source content.")
            if len(content.encode("utf-8")) > MAX_SOURCE_FILE_BYTES:
                raise CandidateWorkspaceError("Candidate source file exceeds the size limit.")
            if _SECRET_VALUE.search(content):
                raise CandidateWorkspaceError("Candidate source appears to contain a hard-coded secret.")
            resulting = content.encode("utf-8")
            try:
                ast.parse(content, filename=target_file)
            except SyntaxError as exc:
                raise CandidateWorkspaceError(f"Candidate Python syntax is invalid: {exc}") from exc
        original_text = original.decode("utf-8") if original is not None else ""
        result_text = resulting.decode("utf-8") if resulting is not None else ""
        patch = "".join(
            difflib.unified_diff(
                original_text.splitlines(keepends=True),
                result_text.splitlines(keepends=True),
                fromfile=f"a/{target_file}" if original is not None else "/dev/null",
                tofile=f"b/{target_file}" if resulting is not None else "/dev/null",
            )
        )
        modification_id = str(uuid.uuid4())
        now = _timestamp()
        original_hash = _sha256(original) if original is not None else None
        resulting_hash = _sha256(resulting) if resulting is not None else None
        if operation == "DELETE":
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = target.with_name(f".{target.name}.{modification_id}.tmp")
            try:
                with temporary.open("xb") as stream:
                    stream.write(resulting)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o600)
                temporary.replace(target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        validation = self.validate(workspace_id)
        if validation["status"] == "INVALID":
            if original is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(original)
            validation_status = "INVALID"
        else:
            validation_status = "VALID"
        try:
            record = self.database.record_candidate_modification(
                modification_id=modification_id,
                candidate_id=self.database.get_candidate_workspace(workspace_id)["candidate_id"],
                workspace_id=workspace_id,
                target_file=target_file,
                operation=operation,
                original_sha256=original_hash,
                resulting_sha256=resulting_hash,
                patch=patch,
                reason=reason,
                hypothesis_id=hypothesis_id,
                validation_status=validation_status,
                timestamp=now,
            )
        except Exception:
            if validation_status == "VALID":
                if original is None:
                    target.unlink(missing_ok=True)
                else:
                    target.write_bytes(original)
            raise
        if validation_status != "VALID":
            raise CandidateWorkspaceError(
                "Modification was rejected because it invalidated the candidate tree."
            )
        if workspace["status"] in {"TESTED", "FAILED"}:
            self.database.update_candidate_workspace_status(
                workspace_id,
                "VALIDATED",
                {"reason": "Source changed after the previous test state."},
            )
        return {**record, "workspace_validation": validation}

    def diff(self, workspace_id: str) -> list[dict]:
        root = self.path_for(workspace_id)
        modifications = self.database.list_candidate_modifications(workspace_id)
        result = []
        for modification in modifications:
            current = self._target_path(root, modification["target_file"])
            current_hash = _sha256(current.read_bytes()) if current.exists() else None
            result.append(
                {
                    **modification,
                    "current_sha256": current_hash,
                    "matches_recorded_result": current_hash
                    == modification["resulting_sha256"],
                }
            )
        return result

    def cleanup(self, workspace_id: str, preserve: bool = False) -> dict:
        workspace = self.database.get_candidate_workspace(workspace_id)
        candidate = self.database.get_kai_candidate(workspace["candidate_id"])
        latest_decision = self.database.latest_candidate_decision(
            workspace["candidate_id"]
        )
        if preserve or (
            latest_decision and latest_decision["decision"] == "ACCEPT_CANDIDATE"
        ):
            self.database.update_candidate_workspace_status(
                workspace_id, "PRESERVED", {"reason": "accepted candidate artifact"}
            )
            return self.database.get_candidate_workspace(workspace_id)
        if candidate["deployment_status"] != "NOT_DEPLOYED":
            raise CandidateWorkspaceError("Cannot clean a candidate with deployment state.")
        path = self.path_for(workspace_id)
        root = self.workspace_root.resolve(strict=True)
        if path == root or not self._is_within(path.resolve(strict=True), root):
            raise CandidateWorkspaceError("Refusing cleanup outside the managed workspace root.")
        shutil.rmtree(path)
        self.database.update_candidate_workspace_status(
            workspace_id, "CLEANED", {"reason": "explicit workspace cleanup"}
        )
        return self.database.get_candidate_workspace(workspace_id)
