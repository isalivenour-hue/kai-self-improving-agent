"""Fixed candidate test/evaluation runner constrained by bubblewrap."""

from __future__ import annotations

import json
import os
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from kai.errors import KaiError
from kai.evaluation_protocol import CPU_PREDICTOR_PROTOCOL

try:
    import resource
except ImportError:
    resource = None

MAX_TIMEOUT_SECONDS = 180
MAX_OUTPUT_BYTES = 128 * 1024
MAX_SANDBOX_FILE_BYTES = 64 * 1024 * 1024
MAX_INPUT_BYTES = 1024 * 1024
MAX_ADDRESS_SPACE_BYTES = 768 * 1024 * 1024
MAX_CPU_SECONDS = 150

_PREDICTOR_WORKER = r'''
import importlib.util
import json
import math
import sys

def main():
    with open("/dev/stdin", "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    spec = importlib.util.spec_from_file_location(
        "kai_candidate_predictor", "/usr/share/" + payload["candidate_module"]
    )
    if spec is None or spec.loader is None:
        raise ValueError("Candidate predictor module is unavailable.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    forecast = getattr(module, "forecast", None)
    if not callable(forecast):
        raise ValueError("Candidate must define forecast(history).")
    output = {"baseline_errors": [], "candidate_errors": []}
    for sequence in payload["sequences"]:
        baseline_errors = []
        candidate_errors = []
        for index in range(payload["warmup"], len(sequence)):
            history = sequence[:index]
            actual = sequence[index]
            baseline_prediction = math.fsum(history) / len(history)
            candidate_prediction = forecast(list(history))
            for name, value in (
                ("baseline", baseline_prediction),
                ("candidate", candidate_prediction),
            ):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(name + " forecast must be numeric.")
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(name + " forecast must be finite and positive.")
            baseline_errors.append(abs(baseline_prediction - actual))
            candidate_errors.append(abs(candidate_prediction - actual))
        output["baseline_errors"].append(baseline_errors)
        output["candidate_errors"].append(candidate_errors)
    sys.stdout.write(json.dumps(output, allow_nan=False, sort_keys=True))

main()
'''


@dataclass(frozen=True)
class CandidateExecutionResult:
    execution_id: str
    candidate_id: str
    workspace_id: str
    procedure_id: str
    status: str
    started_at: str
    completed_at: str
    duration_seconds: float
    exit_code: int | None
    stdout: str
    stderr: str
    environment: dict
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terminate_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        _terminate_process_group(process)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        process.wait()
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None and not stream.closed:
            stream.close()


def _limits() -> None:
    if resource is None:
        raise OSError("Operating-system resource limits are unavailable.")
    resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    resource.setrlimit(
        resource.RLIMIT_AS, (MAX_ADDRESS_SPACE_BYTES, MAX_ADDRESS_SPACE_BYTES)
    )
    # Candidate source is mounted read-only; this bounds only ephemeral files
    # such as SQLite test state inside the sandbox.
    resource.setrlimit(
        resource.RLIMIT_FSIZE,
        (MAX_SANDBOX_FILE_BYTES, MAX_SANDBOX_FILE_BYTES),
    )
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if hasattr(resource, "RLIMIT_NPROC"):
        uid = os.getuid()
        task_count = 1
        for process_path in Path("/proc").iterdir():
            if not process_path.name.isdecimal():
                continue
            try:
                status = (process_path / "status").read_text(encoding="ascii")
            except FileNotFoundError:
                continue
            for line in status.splitlines():
                if line.startswith("Uid:"):
                    if int(line.split()[1]) == uid:
                        try:
                            task_count += len(
                                list((process_path / "task").iterdir())
                            )
                        except FileNotFoundError:
                            pass
                    break
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
        desired_limit = max(64, task_count + 8)
        if hard_limit != resource.RLIM_INFINITY:
            desired_limit = min(desired_limit, hard_limit)
        if soft_limit == resource.RLIM_INFINITY or desired_limit < soft_limit:
            resource.setrlimit(
                resource.RLIMIT_NPROC, (desired_limit, desired_limit)
            )


class CandidateRunner:
    def __init__(self, database, workspace_manager):
        self.database = database
        self.workspace_manager = workspace_manager

    @staticmethod
    def sandbox_available() -> bool:
        return platform.system() == "Linux" and shutil.which("bwrap") is not None

    @staticmethod
    def probe_sandbox() -> dict:
        bwrap = shutil.which("bwrap")
        if platform.system() != "Linux" or bwrap is None:
            return {
                "binary_status": "UNAVAILABLE",
                "probe_status": "UNKNOWN",
                "reason": "Linux bubblewrap is not discoverable on PATH.",
            }
        command = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind",
            "/",
            "/",
            "--tmpfs",
            "/etc",
            "--tmpfs",
            "/home",
            "--tmpfs",
            "/root",
            "--tmpfs",
            "/var",
            "--size",
            str(64 * 1024 * 1024),
            "--tmpfs",
            "/tmp",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--ro-bind",
            "/sys",
            "/sys",
            "--",
            sys.executable,
            "-c",
            "pass",
        ]
        try:
            result = subprocess.run(
                command,
                cwd="/",
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": "/tmp/candidate-home",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "LC_ALL": "C",
                },
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {
                "binary_status": "AVAILABLE",
                "probe_status": "UNAVAILABLE",
                "reason": f"Sandbox readiness probe could not complete: {exc}",
            }
        return {
            "binary_status": "AVAILABLE",
            "probe_status": "READY" if result.returncode == 0 else "UNAVAILABLE",
            "return_code": result.returncode,
            "reason": (
                None
                if result.returncode == 0
                else result.stderr[:1024].decode("utf-8", errors="replace").strip()
            ),
        }

    def _sandbox_command(self, workspace_path: Path, python_args: list[str]) -> list[str]:
        bwrap = shutil.which("bwrap")
        if not bwrap or platform.system() != "Linux":
            raise KaiError("Required bubblewrap Linux sandbox is unavailable.")
        command = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-net",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--ro-bind",
            "/",
            "/",
            "--ro-bind",
            str(workspace_path),
            "/usr/share",
        ]
        for sensitive_path in (
            "/etc",
            "/home",
            "/root",
            "/var",
            "/tmp",
            "/run",
            "/mnt",
            "/media",
            "/boot",
            "/opt",
            "/srv",
        ):
            if Path(sensitive_path).exists():
                mount_size = (
                    64 * 1024 * 1024
                    if sensitive_path == "/tmp"
                    else 4 * 1024 * 1024
                )
                command.extend(["--size", str(mount_size)])
                command.extend(["--tmpfs", sensitive_path])
        command.extend(
            [
                "--chmod",
                "1777",
                "/tmp",
                "--dev",
                "/dev",
                "--size",
                str(16 * 1024 * 1024),
                "--tmpfs",
                "/dev/shm",
                "--proc",
                "/proc",
                "--ro-bind",
                "/sys",
                "/sys",
                "--chdir",
                "/usr/share",
                "--setenv",
                "HOME",
                "/tmp/candidate-home",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "PATH",
                "/usr/bin:/bin",
                "--setenv",
                "PYTHONPATH",
                "/usr/share",
                "--setenv",
                "PYTHONDONTWRITEBYTECODE",
                "1",
                "--setenv",
                "KAI_HOME",
                "/tmp/candidate-state",
                "--setenv",
                "LC_ALL",
                "C",
                "--",
                sys.executable,
                *python_args,
            ]
        )
        return command

    @staticmethod
    def _module_args(module: str | None) -> list[str]:
        if module is None:
            return ["-m", "unittest", "discover", "-s", "tests", "-v"]
        if (
            not isinstance(module, str)
            or not re.fullmatch(r"tests\.test_[a-zA-Z0-9_]+", module)
        ):
            raise KaiError("Candidate test selection must be one tests.test_<name> module.")
        return ["-m", "unittest", module, "-v"]

    @staticmethod
    def _drain_process(
        process: subprocess.Popen,
        input_bytes: bytes,
        timeout_seconds: int,
    ) -> tuple[str, bytes, bytes, str | None]:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise KaiError("Candidate runner pipes could not be initialized.")
        if len(input_bytes) > MAX_INPUT_BYTES:
            _stop_process(process)
            return "RESOURCE_LIMIT", b"", b"", "Runner input exceeds size limit."
        process.stdin.write(input_bytes)
        process.stdin.close()
        selector = selectors.DefaultSelector()
        os.set_blocking(process.stdout.fileno(), False)
        os.set_blocking(process.stderr.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        collected = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = time.monotonic() + timeout_seconds
        status = "SUCCESS"
        reason = None

        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "TIMEOUT"
                reason = "Candidate process exceeded its wall-clock budget."
                _terminate_process_group(process)
                break
            if process.poll() is not None and not selector.select(0):
                break
            for key, _ in selector.select(min(0.1, remaining)):
                try:
                    chunk = os.read(key.fileobj.fileno(), 8192)
                except OSError:
                    chunk = b""
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffer = collected[key.data]
                buffer.extend(chunk)
                if sum(map(len, collected.values())) > MAX_OUTPUT_BYTES:
                    status = "RESOURCE_LIMIT"
                    reason = "Candidate output exceeded the configured byte limit."
                    _terminate_process_group(process)
                    break
            if status != "SUCCESS":
                break
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            process.wait()
            status = "TIMEOUT"
            reason = reason or "Candidate process did not terminate after cancellation."
        selector.close()
        stdout = bytes(collected["stdout"])
        stderr = bytes(collected["stderr"])
        process.stdout.close()
        process.stderr.close()
        if status == "SUCCESS" and process.returncode != 0:
            if stderr.startswith(b"bwrap:"):
                status = "UNKNOWN"
                reason = "Candidate sandbox setup failed: " + stderr[:1024].decode(
                    "utf-8", errors="replace"
                ).strip()
            else:
                status = (
                    "FAILED"
                    if process.returncode and process.returncode > 0
                    else "CRASHED"
                )
                reason = f"Candidate process exited with status {process.returncode}."
        return status, stdout, stderr, reason

    def _run(
        self,
        *,
        candidate_id: str,
        workspace_id: str,
        procedure_id: str,
        python_args: list[str],
        input_payload: dict | None = None,
        timeout_seconds: int = 120,
        persist_result: bool = True,
    ) -> CandidateExecutionResult:
        execution_id = str(uuid.uuid4())
        started_at = _now()
        started = time.monotonic()
        environment = {
            "sandbox": "bubblewrap",
            "network": "unshared/disabled",
            "root_mount": "read-only",
            "workspace_mount": "read-only over /usr/share",
            "home_mount": "empty tmpfs",
            "etc_mount": "empty tmpfs",
            "sensitive_mounts": [
                "/home", "/root", "/etc", "/var", "/tmp", "/run",
                "/mnt", "/media", "/boot", "/opt", "/srv",
            ],
            "temporary_mount": "ephemeral tmpfs capped at 64 MiB",
            "other_temporary_mount_limit_bytes": 4 * 1024 * 1024,
            "shared_memory_mount_limit_bytes": 16 * 1024 * 1024,
            "process_limit": "current uid task count plus 8",
            "memory_limit_bytes": MAX_ADDRESS_SPACE_BYTES,
            "cpu_time_limit_seconds": MAX_CPU_SECONDS,
            "timeout_seconds": timeout_seconds,
            "maximum_output_bytes": MAX_OUTPUT_BYTES,
            "maximum_sandbox_file_bytes": MAX_SANDBOX_FILE_BYTES,
            "python_version": sys.version.split()[0],
            "platform": platform.system(),
        }
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or timeout_seconds < 1
            or timeout_seconds > MAX_TIMEOUT_SECONDS
        ):
            raise KaiError(f"Candidate runner timeout must be in 1..{MAX_TIMEOUT_SECONDS} seconds.")
        process = None
        try:
            workspace_record = self.database.get_candidate_workspace(workspace_id)
            if workspace_record["status"] in {"PRESERVED", "CLEANED"}:
                raise KaiError("Preserved or cleaned candidate workspaces cannot be executed.")
            workspace = self.workspace_manager.path_for(workspace_id)
            validation = self.workspace_manager.validate(workspace_id)
            if validation["status"] != "VALID":
                raise KaiError("Candidate workspace validation failed.")
            if resource is None:
                raise KaiError("Candidate OS resource limits are unavailable.")
            environment["workspace_source_tree_sha256"] = validation[
                "source_tree_sha256"
            ]
            command = self._sandbox_command(workspace, python_args)
            try:
                payload = (
                    json.dumps(
                        input_payload, allow_nan=False, separators=(",", ":")
                    ).encode()
                    if input_payload is not None
                    else b""
                )
            except (TypeError, ValueError) as exc:
                raise KaiError(f"Candidate runner input is not finite JSON: {exc}") from exc
            if len(payload) > MAX_INPUT_BYTES:
                raise KaiError("Candidate runner input exceeds the configured size limit.")
            process = subprocess.Popen(
                command,
                cwd="/",
                env={
                    "PATH": "/usr/bin:/bin",
                    "HOME": "/tmp/candidate-home",
                    "PYTHONDONTWRITEBYTECODE": "1",
                },
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
                preexec_fn=_limits,
            )
            status, stdout_bytes, stderr_bytes, reason = self._drain_process(
                process, payload, timeout_seconds
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            stderr = stderr_bytes.decode("utf-8", errors="replace")
        except FileNotFoundError as exc:
            if process is not None:
                _stop_process(process)
            status = "UNKNOWN"
            stdout = ""
            stderr = ""
            reason = f"Candidate sandbox could not start: {exc}"
            process = None
        except KaiError as exc:
            if process is not None:
                _stop_process(process)
            status = (
                "UNKNOWN"
                if str(exc)
                in {
                    "Required bubblewrap Linux sandbox is unavailable.",
                    "Candidate OS resource limits are unavailable.",
                }
                else "INVALID"
            )
            stdout = ""
            stderr = ""
            reason = str(exc)
            process = None
        except OSError as exc:
            if process is not None:
                _stop_process(process)
            status = "UNKNOWN"
            stdout = ""
            stderr = ""
            reason = f"Candidate sandbox execution unavailable: {exc}"
            process = None
        except subprocess.SubprocessError as exc:
            if process is not None:
                _stop_process(process)
            status = "UNKNOWN"
            stdout = ""
            stderr = ""
            reason = f"Candidate sandbox process could not initialize: {exc}"
            process = None
        completed_at = _now()
        result = CandidateExecutionResult(
            execution_id=execution_id,
            candidate_id=candidate_id,
            workspace_id=workspace_id,
            procedure_id=procedure_id,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=max(0.0, time.monotonic() - started),
            exit_code=process.returncode if process is not None else None,
            stdout=stdout[:MAX_OUTPUT_BYTES],
            stderr=stderr[:MAX_OUTPUT_BYTES],
            environment=environment,
            reason=reason,
        )
        if persist_result:
            self._persist_result(result)
        return result

    def _persist_result(self, result: CandidateExecutionResult) -> None:
        self.database.record_candidate_execution(result.to_dict())

    def run_tests(
        self,
        candidate_id: str,
        workspace_id: str,
        module: str | None = None,
        timeout_seconds: int = 120,
    ) -> CandidateExecutionResult:
        workspace = self.database.get_candidate_workspace(workspace_id)
        if workspace["status"] in {"PRESERVED", "CLEANED"}:
            raise KaiError("Preserved or cleaned candidate workspaces cannot be executed.")
        args = self._module_args(module)
        procedure = (
            f"unittest:{module}" if module else "unittest:discover-tests"
        )
        result = self._run(
            candidate_id=candidate_id,
            workspace_id=workspace_id,
            procedure_id=procedure,
            python_args=args,
            timeout_seconds=timeout_seconds,
            persist_result=False,
        )
        if result.status == "SUCCESS":
            summaries = re.findall(
                r"Ran\s+(\d+)\s+tests?\b", result.stdout + "\n" + result.stderr
            )
            if not summaries or int(summaries[-1]) < 1:
                result = CandidateExecutionResult(
                    **{
                        **result.to_dict(),
                        "status": "INVALID",
                        "reason": (
                            "Candidate unittest procedure completed without "
                            "reporting at least one executed test."
                        ),
                    }
                )
        self._persist_result(result)
        return result

    def run_cpu_predictor_evaluation(
        self,
        candidate_id: str,
        workspace_id: str,
        sequences: list[list[float]],
        warmup: int,
        candidate_module: str = "kai/candidate_algorithm.py",
        timeout_seconds: int = 30,
    ) -> tuple[CandidateExecutionResult, dict | None]:
        if candidate_module != "kai/candidate_algorithm.py":
            raise KaiError("CPU evaluation module path is fixed by the protocol.")
        workspace = self.database.get_candidate_workspace(workspace_id)
        if workspace["status"] in {"PRESERVED", "CLEANED"}:
            raise KaiError("Preserved or cleaned candidate workspaces cannot be executed.")
        payload = {
            "candidate_module": candidate_module,
            "sequences": sequences,
            "warmup": warmup,
        }
        result = self._run(
            candidate_id=candidate_id,
            workspace_id=workspace_id,
            procedure_id="cpu-predictor-evaluation-v1",
            python_args=["-c", _PREDICTOR_WORKER],
            input_payload=payload,
            timeout_seconds=timeout_seconds,
            persist_result=False,
        )
        if result.status != "SUCCESS":
            self._persist_result(result)
            return result, None
        try:
            parsed = json.loads(result.stdout)
            if (
                set(parsed) != {"baseline_errors", "candidate_errors"}
                or len(parsed["baseline_errors"]) != CPU_PREDICTOR_PROTOCOL.sequence_count
                or len(parsed["candidate_errors"]) != CPU_PREDICTOR_PROTOCOL.sequence_count
            ):
                raise ValueError("Candidate evaluation output shape is invalid.")
            from kai.evaluation_protocol import summarize_repeated_errors

            summarize_repeated_errors(
                parsed["baseline_errors"],
                parsed["candidate_errors"],
                CPU_PREDICTOR_PROTOCOL,
            )
            self._persist_result(result)
            return result, parsed
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            invalid = CandidateExecutionResult(
                **{
                    **result.to_dict(),
                    "execution_id": str(uuid.uuid4()),
                    "completed_at": _now(),
                    "status": "INVALID",
                    "reason": f"Candidate evaluation result is malformed: {exc}",
                }
            )
            self._persist_result(invalid)
            return invalid, None
