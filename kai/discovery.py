import os
import platform
import re
import sqlite3
import shutil
import socket
import sys
from pathlib import Path

PCI_VENDOR_NAMES = {
    "0x10de": "NVIDIA",
    "0x1002": "AMD",
    "0x8086": "Intel",
    "0x1022": "AMD",
}


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def _physical_core_count() -> int | None:
    cpu_root = Path("/sys/devices/system/cpu")
    cores = set()
    try:
        cpu_paths = cpu_root.glob("cpu[0-9]*")
        for cpu_path in cpu_paths:
            if not re.fullmatch(r"cpu\d+", cpu_path.name):
                continue
            package = _read_text(
                cpu_path / "topology" / "physical_package_id"
            )
            core = _read_text(cpu_path / "topology" / "core_id")
            if package is not None and core is not None:
                cores.add((package.strip(), core.strip()))
    except OSError:
        return None
    return len(cores) or None


def _cpu_details() -> dict:
    model = None
    flags: list[str] = []
    cpuinfo = _read_text(Path("/proc/cpuinfo"))
    if cpuinfo:
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            if key.strip().lower() in ("model name", "hardware") and model is None:
                model = value.strip()
            elif key.strip().lower() in ("flags", "features") and not flags:
                flags = value.split()
    return {
        "model": model or "unknown",
        "logical_processors": os.cpu_count(),
        "physical_cores": _physical_core_count(),
        "instruction_set_features": flags,
    }


def _memory_details() -> dict:
    values: dict[str, int] = {}
    meminfo = _read_text(Path("/proc/meminfo"))
    if meminfo:
        for line in meminfo.splitlines():
            key, separator, value = line.partition(":")
            if separator and key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree"):
                amount = value.strip().split()
                if amount and amount[0].isdigit():
                    values[key] = int(amount[0]) * 1024
    names = {
        "total_bytes": "MemTotal",
        "available_bytes": "MemAvailable",
        "swap_total_bytes": "SwapTotal",
        "swap_free_bytes": "SwapFree",
    }
    return {target: values.get(source) for target, source in names.items()}


def _gpu_details() -> dict:
    card_paths = sorted(
        path
        for path in Path("/sys/class/drm").glob("card[0-9]*")
        if re.fullmatch(r"card\d+", path.name)
    )
    drm_devices = []
    for card_path in card_paths:
        device_path = card_path / "device"
        vendor_id = _read_text(device_path / "vendor")
        device_id = _read_text(device_path / "device")
        if vendor_id is None:
            continue
        vendor_id = vendor_id.strip().lower()
        device_id = device_id.strip().lower() if device_id else None
        memory = _read_text(device_path / "mem_info_vram_total")
        if memory is not None:
            try:
                memory_bytes = int(memory.strip(), 0)
            except ValueError:
                memory_bytes = None
        else:
            memory_bytes = None
        drm_devices.append(
            {
                "card": card_path.name,
                "vendor": PCI_VENDOR_NAMES.get(vendor_id, "unknown"),
                "vendor_id": vendor_id,
                "device_id": device_id,
                "memory_bytes": memory_bytes,
            }
        )
    if drm_devices:
        classified = all(device["vendor"] != "unknown" for device in drm_devices)
        return {
            "status": "detected_unclassified" if not classified else "detected",
            "devices": drm_devices,
            "evidence": "Device identifiers were read from Linux DRM sysfs.",
        }
    return {
        "status": "not_detected",
        "devices": [],
        "evidence": "No DRM card device was detected; PCI enumeration is performed separately.",
    }


def _tool_details() -> dict:
    tools: dict[str, dict] = {}
    for name in (
        "git",
        "gcc",
        "g++",
        "c++",
        "clang",
        "rustc",
        "cargo",
        "make",
        "cmake",
        "ninja",
        "pip",
        "pip3",
        "pytest",
        "docker",
        "podman",
        "nvcc",
        "nvidia-smi",
        "ollama",
    ):
        path = shutil.which(name)
        tools[name] = {"available": path is not None, "path": path}
    tools["c_compiler"] = {
        "available": any(tools[name]["available"] for name in ("gcc", "clang")),
        "candidates": ["gcc", "clang"],
        "candidate_paths": {name: tools[name]["path"] for name in ("gcc", "clang")},
    }
    tools["cxx_compiler"] = {
        "available": any(tools[name]["available"] for name in ("g++", "c++", "clang")),
        "candidates": ["g++", "c++", "clang"],
        "candidate_paths": {
            name: tools[name]["path"] for name in ("g++", "c++", "clang")
        },
    }
    tools["python"] = {
        "available": True,
        "path": sys.executable,
        "version": platform.python_version(),
    }
    tools["sqlite"] = {
        "available": True,
        "version": sqlite3.sqlite_version,
    }
    return tools


def _network_details() -> dict:
    try:
        interfaces = [
            {"name": name, "index": index}
            for index, name in socket.if_nameindex()
        ]
    except OSError as exc:
        interfaces = []
        discovery = f"Interface discovery failed: {type(exc).__name__}"
    else:
        discovery = "Interfaces are listed; external connectivity was not tested."
    active_routes: list[str] = []
    routes = _read_text(Path("/proc/net/route"))
    if routes:
        for line in routes.splitlines()[1:]:
            columns = line.split()
            if len(columns) > 3 and columns[1] == "00000000":
                active_routes.append(columns[0])
    return {
        "interfaces": interfaces,
        "default_route_interfaces": active_routes,
        "internet_connectivity": "unknown",
        "evidence": discovery,
    }


def _distribution_name() -> str | None:
    if not hasattr(platform, "freedesktop_os_release"):
        return None
    try:
        return platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        return None


def collect_environment(data_dir: Path) -> dict:
    try:
        usage = os.statvfs(data_dir)
        storage = {
            "path": str(data_dir),
            "total_bytes": usage.f_blocks * usage.f_frsize,
            "available_bytes": usage.f_bavail * usage.f_frsize,
        }
    except OSError as exc:
        storage = {
            "path": str(data_dir),
            "total_bytes": None,
            "available_bytes": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    operating_system = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "distribution": _distribution_name(),
        "architecture": platform.machine() or "unknown",
    }
    return {
        "operating_system": operating_system,
        "python": platform.python_version(),
        "tools": _tool_details(),
        "cpu": _cpu_details(),
        "memory": _memory_details(),
        "gpu": _gpu_details(),
        "storage": storage,
        "network": _network_details(),
        "discovery_limitations": [
            "Internet reachability was not tested.",
            "RAM and CPU detail rely on Linux procfs when available.",
            "GPU identification is best-effort and may omit non-DRM accelerators.",
        ],
    }


def capability_summary(environment: dict) -> dict:
    tools = environment["tools"]
    gpu = environment["gpu"]
    ollama_discoverable = tools["ollama"]["available"]
    return {
        "python_runtime": environment["python"],
        "cpu_execution": environment["cpu"]["logical_processors"] is not None,
        "persistent_sqlite_state": True,
        "available_developer_tools": sorted(
            name for name, details in tools.items()
            if details.get("available") and name not in ("python", "sqlite")
        ),
        "gpu_pci_visibility": gpu.get("pci_enumeration", {}).get("status", "unknown"),
        "gpu_kernel_support": gpu.get("kernel_support", {}).get("status", "unknown"),
        "gpu_cuda_library_discovery": gpu.get("runtime_availability", {}).get(
            "library_discovery_status", "unknown"
        ),
        "gpu_cuda_runtime": gpu.get("runtime_availability", {}).get("cuda_status", "unknown"),
        "gpu_compute_usable": gpu.get("usable_capability", {}).get(
            "cuda_compute", "unknown"
        ),
        "gpu": gpu.get("pci_enumeration", {}).get("status", "unknown"),
        "ollama_executable_discoverable": ollama_discoverable,
        "local_llm_runtime": "unknown",
        "external_network": environment["network"]["internet_connectivity"],
        "autonomous_actions": False,
        "language_model": "not configured",
    }
