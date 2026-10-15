import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kai.discovery import collect_environment

EPISTEMIC_CLASSIFICATIONS = (
    "OBSERVATION",
    "FACT",
    "INFERENCE",
    "HYPOTHESIS",
    "UNKNOWN",
)

_PCI_VENDOR_NAMES = {
    "0x10de": "NVIDIA",
    "0x1002": "AMD",
    "0x8086": "Intel",
    "0x1022": "AMD",
    "0x10ec": "Realtek",
}


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None


def _frequency_details() -> dict:
    values: dict[str, list[int]] = {}
    for cpu_path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        if not re.fullmatch(r"cpu\d+", cpu_path.name):
            continue
        for label, filename in (
            ("current_khz", "scaling_cur_freq"),
            ("minimum_khz", "cpuinfo_min_freq"),
            ("maximum_khz", "cpuinfo_max_freq"),
        ):
            raw = _read(cpu_path / "cpufreq" / filename)
            if raw and raw.isdigit():
                values.setdefault(label, []).append(int(raw))
    return {
        label: {"minimum": min(items), "maximum": max(items)}
        for label, items in values.items()
        if items
    }


def _topology() -> dict:
    processors = []
    for cpu_path in sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*"),
        key=lambda path: int(path.name[3:]) if path.name[3:].isdigit() else -1,
    ):
        if not re.fullmatch(r"cpu\d+", cpu_path.name):
            continue
        package = _read(cpu_path / "topology" / "physical_package_id")
        core = _read(cpu_path / "topology" / "core_id")
        online = _read(cpu_path / "online")
        processors.append(
            {
                "logical_cpu": int(cpu_path.name[3:]),
                "package_id": int(package) if package and package.isdigit() else None,
                "core_id": int(core) if core and core.isdigit() else None,
                "online": online != "0" if online is not None else True,
            }
        )
    return {
        "logical_processors": sum(item["online"] for item in processors) or os.cpu_count(),
        "physical_cores": len(
            {
                (item["package_id"], item["core_id"])
                for item in processors
                if item["online"]
                if item["package_id"] is not None and item["core_id"] is not None
            }
        ) or None,
        "processors": processors,
    }


def _cpu_cache() -> list[dict]:
    caches = []
    cpu_root = Path("/sys/devices/system/cpu")
    seen: set[tuple[str | None, str | None, str | None, str | None]] = set()
    for index_path in sorted(cpu_root.glob("cpu0/cache/index*")):
        item = {
            "level": _read(index_path / "level"),
            "type": _read(index_path / "type"),
            "size": _read(index_path / "size"),
            "shared_logical_cpus": _read(index_path / "shared_cpu_list"),
        }
        key = tuple(item.values())
        if key not in seen:
            seen.add(key)
            caches.append(item)
    return caches


def _cpu_vendor_model() -> tuple[str, str]:
    vendor = "unknown"
    model = "unknown"
    cpuinfo = _read(Path("/proc/cpuinfo"))
    if cpuinfo:
        for line in cpuinfo.splitlines():
            key, separator, value = line.partition(":")
            if not separator:
                continue
            key = key.strip().lower()
            if key == "vendor_id" and vendor == "unknown":
                vendor = value.strip()
            if key in ("model name", "hardware") and model == "unknown":
                model = value.strip()
    return vendor, model


def _driver_name(device_path: Path) -> str | None:
    try:
        return (device_path / "driver").resolve(strict=True).name
    except OSError:
        return None


def _pci_devices(pci_root: Path | None = None) -> list[dict]:
    devices = []
    pci_root = pci_root or Path("/sys/bus/pci/devices")
    for device_path in sorted(pci_root.glob("*")):
        vendor_id = _read(device_path / "vendor")
        device_id = _read(device_path / "device")
        class_id = _read(device_path / "class")
        if vendor_id is None or device_id is None or class_id is None:
            continue
        vendor_id, device_id, class_id = (
            vendor_id.lower(),
            device_id.lower(),
            class_id.lower(),
        )
        driver = _driver_name(device_path)
        drm_cards = []
        for drm_device in (device_path / "drm").glob("card[0-9]*"):
            if re.fullmatch(r"card\d+", drm_device.name):
                drm_cards.append(drm_device.name)
        try:
            pci_class = int(class_id, 16)
        except ValueError:
            pci_class = None
        devices.append(
            {
                "address": device_path.name,
                "vendor": _PCI_VENDOR_NAMES.get(vendor_id, "unknown"),
                "vendor_id": vendor_id,
                "device_id": device_id,
                "class_id": class_id,
                "display_controller": pci_class is not None and (pci_class >> 16) == 0x03,
                "compute_accelerator": pci_class is not None and (pci_class >> 16) == 0x12,
                "driver": driver,
                "driver_bound": driver is not None,
                "kernel_visible": True,
                "drm_cards": sorted(drm_cards),
            }
        )
    return devices


def build_gpu_capability_model(
    pci_devices: list[dict],
    nvidia_nodes: list[str],
    nvidia_module: bool,
    nvidia_proc_driver: bool,
    cuda_toolkit: bool,
    runtime_libraries: list[str],
    nvidia_smi_present: bool,
    drm_nodes: list[str],
    render_nodes: list[str],
    node_access: list[dict] | None = None,
) -> dict:
    devices = [
        item for item in pci_devices
        if item["display_controller"] or item.get("compute_accelerator", False)
    ]
    driver_support = [
        {
            "pci_address": device["address"],
            "driver": device["driver"],
            "status": "bound" if device["driver_bound"] else "unbound",
        }
        for device in devices
    ]
    bound_count = sum(device["driver_bound"] for device in devices)
    kernel_status = (
        "unknown" if not devices
        else "driver_bound" if bound_count == len(devices)
        else "no_bound_driver" if bound_count == 0
        else "mixed_driver_binding"
    )
    library_discovery_status = (
        "libraries_detected" if runtime_libraries
        else "not_detected_in_inspected_paths"
    )
    cuda_runtime_status = "unknown"
    graphics_visibility = "kernel_exposed" if drm_nodes or render_nodes else "unknown"
    return {
        "pci_devices": devices,
        "physical_hardware": {
            "status": "pci_enumerated" if devices else "not_visible_in_this_environment",
            "physical_presence": "unknown",
            "evidence": "PCI enumeration is evidence of OS-visible functions, not off-system physical presence.",
        },
        "pci_enumeration": {
            "status": "detected" if devices else "not_visible",
            "evidence": "Graphics/compute PCI functions read from /sys/bus/pci/devices.",
        },
        "kernel_support": {
            "status": kernel_status,
            "driver_functionality": "unknown",
            "nvidia_module_loaded": nvidia_module,
            "nvidia_driver_proc_entry": nvidia_proc_driver,
        },
        "driver_support": driver_support,
        "kernel_visibility": {
            "drm_card_nodes": drm_nodes,
            "drm_render_nodes": render_nodes,
            "nvidia_device_nodes": nvidia_nodes,
            "node_access": node_access or [],
        },
        "runtime_availability": {
            "cuda_toolkit_executable_present": cuda_toolkit,
            "cuda_runtime_libraries": sorted(set(runtime_libraries)),
            "cuda_status": cuda_runtime_status,
            "library_discovery_status": library_discovery_status,
            "nvidia_smi_executable_present": nvidia_smi_present,
            "nvidia_device_node_status": (
                "visible" if nvidia_nodes else "not_visible_in_current_namespace"
            ),
        },
        "graphics_capability": {
            "status": graphics_visibility,
            "usable": "unknown",
            "reason": "DRM node visibility does not prove that graphics operations are usable.",
        },
        "compute_capability": "unknown",
        "usable_capability": {
            "graphics": "unknown",
            "cuda_compute": "unknown",
            "reason": (
                "No graphics or CUDA workload was run; visibility alone does not prove usability."
            ),
        },
        "measured_capability": {
            "status": "not_measured",
            "graphics": "unknown",
            "compute": "unknown",
        },
        "evidence_limitations": [
            "PCI enumeration does not prove physical presence outside this OS environment.",
            "Runtime files or device nodes do not prove a workload can use the device.",
            "CUDA compute capability was not queried by running a GPU utility or workload.",
        ],
    }


def _gpu_subsystem(pci_devices: list[dict]) -> dict:
    nvidia_nodes = [
        str(path)
        for path in (
            Path("/dev/nvidiactl"),
            Path("/dev/nvidia-uvm"),
            *Path("/dev").glob("nvidia[0-9]*"),
        )
        if path.exists()
    ]
    nvidia_module = Path("/sys/module/nvidia").exists()
    nvidia_proc_driver = Path("/proc/driver/nvidia/version").exists()
    cuda_toolkit = shutil.which("nvcc") is not None
    runtime_libraries = _collect_nvidia_runtime_libraries()
    drm_nodes = sorted(
        str(path)
        for path in Path("/dev/dri").glob("card[0-9]*")
        if re.fullmatch(r"card\d+", path.name)
    )
    render_nodes = sorted(str(path) for path in Path("/dev/dri").glob("renderD*"))
    node_access = []
    for path in sorted(set(drm_nodes + render_nodes + nvidia_nodes)):
        node_access.append(
            {
                "path": path,
                "readable_by_current_user": os.access(path, os.R_OK),
                "writable_by_current_user": os.access(path, os.W_OK),
            }
        )
    return build_gpu_capability_model(
        pci_devices,
        nvidia_nodes,
        nvidia_module,
        nvidia_proc_driver,
        cuda_toolkit,
        runtime_libraries,
        shutil.which("nvidia-smi") is not None,
        drm_nodes,
        render_nodes,
        node_access,
    )


def _ram_details() -> dict:
    raw = _read(Path("/proc/meminfo"))
    values: dict[str, int] = {}
    if raw:
        for line in raw.splitlines():
            key, separator, value = line.partition(":")
            if separator and key in {
                "MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "MemFree", "Buffers", "Cached"
            }:
                parts = value.split()
                if parts and parts[0].isdigit():
                    values[key] = int(parts[0]) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    return {
        "total_bytes": total,
        "available_bytes": available,
        "used_bytes": total - available if total is not None and available is not None else None,
        "swap_total_bytes": swap_total,
        "swap_free_bytes": swap_free,
        "swap_used_bytes": swap_total - swap_free
        if swap_total is not None and swap_free is not None else None,
        "used_definition": "total_bytes minus MemAvailable from /proc/meminfo",
        "source": "/proc/meminfo",
    }


def _storage_details(default_path: Path) -> dict:
    block_devices = []
    for block_path in sorted(Path("/sys/class/block").glob("*")):
        if block_path.name.startswith(("loop", "ram")):
            continue
        sectors = _read(block_path / "size")
        device_model = _read(block_path / "device" / "model")
        rotational = _read(block_path / "queue" / "rotational")
        block_devices.append(
            {
                "name": block_path.name,
                "model": device_model,
                "capacity_bytes": int(sectors) * 512 if sectors and sectors.isdigit() else None,
                "rotational": rotational == "1" if rotational in ("0", "1") else None,
                "source": str(block_path),
            }
        )

    mounts = []
    raw_mounts = _read(Path("/proc/self/mounts"))
    if raw_mounts:
        seen_mounts = set()
        for line in raw_mounts.splitlines():
            columns = line.split()
            if len(columns) < 3:
                continue
            source, target, filesystem = (
                value.replace("\\040", " ").replace("\\134", "\\")
                for value in columns[:3]
            )
            key = (source, target, filesystem)
            if key in seen_mounts:
                continue
            seen_mounts.add(key)
            try:
                usage = os.statvfs(target)
                capacity = usage.f_blocks * usage.f_frsize
                free = usage.f_bavail * usage.f_frsize
            except OSError:
                capacity = free = None
            mounts.append(
                {
                    "source": source if source.startswith("/dev/") else None,
                    "mount_point": target,
                    "filesystem": filesystem,
                    "capacity_bytes": capacity,
                    "free_bytes": free,
                }
            )
    try:
        usage = os.statvfs(default_path)
        default_mount = {
            "path": str(default_path),
            "capacity_bytes": usage.f_blocks * usage.f_frsize,
            "free_bytes": usage.f_bavail * usage.f_frsize,
        }
    except OSError as exc:
        default_mount = {"path": str(default_path), "error": type(exc).__name__}
    return {
        "block_devices": block_devices,
        "mount_points": mounts,
        "state_filesystem": default_mount,
        "evidence": "Read-only Linux sysfs, procfs, and statvfs inspection.",
    }


def _networking_tools() -> dict:
    names = ("ip", "ping", "curl", "wget", "nmcli", "iw")
    import shutil

    return {
        name: {
            "available": (path := shutil.which(name)) is not None,
            "path": path,
            "verification": "path lookup only; executable was not run",
        }
        for name in names
    }


def collect_environment_model(data_dir: Path) -> dict:
    base = collect_environment(data_dir)
    topology = _topology()
    vendor, model = _cpu_vendor_model()
    pci_devices = _pci_devices()
    base["operating_system"]["architecture"] = platform.machine() or "unknown"
    base["cpu"] = {
        **base["cpu"],
        "vendor": vendor,
        "model": model,
        "architecture": base["operating_system"]["architecture"],
        "logical_cores": topology["logical_processors"],
        "physical_cores": topology["physical_cores"],
        "topology": topology["processors"],
        "frequency": _frequency_details(),
        "caches": _cpu_cache(),
        "provenance": "Linux procfs and sysfs; missing files remain unknown.",
    }
    base["gpu"] = _gpu_subsystem(pci_devices)
    base["pci_devices"] = pci_devices
    base["memory"] = _ram_details()
    base["storage"] = _storage_details(data_dir)
    base["software_capabilities"] = {
        **base["tools"],
        "networking": _networking_tools(),
        "cuda": {
            "toolkit": base["tools"].get("nvcc", {}).get("available", False),
            "runtime_libraries": base["gpu"]["runtime_availability"]["cuda_runtime_libraries"],
            "usable": "unknown",
        },
        "sqlite": {
            "available": True,
            "version": base["tools"]["sqlite"]["version"],
            "source": "Python sqlite3 standard-library module",
        },
    }
    base["discovery_limitations"] = [
        "All device claims describe visibility from the current OS namespace, not off-system physical presence.",
        "No external programs are run for discovery; tool availability uses PATH and read-only system files.",
        "CPU frequency and cache information depend on exposed Linux sysfs files.",
        "Storage throughput and GPU performance are not inferred from capacity or visibility.",
        "Internet reachability and actual GPU workload usability remain unknown.",
    ]
    return base


def _collect_nvidia_runtime_libraries() -> list[str]:
    machine = platform.machine().lower()
    lib_arch = {
        "x86_64": "x86_64-linux-gnu",
        "aarch64": "aarch64-linux-gnu",
        "arm64": "aarch64-linux-gnu",
    }.get(machine)
    library_dirs = [Path("/usr/lib"), Path("/lib"), Path("/usr/local/cuda/lib64")]
    if lib_arch:
        library_dirs.extend((Path("/usr/lib") / lib_arch, Path("/lib") / lib_arch))
    library_dirs.append(Path("/usr/local/cuda/targets") / machine / "lib")
    libraries = []
    for library_dir in library_dirs:
        if library_dir.is_dir():
            libraries.extend(
                str(path)
                for pattern in ("libcuda.so*", "libcudart.so*")
                for path in library_dir.glob(pattern)
                if path.is_file()
            )
    return sorted(set(libraries))


def environment_fingerprint(environment: dict) -> str:
    cpu = environment.get("cpu", {})
    stable_profile = {
        "architecture": environment.get("operating_system", {}).get("architecture"),
        "cpu_vendor": cpu.get("vendor"),
        "cpu_model": cpu.get("model"),
        "physical_cores": cpu.get("physical_cores"),
        "pci_devices": sorted(
            (
                item.get("address"),
                item.get("vendor_id"),
                item.get("device_id"),
                item.get("class_id"),
            )
            for item in environment.get("pci_devices", [])
        ),
    }
    encoded = json.dumps(stable_profile, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_epistemic_claims(environment: dict, captured_at: str) -> list[dict]:
    claims = []

    def add(
        claim: str,
        classification: str,
        source: str,
        evidence: list[str],
        confidence: float | None,
        verified: bool = False,
        contradictory_evidence: list[str] | None = None,
    ) -> None:
        claims.append(
            {
                "claim": claim,
                "classification": classification,
                "source": source,
                "timestamp": captured_at,
                "confidence": confidence,
                "evidence": evidence,
                "verification_status": (
                    "verified" if verified
                    else "unverified" if classification == "UNKNOWN"
                    else "observed"
                ),
                "contradictory_evidence": contradictory_evidence or [],
            }
        )

    cpu = environment["cpu"]
    operating_system = environment["operating_system"]
    add(
        f"Operating system is {operating_system['distribution'] or operating_system['system']}.",
        "OBSERVATION",
        "/etc/os-release and platform APIs",
        [
            f"distribution={operating_system['distribution']}",
            f"release={operating_system['release']}",
        ],
        0.99 if operating_system["system"] else None,
    )
    add(
        f"Kernel release is {operating_system['release']}.",
        "OBSERVATION",
        "platform.uname()",
        [f"release={operating_system['release']}"],
        0.99,
    )
    if cpu["model"] != "unknown":
        add(
            f"CPU model is {cpu['model']}.",
            "OBSERVATION",
            "Linux /proc/cpuinfo",
            [f"Reported model: {cpu['model']}", f"Vendor: {cpu['vendor']}"],
            0.99,
        )
    else:
        add("CPU model is unknown.", "UNKNOWN", "Linux /proc/cpuinfo", [], None)
    add(
        f"CPU topology reports {cpu['logical_cores']} logical cores and "
        f"{cpu['physical_cores']} physical cores.",
        "OBSERVATION",
        "Linux CPU sysfs topology",
        [f"logical_cores={cpu['logical_cores']}", f"physical_cores={cpu['physical_cores']}"],
        0.95 if cpu["physical_cores"] is not None else None,
    )
    if cpu.get("frequency"):
        add(
            "CPU frequency ranges were read from Linux cpufreq sysfs.",
            "OBSERVATION",
            "/sys/devices/system/cpu/*/cpufreq",
            [json.dumps(cpu["frequency"], sort_keys=True)],
            0.95,
        )
    if cpu.get("caches"):
        add(
            "CPU cache levels and sharing were read from Linux sysfs.",
            "OBSERVATION",
            "/sys/devices/system/cpu/cpu0/cache",
            [json.dumps(cpu["caches"], sort_keys=True)],
            0.95,
        )
    for device in environment["gpu"]["pci_devices"]:
        add(
            f"PCI graphics/compute function {device['address']} ({device['vendor_id']}:{device['device_id']}) "
            "is enumerated in this OS environment.",
            "OBSERVATION",
            f"/sys/bus/pci/devices/{device['address']}",
            [
                f"vendor={device['vendor_id']}",
                f"device={device['device_id']}",
                f"class={device['class_id']}",
            ],
            0.99,
        )
        if device["driver"]:
            add(
                f"Kernel driver {device['driver']} is bound to PCI function {device['address']}.",
                "OBSERVATION",
                f"/sys/bus/pci/devices/{device['address']}/driver",
                [f"driver link resolves to {device['driver']}"],
                0.99,
            )
        else:
            add(
                f"No bound kernel driver is visible for PCI function {device['address']}.",
                "OBSERVATION",
                f"/sys/bus/pci/devices/{device['address']}/driver",
                ["No driver symlink was visible."],
                0.9,
            )
    graphics_compute_ids = {
        (device["address"], device["vendor_id"], device["device_id"])
        for device in environment["gpu"]["pci_devices"]
    }
    for device in environment["pci_devices"]:
        if (device["address"], device["vendor_id"], device["device_id"]) in graphics_compute_ids:
            continue
        add(
            f"PCI function {device['address']} ({device['vendor_id']}:{device['device_id']}) "
            "is enumerated in this OS environment.",
            "OBSERVATION",
            f"/sys/bus/pci/devices/{device['address']}",
            [
                f"class={device['class_id']}",
                f"driver={device['driver']}",
                f"kernel_visible={device['kernel_visible']}",
            ],
            0.99,
        )
    if not environment["gpu"]["pci_devices"]:
        add(
            "No graphics/compute-class PCI function is visible in this OS environment.",
            "OBSERVATION",
            "/sys/bus/pci/devices",
            ["PCI enumeration returned no graphics/compute-class function."],
            0.95,
            contradictory_evidence=[
                "This does not establish that no GPU exists outside the current OS namespace."
            ],
        )
    add(
        "The physical presence of GPUs outside the current OS-visible PCI namespace is unknown.",
        "UNKNOWN",
        "Current operating-system PCI/sysfs visibility",
        [
            f"visible_display_pci_functions={len(environment['gpu']['pci_devices'])}",
            "This environment may hide hardware from the current process.",
        ],
        None,
    )
    if environment["gpu"]["compute_capability"] == "unknown":
        add(
            "GPU compute capability is unknown.",
            "UNKNOWN",
            "No safe runtime-level compute-capability query was performed.",
            environment["gpu"]["evidence_limitations"],
            None,
        )
    gpu_kernel = environment["gpu"]["kernel_support"]
    add(
        f"GPU kernel-support state is {gpu_kernel['status']}.",
        "OBSERVATION" if gpu_kernel["status"] != "unknown" else "UNKNOWN",
        "PCI driver links and Linux module visibility",
        [
            f"nvidia_module_loaded={gpu_kernel['nvidia_module_loaded']}",
            f"nvidia_driver_proc_entry={gpu_kernel['nvidia_driver_proc_entry']}",
        ],
        0.95 if gpu_kernel["status"] != "unknown" else None,
    )
    if gpu_kernel["driver_functionality"] == "unknown":
        add(
            "GPU driver functionality has not been verified by a workload.",
            "UNKNOWN",
            "No driver API or GPU workload was invoked",
            [
                f"driver_bindings={environment['gpu']['driver_support']}",
                "A bound driver is not proof that it operates correctly.",
            ],
            None,
        )
    add(
        "GPU device-node and current-user access visibility was inspected.",
        "OBSERVATION",
        "/dev/dri and /dev/nvidia* metadata plus access checks",
        [
            f"kernel_visibility={json.dumps(environment['gpu']['kernel_visibility'], sort_keys=True)}",
        ],
        0.99,
    )
    runtime = environment["gpu"]["runtime_availability"]
    add(
        f"CUDA runtime usability is {runtime['cuda_status']}.",
        "UNKNOWN",
        "Read-only runtime metadata inspection; no runtime was invoked",
        [
            f"library_discovery_status={runtime['library_discovery_status']}",
            f"libraries={runtime['cuda_runtime_libraries']}",
            f"toolkit_executable_present={runtime['cuda_toolkit_executable_present']}",
        ],
        None,
    )
    if runtime["library_discovery_status"] == "not_detected_in_inspected_paths":
        add(
            "No CUDA runtime library was found in the inspected library paths.",
            "OBSERVATION",
            "Read-only inspection of selected system and CUDA library directories",
            ["No matching libcuda.so or libcudart.so file was found in those paths."],
            0.9,
            contradictory_evidence=[
                "A runtime may exist in an uninspected custom or container path."
            ],
        )
    if environment["gpu"]["usable_capability"]["cuda_compute"] == "unknown":
        add(
            "Usable CUDA acceleration is unknown.",
            "UNKNOWN",
            "CUDA device/runtime visibility inspection",
            [
                f"device_nodes={environment['gpu']['kernel_visibility']['nvidia_device_nodes']}",
                f"runtime_libraries={environment['gpu']['runtime_availability']['cuda_runtime_libraries']}",
                "No CUDA workload was executed.",
            ],
            None,
        )
    if environment["gpu"]["graphics_capability"]["usable"] == "unknown":
        add(
            "Usable graphics capability is unknown.",
            "UNKNOWN",
            "No graphics workload was run.",
            [json.dumps(environment["gpu"]["graphics_capability"], sort_keys=True)],
            None,
        )
    gpu_name = any(
        device["vendor"] == "NVIDIA" for device in environment["gpu"]["pci_devices"]
    )
    if not gpu_name:
        add(
            "An NVIDIA graphics/compute PCI function is not visible in this environment.",
            "OBSERVATION",
            "/sys/bus/pci/devices",
            [
                "No NVIDIA vendor ID was present in the enumerated graphics/compute PCI functions.",
                "This does not establish whether hardware is physically installed or hidden.",
            ],
            0.95,
            contradictory_evidence=[
                "A GPU could be hidden from this OS or visible only outside its namespace."
            ],
        )
    ram = environment["memory"]
    add(
        "RAM availability was sampled from the current operating-system view.",
        "OBSERVATION",
        ram["source"],
        [f"total_bytes={ram['total_bytes']}", f"available_bytes={ram['available_bytes']}"],
        0.99 if ram["total_bytes"] is not None else None,
    )
    storage = environment["storage"]
    add(
        "Storage devices, filesystems, and mount capacity were enumerated from the current environment.",
        "OBSERVATION",
        "Linux /sys/class/block, /proc/self/mounts, and statvfs",
        [
            f"block_device_count={len(storage['block_devices'])}",
            f"mount_count={len(storage['mount_points'])}",
            f"state_filesystem={json.dumps(storage['state_filesystem'], sort_keys=True)}",
        ],
        0.95,
    )
    add(
        "Internet connectivity is unknown because no network request was made.",
        "UNKNOWN",
        "Network interfaces and routes only",
        ["Interface/route enumeration does not verify end-to-end connectivity."],
        None,
    )
    add(
        "Network interfaces and any exposed default routes were enumerated.",
        "OBSERVATION",
        "socket.if_nameindex and Linux /proc/net/route",
        [
            f"interfaces={json.dumps(environment['network']['interfaces'], sort_keys=True)}",
            f"default_route_interfaces={environment['network']['default_route_interfaces']}",
        ],
        0.95,
    )
    def walk_capabilities(prefix: str, item: dict) -> None:
        if "available" in item:
            add(
                f"Software capability {prefix} is "
                f"{'discoverable' if item['available'] else 'not found in its inspected location'}.",
                "OBSERVATION",
                "Read-only PATH or runtime-file inspection",
                [
                    f"path={item.get('path')}",
                    f"candidates={item.get('candidate_paths', item.get('candidates'))}",
                ],
                0.99,
            )
        for key, value in item.items():
            if isinstance(value, dict):
                walk_capabilities(f"{prefix}.{key}" if prefix else key, value)

    walk_capabilities("", environment["software_capabilities"])
    add(
        "Local language-model runtime usability is unknown.",
        "UNKNOWN",
        "No local inference runtime was invoked.",
        [
            f"ollama_executable_discoverable={environment['tools']['ollama']['available']}",
            "Executable discovery does not verify a running service or usable model.",
        ],
        None,
    )
    add(
        f"KAI software version is {environment.get('kai_version', 'unknown')}.",
        "FACT",
        "kai.__version__",
        [f"version={environment.get('kai_version', 'unknown')}"],
        1.0,
        verified=True,
    )
    return claims


def compare_environments(previous: dict | None, current: dict) -> dict:
    if previous is None:
        return {
            "baseline": "first_snapshot",
            "same_environment_profile": None,
            "added_capabilities": [],
            "removed_capabilities": [],
            "changed_hardware": [],
            "changed_software": [],
            "changed_resources": [],
            "changed_drivers": [],
            "changed_runtime_availability": [],
            "changed_kernel_visibility": [],
            "other_changes": [],
        }
    old_env = previous["environment"]
    old_caps = _capability_set(old_env)
    new_caps = _capability_set(current["environment"])
    changes = {
        "baseline": previous["snapshot_id"],
        "same_environment_profile": (
            previous["environment_fingerprint"] == current["environment_fingerprint"]
        ),
        "added_capabilities": sorted(new_caps - old_caps),
        "removed_capabilities": sorted(old_caps - new_caps),
        "changed_hardware": [],
        "changed_software": [],
        "changed_resources": [],
        "changed_drivers": [],
        "changed_runtime_availability": [],
        "changed_kernel_visibility": [],
        "other_changes": [],
    }
    old = old_env
    new = current["environment"]
    cpu_identity_fields = ("vendor", "model", "architecture", "logical_cores", "physical_cores", "topology", "caches")
    old_cpu = {key: old.get("cpu", {}).get(key) for key in cpu_identity_fields}
    new_cpu = {key: new.get("cpu", {}).get(key) for key in cpu_identity_fields}
    if old_cpu != new_cpu:
        changes["changed_hardware"].append({"component": "cpu", "before": old_cpu, "after": new_cpu})
    pci_hardware_fields = ("address", "vendor_id", "device_id", "class_id")
    old_pci = [
        {key: device.get(key) for key in pci_hardware_fields}
        for device in old.get("pci_devices", [])
    ]
    new_pci = [
        {key: device.get(key) for key in pci_hardware_fields}
        for device in new.get("pci_devices", [])
    ]
    if old_pci != new_pci:
        changes["changed_hardware"].append(
            {"component": "pci_devices", "before": old_pci, "after": new_pci}
        )
    old_drivers = [
        (device.get("address"), device.get("driver"))
        for device in old.get("pci_devices", [])
    ]
    new_drivers = [
        (device.get("address"), device.get("driver"))
        for device in new.get("pci_devices", [])
    ]
    if old_drivers != new_drivers:
        changes["changed_drivers"].append(
            {"component": "pci_driver_bindings", "before": old_drivers, "after": new_drivers}
        )
    if old.get("gpu", {}).get("driver_support") != new.get("gpu", {}).get("driver_support"):
        changes["changed_drivers"].append(
            {
                "before": old.get("gpu", {}).get("driver_support"),
                "after": new.get("gpu", {}).get("driver_support"),
            }
        )
    if old.get("gpu", {}).get("runtime_availability") != new.get("gpu", {}).get("runtime_availability"):
        changes["changed_runtime_availability"].append(
            {
                "before": old.get("gpu", {}).get("runtime_availability"),
                "after": new.get("gpu", {}).get("runtime_availability"),
            }
        )
    if old.get("gpu", {}).get("kernel_visibility") != new.get("gpu", {}).get("kernel_visibility"):
        changes["changed_kernel_visibility"].append(
            {
                "before": old.get("gpu", {}).get("kernel_visibility"),
                "after": new.get("gpu", {}).get("kernel_visibility"),
            }
        )
    if old.get("software_capabilities") != new.get("software_capabilities"):
        changes["changed_software"].append({"component": "software_capabilities"})
    if old.get("kai_version") != new.get("kai_version"):
        changes["changed_software"].append(
            {
                "component": "kai_version",
                "before": old.get("kai_version"),
                "after": new.get("kai_version"),
            }
        )
    if old.get("operating_system") != new.get("operating_system"):
        changes["changed_software"].append({"component": "operating_system"})
    for section in ("memory", "storage"):
        if old.get(section) != new.get(section):
            changes["changed_resources"].append({"component": section})
    if old.get("cpu", {}).get("frequency") != new.get("cpu", {}).get("frequency"):
        changes["changed_resources"].append({"component": "cpu_frequency"})
    if old.get("network") != new.get("network"):
        changes["other_changes"].append({"component": "network"})
    return changes


def _capability_set(environment: dict) -> set[str]:
    capabilities = set()

    def add_discovered_tools(name: str, item: dict) -> None:
        if item.get("available") is True:
            capabilities.add(f"tool:{name}")
        for child_name, child in item.items():
            if isinstance(child, dict):
                add_discovered_tools(f"{name}.{child_name}", child)

    for name, item in environment.get("software_capabilities", {}).items():
        if isinstance(item, dict):
            add_discovered_tools(name, item)
    for device in environment.get("gpu", {}).get("pci_devices", []):
        capabilities.add(f"pci-device:{device['vendor_id']}:{device['device_id']}")
        if device.get("driver_bound"):
            capabilities.add(f"driver:{device['driver']}")
    gpu = environment.get("gpu", {})
    runtime = gpu.get("runtime_availability", {})
    if runtime.get("library_discovery_status") == "libraries_detected":
        capabilities.add("runtime:cuda-libraries")
    if runtime.get("cuda_toolkit_executable_present"):
        capabilities.add("toolkit:cuda")
    if gpu.get("kernel_visibility", {}).get("nvidia_device_nodes"):
        capabilities.add("runtime:nvidia-device-nodes")
    for node in environment.get("gpu", {}).get("kernel_visibility", {}).get("drm_card_nodes", []):
        capabilities.add(f"drm:{Path(node).name}")
    return capabilities


def measure_baseline(environment: dict, database_path: Path) -> dict:
    start = time.perf_counter()
    value = 0
    iterations = 100_000
    for index in range(iterations):
        value = (value + index * 17) % 1_000_003
    duration = time.perf_counter() - start
    cpu_measurement = {
        "status": "measured",
        "metric": "simple Python integer-loop iterations per second",
        "iterations": iterations,
        "duration_seconds": duration,
        "operations_per_second": iterations / duration if duration else None,
        "result_checksum": value,
        "not_a_model_or_ML_benchmark": True,
    }

    disk = {"status": "unavailable", "reason": "KAI state database does not exist."}
    try:
        descriptor = os.open(database_path, os.O_RDONLY)
    except OSError as exc:
        disk = {"status": "unavailable", "reason": f"Read-only open failed: {type(exc).__name__}"}
    else:
        try:
            start = time.perf_counter()
            data = os.read(descriptor, 64 * 1024)
            duration = time.perf_counter() - start
            disk = {
                "status": "measured",
                "metric": "single bounded read of existing SQLite state file",
                "bytes_read": len(data),
                "duration_seconds": duration,
                "bytes_per_second": len(data) / duration if duration else None,
                "caveat": "OS cache state is uncontrolled; not a storage-device benchmark.",
            }
        finally:
            os.close(descriptor)
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "cpu": cpu_measurement,
        "memory_availability": {
            "status": "discovered",
            "available_bytes": environment["memory"]["available_bytes"],
            "source": environment["memory"]["source"],
        },
        "storage_capacity": {
            "status": "discovered",
            "mount": environment["storage"]["state_filesystem"],
        },
        "disk_read": disk,
        "gpu": {
            "status": "not_measured",
            "compute_capability": environment["gpu"]["compute_capability"],
            "usable_capability": environment["gpu"]["usable_capability"],
            "reason": "No GPU workload was executed.",
        },
    }


def build_self_model(
    identity: dict,
    environment: dict,
    fingerprint: str,
    capabilities: dict,
    goals: list[dict],
    claims: list[dict],
    version: str,
    schema_version: int,
) -> dict:
    observed = sum(claim["classification"] != "UNKNOWN" for claim in claims)
    confidence = observed / len(claims) if claims else 0.0
    unknowns = [
        claim["claim"] for claim in claims if claim["classification"] == "UNKNOWN"
    ]
    unavailable = [
        name
        for name, details in environment["software_capabilities"].items()
        if isinstance(details, dict) and details.get("available") is False
    ]
    return {
        "identity": identity,
        "software_version": version,
        "architecture_version": schema_version,
        "environment_fingerprint": fingerprint,
        "environment_fingerprint_semantics": (
            "Hardware/architecture profile hash without serials, MAC addresses, "
            "hostnames, or a claim of unique physical-machine identity."
        ),
        "available_capabilities": capabilities,
        "unavailable_capabilities": [
            name
            for name, state in (
                ("autonomous_actions", capabilities.get("autonomous_actions")),
                ("language_model_integration", capabilities.get("language_model")),
            )
            if state is False or state == "not configured"
        ],
        "not_discovered_tools": unavailable,
        "unknown_capabilities": [
            name
            for name, state in (
                ("GPU compute usability", capabilities.get("gpu_compute_usable")),
                ("graphics usability", environment["gpu"]["graphics_capability"]["usable"]),
                ("external network", capabilities.get("external_network")),
                ("local LLM runtime", capabilities.get("local_llm_runtime")),
            )
            if state == "unknown"
        ],
        "current_resources": {
            "cpu": environment["cpu"],
            "memory": environment["memory"],
            "storage": environment["storage"]["state_filesystem"],
            "gpu": {
                "pci_devices": environment["gpu"]["pci_devices"],
                "kernel_support": environment["gpu"]["kernel_support"],
                "runtime_availability": environment["gpu"]["runtime_availability"],
                "usable_capability": environment["gpu"]["usable_capability"],
            },
        },
        "known_limitations": environment["discovery_limitations"]
        + [
            "KAI does not autonomously execute actions.",
            "A discovered capability is not evidence of measured performance or usability.",
            "Language model responses are bounded local inference, not evidence of general intelligence.",
        ]
        + (
            ["No language model is configured."]
            if capabilities.get("language_model") == "not configured"
            else []
        ),
        "current_goals": [goal for goal in goals if goal["status"] == "active"],
        "active_components": sorted(
            module.removeprefix("kai.")
            for module in sys.modules
            if module == "kai" or module.startswith("kai.")
        ),
        "dependencies": {
            "external_runtime_dependencies": [],
            "standard_library": ["sqlite3", "argparse", "json", "pathlib", "hashlib"],
        },
        "environment_identity": {
            "fingerprint": fingerprint,
            "same_as_previous_profile": None,
        },
        "confidence": confidence,
        "confidence_semantics": (
            "Evidence coverage ratio (non-UNKNOWN claims / total claims), not a "
            "calibrated probability that KAI's complete self-description is correct."
        ),
        "uncertainties": unknowns,
    }
