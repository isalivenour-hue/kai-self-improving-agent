import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kai.config import KaiConfig
from kai.database import SCHEMA_VERSION, Database
from kai.discovery import _gpu_details, _tool_details, capability_summary
from kai.environment_model import (
    _pci_devices,
    build_epistemic_claims,
    build_gpu_capability_model,
    build_self_model,
    compare_environments,
    environment_fingerprint,
)
from kai.errors import KaiError
from kai.runtime import KaiRuntime


def fake_environment(gpu_devices=None, driver_bound=True, runtime_libraries=None):
    gpu_devices = gpu_devices or 0
    pci_devices = [
        {
            "address": f"0000:01:00.{index}",
            "vendor": "NVIDIA",
            "vendor_id": "0x10de",
            "device_id": f"0x{index:04x}",
            "class_id": "0x030000",
            "display_controller": True,
            "driver": "nvidia" if driver_bound else None,
            "driver_bound": driver_bound,
            "kernel_visible": True,
            "drm_cards": ["card0"] if driver_bound else [],
        }
        for index in range(gpu_devices)
    ]
    gpu = build_gpu_capability_model(
        pci_devices,
        ["/dev/nvidiactl"] if driver_bound and gpu_devices else [],
        driver_bound and bool(gpu_devices),
        driver_bound and bool(gpu_devices),
        bool(runtime_libraries),
        runtime_libraries or [],
        False,
        ["/dev/dri/card0"] if driver_bound and gpu_devices else [],
        ["/dev/dri/renderD128"] if driver_bound and gpu_devices else [],
    )
    tools = {
        name: {"available": False, "path": None}
        for name in ("git", "gcc", "clang", "docker", "podman", "nvcc", "nvidia-smi", "ollama")
    }
    tools["python"] = {"available": True, "path": "/fake/python", "version": "3.14.4"}
    tools["sqlite"] = {"available": True, "version": "3.46.1"}
    return {
        "operating_system": {
            "system": "Linux",
            "distribution": "FakeOS 1",
            "release": "fake-kernel",
            "architecture": "x86_64",
        },
        "kai_version": "0.1.0",
        "python": "3.14.4",
        "tools": tools,
        "cpu": {
            "vendor": "FakeVendor",
            "model": "Fake CPU",
            "architecture": "x86_64",
            "logical_cores": 4,
            "logical_processors": 4,
            "physical_cores": 2,
            "topology": [{"logical_cpu": 0, "package_id": 0, "core_id": 0, "online": True}],
            "frequency": {},
            "caches": [],
        },
        "pci_devices": pci_devices,
        "gpu": gpu,
        "memory": {
            "total_bytes": 8_000_000,
            "available_bytes": 4_000_000,
            "used_bytes": 4_000_000,
            "swap_total_bytes": 0,
            "swap_free_bytes": 0,
            "swap_used_bytes": 0,
            "source": "fake meminfo",
        },
        "storage": {
            "block_devices": [],
            "mount_points": [],
            "state_filesystem": {
                "path": "/fake/kai",
                "capacity_bytes": 10_000,
                "free_bytes": 5_000,
            },
        },
        "software_capabilities": {
            **tools,
            "networking": {"ip": {"available": False, "path": None}},
            "cuda": {
                "toolkit": tools["nvcc"]["available"],
                "runtime_libraries": runtime_libraries or [],
                "usable": "unknown",
            },
        },
        "network": {
            "interfaces": [{"name": "fake0", "index": 1}],
            "default_route_interfaces": [],
            "internet_connectivity": "unknown",
        },
        "discovery_limitations": ["Fake environment does not verify Internet."],
    }


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Database(Path(self.temporary_directory.name) / "state.sqlite3")
        self.database.initialize()

    def test_identity_survives_reinitialization(self):
        first = self.database.identity()
        self.database.initialize()
        self.assertEqual(self.database.identity()["identity_id"], first["identity_id"])

    def test_version_zero_schema_migrates(self):
        path = Path(self.temporary_directory.name) / "version-zero.sqlite3"
        database = Database(path)
        with database.connection() as connection:
            connection.execute(
                "CREATE TABLE schema_info "
                "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "version INTEGER NOT NULL)"
            )
            connection.execute("INSERT INTO schema_info VALUES (1, 0)")
            connection.commit()
        database.initialize()
        self.assertEqual(database.schema_version(), SCHEMA_VERSION)
        self.assertEqual(database.integrity_check(), "ok")

    def test_memory_keeps_epistemic_label_and_provenance(self):
        self.database.add_memory("hypothesis", "Example claim", "test fixture", 0.4)
        memory = self.database.list_memories()[0]
        self.assertEqual(memory["kind"], "hypothesis")
        self.assertEqual(memory["source"], "test fixture")
        self.assertEqual(memory["confidence"], 0.4)
        event = self.database.recent_events()[0]
        self.assertEqual(event["event_type"], "CLAIM_CREATED")
        self.assertEqual(event["details"], {"memory_id": memory["memory_id"], "kind": "hypothesis"})

    def test_memory_search_is_literal_and_case_insensitive(self):
        self.database.add_memory("observation", "KAI stores SQLite state", "test", None)
        self.assertEqual(len(self.database.list_memories(query="sqlite")), 1)
        self.assertEqual(self.database.list_memories(query="' OR 1=1 --"), [])

    def test_memory_rejects_empty_content_and_invalid_confidence(self):
        with self.assertRaisesRegex(KaiError, "cannot be empty"):
            self.database.add_memory("fact", "  ", "test", None)
        with self.assertRaisesRegex(KaiError, "between 0.0 and 1.0"):
            self.database.add_memory("fact", "Example", "test", 1.1)

    def test_goals_persist_with_priorities_and_criteria(self):
        self.database.add_goal("Test persistence", 2, "Goal is listed after reloading")
        goal = self.database.list_goals()[0]
        self.assertEqual(goal["title"], "Test persistence")
        self.assertEqual(goal["priority"], 2)
        self.assertEqual(goal["success_criteria"], "Goal is listed after reloading")
        self.assertEqual(self.database.recent_events()[0]["event_type"], "GOAL_CREATED")

    def test_database_integrity_and_version(self):
        self.assertEqual(self.database.integrity_check(), "ok")
        self.assertEqual(self.database.schema_version(), SCHEMA_VERSION)

    def test_v2_migration_preserves_kai_identity_and_adds_experience_tables(self):
        path = Path(self.temporary_directory.name) / "v2.sqlite3"
        database = Database(path)
        with database.connection() as connection:
            connection.execute(
                "CREATE TABLE schema_info "
                "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "version INTEGER NOT NULL)"
            )
            connection.execute("INSERT INTO schema_info VALUES (1, 0)")
            Database._migrate_to_v1(connection)
            Database._migrate_to_v2(connection)
            connection.commit()
        original_identity = database.identity()["identity_id"]
        database.initialize()
        self.assertEqual(database.schema_version(), SCHEMA_VERSION)
        self.assertEqual(database.identity()["identity_id"], original_identity)
        self.assertEqual(database.environment_snapshot_count(), 0)

    def test_newer_schema_is_rejected(self):
        with self.database.connection() as connection:
            connection.execute("UPDATE schema_info SET version = 99 WHERE singleton = 1")
            connection.commit()
        with self.assertRaisesRegex(KaiError, "newer than this KAI version"):
            self.database.initialize()


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.runtime = KaiRuntime(KaiConfig(Path(self.temporary_directory.name)))
        self.runtime.database.initialize()
        discovery = patch(
            "kai.runtime.collect_environment_model",
            return_value=fake_environment(),
        )
        discovery.start()
        self.addCleanup(discovery.stop)

    def test_boot_persists_identity_and_records_event(self):
        first = self.runtime.boot()
        second = self.runtime.boot()
        self.assertEqual(first["identity"]["identity_id"], second["identity"]["identity_id"])
        self.assertGreaterEqual(self.runtime.database.event_count(), 6)
        self.assertEqual(
            self.runtime.database.recent_events(limit=1)[0]["event_id"],
            second["event_id"],
        )
        persisted_self_model = {
            item["key"]: item["value"]
            for item in self.runtime.database.working_memory()
        }["self_model"]
        self.assertEqual(
            persisted_self_model["experience_state"]["latest_committed_event_id"],
            second["event_id"],
        )
        log_lines = self.runtime.config.log_path.read_text().splitlines()
        self.assertEqual(len(log_lines), 2)
        self.assertEqual(json.loads(log_lines[-1])["level"], "INFO")

    def test_self_model_reports_configured_local_llm_readiness(self):
        class ReadyLocalProvider:
            def status(self):
                return {
                    "provider": "ollama",
                    "model": "qwen2.5:0.5b",
                    "status": "READY",
                    "local_only": True,
                }

        self.runtime._configured_llm_provider = ReadyLocalProvider()
        report = self.runtime.status()
        self.assertEqual(report["self_model"]["language_model"]["status"], "READY")
        self.assertEqual(
            report["self_model"]["available_capabilities"]["language_model"],
            "configured",
        )
        self.assertEqual(
            report["self_model"]["available_capabilities"]["local_llm_runtime"],
            "READY",
        )

    def test_status_discloses_limits_and_uncertainties(self):
        report = self.runtime.status()
        self.assertIn("autonomous_actions", report["capabilities"])
        self.assertTrue(
            any("language model" in item.lower() for item in report["current_limitations"])
        )
        self.assertTrue(report["uncertainties"])
        self.assertTrue(self.runtime.config.database_path.exists())
        self.assertTrue(report["claims"])
        self.assertEqual(report["self_model"]["identity"]["identity_id"], report["identity"]["identity_id"])
        self.assertEqual(report["known_state"]["environment_snapshot_count"], 1)

    def test_awareness_status_is_provenance_bearing_and_keeps_unverified_state_unknown(self):
        report = self.runtime.awareness_status()
        self.assertEqual(report["self_model"]["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["fields"]["software.version"]["value"], "0.1.0")
        self.assertEqual(
            report["fields"]["database.schema_version"]["source"],
            "SQLite schema_info",
        )
        self.assertEqual(report["fields"]["tests.execution"]["state"], "UNKNOWN")
        self.assertEqual(
            report["capabilities"]["local_model_generation"]["status"],
            "IMPLEMENTED_UNVERIFIED",
        )
        self.assertEqual(report["evaluation"]["database_integrity"], "PASS")
        self.assertEqual(report["evaluation"]["awareness_benchmark"]["status"], "UNKNOWN")
        gaps = {gap["id"]: gap for gap in report["knowledge_gaps"]}
        self.assertIn("test-suite-execution-unverified", gaps)
        self.assertIn("local-model-generation-unverified", gaps)
        self.assertIn("gpu-inference-usability-unknown", gaps)
        self.assertIn("awareness-benchmark-not-passed", gaps)
        self.assertTrue(gaps["test-suite-execution-unverified"]["links"])
        self.assertFalse(report["privacy"]["secrets_included"])
        self.assertTrue(all(
            field["source"] and field["observed_at"]
            for field in report["fields"].values()
        ))

    def test_report_values_are_serializable(self):
        self.runtime.boot()
        json.dumps(self.runtime.status())

    def test_capability_summary_preserves_unknown_network_state(self):
        environment = {
            "python": "3.x",
            "tools": {"ollama": {"available": False}},
            "cpu": {"logical_processors": 2},
            "gpu": {"status": "unknown"},
            "network": {"internet_connectivity": "unknown"},
        }
        capabilities = capability_summary(environment)
        self.assertEqual(capabilities["external_network"], "unknown")
        self.assertEqual(capabilities["gpu"], "unknown")
        self.assertEqual(capabilities["local_llm_runtime"], "unknown")

    def test_gpu_sysfs_identifies_pci_vendor(self):
        with tempfile.TemporaryDirectory() as directory:
            drm = Path(directory) / "drm"
            device = drm / "card0" / "device"
            device.mkdir(parents=True)
            (device / "vendor").write_text("0x8086\n")
            (device / "device").write_text("0x0416\n")
            (device / "mem_info_vram_total").write_text("0x1000\n")
            with patch("kai.discovery.Path", wraps=Path) as path_factory:
                path_factory.side_effect = lambda value: drm if value == "/sys/class/drm" else Path(value)
                gpu = _gpu_details()
        self.assertEqual(gpu["status"], "detected")
        self.assertEqual(gpu["devices"][0]["vendor"], "Intel")
        self.assertEqual(gpu["devices"][0]["device_id"], "0x0416")
        self.assertEqual(gpu["devices"][0]["memory_bytes"], 4096)

    def test_tool_discovery_reports_absence_explicitly(self):
        with patch("kai.discovery.shutil.which", return_value=None):
            tools = _tool_details()
        self.assertFalse(tools["ollama"]["available"])
        self.assertIsNone(tools["ollama"]["path"])
        self.assertFalse(tools["c_compiler"]["available"])

    def test_corrupt_database_fails_visibly(self):
        self.runtime.database.path.write_text("not a SQLite database")
        with self.assertRaises(KaiError):
            self.runtime.database.initialize()

    def test_epistemic_claim_persists_provenance_unknown_and_contradiction(self):
        claim = {
            "claim": "Usable accelerator state is unknown.",
            "classification": "UNKNOWN",
            "source": "No safe runtime test",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "confidence": None,
            "evidence": ["No workload was run."],
            "verification_status": "unverified",
            "contradictory_evidence": ["Kernel reports a graphics node."],
        }
        saved = self.runtime.database.save_environment_snapshot(
            claim["timestamp"], "fingerprint-a", {"fixture": True}, [claim]
        )
        stored = self.runtime.database.claims_for_snapshot(saved["snapshot_id"])[0]
        self.assertEqual(stored["classification"], "UNKNOWN")
        self.assertEqual(stored["source"], claim["source"])
        self.assertEqual(stored["confidence"], None)
        self.assertEqual(stored["evidence"], claim["evidence"])
        self.assertEqual(stored["contradictory_evidence"], claim["contradictory_evidence"])

    def test_unknown_cannot_be_marked_verified(self):
        claim = {
            "claim": "GPU usable",
            "classification": "UNKNOWN",
            "source": "not tested",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "confidence": None,
            "evidence": [],
            "verification_status": "verified",
        }
        with self.assertRaisesRegex(KaiError, "UNKNOWN claims"):
            self.runtime.database.save_environment_snapshot(
                "2026-01-01T00:00:00+00:00", "fp", {}, [claim]
            )
        self.assertEqual(self.runtime.database.environment_snapshot_count(), 0)

    def test_fact_requires_evidence_and_verified_status(self):
        claim = {
            "claim": "Fact claim",
            "classification": "FACT",
            "source": "test",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "confidence": 1.0,
            "evidence": [],
            "verification_status": "verified",
        }
        with self.assertRaisesRegex(KaiError, "FACT claims require evidence"):
            self.runtime.database.save_environment_snapshot(
                "2026-01-01T00:00:00+00:00", "fp", {}, [claim]
            )
        self.assertEqual(self.runtime.database.environment_snapshot_count(), 0)

    def test_contradicted_claim_requires_counterevidence(self):
        claim = {
            "claim": "Driver appears usable.",
            "classification": "HYPOTHESIS",
            "source": "fixture",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "confidence": 0.4,
            "evidence": ["A driver is bound."],
            "verification_status": "contradicted",
        }
        with self.assertRaisesRegex(KaiError, "contradictory evidence"):
            self.runtime.database.save_environment_snapshot(
                "2026-01-01T00:00:00+00:00", "fp", {}, [claim]
            )

    def test_claim_confidence_must_be_finite_and_in_range(self):
        claim = {
            "claim": "Observation",
            "classification": "OBSERVATION",
            "source": "fixture",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "confidence": float("nan"),
            "evidence": ["fixture evidence"],
            "verification_status": "observed",
        }
        with self.assertRaisesRegex(KaiError, "confidence must be between"):
            self.runtime.database.save_environment_snapshot(
                "2026-01-01T00:00:00+00:00", "fp", {}, [claim]
            )

    def test_all_epistemic_classes_are_persisted_without_promotion(self):
        claims = []
        for classification in (
            "OBSERVATION", "FACT", "INFERENCE", "HYPOTHESIS", "UNKNOWN"
        ):
            claims.append(
                {
                    "claim": f"{classification} fixture",
                    "classification": classification,
                    "source": "deterministic fixture",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "confidence": None if classification == "UNKNOWN" else 0.7,
                    "evidence": ["fixture evidence"],
                    "verification_status": (
                        "unverified" if classification == "UNKNOWN"
                        else "verified" if classification == "FACT"
                        else "observed" if classification in ("OBSERVATION", "INFERENCE")
                        else "unverified"
                    ),
                }
            )
        saved = self.runtime.database.save_environment_snapshot(
            "2026-01-01T00:00:00+00:00", "fp", {}, claims
        )
        stored = self.runtime.database.claims_for_snapshot(saved["snapshot_id"])
        self.assertEqual(
            [claim["classification"] for claim in stored],
            ["OBSERVATION", "FACT", "INFERENCE", "HYPOTHESIS", "UNKNOWN"],
        )


class EnvironmentModelTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.runtime = KaiRuntime(KaiConfig(Path(self.temporary_directory.name)))
        self.runtime.database.initialize()

    def test_gpu_layer_states_do_not_imply_each_other(self):
        absent = fake_environment()["gpu"]
        self.assertEqual(absent["physical_hardware"]["status"], "not_visible_in_this_environment")
        self.assertEqual(absent["physical_hardware"]["physical_presence"], "unknown")
        self.assertEqual(absent["kernel_support"]["status"], "unknown")
        self.assertEqual(absent["runtime_availability"]["cuda_status"], "unknown")
        self.assertEqual(
            absent["runtime_availability"]["library_discovery_status"],
            "not_detected_in_inspected_paths",
        )
        self.assertEqual(absent["usable_capability"]["cuda_compute"], "unknown")

        unbound_device = fake_environment(gpu_devices=1, driver_bound=False)["gpu"]
        self.assertEqual(unbound_device["pci_enumeration"]["status"], "detected")
        self.assertEqual(unbound_device["kernel_support"]["status"], "no_bound_driver")
        self.assertEqual(unbound_device["runtime_availability"]["cuda_status"], "unknown")
        self.assertEqual(
            unbound_device["runtime_availability"]["library_discovery_status"],
            "not_detected_in_inspected_paths",
        )
        self.assertEqual(unbound_device["usable_capability"]["cuda_compute"], "unknown")

        runtime_without_device = fake_environment(
            runtime_libraries=["/fake/libcuda.so"]
        )["gpu"]
        self.assertEqual(
            runtime_without_device["runtime_availability"]["library_discovery_status"],
            "libraries_detected",
        )
        self.assertEqual(runtime_without_device["pci_enumeration"]["status"], "not_visible")
        self.assertEqual(runtime_without_device["usable_capability"]["cuda_compute"], "unknown")

        driver_without_runtime = fake_environment(gpu_devices=1)["gpu"]
        self.assertEqual(driver_without_runtime["kernel_support"]["status"], "driver_bound")
        self.assertEqual(
            driver_without_runtime["runtime_availability"]["cuda_status"],
            "unknown",
        )
        self.assertEqual(
            driver_without_runtime["runtime_availability"]["library_discovery_status"],
            "not_detected_in_inspected_paths",
        )
        self.assertEqual(driver_without_runtime["usable_capability"]["cuda_compute"], "unknown")

    def test_claims_keep_observation_inference_fact_and_unknown_separate(self):
        environment = fake_environment(gpu_devices=1)
        claims = build_epistemic_claims(environment, "2026-01-01T00:00:00+00:00")
        pci_claim = next(claim for claim in claims if "PCI graphics/compute function" in claim["claim"])
        self.assertEqual(pci_claim["classification"], "OBSERVATION")
        unknown = next(claim for claim in claims if "Usable CUDA acceleration" in claim["claim"])
        self.assertEqual(unknown["classification"], "UNKNOWN")
        self.assertEqual(unknown["verification_status"], "unverified")
        version_claim = next(claim for claim in claims if "KAI software version" in claim["claim"])
        self.assertEqual(version_claim["classification"], "FACT")
        self.assertEqual(version_claim["verification_status"], "verified")
        self.assertIn("contradictory_evidence", pci_claim)
        self.assertTrue(all(claim["source"] and claim["timestamp"] for claim in claims))

    def test_fake_pci_inventory_keeps_multiple_devices_and_driver_state(self):
        with tempfile.TemporaryDirectory() as directory:
            pci_root = Path(directory)
            devices = (
                ("0000:01:00.0", "0x10de", "0x1234", "0x030000", "nvidia"),
                ("0000:02:00.0", "0x8086", "0x5678", "0x030000", None),
                ("0000:03:00.0", "0x10de", "0xabcd", "0x120000", None),
            )
            for address, vendor, device, class_id, driver in devices:
                path = pci_root / address
                path.mkdir()
                (path / "vendor").write_text(vendor)
                (path / "device").write_text(device)
                (path / "class").write_text(class_id)
                if driver:
                    driver_path = pci_root / "drivers" / driver
                    driver_path.mkdir(parents=True, exist_ok=True)
                    (path / "driver").symlink_to(driver_path)
            inventory = _pci_devices(pci_root)
        self.assertEqual(len(inventory), 3)
        self.assertEqual([device["driver_bound"] for device in inventory], [True, False, False])
        self.assertEqual(inventory[2]["compute_accelerator"], True)

    def test_fingerprint_ignores_volatile_resources_and_sensitive_ids(self):
        first = fake_environment()
        first["serial_number"] = "secret-one"
        first["mac_address"] = "00:00:00:00:00:01"
        second = fake_environment()
        second["memory"]["available_bytes"] = 123
        second["serial_number"] = "secret-two"
        second["mac_address"] = "00:00:00:00:00:02"
        self.assertEqual(environment_fingerprint(first), environment_fingerprint(second))
        second["pci_devices"] = [{"address": "0000:02:00.0", "vendor_id": "0x10de",
                                  "device_id": "0x1234", "class_id": "0x030000"}]
        self.assertNotEqual(environment_fingerprint(first), environment_fingerprint(second))

    def test_snapshot_diff_reports_hardware_driver_runtime_and_resource_changes(self):
        before_env = fake_environment(gpu_devices=1, driver_bound=False)
        after_env = fake_environment(gpu_devices=1, driver_bound=True, runtime_libraries=["libcuda.so"])
        after_env["pci_devices"][0]["device_id"] = "0x9999"
        after_env["memory"]["available_bytes"] -= 500
        before_env["software_capabilities"]["docker"]["available"] = True
        after_env["software_capabilities"]["python"]["version"] = "3.15"
        after = {
            "environment": after_env,
            "environment_fingerprint": environment_fingerprint(after_env),
        }
        previous = {
            "snapshot_id": 1,
            "environment": before_env,
            "environment_fingerprint": environment_fingerprint(before_env),
        }
        changes = compare_environments(previous, after)
        self.assertFalse(changes["same_environment_profile"])
        self.assertTrue(changes["changed_hardware"])
        self.assertTrue(changes["changed_drivers"])
        self.assertTrue(changes["changed_runtime_availability"])
        self.assertTrue(changes["changed_kernel_visibility"])
        self.assertTrue(changes["changed_resources"])
        self.assertTrue(changes["changed_software"])
        self.assertIn("runtime:cuda-libraries", changes["added_capabilities"])
        self.assertIn("tool:docker", changes["removed_capabilities"])
        self.assertEqual(compare_environments(None, after)["baseline"], "first_snapshot")

    def test_self_model_is_generated_from_fake_state(self):
        environment = fake_environment()
        claims = build_epistemic_claims(environment, "2026-01-01T00:00:00+00:00")
        identity = {"identity_id": "stable-kai-id", "name": "KAI"}
        model = build_self_model(
            identity, environment, "fingerprint", capability_summary({
                **environment,
                "gpu": environment["gpu"],
            }), [], claims, "0.1.0", SCHEMA_VERSION
        )
        self.assertEqual(model["identity"], identity)
        self.assertEqual(model["environment_fingerprint"], "fingerprint")
        self.assertEqual(model["software_version"], "0.1.0")
        self.assertEqual(model["architecture_version"], SCHEMA_VERSION)
        self.assertIn("memory", model["current_resources"])
        self.assertTrue(model["uncertainties"])

    def test_snapshot_comparison_inference_is_not_promoted(self):
        first = fake_environment()
        second = fake_environment()
        previous = {
            "snapshot_id": 1,
            "environment": first,
            "environment_fingerprint": environment_fingerprint(first),
        }
        current = {
            "environment": second,
            "environment_fingerprint": environment_fingerprint(second),
        }
        changes = compare_environments(previous, current)
        self.assertTrue(changes["same_environment_profile"])
        claim = {
            "claim": "The profiles match.",
            "classification": "INFERENCE",
            "source": "fingerprint comparison",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "confidence": 0.7,
            "evidence": ["Matching profile fingerprints."],
            "verification_status": "observed",
        }
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "inference.sqlite3")
            database.initialize()
            saved = database.save_environment_snapshot(
                claim["timestamp"], "fp", {}, [claim]
            )
            stored = database.claims_for_snapshot(saved["snapshot_id"])[0]
        self.assertEqual(stored["classification"], "INFERENCE")
        self.assertNotEqual(stored["classification"], "FACT")

    def test_runtime_keeps_identity_stable_when_environment_changes(self):
        first = fake_environment()
        second = fake_environment(gpu_devices=1, driver_bound=False)
        with patch("kai.runtime.collect_environment_model", side_effect=[first, second]):
            boot = self.runtime.boot()
            status = self.runtime.status()
        self.assertEqual(boot["identity"]["identity_id"], status["identity"]["identity_id"])
        self.assertFalse(status["changes"]["same_environment_profile"])
        self.assertTrue(status["changes"]["changed_hardware"])
        self.assertFalse(status["self_model"]["environment_identity"]["same_as_previous_profile"])
        inference = next(
            claim for claim in status["claims"]
            if claim["classification"] == "INFERENCE"
        )
        self.assertIn("new or substantially changed", inference["claim"])
        changed_event = next(
            event
            for event in self.runtime.database.recent_events()
            if event["event_type"] == "ENVIRONMENT_CHANGED"
        )
        self.assertEqual(
            changed_event["environment_snapshot_id"], status["snapshot_id"]
        )
        self.assertEqual(
            self.runtime.database.get_experience(status["experience_id"])["status"],
            "COMPLETED",
        )
        working = {
            entry["key"]: entry["value"]
            for entry in self.runtime.database.working_memory()
        }
        self.assertEqual(
            working["self_model"]["environment_fingerprint"],
            status["environment_fingerprint"],
        )

    def test_boot_recovers_interrupted_measurement_once(self):
        self.runtime.database.record_event(
            "MEASUREMENT_STARTED", "fixture start", source="test"
        )
        with patch(
            "kai.runtime.collect_environment_model",
            return_value=fake_environment(),
        ):
            boot = self.runtime.boot()
        self.assertEqual(len(boot["recovered_event_ids"]), 1)
        self.assertEqual(
            len(self.runtime.database.recover_interrupted_operations()), 0
        )

    def test_runtime_persists_claims_and_safe_measurements(self):
        with patch(
            "kai.runtime.collect_environment_model",
            side_effect=[fake_environment(), fake_environment()],
        ):
            result = self.runtime.measure()
            status = self.runtime.status()
        saved = self.runtime.database.latest_environment_snapshot()
        self.assertNotEqual(saved["snapshot_id"], result["snapshot_id"])
        self.assertEqual(saved["measurements"]["cpu"]["status"], "measured")
        self.assertIn(saved["measurements"]["disk_read"]["status"], {"measured", "unavailable"})
        self.assertTrue(self.runtime.database.claims_for_snapshot(saved["snapshot_id"]))
        self.assertEqual(status["measurements"]["cpu"]["status"], "measured")
        self.assertGreaterEqual(
            status["self_model"]["experience_state"]["experience_count"], 3
        )

    def test_cpu_frequency_change_is_a_resource_not_hardware_identity_change(self):
        before = fake_environment()
        after = fake_environment()
        before["cpu"]["frequency"] = {"maximum_khz": {"minimum": 1000, "maximum": 1000}}
        after["cpu"]["frequency"] = {"maximum_khz": {"minimum": 2000, "maximum": 2000}}
        previous = {
            "snapshot_id": 1,
            "environment": before,
            "environment_fingerprint": environment_fingerprint(before),
        }
        current = {
            "environment": after,
            "environment_fingerprint": environment_fingerprint(after),
        }
        changes = compare_environments(previous, current)
        self.assertFalse(changes["changed_hardware"])
        self.assertIn(
            "cpu_frequency",
            [item["component"] for item in changes["changed_resources"]],
        )

    def test_snapshot_retention_keeps_newest_snapshots(self):
        for index in range(3):
            self.runtime.database.save_environment_snapshot(
                f"2026-01-01T00:00:0{index}+00:00",
                f"fp-{index}",
                {"snapshot": index},
                [],
                retention=2,
            )
        self.assertEqual(self.runtime.database.environment_snapshot_count(), 2)
        self.assertEqual(
            self.runtime.database.latest_environment_snapshot()["environment_fingerprint"],
            "fp-2",
        )


if __name__ == "__main__":
    unittest.main()
