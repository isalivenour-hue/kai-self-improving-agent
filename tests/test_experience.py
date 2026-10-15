import math
import tempfile
import unittest
from pathlib import Path

from kai.database import Database
from kai.errors import KaiError


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.database = Database(Path(self.temporary_directory.name) / "state.sqlite3")
        self.database.initialize()

    def test_events_keep_provenance_causality_and_transactional_order(self):
        first = self.database.record_event(
            "CUSTOM_EXTENSION_EVENT",
            "First event",
            {"value": 1},
            source="test",
            occurred_at="2026-01-01T03:00:00+03:00",
        )
        second = self.database.record_event(
            "ERROR",
            "Second event",
            {"value": 2},
            source="test",
            causal_parent_id=first,
            confidence=0.8,
            epistemic_status="OBSERVATION",
            occurred_at="2026-01-01T00:00:01+00:00",
        )
        events = self.database.recent_events()
        self.assertEqual([event["event_id"] for event in events], [second, first])
        self.assertEqual(events[0]["causal_parent_id"], first)
        self.assertEqual(events[0]["payload"], {"value": 2})
        self.assertEqual(events[1]["occurred_at"], "2026-01-01T00:00:00+00:00")
        self.assertEqual(
            [event["event_id"] for event in self.database.recent_events(since="2026-01-01T00:00:01Z")],
            [second],
        )
        self.assertIn("CUSTOM_EXTENSION_EVENT", self.database.event_types())

    def test_partial_experiences_remain_open_and_persisted(self):
        experience_id = self.database.create_experience(
            {"task": "observe"},
            observation={"cpu": "visible"},
            timestamp="2026-01-01T00:00:00Z",
        )
        episode = self.database.get_experience(experience_id)
        self.assertEqual(episode["context"], {"task": "observe"})
        self.assertEqual(episode["observation"], {"cpu": "visible"})
        self.assertIsNone(episode["prediction_id"])
        self.assertEqual(episode["status"], "OPEN")

    def test_prediction_evaluation_records_error_and_auditable_learning(self):
        experience_id = self.database.create_experience(
            {"measurement": "cpu"}, belief={"capacity": "moderate"}
        )
        prediction_id = self.database.create_prediction(
            "CPU baseline should take about one second.",
            1.0,
            context={"iterations": 100000},
            confidence=0.65,
            basis="Prior local measurement",
            observation_window="next explicit baseline",
            experience_id=experience_id,
            timestamp="2026-01-01T00:00:00Z",
        )
        result = self.database.evaluate_prediction(
            prediction_id,
            1.2,
            tolerance=0.25,
            timestamp="2026-01-01T00:00:02Z",
        )
        self.assertEqual(result["status"], "PARTIALLY_CONFIRMED")
        self.assertAlmostEqual(result["difference"]["difference"], 0.2)
        self.assertAlmostEqual(result["difference"]["error_magnitude"], 0.2)
        prediction = self.database.get_prediction(prediction_id)
        self.assertEqual(prediction["status"], "PARTIALLY_CONFIRMED")
        self.assertEqual(prediction["observed_outcome"], 1.2)
        episode = self.database.get_experience(experience_id)
        self.assertEqual(episode["status"], "COMPLETED")
        self.assertEqual(episode["outcome"], 1.2)
        learning = self.database.list_learning_records()[0]
        self.assertEqual(learning["experience_id"], experience_id)
        self.assertEqual(learning["before"], {"capacity": "moderate"})
        self.assertEqual(learning["happened"], 1.2)
        self.assertEqual(learning["predicted"], 1.0)
        self.assertEqual(learning["source_experiences"], [experience_id])
        events = self.database.recent_events()
        learned_event = next(item for item in events if item["event_type"] == "LEARNING_OCCURRED")
        evaluated_event = next(
            item for item in events if item["event_type"] == "PREDICTION_EVALUATED"
        )
        self.assertEqual(learned_event["causal_parent_id"], evaluated_event["event_id"])

    def test_exact_refuted_unknown_and_expired_prediction_states(self):
        def create_prediction(expected):
            return self.database.create_prediction(
                "Testable hypothesis",
                expected,
                context={},
                confidence=0.5,
                basis="Deterministic fixture",
            )

        confirmed_id = create_prediction("ready")
        self.assertEqual(
            self.database.evaluate_prediction(confirmed_id, "ready")["status"],
            "CONFIRMED",
        )
        null_id = create_prediction(None)
        self.assertEqual(
            self.database.evaluate_prediction(null_id, None)["status"],
            "CONFIRMED",
        )
        boolean_mismatch_id = create_prediction(1)
        self.assertEqual(
            self.database.evaluate_prediction(boolean_mismatch_id, True)["status"],
            "REFUTED",
        )
        refuted_id = create_prediction("ready")
        self.assertEqual(
            self.database.evaluate_prediction(refuted_id, "not ready")["status"],
            "REFUTED",
        )
        prior_learning_count = len(self.database.list_learning_records())
        unknown_id = create_prediction("ready")
        unknown = self.database.evaluate_prediction(
            unknown_id, None, outcome_available=False
        )
        self.assertEqual(unknown["status"], "UNKNOWN")
        self.assertIsNone(unknown["learning_id"])
        unknown_episode = self.database.get_experience(unknown["experience_id"])
        self.assertEqual(unknown_episode["status"], "OPEN")
        self.assertEqual(
            len(self.database.list_learning_records()), prior_learning_count
        )
        expired_id = create_prediction("ready")
        self.database.expire_prediction(expired_id)
        self.assertEqual(self.database.get_prediction(expired_id)["status"], "EXPIRED")

    def test_failed_prediction_evaluation_rolls_back_all_linked_writes(self):
        prediction_id = self.database.create_prediction(
            "Finite measurement",
            1.0,
            context={},
            confidence=0.5,
            basis="test",
        )
        before = self.database.event_count()
        with self.assertRaises(KaiError):
            self.database.evaluate_prediction(prediction_id, math.nan)
        self.assertEqual(self.database.get_prediction(prediction_id)["status"], "PENDING")
        self.assertEqual(self.database.event_count(), before)
        self.assertEqual(self.database.list_learning_records(), [])

    def test_layered_memory_and_consolidation_preserve_candidate_uncertainty(self):
        self.database.set_working_memory("focus", {"goal": "observe"}, "test")
        self.database.add_layered_memory(
            "procedural",
            "Compare repeated readings using the same bounded measurement.",
            "explicit test strategy",
            epistemic_status="HYPOTHESIS",
        )
        for _ in range(3):
            self.database.create_experience(
                {"source": "fixture"}, observation="CUDA not visible in this namespace"
            )
        candidates = self.database.consolidate_observations(minimum_occurrences=3)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(self.database.consolidate_observations(3), [])
        semantic = self.database.list_layered_memory("semantic")[0]
        self.assertEqual(semantic["epistemic_status"], "HYPOTHESIS")
        self.assertEqual(semantic["verification_status"], "candidate")
        self.assertIsNone(semantic["confidence"])
        self.assertEqual(len(semantic["source_experiences"]), 3)
        self.assertEqual(len(self.database.list_layered_memory("procedural")), 1)
        self.database.verify_semantic_memory(
            semantic["memory_id"],
            ["Independent CUDA runtime probe succeeded."],
            "explicit test verification",
        )
        verified = self.database.list_layered_memory("semantic")[0]
        self.assertEqual(verified["epistemic_status"], "FACT")
        self.assertEqual(verified["verification_status"], "verified")
        self.assertEqual(verified["source"], "deterministic episodic consolidation")
        self.assertEqual(self.database.working_memory()[0]["key"], "focus")

    def test_unknown_memory_cannot_be_silently_verified(self):
        memory_id = self.database.add_layered_memory(
            "semantic",
            "CUDA usability is unknown.",
            "test",
            epistemic_status="UNKNOWN",
            verification_status="unverified",
        )
        with self.assertRaisesRegex(KaiError, "UNKNOWN memory"):
            self.database.add_layered_memory(
                "semantic",
                "CUDA remains unknown.",
                "test",
                epistemic_status="UNKNOWN",
                verification_status="verified",
                evidence=["evidence"],
            )
        self.database.verify_semantic_memory(memory_id, ["new evidence"], "test")
        claim = self.database.list_layered_memory("semantic")[0]
        self.assertEqual(claim["epistemic_status"], "FACT")
        verified_event = next(
            event
            for event in self.database.recent_events()
            if event["event_type"] == "CLAIM_VERIFIED"
        )
        self.assertEqual(verified_event["payload"]["memory_id"], memory_id)

    def test_unverified_legacy_fact_label_stays_a_candidate(self):
        self.database.add_memory(
            "fact", "User supplied statement", "user-provided", confidence=0.9
        )
        candidate = self.database.list_layered_memory("semantic")[0]
        self.assertEqual(candidate["epistemic_status"], "HYPOTHESIS")
        self.assertEqual(candidate["verification_status"], "candidate")

    def test_contradiction_is_preserved_as_an_auditable_transition(self):
        memory_id = self.database.add_layered_memory(
            "semantic",
            "The runtime is probably available.",
            "prior observation",
            epistemic_status="INFERENCE",
            verification_status="unverified",
        )
        self.database.contradict_semantic_memory(
            memory_id, ["Runtime library could not be loaded."], "test probe"
        )
        claim = self.database.list_layered_memory("semantic")[0]
        self.assertEqual(claim["verification_status"], "contradicted")
        event = next(
            item
            for item in self.database.recent_events()
            if item["event_type"] == "CLAIM_CONTRADICTED"
        )
        self.assertEqual(
            event["payload"]["contradictory_evidence"],
            ["Runtime library could not be loaded."],
        )

    def test_snapshot_event_and_experience_are_committed_together(self):
        saved = self.database.save_environment_snapshot(
            "2026-01-01T00:00:00Z",
            "profile-a",
            {"operating_system": {"distribution": "FakeOS"}},
            [],
        )
        event = next(
            item
            for item in self.database.recent_events()
            if item["event_id"] == saved["event_id"]
        )
        experience = self.database.get_experience(saved["experience_id"])
        self.assertEqual(event["event_type"], "ENVIRONMENT_OBSERVED")
        self.assertEqual(event["environment_snapshot_id"], saved["snapshot_id"])
        self.assertEqual(experience["causal_event_id"], saved["event_id"])
        self.assertEqual(
            self.database.working_memory()[0]["value"]["experience_id"],
            saved["experience_id"],
        )

    def test_interrupted_transaction_rolls_back_and_recovery_is_repeatable(self):
        before = self.database.event_count()
        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            with self.database.connection() as connection:
                connection.execute(
                    "INSERT INTO events (occurred_at, event_type, summary, details_json) "
                    "VALUES ('2026-01-01T00:00:00+00:00', 'ERROR', 'partial', '{}')"
                )
                raise RuntimeError("simulated interruption")
        self.assertEqual(self.database.event_count(), before)
        started_id = self.database.record_event(
            "MEASUREMENT_STARTED", "measurement started", source="test"
        )
        recovered = self.database.recover_interrupted_operations()
        self.assertEqual(len(recovered), 1)
        recovery = next(event for event in self.database.recent_events() if event["event_id"] == recovered[0])
        self.assertEqual(recovery["causal_parent_id"], started_id)
        self.assertEqual(self.database.recover_interrupted_operations(), [])
        self.assertEqual(self.database.integrity_check(), "ok")


if __name__ == "__main__":
    unittest.main()
