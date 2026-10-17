import json
import hashlib
import math
import re
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from kai.errors import KaiError
from kai.evaluation_protocol import (
    CPU_PREDICTOR_PROTOCOL,
    EvaluationProtocol,
    validate_protocol_comparison,
)
from kai.experience import EVENT_TYPES, Event, Experience, LearningRecord, Prediction
from kai.experiment_registry import (
    CPU_BASELINE_EXPERIMENT,
    ExperimentResult,
    classify_evidence_quality,
    validate_experiment_result,
)
from kai.predictive import (
    CPU_BASELINE_DOMAIN,
    CPU_BASELINE_MODEL_TYPE,
    CPU_BASELINE_TOLERANCE_SECONDS,
    CPU_BASELINE_UPDATE_RULE,
    confidence_from_statistics,
    Experiment,
    initial_parameters,
    PredictiveModel,
    prediction_context_key,
    prediction_statistics,
    uncertainty,
    update_running_mean,
    validate_cpu_baseline_context,
)
from kai.self_improvement import (
    IMPROVEMENT_EVIDENCE_MINIMUM,
    IMPROVEMENT_DECISIONS,
    IMPROVEMENT_EVALUATION_STATUSES,
    OPPORTUNITY_STATUSES,
    SELF_EPISTEMIC_STATES,
    classify_candidate_comparison,
    validate_improvement_decision,
    validate_self_observation,
)

SCHEMA_VERSION = 10
MAX_LIST_LIMIT = 1000
IDENTITY_NAME = "KAI"
IDENTITY_PRINCIPLES = [
    "Prefer truth over confidence.",
    "Distinguish stored self-model information from consciousness.",
    "Record uncertainty instead of inventing facts.",
]
ENGINEERING_TASK_STATES = {
    "OBJECTIVE",
    "REPOSITORY_INSPECTION",
    "PLAN",
    "PROPOSED_CHANGES",
    "CANDIDATE_WORKSPACE",
    "PATCH_APPLICATION",
    "STATIC_VALIDATION",
    "TESTING",
    "BENCHMARKING",
    "EVALUATION",
    "ACCEPTED",
    "REJECTED",
    "INCONCLUSIVE",
    "FAILED",
    "BLOCKED",
    "UNKNOWN",
}
ENGINEERING_TASK_STAGES = ENGINEERING_TASK_STATES | {"MODEL_REVIEW"}
LLM_TASK_CATEGORIES = {
    "general",
    "planning",
    "coding",
    "code_review",
    "summarization",
    "classification",
    "reasoning",
    "structured_output",
}


def _validate_llm_target(provider: str, model_id: str) -> None:
    if (
        not isinstance(provider, str)
        or not provider
        or not provider.replace("_", "").isalnum()
    ):
        raise KaiError("LLM provider name must be a non-empty identifier.")
    if (
        not isinstance(model_id, str)
        or not model_id.strip()
        or len(model_id) > 200
        or any(ord(char) < 32 for char in model_id)
    ):
        raise KaiError("LLM model ID must be non-empty text of at most 200 characters.")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_timestamp(value: str) -> str:
    if isinstance(value, str) and value.endswith(("Z", "z")):
        value = value[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise KaiError("Timestamp must be an ISO-8601 value with a timezone.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise KaiError("Timestamp must include a timezone.")
    return timestamp.astimezone(timezone.utc).isoformat()


def _insert_event(
    connection: sqlite3.Connection,
    event_type: str,
    summary: str,
    details: dict,
    source: str = "kai",
    environment_snapshot_id: int | None = None,
    causal_parent_id: int | None = None,
    experiment_id: int | None = None,
    confidence: float | None = None,
    epistemic_status: str = "OBSERVATION",
    occurred_at: str | None = None,
    execution_id: int | None = None,
    experiment_definition_key: str | None = None,
    experiment_definition_id: str | None = None,
    experiment_version: int | None = None,
) -> int:
    payload = _encode_json(details)
    cursor = connection.execute(
        "INSERT INTO events (occurred_at, event_type, summary, details_json, source, "
        "environment_snapshot_id, causal_parent_id, experiment_id, execution_id, "
        "experiment_definition_key, experiment_definition_id, experiment_version, "
        "payload_json, confidence, epistemic_status) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            occurred_at or utc_now(),
            event_type,
            summary,
            payload,
            source,
            environment_snapshot_id,
            causal_parent_id,
            experiment_id,
            execution_id,
            experiment_definition_key,
            experiment_definition_id,
            experiment_version,
            payload,
            confidence,
            epistemic_status,
        ),
    )
    return int(cursor.lastrowid)


def _json_optional(value: object | None) -> str | None:
    if value is None:
        return None
    try:
        return _encode_json(value)
    except (TypeError, ValueError) as exc:
        raise KaiError(f"Value is not finite JSON data: {exc}") from exc


def _encode_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _validate_confidence(value: float | None, label: str) -> None:
    if value is not None and (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise KaiError(f"{label} confidence must be between 0.0 and 1.0.")


def _validate_limit(limit: int, label: str) -> None:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIST_LIMIT:
        raise KaiError(f"{label} limit must be between 1 and {MAX_LIST_LIMIT}.")


def _decode_json(value: str | None) -> object | None:
    return json.loads(value) if value is not None else None


def _experience_row(row: sqlite3.Row) -> Experience:
    return {
        "experience_id": row["experience_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "status": row["status"],
        "context": json.loads(row["context_json"]),
        "observation": _decode_json(row["observation_json"]),
        "belief": _decode_json(row["belief_json"]),
        "prediction_id": row["prediction_id"],
        "action": _decode_json(row["action_json"]),
        "outcome": _decode_json(row["outcome_json"]),
        "evaluation": _decode_json(row["evaluation_json"]),
        "learning": _decode_json(row["learning_json"]),
        "causal_event_id": row["causal_event_id"],
        "source_event_ids": json.loads(row["source_event_ids_json"]),
    }


def _prediction_row(row: sqlite3.Row) -> Prediction:
    return {
        "prediction_id": row["prediction_id"],
        "experience_id": row["experience_id"],
        "created_at": row["created_at"],
        "context": json.loads(row["context_json"]),
        "hypothesis": row["hypothesis"],
        "expected_outcome": json.loads(row["expected_outcome_json"]),
        "confidence": row["confidence"],
        "basis": row["basis"],
        "observation_window": row["observation_window"],
        "status": row["status"],
        "evaluated_at": row["evaluated_at"],
        "observed_outcome": _decode_json(row["observed_outcome_json"]),
        "difference": _decode_json(row["difference_json"]),
        "evaluation": row["evaluation"],
        "created_event_id": row["created_event_id"],
        "evaluated_event_id": row["evaluated_event_id"],
        "model_id": row["model_id"],
        "model_version": row["model_version"],
        "experiment_id": row["experiment_id"],
        "execution_id": row["execution_id"],
        "experiment_definition_id": row["experiment_definition_id"],
        "experiment_version": row["experiment_version"],
        "environment_snapshot_id": row["environment_snapshot_id"],
    }


def _experiment_row(row: sqlite3.Row) -> Experiment:
    return {
        "experiment_id": row["execution_id"],
        "execution_id": row["execution_id"],
        "experience_id": row["experience_id"],
        "domain": row["domain"],
        "question": row["question"],
        "hypothesis": row["hypothesis"],
        "experiment_type": row["experiment_type"],
        "resource_estimate": json.loads(row["resource_estimate_json"]),
        "status": row["status"],
        "uncertainty_before": json.loads(row["uncertainty_before_json"]),
        "uncertainty_after": _decode_json(row["uncertainty_after_json"]),
        "information_gain": _decode_json(row["information_gain_json"]),
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "output": _decode_json(row["output_json"]),
    }


def _experiment_definition_row(row: sqlite3.Row) -> dict:
    return {
        "definition_key": row["definition_key"],
        "experiment_id": row["experiment_id"],
        "domain": row["domain"],
        "name": row["name"],
        "description": row["description"],
        "version": row["version"],
        "status": row["status"],
        "procedure_id": row["procedure_id"],
        "parameter_specification": json.loads(row["parameter_specification_json"]),
        "context_specification": json.loads(row["context_specification_json"]),
        "outcome_specification": json.loads(row["outcome_specification_json"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _experiment_execution_row(row: sqlite3.Row) -> dict:
    return {
        "execution_id": row["execution_id"],
        "experiment_id": row["experiment_definition_id"],
        "experiment_version": row["experiment_version"],
        "procedure_id": row["procedure_id"],
        "started_at": row["started_at"],
        "completed_at": row["completed_at"],
        "execution_status": row["execution_status"],
        "legacy_status": row["status"],
        "environment_snapshot_id": row["environment_snapshot_id"],
        "context_profile": row["context_profile"],
        "context": json.loads(row["context_json"]),
        "parameters": json.loads(row["parameters_json"]),
        "raw_result": _decode_json(row["raw_result_json"]),
        "output": _decode_json(row["output_json"]),
        "failure_information": row["failure_information"],
        "result": (
            {
                "outcome_value": row["outcome_value"],
                "unit": row["unit"],
                "sample_count": row["sample_count"],
                "duration_seconds": row["duration_seconds"],
                "metadata": json.loads(row["metadata_json"]),
                "measurement_status": row["measurement_status"],
            }
            if row["result_execution_id"] is not None
            else None
        ),
        "evaluation": (
            {
                "prediction_id": row["evaluation_prediction_id"],
                "expected_value": row["expected_value"],
                "observed_value": row["observed_value"],
                "error": row["error"],
                "absolute_error": row["absolute_error"],
                "relative_error": row["relative_error"],
                "status": row["evaluation_status"],
                "evidence_quality": row["evidence_quality"],
                "evidence": json.loads(row["evidence_json"]),
            }
            if row["evaluation_execution_id"] is not None
            else None
        ),
    }


def _learning_row(row: sqlite3.Row) -> LearningRecord:
    return {
        "learning_id": row["learning_id"],
        "experience_id": row["experience_id"],
        "created_at": row["created_at"],
        "before": json.loads(row["before_json"]),
        "happened": json.loads(row["happened_json"]),
        "predicted": json.loads(row["predicted_json"]),
        "error": json.loads(row["error_json"]),
        "changed": json.loads(row["changed_json"]),
        "why": row["why"],
        "evidence": json.loads(row["evidence_json"]),
        "confidence": row["confidence"],
        "source_experiences": json.loads(row["source_experiences_json"]),
    }


def _memory_item_row(row: sqlite3.Row) -> dict:
    return {
        "memory_id": row["memory_id"],
        "layer": row["layer"],
        "created_at": row["created_at"],
        "content": row["content"],
        "source": row["source"],
        "epistemic_status": row["epistemic_status"],
        "verification_status": row["verification_status"],
        "confidence": row["confidence"],
        "evidence": json.loads(row["evidence_json"]),
        "source_experiences": json.loads(row["source_experiences_json"]),
    }


def _validate_references(
    connection: sqlite3.Connection, table: str, column: str, identifiers: list[int]
) -> None:
    if table not in {"events", "experiences"} or column not in {
        "event_id", "experience_id"
    }:
        raise KaiError("Invalid internal reference target.")
    for identifier in identifiers:
        if connection.execute(
            f"SELECT 1 FROM {table} WHERE {column} = ?", (identifier,)
        ).fetchone() is None:
            raise KaiError(f"Referenced {table[:-1]} {identifier} does not exist.")


def _latest_experience_event_id(
    connection: sqlite3.Connection, experience_ids: list[int]
) -> int | None:
    _validate_references(connection, "experiences", "experience_id", experience_ids)
    event_ids = []
    for experience_id in experience_ids:
        row = connection.execute(
            "SELECT source_event_ids_json FROM experiences WHERE experience_id = ?",
            (experience_id,),
        ).fetchone()
        event_ids.extend(json.loads(row["source_event_ids_json"]))
    return max(event_ids) if event_ids else None


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            connection = sqlite3.connect(self.path, timeout=5.0)
        except sqlite3.Error as exc:
            raise KaiError(f"Cannot open state database {self.path}: {exc}") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
        except sqlite3.Error as exc:
            connection.rollback()
            raise KaiError(f"State database operation failed: {exc}") from exc
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS schema_info "
                    "(singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                    "version INTEGER NOT NULL)"
                )
                row = connection.execute(
                    "SELECT version FROM schema_info WHERE singleton = 1"
                ).fetchone()
                version = int(row["version"]) if row else 0
                if version > SCHEMA_VERSION:
                    raise KaiError(
                        f"State schema version {version} is newer than this KAI "
                        f"version supports ({SCHEMA_VERSION})."
                    )
                if version < 1:
                    self._migrate_to_v1(connection)
                    version = 1
                if version < 2:
                    self._migrate_to_v2(connection)
                    version = 2
                if version < 3:
                    self._migrate_to_v3(connection)
                    version = 3
                if version < 4:
                    self._migrate_to_v4(connection)
                    version = 4
                if version < 5:
                    self._migrate_to_v5(connection)
                    version = 5
                if version < 6:
                    self._migrate_to_v6(connection)
                    version = 6
                if version < 7:
                    self._migrate_to_v7(connection)
                    version = 7
                if version < 8:
                    self._migrate_to_v8(connection)
                    version = 8
                if version < 9:
                    self._migrate_to_v9(connection)
                    version = 9
                if version < 10:
                    self._migrate_to_v10(connection)
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot initialize state database: {exc}") from exc
        self.integrity_check()

    @staticmethod
    def _migrate_to_v1(connection: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE identity (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                identity_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                principles_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                summary TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE memories (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN
                    ('fact', 'observation', 'inference', 'hypothesis',
                     'speculation', 'unknown')),
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL CHECK (confidence IS NULL OR
                    (confidence >= 0.0 AND confidence <= 1.0))
            )
            """,
            """
            CREATE TABLE goals (
                goal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                title TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active'
                    CHECK (status IN ('active', 'completed', 'paused')),
                success_criteria TEXT NOT NULL DEFAULT ''
            )
            """,
        )
        for statement in statements:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO schema_info (singleton, version) VALUES (1, 1) "
            "ON CONFLICT(singleton) DO UPDATE SET version = excluded.version"
        )
        connection.execute(
            "INSERT INTO identity "
            "(singleton, identity_id, name, created_at, principles_json) "
            "VALUES (1, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                IDENTITY_NAME,
                utc_now(),
                json.dumps(IDENTITY_PRINCIPLES),
            ),
        )

    @staticmethod
    def _migrate_to_v2(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE environment_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                environment_fingerprint TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                measurements_json TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE epistemic_claims (
                claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL REFERENCES environment_snapshots(snapshot_id)
                    ON DELETE CASCADE,
                claim TEXT NOT NULL,
                classification TEXT NOT NULL CHECK (classification IN
                    ('OBSERVATION', 'FACT', 'INFERENCE', 'HYPOTHESIS', 'UNKNOWN')),
                source TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                confidence REAL CHECK (confidence IS NULL OR
                    (confidence >= 0.0 AND confidence <= 1.0)),
                evidence_json TEXT NOT NULL,
                verification_status TEXT NOT NULL CHECK (verification_status IN
                    ('observed', 'verified', 'unverified', 'contradicted')),
                contradictory_evidence_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX epistemic_claims_snapshot_idx "
            "ON epistemic_claims(snapshot_id, claim_id)"
        )
        connection.execute(
            "UPDATE schema_info SET version = 2 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v3(connection: sqlite3.Connection) -> None:
        connection.execute(
            "ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'legacy'"
        )
        connection.execute(
            "ALTER TABLE events ADD COLUMN environment_snapshot_id INTEGER "
            "REFERENCES environment_snapshots(snapshot_id) ON DELETE SET NULL"
        )
        connection.execute(
            "ALTER TABLE events ADD COLUMN causal_parent_id INTEGER "
            "REFERENCES events(event_id) ON DELETE SET NULL"
        )
        connection.execute(
            "ALTER TABLE events ADD COLUMN payload_json TEXT NOT NULL DEFAULT '{}'"
        )
        connection.execute(
            "ALTER TABLE events ADD COLUMN confidence REAL CHECK "
            "(confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0))"
        )
        connection.execute(
            "ALTER TABLE events ADD COLUMN epistemic_status TEXT NOT NULL "
            "DEFAULT 'OBSERVATION' CHECK (epistemic_status IN "
            "('OBSERVATION', 'FACT', 'INFERENCE', 'HYPOTHESIS', 'UNKNOWN'))"
        )
        connection.execute(
            "UPDATE events SET payload_json = details_json"
        )
        connection.execute(
            """
            CREATE TABLE experiences (
                experience_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('OPEN', 'COMPLETED')),
                context_json TEXT NOT NULL DEFAULT '{}',
                observation_json TEXT,
                belief_json TEXT,
                action_json TEXT,
                outcome_json TEXT,
                evaluation_json TEXT,
                learning_json TEXT,
                causal_event_id INTEGER REFERENCES events(event_id)
                    ON DELETE SET NULL,
                source_event_ids_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id INTEGER REFERENCES experiences(experience_id)
                    ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                context_json TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                expected_outcome_json TEXT NOT NULL,
                confidence REAL CHECK (confidence IS NULL OR
                    (confidence >= 0.0 AND confidence <= 1.0)),
                basis TEXT NOT NULL,
                observation_window TEXT,
                status TEXT NOT NULL CHECK (status IN
                    ('PENDING', 'CONFIRMED', 'PARTIALLY_CONFIRMED', 'REFUTED',
                     'EXPIRED', 'UNKNOWN')),
                evaluated_at TEXT,
                observed_outcome_json TEXT,
                difference_json TEXT,
                evaluation TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX predictions_status_idx ON predictions(status, prediction_id)"
        )
        connection.execute(
            "ALTER TABLE experiences ADD COLUMN prediction_id INTEGER "
            "REFERENCES predictions(prediction_id) ON DELETE SET NULL"
        )
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN created_event_id INTEGER "
            "REFERENCES events(event_id) ON DELETE SET NULL"
        )
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN evaluated_event_id INTEGER "
            "REFERENCES events(event_id) ON DELETE SET NULL"
        )
        connection.execute(
            """
            CREATE TABLE learning_records (
                learning_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id INTEGER REFERENCES experiences(experience_id)
                    ON DELETE SET NULL,
                created_at TEXT NOT NULL,
                before_json TEXT NOT NULL,
                happened_json TEXT NOT NULL,
                predicted_json TEXT NOT NULL,
                error_json TEXT NOT NULL,
                changed_json TEXT NOT NULL,
                why TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                confidence REAL CHECK (confidence IS NULL OR
                    (confidence >= 0.0 AND confidence <= 1.0)),
                source_experiences_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE memory_items (
                memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                layer TEXT NOT NULL CHECK (layer IN ('semantic', 'procedural')),
                created_at TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                epistemic_status TEXT NOT NULL CHECK (epistemic_status IN
                    ('OBSERVATION', 'FACT', 'INFERENCE', 'HYPOTHESIS', 'UNKNOWN')),
                verification_status TEXT NOT NULL CHECK (verification_status IN
                    ('candidate', 'unverified', 'verified', 'contradicted')),
                confidence REAL CHECK (confidence IS NULL OR
                    (confidence >= 0.0 AND confidence <= 1.0)),
                evidence_json TEXT NOT NULL DEFAULT '[]',
                source_experiences_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        connection.execute(
            "CREATE INDEX memory_items_layer_idx "
            "ON memory_items(layer, memory_id DESC)"
        )
        connection.execute(
            """
            CREATE TABLE working_memory (
                memory_key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                source TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "UPDATE schema_info SET version = 3 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v4(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE predictive_models (
                model_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                context_key TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'REVERTED')),
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                confidence_json TEXT NOT NULL,
                performance_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX predictive_models_domain_idx "
            "ON predictive_models(domain, context_key)"
        )
        connection.execute(
            """
            CREATE TABLE predictive_model_versions (
                model_id TEXT NOT NULL REFERENCES predictive_models(model_id)
                    ON DELETE CASCADE,
                version INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                confidence_json TEXT NOT NULL,
                performance_json TEXT NOT NULL,
                evidence_refs_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'REVERTED')),
                PRIMARY KEY(model_id, version)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experience_id INTEGER NOT NULL REFERENCES experiences(experience_id)
                    ON DELETE CASCADE,
                domain TEXT NOT NULL,
                question TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                experiment_type TEXT NOT NULL,
                resource_estimate_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN
                    ('RUNNING', 'COMPLETED', 'FAILED', 'INTERRUPTED')),
                uncertainty_before_json TEXT NOT NULL,
                uncertainty_after_json TEXT,
                information_gain_json TEXT,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                output_json TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX experiments_domain_idx "
            "ON experiments(domain, experiment_id DESC)"
        )
        connection.execute(
            """
            CREATE TABLE predictive_model_updates (
                update_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL REFERENCES predictive_models(model_id),
                experience_id INTEGER NOT NULL REFERENCES experiences(experience_id),
                prediction_id INTEGER REFERENCES predictions(prediction_id),
                experiment_id INTEGER REFERENCES experiments(experiment_id),
                created_at TEXT NOT NULL,
                version_before INTEGER NOT NULL,
                version_after INTEGER NOT NULL,
                update_rule TEXT NOT NULL,
                parameters_before_json TEXT NOT NULL,
                parameters_after_json TEXT NOT NULL,
                input_evidence_json TEXT NOT NULL,
                prediction_error_json TEXT,
                reverses_update_id INTEGER REFERENCES predictive_model_updates(update_id)
            )
            """
        )
        connection.execute(
            "ALTER TABLE predictive_model_updates ADD COLUMN update_event_id INTEGER "
            "REFERENCES events(event_id) ON DELETE SET NULL"
        )
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN model_id TEXT "
            "REFERENCES predictive_models(model_id)"
        )
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN model_version INTEGER"
        )
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN experiment_id INTEGER "
            "REFERENCES experiments(experiment_id)"
        )
        connection.execute(
            "ALTER TABLE experiences ADD COLUMN experiment_id INTEGER "
            "REFERENCES experiments(experiment_id)"
        )
        connection.execute(
            "ALTER TABLE events ADD COLUMN experiment_id INTEGER "
            "REFERENCES experiments(experiment_id) ON DELETE SET NULL"
        )
        connection.execute(
            "CREATE INDEX predictions_model_idx "
            "ON predictions(model_id, model_version, prediction_id)"
        )
        connection.execute(
            "UPDATE schema_info SET version = 4 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v5(connection: sqlite3.Connection) -> None:
        connection.execute(
            "ALTER TABLE experiments RENAME TO experiment_executions"
        )
        connection.execute(
            "ALTER TABLE experiment_executions RENAME COLUMN experiment_id TO execution_id"
        )
        connection.execute("DROP INDEX experiments_domain_idx")
        connection.execute(
            """
            CREATE TABLE experiments (
                definition_key TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                status TEXT NOT NULL CHECK (status IN
                    ('DEFINED', 'ENABLED', 'DISABLED', 'DEPRECATED')),
                procedure_id TEXT NOT NULL,
                parameter_specification_json TEXT NOT NULL,
                context_specification_json TEXT NOT NULL,
                outcome_specification_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(experiment_id, version)
            )
            """
        )
        connection.execute(
            "CREATE INDEX experiments_status_idx ON experiments(status, experiment_id)"
        )

        definition = CPU_BASELINE_EXPERIMENT.to_dict()
        now = utc_now()
        definition_key = f"{definition['experiment_id']}@{definition['version']}"
        connection.execute(
            "INSERT INTO experiments "
            "(definition_key, experiment_id, domain, name, description, version, "
            "status, procedure_id, parameter_specification_json, "
            "context_specification_json, outcome_specification_json, created_at, "
            "updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                definition_key,
                definition["experiment_id"],
                definition["domain"],
                definition["name"],
                definition["description"],
                definition["version"],
                definition["status"],
                definition["procedure_id"],
                _encode_json(definition["parameter_specification"]),
                _encode_json(definition["context_specification"]),
                _encode_json(definition["outcome_specification"]),
                now,
                now,
            ),
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN experiment_definition_key "
            "TEXT REFERENCES experiments(definition_key)"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN experiment_definition_id TEXT"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN experiment_version INTEGER"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN procedure_id TEXT"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN execution_status TEXT "
            "NOT NULL DEFAULT 'UNKNOWN' CHECK (execution_status IN "
            "('RUNNING', 'SUCCEEDED', 'FAILED', 'ABORTED', 'UNKNOWN'))"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN environment_snapshot_id "
            "INTEGER REFERENCES environment_snapshots(snapshot_id) ON DELETE SET NULL"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN context_profile TEXT"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN parameters_json TEXT "
            "NOT NULL DEFAULT '{}'"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN context_json TEXT "
            "NOT NULL DEFAULT '{}'"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN raw_result_json TEXT"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN failure_information TEXT"
        )
        connection.execute(
            "ALTER TABLE experiment_executions ADD COLUMN update_reason TEXT "
            "CHECK (update_reason IS NULL OR update_reason IN "
            "('INITIAL_SEED', 'OBSERVATION_UPDATE', 'CORRECTION', 'ROLLBACK'))"
        )
        for table in ("events", "predictions", "predictive_model_updates"):
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN execution_id INTEGER "
                "REFERENCES experiment_executions(execution_id) ON DELETE SET NULL"
            )
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN experiment_definition_key TEXT "
                "REFERENCES experiments(definition_key)"
            )
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN experiment_definition_id TEXT"
            )
            connection.execute(
                f"ALTER TABLE {table} ADD COLUMN experiment_version INTEGER"
            )
        connection.execute(
            "ALTER TABLE predictions ADD COLUMN environment_snapshot_id INTEGER "
            "REFERENCES environment_snapshots(snapshot_id) ON DELETE SET NULL"
        )
        connection.execute(
            "ALTER TABLE predictive_model_versions ADD COLUMN source_execution_id "
            "INTEGER REFERENCES experiment_executions(execution_id)"
        )
        connection.execute(
            "ALTER TABLE predictive_model_updates ADD COLUMN update_reason TEXT "
            "NOT NULL DEFAULT 'OBSERVATION_UPDATE' CHECK (update_reason IN "
            "('INITIAL_SEED', 'OBSERVATION_UPDATE', 'CORRECTION', 'ROLLBACK'))"
        )
        connection.execute(
            """
            CREATE TABLE experiment_results (
                execution_id INTEGER PRIMARY KEY REFERENCES experiment_executions(execution_id)
                    ON DELETE CASCADE,
                outcome_value REAL,
                unit TEXT NOT NULL,
                sample_count INTEGER NOT NULL CHECK (sample_count >= 0),
                duration_seconds REAL CHECK (duration_seconds IS NULL OR duration_seconds >= 0),
                metadata_json TEXT NOT NULL,
                measurement_status TEXT NOT NULL CHECK (measurement_status IN
                    ('MEASURED', 'UNKNOWN', 'INVALID'))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE experiment_evaluations (
                execution_id INTEGER PRIMARY KEY REFERENCES experiment_executions(execution_id)
                    ON DELETE CASCADE,
                prediction_id INTEGER REFERENCES predictions(prediction_id),
                expected_value REAL,
                observed_value REAL,
                error REAL,
                absolute_error REAL,
                relative_error REAL,
                evaluation_status TEXT NOT NULL CHECK (evaluation_status IN
                    ('NOT_EVALUATED', 'EVALUATED', 'UNKNOWN', 'INVALID')),
                evidence_quality TEXT NOT NULL CHECK (evidence_quality IN
                    ('NONE', 'INSUFFICIENT', 'PRELIMINARY', 'ADEQUATE')),
                evidence_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX experiment_executions_definition_idx "
            "ON experiment_executions(experiment_definition_id, execution_id DESC)"
        )
        connection.execute(
            "CREATE INDEX experiment_executions_status_idx "
            "ON experiment_executions(execution_status, execution_id DESC)"
        )

        execution_rows = connection.execute(
            "SELECT execution_id, experience_id, status, started_at, completed_at, "
            "output_json, experiment_type FROM experiment_executions"
        ).fetchall()
        for row in execution_rows:
            experience = connection.execute(
                "SELECT context_json, prediction_id FROM experiences WHERE experience_id = ?",
                (row["experience_id"],),
            ).fetchone()
            context = json.loads(experience["context_json"]) if experience else {}
            output = _decode_json(row["output_json"]) or {}
            legacy_status = row["status"]
            execution_status = {
                "RUNNING": "RUNNING",
                "COMPLETED": "SUCCEEDED",
                "FAILED": "FAILED",
                "INTERRUPTED": "UNKNOWN",
            }.get(legacy_status, "UNKNOWN")
            connection.execute(
                "UPDATE experiment_executions SET experiment_definition_key = ?, "
                "experiment_definition_id = ?, experiment_version = ?, procedure_id = ?, "
                "execution_status = ?, environment_snapshot_id = ?, context_profile = ?, "
                "parameters_json = ?, context_json = ? WHERE execution_id = ?",
                (
                    definition_key,
                    definition["experiment_id"],
                    definition["version"],
                    definition["procedure_id"],
                    execution_status,
                    context.get("environment_snapshot_id"),
                    context.get("environment_profile"),
                    _encode_json({"iterations": 100_000}),
                    _encode_json(context),
                    row["execution_id"],
                ),
            )
            execution_id = row["execution_id"]
            prediction_id = experience["prediction_id"] if experience else None
            prediction = (
                connection.execute(
                    "SELECT * FROM predictions WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
                if prediction_id is not None
                else None
            )
            observed = output.get("measured_runtime_seconds")
            valid_observed = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isfinite(observed)
                and observed >= 0
            )
            if legacy_status == "COMPLETED" and valid_observed:
                connection.execute(
                    "INSERT INTO experiment_results "
                    "(execution_id, outcome_value, unit, sample_count, duration_seconds, "
                    "metadata_json, measurement_status) VALUES (?, ?, 'seconds', 1, ?, ?, 'MEASURED')",
                    (
                        execution_id,
                        observed,
                        observed,
                        _encode_json({"migrated_from_schema": 4, "legacy_output": output}),
                    ),
                )
                expected = None
                if prediction:
                    candidate = _decode_json(prediction["expected_outcome_json"])
                    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                        expected = float(candidate)
                evaluation_status = "EVALUATED" if expected is not None else "NOT_EVALUATED"
                error = observed - expected if expected is not None else None
                quality, evidence = classify_evidence_quality(
                    sample_count=1,
                    context_consistency=True,
                    measurement_status="MEASURED",
                    prediction_available=prediction is not None,
                    evaluation_available=expected is not None,
                )
                connection.execute(
                    "INSERT INTO experiment_evaluations "
                    "(execution_id, prediction_id, expected_value, observed_value, error, "
                    "absolute_error, relative_error, evaluation_status, evidence_quality, "
                    "evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        execution_id,
                        prediction_id,
                        expected,
                        observed,
                        error,
                        abs(error) if error is not None else None,
                        error / abs(expected) if expected not in (None, 0) else None,
                        evaluation_status,
                        quality,
                        _encode_json(evidence),
                        row["completed_at"] or now,
                    ),
                )
            else:
                evaluation_status = (
                    "UNKNOWN" if prediction is not None else "NOT_EVALUATED"
                )
                quality, evidence = classify_evidence_quality(
                    sample_count=0,
                    context_consistency=False,
                    measurement_status="UNKNOWN",
                    prediction_available=prediction is not None,
                    evaluation_available=False,
                )
                connection.execute(
                    "INSERT INTO experiment_evaluations "
                    "(execution_id, prediction_id, evaluation_status, evidence_quality, "
                    "evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        execution_id,
                        prediction_id,
                        evaluation_status,
                        quality,
                        _encode_json(evidence),
                        row["completed_at"] or row["started_at"],
                    ),
                )

        connection.execute(
            "UPDATE events SET execution_id = experiment_id, "
            "experiment_definition_key = ?, experiment_definition_id = ?, "
            "experiment_version = ? WHERE experiment_id IS NOT NULL",
            (definition_key, definition["experiment_id"], definition["version"]),
        )
        prediction_rows = connection.execute(
            "SELECT prediction_id, experience_id, experiment_id FROM predictions "
            "WHERE model_id IS NOT NULL"
        ).fetchall()
        for row in prediction_rows:
            context_row = connection.execute(
                "SELECT context_json FROM experiences WHERE experience_id = ?",
                (row["experience_id"],),
            ).fetchone()
            context = json.loads(context_row["context_json"]) if context_row else {}
            connection.execute(
                "UPDATE predictions SET execution_id = ?, "
                "experiment_definition_key = ?, experiment_definition_id = ?, "
                "experiment_version = ?, environment_snapshot_id = ? "
                "WHERE prediction_id = ?",
                (
                    row["experiment_id"],
                    definition_key,
                    definition["experiment_id"],
                    definition["version"],
                    context.get("environment_snapshot_id"),
                    row["prediction_id"],
                ),
            )
        connection.execute(
            "UPDATE predictive_model_updates SET execution_id = experiment_id, "
            "experiment_definition_key = ?, experiment_definition_id = ?, "
            "experiment_version = ?, update_reason = CASE "
            "WHEN version_before = 0 THEN 'INITIAL_SEED' "
            "WHEN update_rule LIKE 'rollback:%' THEN 'ROLLBACK' "
            "ELSE 'OBSERVATION_UPDATE' END WHERE experiment_id IS NOT NULL",
            (definition_key, definition["experiment_id"], definition["version"]),
        )
        connection.execute(
            "UPDATE predictive_model_versions SET source_execution_id = "
            "(SELECT experiment_id FROM predictive_model_updates "
            "WHERE predictive_model_updates.model_id = predictive_model_versions.model_id "
            "AND predictive_model_updates.version_after = predictive_model_versions.version) "
            "WHERE EXISTS (SELECT 1 FROM predictive_model_updates "
            "WHERE predictive_model_updates.model_id = predictive_model_versions.model_id "
            "AND predictive_model_updates.version_after = predictive_model_versions.version)"
        )
        connection.execute(
            "UPDATE schema_info SET version = 5 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v6(connection: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE self_model_versions (
                self_model_version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                identity_id TEXT NOT NULL REFERENCES identity(identity_id)
                    ON DELETE RESTRICT,
                created_at TEXT NOT NULL,
                kai_version TEXT NOT NULL,
                architecture_version TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                environment_fingerprint TEXT,
                source_snapshot_sha256 TEXT,
                epistemic_summary_json TEXT NOT NULL,
                model_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE self_observations (
                observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                self_model_version_id INTEGER REFERENCES
                    self_model_versions(self_model_version_id) ON DELETE RESTRICT,
                observed_at TEXT NOT NULL,
                subject TEXT NOT NULL,
                observation_type TEXT NOT NULL,
                value_json TEXT NOT NULL,
                source TEXT NOT NULL,
                epistemic_status TEXT NOT NULL CHECK (epistemic_status IN
                    ('OBSERVED', 'INFERRED', 'UNKNOWN')),
                evidence_status TEXT NOT NULL CHECK (evidence_status IN
                    ('observed', 'verified', 'unverified', 'contradicted', 'unknown')),
                confidence REAL CHECK (confidence IS NULL OR
                    (confidence >= 0.0 AND confidence <= 1.0)),
                kai_version TEXT NOT NULL,
                environment_context_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE improvement_opportunities (
                opportunity_id INTEGER PRIMARY KEY AUTOINCREMENT,
                detected_at TEXT NOT NULL,
                source_observation_ids_json TEXT NOT NULL,
                affected_subsystem TEXT NOT NULL,
                current_behavior TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                severity TEXT NOT NULL CHECK (severity IN
                    ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
                expected_impact TEXT NOT NULL,
                confidence REAL CHECK (confidence IS NULL OR
                    (confidence >= 0.0 AND confidence <= 1.0)),
                status TEXT NOT NULL CHECK (status IN
                    ('OPEN', 'INVESTIGATING', 'PROPOSED', 'EXPERIMENTING',
                     'ACCEPTED', 'REJECTED', 'DEFERRED', 'INVALIDATED')),
                version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1)
            )
            """,
            """
            CREATE TABLE improvement_opportunity_history (
                transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                opportunity_id INTEGER NOT NULL REFERENCES
                    improvement_opportunities(opportunity_id) ON DELETE RESTRICT,
                transitioned_at TEXT NOT NULL,
                from_status TEXT,
                to_status TEXT NOT NULL CHECK (to_status IN
                    ('OPEN', 'INVESTIGATING', 'PROPOSED', 'EXPERIMENTING',
                     'ACCEPTED', 'REJECTED', 'DEFERRED', 'INVALIDATED')),
                reason TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE improvement_hypotheses (
                hypothesis_id INTEGER PRIMARY KEY AUTOINCREMENT,
                opportunity_id INTEGER NOT NULL REFERENCES
                    improvement_opportunities(opportunity_id) ON DELETE RESTRICT,
                created_at TEXT NOT NULL,
                observed_evidence_json TEXT NOT NULL,
                interpretation TEXT NOT NULL,
                proposed_intervention TEXT NOT NULL,
                expected_outcome_json TEXT NOT NULL,
                uncertainty_json TEXT NOT NULL,
                negative_consequences_json TEXT NOT NULL,
                epistemic_status TEXT NOT NULL DEFAULT 'HYPOTHESIS'
                    CHECK (epistemic_status = 'HYPOTHESIS')
            )
            """,
            """
            CREATE TABLE improvement_proposals (
                proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id INTEGER NOT NULL REFERENCES
                    improvement_hypotheses(hypothesis_id) ON DELETE RESTRICT,
                proposal_version INTEGER NOT NULL CHECK (proposal_version >= 1),
                created_at TEXT NOT NULL,
                affected_component TEXT NOT NULL,
                current_implementation TEXT NOT NULL,
                proposed_implementation TEXT NOT NULL,
                reason TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                expected_benefit TEXT NOT NULL,
                expected_cost TEXT NOT NULL,
                risk TEXT NOT NULL,
                dependencies_json TEXT NOT NULL,
                test_requirements_json TEXT NOT NULL,
                benchmark_requirements_json TEXT NOT NULL,
                rollback_strategy TEXT NOT NULL,
                UNIQUE(hypothesis_id, proposal_version)
            )
            """,
            """
            CREATE TABLE kai_candidates (
                candidate_id TEXT PRIMARY KEY,
                proposal_id INTEGER NOT NULL REFERENCES
                    improvement_proposals(proposal_id) ON DELETE RESTRICT,
                parent_candidate_id TEXT REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                candidate_version INTEGER NOT NULL CHECK (candidate_version >= 1),
                parent_kai_version TEXT NOT NULL,
                source_snapshot_ref TEXT NOT NULL,
                source_snapshot_sha256 TEXT NOT NULL,
                candidate_specification_json TEXT NOT NULL,
                isolation_metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                deployment_status TEXT NOT NULL DEFAULT 'NOT_DEPLOYED'
                    CHECK (deployment_status = 'NOT_DEPLOYED'),
                UNIQUE(proposal_id, candidate_version)
            )
            """,
            """
            CREATE TABLE candidate_assessments (
                assessment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                assessed_at TEXT NOT NULL,
                test_status TEXT NOT NULL CHECK (test_status IN
                    ('NOT_RUN', 'PASSED', 'FAILED')),
                benchmark_status TEXT NOT NULL CHECK (benchmark_status IN
                    ('NOT_RUN', 'COMPLETED', 'FAILED')),
                evaluation_status TEXT NOT NULL CHECK (evaluation_status IN
                    ('UNKNOWN', 'FAILED', 'REGRESSED', 'INCONCLUSIVE',
                     'IMPROVED', 'STRONGLY_IMPROVED')),
                evidence_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE improvement_evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                created_at TEXT NOT NULL,
                hypothesis_satisfied INTEGER CHECK (hypothesis_satisfied IN (0, 1)),
                evaluation_status TEXT NOT NULL CHECK (evaluation_status IN
                    ('UNKNOWN', 'FAILED', 'REGRESSED', 'INCONCLUSIVE',
                     'IMPROVED', 'STRONGLY_IMPROVED')),
                baseline_metrics_json TEXT NOT NULL,
                candidate_metrics_json TEXT NOT NULL,
                comparison_json TEXT NOT NULL,
                test_status TEXT NOT NULL CHECK (test_status IN
                    ('NOT_RUN', 'PASSED', 'FAILED')),
                benchmark_status TEXT NOT NULL CHECK (benchmark_status IN
                    ('NOT_RUN', 'COMPLETED', 'FAILED')),
                regression_count INTEGER NOT NULL CHECK (regression_count >= 0),
                failure_count INTEGER NOT NULL CHECK (failure_count >= 0),
                evidence_quality TEXT NOT NULL CHECK (evidence_quality IN
                    ('NONE', 'INSUFFICIENT', 'PRELIMINARY', 'ADEQUATE')),
                evidence_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE improvement_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                evaluation_id INTEGER REFERENCES improvement_evaluations(evaluation_id)
                    ON DELETE RESTRICT,
                decided_at TEXT NOT NULL,
                decision TEXT NOT NULL CHECK (decision IN
                    ('REJECT', 'DEFER', 'REQUEST_MORE_EVIDENCE',
                     'ACCEPT_CANDIDATE', 'READY_FOR_DEPLOYMENT')),
                reason TEXT NOT NULL,
                evidence_json TEXT NOT NULL
            )
            """,
        )
        for statement in statements:
            connection.execute(statement)
        indexes = (
            "CREATE INDEX self_observations_subject_idx "
            "ON self_observations(subject, observation_id DESC)",
            "CREATE INDEX self_model_versions_identity_idx "
            "ON self_model_versions(identity_id, self_model_version_id DESC)",
            "CREATE INDEX improvement_opportunities_status_idx "
            "ON improvement_opportunities(status, opportunity_id DESC)",
            "CREATE INDEX improvement_hypotheses_opportunity_idx "
            "ON improvement_hypotheses(opportunity_id, hypothesis_id DESC)",
            "CREATE INDEX improvement_proposals_hypothesis_idx "
            "ON improvement_proposals(hypothesis_id, proposal_version DESC)",
            "CREATE INDEX kai_candidates_proposal_idx "
            "ON kai_candidates(proposal_id, candidate_version DESC)",
            "CREATE INDEX candidate_assessments_candidate_idx "
            "ON candidate_assessments(candidate_id, assessment_id DESC)",
            "CREATE INDEX improvement_evaluations_candidate_idx "
            "ON improvement_evaluations(candidate_id, evaluation_id DESC)",
            "CREATE INDEX improvement_decisions_candidate_idx "
            "ON improvement_decisions(candidate_id, decision_id DESC)",
        )
        for statement in indexes:
            connection.execute(statement)
        immutable_tables = (
            "self_model_versions",
            "self_observations",
            "improvement_opportunity_history",
            "improvement_hypotheses",
            "improvement_proposals",
            "kai_candidates",
            "candidate_assessments",
            "improvement_evaluations",
            "improvement_decisions",
        )
        for table in immutable_tables:
            connection.execute(
                f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'historical improvement evidence is immutable'); END"
            )
            connection.execute(
                f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'historical improvement evidence is immutable'); END"
            )
        connection.execute(
            """
            CREATE TRIGGER opportunity_evidence_immutable
            BEFORE UPDATE ON improvement_opportunities
            WHEN NEW.detected_at != OLD.detected_at
              OR NEW.source_observation_ids_json != OLD.source_observation_ids_json
              OR NEW.affected_subsystem != OLD.affected_subsystem
              OR NEW.current_behavior != OLD.current_behavior
              OR NEW.evidence_json != OLD.evidence_json
              OR NEW.severity != OLD.severity
              OR NEW.expected_impact != OLD.expected_impact
              OR NEW.confidence IS NOT OLD.confidence
            BEGIN
                SELECT RAISE(ABORT, 'opportunity evidence is immutable');
            END
            """
        )
        connection.execute(
            "CREATE TRIGGER opportunities_no_delete BEFORE DELETE ON "
            "improvement_opportunities BEGIN "
            "SELECT RAISE(ABORT, 'improvement opportunities are historical evidence'); END"
        )
        connection.execute(
            "UPDATE schema_info SET version = 6 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v7(connection: sqlite3.Connection) -> None:
        statements = (
            """
            CREATE TABLE evaluation_protocols (
                protocol_id TEXT NOT NULL,
                version INTEGER NOT NULL CHECK (version >= 1),
                definition_sha256 TEXT NOT NULL,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(protocol_id, version)
            )
            """,
            """
            CREATE TABLE candidate_workspaces (
                workspace_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL UNIQUE REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                relative_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                source_revision TEXT,
                source_tree_sha256 TEXT NOT NULL,
                configuration_sha256 TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                python_version TEXT NOT NULL,
                operating_system TEXT NOT NULL,
                environment_fingerprint TEXT,
                metadata_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN
                    ('ACTIVE', 'VALIDATED', 'TESTED', 'FAILED', 'PRESERVED', 'CLEANED'))
            )
            """,
            """
            CREATE TABLE candidate_workspace_events (
                workspace_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL REFERENCES candidate_workspaces(workspace_id)
                    ON DELETE RESTRICT,
                occurred_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE candidate_modifications (
                modification_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                workspace_id TEXT NOT NULL REFERENCES candidate_workspaces(workspace_id)
                    ON DELETE RESTRICT,
                target_file TEXT NOT NULL,
                operation TEXT NOT NULL CHECK (operation IN ('ADD', 'MODIFY', 'DELETE')),
                original_sha256 TEXT,
                resulting_sha256 TEXT,
                patch TEXT NOT NULL,
                reason TEXT NOT NULL,
                hypothesis_id INTEGER NOT NULL REFERENCES
                    improvement_hypotheses(hypothesis_id) ON DELETE RESTRICT,
                created_at TEXT NOT NULL,
                validation_status TEXT NOT NULL CHECK (validation_status IN
                    ('VALID', 'INVALID'))
            )
            """,
            """
            CREATE TABLE candidate_executions (
                execution_id TEXT PRIMARY KEY,
                candidate_id TEXT NOT NULL REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                workspace_id TEXT NOT NULL REFERENCES candidate_workspaces(workspace_id)
                    ON DELETE RESTRICT,
                procedure_id TEXT NOT NULL CHECK (procedure_id IN
                    ('unittest:discover-tests', 'cpu-predictor-evaluation-v1')
                    OR procedure_id GLOB 'unittest:tests.test_[A-Za-z0-9_]*'),
                status TEXT NOT NULL CHECK (status IN
                    ('SUCCESS', 'FAILED', 'TIMEOUT', 'RESOURCE_LIMIT', 'INVALID',
                     'CRASHED', 'UNKNOWN')),
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                duration_seconds REAL NOT NULL CHECK (duration_seconds >= 0),
                exit_code INTEGER,
                stdout TEXT NOT NULL,
                stderr TEXT NOT NULL,
                environment_json TEXT NOT NULL,
                reason TEXT
            )
            """,
            "ALTER TABLE improvement_evaluations ADD COLUMN evaluation_protocol_id TEXT",
            "ALTER TABLE improvement_evaluations ADD COLUMN baseline_snapshot_sha256 TEXT NOT NULL DEFAULT ''",
            "CREATE INDEX candidate_workspaces_candidate_idx "
            "ON candidate_workspaces(candidate_id, created_at)",
            "CREATE INDEX candidate_modifications_candidate_idx "
            "ON candidate_modifications(candidate_id, created_at)",
            "CREATE INDEX candidate_executions_candidate_idx "
            "ON candidate_executions(candidate_id, started_at)",
            "CREATE INDEX candidate_workspace_events_idx "
            "ON candidate_workspace_events(workspace_id, workspace_event_id)",
        )
        for statement in statements:
            connection.execute(statement)
        protocol = CPU_PREDICTOR_PROTOCOL
        connection.execute(
            "INSERT INTO evaluation_protocols "
            "(protocol_id, version, definition_sha256, definition_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                protocol.protocol_id,
                protocol.version,
                protocol.definition_sha256,
                _encode_json(protocol.to_dict()),
                utc_now(),
            ),
        )
        for table in (
            "evaluation_protocols",
            "candidate_workspace_events",
            "candidate_modifications",
            "candidate_executions",
        ):
            connection.execute(
                f"CREATE TRIGGER {table}_no_update BEFORE UPDATE ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'candidate history is immutable'); END"
            )
            connection.execute(
                f"CREATE TRIGGER {table}_no_delete BEFORE DELETE ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'candidate history is immutable'); END"
            )
        for table in (
            "candidate_workspaces",
            "improvement_evaluations",
        ):
            connection.execute(
                f"CREATE TRIGGER {table}_no_delete_v7 BEFORE DELETE ON {table} "
                "BEGIN SELECT RAISE(ABORT, 'candidate history is immutable'); END"
            )
        connection.execute(
            """
            CREATE TRIGGER candidate_workspace_provenance_immutable
            BEFORE UPDATE ON candidate_workspaces
            WHEN NEW.workspace_id != OLD.workspace_id
              OR NEW.candidate_id != OLD.candidate_id
              OR NEW.relative_path != OLD.relative_path
              OR NEW.created_at != OLD.created_at
              OR NEW.source_revision IS NOT OLD.source_revision
              OR NEW.source_tree_sha256 != OLD.source_tree_sha256
              OR NEW.configuration_sha256 != OLD.configuration_sha256
              OR NEW.schema_version != OLD.schema_version
              OR NEW.python_version != OLD.python_version
              OR NEW.operating_system != OLD.operating_system
              OR NEW.environment_fingerprint IS NOT OLD.environment_fingerprint
              OR NEW.metadata_json != OLD.metadata_json
            BEGIN
                SELECT RAISE(ABORT, 'candidate workspace provenance is immutable');
            END
            """
        )
        connection.execute(
            "UPDATE schema_info SET version = 7 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v8(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE engineering_tasks (
                task_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                objective TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN (
                    'OBJECTIVE', 'REPOSITORY_INSPECTION', 'PLAN',
                    'PROPOSED_CHANGES', 'CANDIDATE_WORKSPACE',
                    'PATCH_APPLICATION', 'STATIC_VALIDATION', 'TESTING',
                    'BENCHMARKING', 'EVALUATION', 'ACCEPTED', 'REJECTED',
                    'INCONCLUSIVE', 'FAILED', 'BLOCKED', 'UNKNOWN'
                )),
                provider_json TEXT NOT NULL,
                inspection_json TEXT NOT NULL DEFAULT '{}',
                plan_json TEXT NOT NULL DEFAULT '{}',
                candidate_id TEXT REFERENCES kai_candidates(candidate_id)
                    ON DELETE RESTRICT,
                workspace_id TEXT REFERENCES candidate_workspaces(workspace_id)
                    ON DELETE RESTRICT,
                result_json TEXT NOT NULL DEFAULT '{}',
                max_repair_attempts INTEGER NOT NULL DEFAULT 1
                    CHECK (max_repair_attempts BETWEEN 0 AND 2),
                repair_attempts INTEGER NOT NULL DEFAULT 0
                    CHECK (repair_attempts BETWEEN 0 AND 2),
                CHECK (repair_attempts <= max_repair_attempts)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE engineering_task_events (
                task_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES engineering_tasks(task_id)
                    ON DELETE RESTRICT,
                occurred_at TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                details_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX engineering_tasks_updated_idx "
            "ON engineering_tasks(updated_at DESC)"
        )
        connection.execute(
            "CREATE INDEX engineering_task_events_idx "
            "ON engineering_task_events(task_id, task_event_id)"
        )
        connection.execute(
            """
            CREATE TRIGGER engineering_task_events_no_update
            BEFORE UPDATE ON engineering_task_events
            BEGIN SELECT RAISE(ABORT, 'engineering task history is immutable'); END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER engineering_task_events_no_delete
            BEFORE DELETE ON engineering_task_events
            BEGIN SELECT RAISE(ABORT, 'engineering task history is immutable'); END
            """
        )
        connection.execute(
            "UPDATE schema_info SET version = 8 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v9(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE llm_model_selection (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE llm_task_routes (
                task_category TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE llm_fallbacks (
                position INTEGER PRIMARY KEY CHECK (position >= 0),
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(provider, model_id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE llm_request_history (
                request_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
                task_category TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_id TEXT NOT NULL,
                outcome TEXT NOT NULL CHECK (outcome IN ('SUCCESS', 'FAILED')),
                token_usage_json TEXT NOT NULL,
                error_category TEXT,
                fallback_from_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX llm_request_history_started_idx "
            "ON llm_request_history(started_at DESC)"
        )
        connection.execute(
            """
            CREATE TRIGGER llm_request_history_no_update
            BEFORE UPDATE ON llm_request_history
            BEGIN SELECT RAISE(ABORT, 'LLM request history is immutable'); END
            """
        )
        connection.execute(
            """
            CREATE TRIGGER llm_request_history_no_delete
            BEFORE DELETE ON llm_request_history
            BEGIN SELECT RAISE(ABORT, 'LLM request history is immutable'); END
            """
        )
        connection.execute(
            "UPDATE schema_info SET version = 9 WHERE singleton = 1"
        )

    @staticmethod
    def _migrate_to_v10(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE INDEX llm_request_history_model_latest_idx "
            "ON llm_request_history(provider, model_id, completed_at DESC, request_id)"
        )
        connection.execute(
            "UPDATE schema_info SET version = 10 WHERE singleton = 1"
        )

    def create_engineering_task(
        self,
        *,
        task_id: str,
        objective: str,
        provider: dict,
        max_repair_attempts: int = 1,
    ) -> dict:
        if not isinstance(objective, str) or not objective.strip():
            raise KaiError("Engineering objective must be non-empty.")
        if len(objective) > 4_000:
            raise KaiError("Engineering objective exceeds the 4000-character limit.")
        try:
            if str(uuid.UUID(task_id)) != task_id:
                raise ValueError("task ID must be a canonical UUID")
            provider_json = _encode_json(provider)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Invalid engineering task metadata: {exc}") from exc
        if (
            isinstance(max_repair_attempts, bool)
            or not isinstance(max_repair_attempts, int)
            or max_repair_attempts < 0
            or max_repair_attempts > 2
        ):
            raise KaiError("Engineering repair attempts must be between zero and two.")
        now = utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "INSERT INTO engineering_tasks "
                    "(task_id, created_at, updated_at, objective, status, provider_json, "
                    "max_repair_attempts) VALUES (?, ?, ?, ?, 'OBJECTIVE', ?, ?)",
                    (
                        task_id,
                        now,
                        now,
                        objective.strip(),
                        provider_json,
                        max_repair_attempts,
                    ),
                )
                connection.execute(
                    "INSERT INTO engineering_task_events "
                    "(task_id, occurred_at, stage, status, details_json) "
                    "VALUES (?, ?, 'OBJECTIVE', 'OBJECTIVE', '{}')",
                    (task_id, now),
                )
                connection.commit()
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise KaiError(f"Cannot create engineering task: {exc}") from exc
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot create engineering task: {exc}") from exc
        return self.get_engineering_task(task_id)

    def get_llm_model_selection(self, *, read_only: bool = False) -> dict | None:
        if read_only and not self.path.is_file():
            return None
        if read_only:
            uri = self.path.resolve().as_uri() + "?mode=ro"
            try:
                connection = sqlite3.connect(uri, uri=True, timeout=5.0)
                connection.row_factory = sqlite3.Row
                try:
                    row = connection.execute(
                        "SELECT provider, model_id, updated_at "
                        "FROM llm_model_selection WHERE singleton = 1"
                    ).fetchone()
                finally:
                    connection.close()
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc).lower():
                    return None
                raise KaiError(f"Cannot read active model selection: {exc}") from exc
        else:
            with self.connection() as connection:
                row = connection.execute(
                    "SELECT provider, model_id, updated_at "
                    "FROM llm_model_selection WHERE singleton = 1"
                ).fetchone()
        return dict(row) if row is not None else None

    def set_llm_model_selection(self, provider: str, model_id: str) -> dict:
        _validate_llm_target(provider, model_id)
        updated_at = utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO llm_model_selection(singleton, provider, model_id, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    provider = excluded.provider,
                    model_id = excluded.model_id,
                    updated_at = excluded.updated_at
                """,
                (provider, model_id, updated_at),
            )
            connection.commit()
        return {
            "provider": provider,
            "model_id": model_id,
            "updated_at": updated_at,
        }

    def get_llm_task_route(self, task_category: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT task_category, provider, model_id, updated_at "
                "FROM llm_task_routes WHERE task_category = ?",
                (task_category,),
            ).fetchone()
        return dict(row) if row is not None else None

    def set_llm_task_route(
        self, task_category: str, provider: str, model_id: str
    ) -> dict:
        _validate_llm_target(provider, model_id)
        if task_category not in LLM_TASK_CATEGORIES:
            raise KaiError(f"Unsupported LLM task category: {task_category!r}.")
        updated_at = utc_now()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO llm_task_routes(task_category, provider, model_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(task_category) DO UPDATE SET
                    provider = excluded.provider,
                    model_id = excluded.model_id,
                    updated_at = excluded.updated_at
                """,
                (task_category, provider, model_id, updated_at),
            )
            connection.commit()
        return {
            "task_category": task_category,
            "provider": provider,
            "model_id": model_id,
            "updated_at": updated_at,
        }

    def list_llm_task_routes(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT task_category, provider, model_id, updated_at "
                "FROM llm_task_routes ORDER BY task_category"
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_llm_task_route(self, task_category: str) -> None:
        if task_category not in LLM_TASK_CATEGORIES:
            raise KaiError(f"Unsupported LLM task category: {task_category!r}.")
        with self.connection() as connection:
            connection.execute(
                "DELETE FROM llm_task_routes WHERE task_category = ?",
                (task_category,),
            )
            connection.commit()

    def add_llm_fallback(self, provider: str, model_id: str) -> list[dict]:
        _validate_llm_target(provider, model_id)
        now = utc_now()
        with self.connection() as connection:
            count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM llm_fallbacks"
                ).fetchone()[0]
            )
            if count >= 3:
                raise KaiError("At most three ordered LLM fallbacks may be configured.")
            existing = connection.execute(
                "SELECT 1 FROM llm_fallbacks WHERE provider = ? AND model_id = ?",
                (provider, model_id),
            ).fetchone()
            if existing is not None:
                raise KaiError("That fallback model is already configured.")
            position = connection.execute(
                "SELECT COALESCE(MAX(position), -1) + 1 FROM llm_fallbacks"
            ).fetchone()[0]
            connection.execute(
                "INSERT INTO llm_fallbacks(position, provider, model_id, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (position, provider, model_id, now),
            )
            connection.commit()
        return self.list_llm_fallbacks()

    def list_llm_fallbacks(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT position, provider, model_id, updated_at "
                "FROM llm_fallbacks ORDER BY position"
            ).fetchall()
        return [dict(row) for row in rows]

    def clear_llm_fallbacks(self) -> None:
        with self.connection() as connection:
            connection.execute("DELETE FROM llm_fallbacks")
            connection.commit()

    def record_llm_request(self, request: dict) -> None:
        try:
            token_usage = request.get("token_usage") or {}
            if not isinstance(token_usage, dict):
                raise ValueError("token usage must be an object")
            safe_usage = {
                key: value
                for key, value in token_usage.items()
                if key in {"input_tokens", "output_tokens", "total_tokens"}
                and isinstance(value, int)
                and not isinstance(value, bool)
                and value >= 0
            }
            fallback_from = request.get("fallback_from") or []
            if not isinstance(fallback_from, list):
                raise ValueError("fallback history must be a list")
            connection_values = (
                request["request_id"],
                request["started_at"],
                request["completed_at"],
                request["duration_ms"],
                request["task_category"],
                request["provider"],
                request["model_id"],
                request["outcome"],
                _encode_json(safe_usage),
                request.get("error_category"),
                _encode_json(fallback_from),
            )
            if request["outcome"] not in {"SUCCESS", "FAILED"}:
                raise ValueError("invalid request outcome")
            if (
                isinstance(request["duration_ms"], bool)
                or not isinstance(request["duration_ms"], int)
                or request["duration_ms"] < 0
            ):
                raise ValueError("invalid duration")
        except (KeyError, TypeError, ValueError) as exc:
            raise KaiError(f"Invalid LLM request metadata: {exc}") from exc
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO llm_request_history(
                    request_id, started_at, completed_at, duration_ms,
                    task_category, provider, model_id, outcome,
                    token_usage_json, error_category, fallback_from_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                connection_values,
            )
            connection.commit()

    def list_llm_requests(self, limit: int = 50) -> list[dict]:
        _validate_limit(limit, "LLM request history")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM llm_request_history "
                "ORDER BY started_at DESC, request_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "request_id": row["request_id"],
                "started_at": row["started_at"],
                "completed_at": row["completed_at"],
                "duration_ms": row["duration_ms"],
                "task_category": row["task_category"],
                "provider": row["provider"],
                "model_id": row["model_id"],
                "outcome": row["outcome"],
                "token_usage": json.loads(row["token_usage_json"]),
                "error_category": row["error_category"],
                "fallback_from": json.loads(row["fallback_from_json"]),
            }
            for row in rows
        ]

    def latest_llm_request(self, provider: str, model_id: str) -> dict | None:
        """Return the newest immutable outcome for the exact provider/model pair."""
        _validate_llm_target(provider, model_id)
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM llm_request_history "
                "WHERE provider = ? AND model_id = ? "
                "ORDER BY completed_at DESC, request_id LIMIT 1",
                (provider, model_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "request_id": row["request_id"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "duration_ms": row["duration_ms"],
            "task_category": row["task_category"],
            "provider": row["provider"],
            "model_id": row["model_id"],
            "outcome": row["outcome"],
            "token_usage": json.loads(row["token_usage_json"]),
            "error_category": row["error_category"],
            "fallback_from": json.loads(row["fallback_from_json"]),
        }

    def record_engineering_task_event(
        self,
        task_id: str,
        *,
        stage: str,
        status: str,
        details: dict,
        updates: dict | None = None,
    ) -> dict:
        if stage not in ENGINEERING_TASK_STAGES or status not in ENGINEERING_TASK_STATES:
            raise KaiError("Engineering task stage or status is unsupported.")
        if not isinstance(details, dict):
            raise KaiError("Engineering task event details must be an object.")
        permitted_updates = {
            "inspection": "inspection_json",
            "plan": "plan_json",
            "provider": "provider_json",
            "candidate_id": "candidate_id",
            "workspace_id": "workspace_id",
            "result": "result_json",
            "repair_attempts": "repair_attempts",
        }
        if updates is None:
            updates = {}
        if not isinstance(updates, dict) or set(updates) - set(permitted_updates):
            raise KaiError("Engineering task update contains unsupported fields.")
        try:
            details_json = _encode_json(details)
            update_values = {}
            for key, value in updates.items():
                if key in {"inspection", "plan", "provider", "result"}:
                    update_values[permitted_updates[key]] = _encode_json(value)
                else:
                    update_values[permitted_updates[key]] = value
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Engineering task data must be finite JSON: {exc}") from exc
        if "repair_attempts" in update_values:
            attempts = update_values["repair_attempts"]
            if (
                isinstance(attempts, bool)
                or not isinstance(attempts, int)
                or attempts < 0
                or attempts > 2
            ):
                raise KaiError("Engineering repair attempts must be between zero and two.")
        now = utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT max_repair_attempts, repair_attempts FROM engineering_tasks "
                    "WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
                if row is None:
                    raise KaiError(f"Engineering task {task_id!r} does not exist.")
                if "repair_attempts" in update_values:
                    attempts = update_values["repair_attempts"]
                    if attempts < row["repair_attempts"] or attempts > row["max_repair_attempts"]:
                        raise KaiError("Engineering repair attempt count cannot decrease or exceed its budget.")
                assignments = ["status = ?", "updated_at = ?"]
                values: list = [status, now]
                for column, value in update_values.items():
                    assignments.append(f"{column} = ?")
                    values.append(value)
                values.append(task_id)
                connection.execute(
                    f"UPDATE engineering_tasks SET {', '.join(assignments)} "
                    "WHERE task_id = ?",
                    values,
                )
                connection.execute(
                    "INSERT INTO engineering_task_events "
                    "(task_id, occurred_at, stage, status, details_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (task_id, now, stage, status, details_json),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise KaiError(f"Cannot record engineering task event: {exc}") from exc
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot record engineering task event: {exc}") from exc
        return self.get_engineering_task(task_id)

    def get_engineering_task(self, task_id: str) -> dict:
        with self.connection() as connection:
            task = connection.execute(
                "SELECT * FROM engineering_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if task is None:
                raise KaiError(f"Engineering task {task_id!r} does not exist.")
            events = connection.execute(
                "SELECT task_event_id, occurred_at, stage, status, details_json "
                "FROM engineering_task_events WHERE task_id = ? "
                "ORDER BY task_event_id",
                (task_id,),
            ).fetchall()
        return {
            "task_id": task["task_id"],
            "created_at": task["created_at"],
            "updated_at": task["updated_at"],
            "objective": task["objective"],
            "status": task["status"],
            "provider": json.loads(task["provider_json"]),
            "inspection": json.loads(task["inspection_json"]),
            "plan": json.loads(task["plan_json"]),
            "candidate_id": task["candidate_id"],
            "workspace_id": task["workspace_id"],
            "result": json.loads(task["result_json"]),
            "max_repair_attempts": task["max_repair_attempts"],
            "repair_attempts": task["repair_attempts"],
            "events": [
                {
                    "task_event_id": row["task_event_id"],
                    "occurred_at": row["occurred_at"],
                    "stage": row["stage"],
                    "status": row["status"],
                    "details": json.loads(row["details_json"]),
                }
                for row in events
            ],
        }

    def list_engineering_tasks(self, limit: int = 50) -> list[dict]:
        _validate_limit(limit, "Engineering task")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT task_id FROM engineering_tasks "
                "ORDER BY created_at DESC, task_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get_engineering_task(row["task_id"]) for row in rows]

    def engineering_task_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM engineering_tasks"
            ).fetchone()
        return int(row[0])

    def integrity_check(self) -> str:
        with self.connection() as connection:
            try:
                rows = connection.execute("PRAGMA quick_check").fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"State database integrity check failed: {exc}") from exc
        result = "; ".join(str(row[0]) for row in rows)
        if result != "ok":
            raise KaiError(f"State database integrity check reported: {result}")
        return result

    def schema_version(self) -> int:
        with self.connection() as connection:
            try:
                row = connection.execute(
                    "SELECT version FROM schema_info WHERE singleton = 1"
                ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read state schema version: {exc}") from exc
        if row is None:
            raise KaiError("State schema version is missing.")
        return int(row["version"])

    def identity(self) -> dict:
        with self.connection() as connection:
            try:
                row = connection.execute(
                    "SELECT identity_id, name, created_at, principles_json "
                    "FROM identity WHERE singleton = 1"
                ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read KAI identity: {exc}") from exc
        if row is None:
            raise KaiError("KAI identity is missing from persistent state.")
        return {
            "identity_id": row["identity_id"],
            "name": row["name"],
            "created_at": row["created_at"],
            "principles": json.loads(row["principles_json"]),
        }

    def record_event(
        self,
        event_type: str,
        summary: str,
        details: dict | None = None,
        *,
        source: str = "kai",
        environment_snapshot_id: int | None = None,
        causal_parent_id: int | None = None,
        experiment_id: int | None = None,
        confidence: float | None = None,
        epistemic_status: str = "OBSERVATION",
        occurred_at: str | None = None,
    ) -> int:
        if (
            not isinstance(event_type, str)
            or not isinstance(summary, str)
            or not isinstance(source, str)
            or not event_type.strip()
            or not summary.strip()
            or not source.strip()
        ):
            raise KaiError("Event type, summary, and source must be non-empty.")
        if details is not None and not isinstance(details, dict):
            raise KaiError("Event payload must be an object.")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise KaiError("Event confidence must be between 0.0 and 1.0.")
        if epistemic_status not in {
            "OBSERVATION", "FACT", "INFERENCE", "HYPOTHESIS", "UNKNOWN"
        }:
            raise KaiError(f"Invalid event epistemic status: {epistemic_status}")
        timestamp = _utc_timestamp(occurred_at) if occurred_at else utc_now()
        with self.connection() as connection:
            try:
                event_id = _insert_event(
                    connection,
                    event_type,
                    summary,
                    details or {},
                    source,
                    environment_snapshot_id,
                    causal_parent_id,
                    experiment_id,
                    confidence,
                    epistemic_status,
                    timestamp,
                )
                connection.commit()
                return event_id
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot record event: {exc}") from exc

    def recent_events(
        self,
        limit: int = 20,
        event_type: str | None = None,
        since: str | None = None,
    ) -> list[Event]:
        _validate_limit(limit, "Event")
        timestamp = _utc_timestamp(since) if since else None
        with self.connection() as connection:
            try:
                clauses = []
                parameters: list[object] = []
                if event_type:
                    clauses.append("event_type = ?")
                    parameters.append(event_type)
                if timestamp:
                    clauses.append("occurred_at >= ?")
                    parameters.append(timestamp)
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                parameters.append(limit)
                rows = connection.execute(
                    "SELECT event_id, occurred_at, event_type, summary, details_json, "
                    "source, environment_snapshot_id, causal_parent_id, experiment_id, confidence, "
                    "execution_id, experiment_definition_id, experiment_version, "
                    "epistemic_status, payload_json FROM events "
                    f"{where} ORDER BY event_id DESC LIMIT ?",
                    parameters,
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read events: {exc}") from exc
        return [
            {
                "event_id": row["event_id"],
                "occurred_at": row["occurred_at"],
                "timestamp": row["occurred_at"],
                "event_type": row["event_type"],
                "summary": row["summary"],
                "details": json.loads(row["payload_json"]),
                "source": row["source"],
                "environment_snapshot_id": row["environment_snapshot_id"],
                "causal_parent_id": row["causal_parent_id"],
                "experiment_id": row["experiment_id"],
                "confidence": row["confidence"],
                "epistemic_status": row["epistemic_status"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def event_count(self) -> int:
        with self.connection() as connection:
            try:
                return int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot count events: {exc}") from exc

    def event_types(self) -> list[str]:
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT DISTINCT event_type FROM events ORDER BY event_type"
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot list event types: {exc}") from exc
        return sorted(set(EVENT_TYPES).union(row["event_type"] for row in rows))

    def history_summary(self) -> dict:
        with self.connection() as connection:
            try:
                experience_count = connection.execute(
                    "SELECT COUNT(*) FROM experiences"
                ).fetchone()[0]
                learning_count = connection.execute(
                    "SELECT COUNT(*) FROM learning_records"
                ).fetchone()[0]
                prediction_rows = connection.execute(
                    "SELECT status, COUNT(*) AS count FROM predictions GROUP BY status"
                ).fetchall()
                latest_event_id = connection.execute(
                    "SELECT coalesce(MAX(event_id), 0) FROM events"
                ).fetchone()[0]
                latest_experience_id = connection.execute(
                    "SELECT coalesce(MAX(experience_id), 0) FROM experiences"
                ).fetchone()[0]
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot summarize persistent history: {exc}") from exc
        return {
            "experience_count": int(experience_count),
            "learning_record_count": int(learning_count),
            "prediction_counts": {
                row["status"]: int(row["count"]) for row in prediction_rows
            },
            "latest_committed_event_id": int(latest_event_id),
            "latest_experience_id": int(latest_experience_id),
        }

    def save_self_model_version(
        self,
        *,
        identity_id: str,
        kai_version: str,
        architecture_version: str,
        schema_version: int,
        environment_fingerprint: str | None,
        source_snapshot_sha256: str | None,
        epistemic_summary: dict,
        model: dict,
        observations: list[dict],
        timestamp: str | None = None,
    ) -> dict:
        if not isinstance(identity_id, str) or not identity_id.strip():
            raise KaiError("Self-model version requires a KAI identity.")
        for label, value in (
            ("KAI version", kai_version),
            ("architecture version", architecture_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise KaiError(f"Self-model {label} must be non-empty.")
        if (
            isinstance(schema_version, bool)
            or not isinstance(schema_version, int)
            or schema_version < 1
        ):
            raise KaiError("Self-model schema version must be a positive integer.")
        try:
            _encode_json(epistemic_summary)
            _encode_json(model)
            if not isinstance(observations, list):
                raise ValueError("Self-model observations must be a list.")
            normalized_observations = []
            for item in observations:
                validate_self_observation(item)
                normalized_observations.append(
                    {
                        **item,
                        "observed_at": _utc_timestamp(item["observed_at"]),
                        "environment_context_json": _encode_json(
                            item["environment_context"]
                        ),
                        "value_json": _encode_json(item["value"]),
                    }
                )
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Invalid self-model version: {exc}") from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "INSERT INTO self_model_versions "
                    "(identity_id, created_at, kai_version, architecture_version, "
                    "schema_version, environment_fingerprint, source_snapshot_sha256, "
                    "epistemic_summary_json, model_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identity_id,
                        now,
                        kai_version,
                        architecture_version,
                        schema_version,
                        environment_fingerprint,
                        source_snapshot_sha256,
                        _encode_json(epistemic_summary),
                        _encode_json(model),
                    ),
                )
                version_id = int(cursor.lastrowid)
                for item in normalized_observations:
                    connection.execute(
                        "INSERT INTO self_observations "
                        "(self_model_version_id, observed_at, subject, observation_type, "
                        "value_json, source, epistemic_status, evidence_status, confidence, "
                        "kai_version, environment_context_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            version_id,
                            item["observed_at"],
                            item["subject"],
                            item["observation_type"],
                            item["value_json"],
                            item["source"],
                            item["epistemic_status"],
                            item["evidence_status"],
                            item.get("confidence"),
                            item["kai_version"],
                            item["environment_context_json"],
                        ),
                    )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot persist self-model version: {exc}") from exc
        return self.get_self_model_version(version_id)

    def get_self_model_version(self, version_id: int | None = None) -> dict | None:
        with self.connection() as connection:
            try:
                if version_id is None:
                    row = connection.execute(
                        "SELECT * FROM self_model_versions "
                        "ORDER BY self_model_version_id DESC LIMIT 1"
                    ).fetchone()
                else:
                    row = connection.execute(
                        "SELECT * FROM self_model_versions "
                        "WHERE self_model_version_id = ?",
                        (version_id,),
                    ).fetchone()
                if row is None:
                    return None
                observations = connection.execute(
                    "SELECT * FROM self_observations WHERE self_model_version_id = ? "
                    "ORDER BY observation_id",
                    (row["self_model_version_id"],),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot load self-model version: {exc}") from exc
        return {
            "self_model_version_id": row["self_model_version_id"],
            "identity_id": row["identity_id"],
            "created_at": row["created_at"],
            "kai_version": row["kai_version"],
            "architecture_version": row["architecture_version"],
            "schema_version": row["schema_version"],
            "environment_fingerprint": row["environment_fingerprint"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "epistemic_summary": json.loads(row["epistemic_summary_json"]),
            "model": json.loads(row["model_json"]),
            "observations": [self._self_observation_row(item) for item in observations],
        }

    @staticmethod
    def _self_observation_row(row: sqlite3.Row) -> dict:
        return {
            "observation_id": row["observation_id"],
            "self_model_version_id": row["self_model_version_id"],
            "observed_at": row["observed_at"],
            "subject": row["subject"],
            "observation_type": row["observation_type"],
            "value": json.loads(row["value_json"]),
            "source": row["source"],
            "epistemic_status": row["epistemic_status"],
            "evidence_status": row["evidence_status"],
            "confidence": row["confidence"],
            "kai_version": row["kai_version"],
            "environment_context": json.loads(row["environment_context_json"]),
        }

    def list_self_model_versions(self, limit: int = 20) -> list[dict]:
        _validate_limit(limit, "Self-model history")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT self_model_version_id FROM self_model_versions "
                "ORDER BY self_model_version_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            self.get_self_model_version(row["self_model_version_id"])
            for row in rows
        ]

    def list_self_observations(
        self,
        limit: int = 50,
        *,
        subject: str | None = None,
        epistemic_status: str | None = None,
    ) -> list[dict]:
        _validate_limit(limit, "Self-observation")
        if epistemic_status is not None and epistemic_status not in SELF_EPISTEMIC_STATES:
            raise KaiError("Unsupported self-observation epistemic status.")
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM self_observations "
                    "WHERE (? IS NULL OR subject = ?) "
                    "AND (? IS NULL OR epistemic_status = ?) "
                    "ORDER BY observation_id DESC LIMIT ?",
                    (subject, subject, epistemic_status, epistemic_status, limit),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot list self-observations: {exc}") from exc
        return [self._self_observation_row(row) for row in rows]

    def record_self_observation(self, observation: dict) -> dict:
        try:
            validate_self_observation(observation)
            observed_at = _utc_timestamp(observation["observed_at"])
            value_json = _encode_json(observation["value"])
            context_json = _encode_json(observation["environment_context"])
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Invalid self-observation: {exc}") from exc
        with self.connection() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO self_observations "
                    "(self_model_version_id, observed_at, subject, observation_type, "
                    "value_json, source, epistemic_status, evidence_status, confidence, "
                    "kai_version, environment_context_json) "
                    "VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        observed_at,
                        observation["subject"],
                        observation["observation_type"],
                        value_json,
                        observation["source"],
                        observation["epistemic_status"],
                        observation["evidence_status"],
                        observation.get("confidence"),
                        observation["kai_version"],
                        context_json,
                    ),
                )
                observation_id = int(cursor.lastrowid)
                row = connection.execute(
                    "SELECT * FROM self_observations WHERE observation_id = ?",
                    (observation_id,),
                ).fetchone()
                connection.commit()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot persist self-observation: {exc}") from exc
        return self._self_observation_row(row)

    def self_model_version_count(self) -> int:
        with self.connection() as connection:
            try:
                return int(
                    connection.execute(
                        "SELECT COUNT(*) FROM self_model_versions"
                    ).fetchone()[0]
                )
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot count self-model versions: {exc}") from exc

    def create_improvement_opportunity(
        self,
        *,
        source_observation_ids: list[int],
        affected_subsystem: str,
        current_behavior: str,
        evidence: dict | list,
        severity: str,
        expected_impact: str,
        confidence: float | None,
        timestamp: str | None = None,
    ) -> dict:
        for label, value in (
            ("affected subsystem", affected_subsystem),
            ("current behavior", current_behavior),
            ("expected impact", expected_impact),
        ):
            if not isinstance(value, str) or not value.strip():
                raise KaiError(f"Opportunity {label} must be non-empty.")
        if severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
            raise KaiError("Opportunity severity is unsupported.")
        _validate_confidence(confidence, "Opportunity")
        if not isinstance(source_observation_ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in source_observation_ids
        ):
            raise KaiError("Opportunity source observation IDs must be positive integers.")
        try:
            evidence_json = _encode_json(evidence)
            source_json = _encode_json(source_observation_ids)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Opportunity evidence must be finite JSON: {exc}") from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                for observation_id in source_observation_ids:
                    if connection.execute(
                        "SELECT 1 FROM self_observations WHERE observation_id = ?",
                        (observation_id,),
                    ).fetchone() is None:
                        raise KaiError(
                            f"Self-observation {observation_id} does not exist."
                        )
                cursor = connection.execute(
                    "INSERT INTO improvement_opportunities "
                    "(detected_at, source_observation_ids_json, affected_subsystem, "
                    "current_behavior, evidence_json, severity, expected_impact, "
                    "confidence, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN')",
                    (
                        now,
                        source_json,
                        affected_subsystem,
                        current_behavior,
                        evidence_json,
                        severity,
                        expected_impact,
                        confidence,
                    ),
                )
                opportunity_id = int(cursor.lastrowid)
                connection.execute(
                    "INSERT INTO improvement_opportunity_history "
                    "(opportunity_id, transitioned_at, from_status, to_status, reason, "
                    "evidence_json) VALUES (?, ?, NULL, 'OPEN', ?, ?)",
                    (
                        opportunity_id,
                        now,
                        "Opportunity recorded from supplied evidence.",
                        evidence_json,
                    ),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot create improvement opportunity: {exc}") from exc
        return self.get_improvement_opportunity(opportunity_id)

    def update_improvement_opportunity_status(
        self,
        opportunity_id: int,
        status: str,
        reason: str,
        evidence: dict | list | None = None,
        timestamp: str | None = None,
    ) -> dict:
        if status not in OPPORTUNITY_STATUSES:
            raise KaiError("Unsupported improvement opportunity status.")
        if not isinstance(reason, str) or not reason.strip():
            raise KaiError("Opportunity status transition requires a reason.")
        try:
            evidence_json = _encode_json(evidence if evidence is not None else {})
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Opportunity transition evidence must be finite JSON: {exc}") from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        allowed = {
            "OPEN": {"INVESTIGATING", "PROPOSED", "DEFERRED", "INVALIDATED"},
            "INVESTIGATING": {"PROPOSED", "EXPERIMENTING", "DEFERRED", "REJECTED", "INVALIDATED"},
            "PROPOSED": {"INVESTIGATING", "EXPERIMENTING", "DEFERRED", "REJECTED", "INVALIDATED"},
            "EXPERIMENTING": {"PROPOSED", "ACCEPTED", "REJECTED", "DEFERRED", "INVALIDATED"},
            "DEFERRED": {"OPEN", "INVESTIGATING", "INVALIDATED"},
            "ACCEPTED": set(),
            "REJECTED": set(),
            "INVALIDATED": set(),
        }
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT status FROM improvement_opportunities WHERE opportunity_id = ?",
                    (opportunity_id,),
                ).fetchone()
                if row is None:
                    raise KaiError(f"Improvement opportunity {opportunity_id} does not exist.")
                previous = row["status"]
                if status not in allowed[previous]:
                    raise KaiError(
                        f"Opportunity cannot transition from {previous} to {status}."
                    )
                connection.execute(
                    "UPDATE improvement_opportunities SET status = ?, version = version + 1 "
                    "WHERE opportunity_id = ?",
                    (status, opportunity_id),
                )
                connection.execute(
                    "INSERT INTO improvement_opportunity_history "
                    "(opportunity_id, transitioned_at, from_status, to_status, reason, "
                    "evidence_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (opportunity_id, now, previous, status, reason, evidence_json),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot transition improvement opportunity: {exc}") from exc
        return self.get_improvement_opportunity(opportunity_id)

    def get_improvement_opportunity(self, opportunity_id: int) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM improvement_opportunities WHERE opportunity_id = ?",
                (opportunity_id,),
            ).fetchone()
            if row is None:
                raise KaiError(f"Improvement opportunity {opportunity_id} does not exist.")
            history = connection.execute(
                "SELECT * FROM improvement_opportunity_history WHERE opportunity_id = ? "
                "ORDER BY transition_id",
                (opportunity_id,),
            ).fetchall()
            hypotheses = connection.execute(
                "SELECT hypothesis_id FROM improvement_hypotheses WHERE opportunity_id = ? "
                "ORDER BY hypothesis_id",
                (opportunity_id,),
            ).fetchall()
        return {
            "opportunity_id": row["opportunity_id"],
            "detected_at": row["detected_at"],
            "source_observation_ids": json.loads(row["source_observation_ids_json"]),
            "affected_subsystem": row["affected_subsystem"],
            "current_behavior": row["current_behavior"],
            "evidence": json.loads(row["evidence_json"]),
            "severity": row["severity"],
            "expected_impact": row["expected_impact"],
            "confidence": row["confidence"],
            "status": row["status"],
            "version": row["version"],
            "history": [
                {
                    "transition_id": item["transition_id"],
                    "transitioned_at": item["transitioned_at"],
                    "from_status": item["from_status"],
                    "to_status": item["to_status"],
                    "reason": item["reason"],
                    "evidence": json.loads(item["evidence_json"]),
                }
                for item in history
            ],
            "hypothesis_ids": [item["hypothesis_id"] for item in hypotheses],
        }

    def list_improvement_opportunities(
        self, limit: int = 50, status: str | None = None
    ) -> list[dict]:
        _validate_limit(limit, "Improvement opportunity")
        if status is not None and status not in OPPORTUNITY_STATUSES:
            raise KaiError("Unsupported improvement opportunity status.")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT opportunity_id FROM improvement_opportunities "
                "WHERE (? IS NULL OR status = ?) ORDER BY opportunity_id DESC LIMIT ?",
                (status, status, limit),
            ).fetchall()
        return [self.get_improvement_opportunity(row["opportunity_id"]) for row in rows]

    def create_improvement_hypothesis(
        self,
        *,
        opportunity_id: int,
        observed_evidence: dict | list,
        interpretation: str,
        proposed_intervention: str,
        expected_outcome: dict | list,
        uncertainty: dict | list,
        negative_consequences: dict | list,
        timestamp: str | None = None,
    ) -> dict:
        for label, value in (
            ("interpretation", interpretation),
            ("proposed intervention", proposed_intervention),
        ):
            if not isinstance(value, str) or not value.strip():
                raise KaiError(f"Hypothesis {label} must be non-empty.")
        try:
            json_values = [
                _encode_json(value)
                for value in (
                    observed_evidence,
                    expected_outcome,
                    uncertainty,
                    negative_consequences,
                )
            ]
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Hypothesis fields must be finite JSON: {exc}") from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM improvement_opportunities WHERE opportunity_id = ?",
                    (opportunity_id,),
                ).fetchone() is None:
                    raise KaiError(f"Improvement opportunity {opportunity_id} does not exist.")
                cursor = connection.execute(
                    "INSERT INTO improvement_hypotheses "
                    "(opportunity_id, created_at, observed_evidence_json, interpretation, "
                    "proposed_intervention, expected_outcome_json, uncertainty_json, "
                    "negative_consequences_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        opportunity_id,
                        now,
                        json_values[0],
                        interpretation,
                        proposed_intervention,
                        json_values[1],
                        json_values[2],
                        json_values[3],
                    ),
                )
                hypothesis_id = int(cursor.lastrowid)
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot create improvement hypothesis: {exc}") from exc
        return self.get_improvement_hypothesis(hypothesis_id)

    def get_improvement_hypothesis(self, hypothesis_id: int) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM improvement_hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
        if row is None:
            raise KaiError(f"Improvement hypothesis {hypothesis_id} does not exist.")
        return {
            "hypothesis_id": row["hypothesis_id"],
            "opportunity_id": row["opportunity_id"],
            "created_at": row["created_at"],
            "observed_evidence": json.loads(row["observed_evidence_json"]),
            "interpretation": row["interpretation"],
            "proposed_intervention": row["proposed_intervention"],
            "expected_outcome": json.loads(row["expected_outcome_json"]),
            "uncertainty": json.loads(row["uncertainty_json"]),
            "negative_consequences": json.loads(row["negative_consequences_json"]),
            "epistemic_status": row["epistemic_status"],
        }

    def create_improvement_proposal(
        self,
        *,
        hypothesis_id: int,
        affected_component: str,
        current_implementation: str,
        proposed_implementation: str,
        reason: str,
        evidence: dict | list,
        expected_benefit: str,
        expected_cost: str,
        risk: str,
        dependencies: list,
        test_requirements: list,
        benchmark_requirements: list,
        rollback_strategy: str,
        proposal_version: int = 1,
        timestamp: str | None = None,
    ) -> dict:
        for label, value in (
            ("affected component", affected_component),
            ("current implementation", current_implementation),
            ("proposed implementation", proposed_implementation),
            ("reason", reason),
            ("expected benefit", expected_benefit),
            ("expected cost", expected_cost),
            ("risk", risk),
            ("rollback strategy", rollback_strategy),
        ):
            if not isinstance(value, str) or not value.strip():
                raise KaiError(f"Proposal {label} must be non-empty.")
        if (
            isinstance(proposal_version, bool)
            or not isinstance(proposal_version, int)
            or proposal_version < 1
        ):
            raise KaiError("Proposal version must be a positive integer.")
        try:
            json_fields = [
                _encode_json(value)
                for value in (evidence, dependencies, test_requirements, benchmark_requirements)
            ]
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Proposal fields must be finite JSON: {exc}") from exc
        if not all(isinstance(value, list) for value in (dependencies, test_requirements, benchmark_requirements)):
            raise KaiError("Proposal dependencies and requirements must be lists.")
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM improvement_hypotheses WHERE hypothesis_id = ?",
                    (hypothesis_id,),
                ).fetchone() is None:
                    raise KaiError(f"Improvement hypothesis {hypothesis_id} does not exist.")
                cursor = connection.execute(
                    "INSERT INTO improvement_proposals "
                    "(hypothesis_id, proposal_version, created_at, affected_component, "
                    "current_implementation, proposed_implementation, reason, evidence_json, "
                    "expected_benefit, expected_cost, risk, dependencies_json, "
                    "test_requirements_json, benchmark_requirements_json, rollback_strategy) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        hypothesis_id,
                        proposal_version,
                        now,
                        affected_component,
                        current_implementation,
                        proposed_implementation,
                        reason,
                        json_fields[0],
                        expected_benefit,
                        expected_cost,
                        risk,
                        json_fields[1],
                        json_fields[2],
                        json_fields[3],
                        rollback_strategy,
                    ),
                )
                proposal_id = int(cursor.lastrowid)
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot create improvement proposal: {exc}") from exc
        return self.get_improvement_proposal(proposal_id)

    def get_improvement_proposal(self, proposal_id: int) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM improvement_proposals WHERE proposal_id = ?",
                (proposal_id,),
            ).fetchone()
        if row is None:
            raise KaiError(f"Improvement proposal {proposal_id} does not exist.")
        return {
            "proposal_id": row["proposal_id"],
            "hypothesis_id": row["hypothesis_id"],
            "proposal_version": row["proposal_version"],
            "created_at": row["created_at"],
            "affected_component": row["affected_component"],
            "current_implementation": row["current_implementation"],
            "proposed_implementation": row["proposed_implementation"],
            "reason": row["reason"],
            "evidence": json.loads(row["evidence_json"]),
            "expected_benefit": row["expected_benefit"],
            "expected_cost": row["expected_cost"],
            "risk": row["risk"],
            "dependencies": json.loads(row["dependencies_json"]),
            "test_requirements": json.loads(row["test_requirements_json"]),
            "benchmark_requirements": json.loads(row["benchmark_requirements_json"]),
            "rollback_strategy": row["rollback_strategy"],
        }

    def create_kai_candidate(
        self,
        *,
        candidate_id: str,
        proposal_id: int,
        candidate_version: int,
        parent_kai_version: str,
        source_snapshot_ref: str,
        source_snapshot_sha256: str,
        candidate_specification: dict,
        isolation_metadata: dict,
        parent_candidate_id: str | None = None,
        timestamp: str | None = None,
    ) -> dict:
        for label, value in (
            ("candidate ID", candidate_id),
            ("parent KAI version", parent_kai_version),
            ("source snapshot reference", source_snapshot_ref),
            ("source snapshot digest", source_snapshot_sha256),
        ):
            if not isinstance(value, str) or not value.strip():
                raise KaiError(f"{label.capitalize()} must be non-empty.")
        if (
            isinstance(candidate_version, bool)
            or not isinstance(candidate_version, int)
            or candidate_version < 1
        ):
            raise KaiError("Candidate version must be a positive integer.")
        if (
            not isinstance(isolation_metadata, dict)
            or isolation_metadata.get("isolated") is not True
            or isolation_metadata.get("production_files_modified") is not False
            or isolation_metadata.get("production_model_mutation") is not False
        ):
            raise KaiError("Candidate must declare isolation from production.")
        try:
            specification_json = _encode_json(candidate_specification)
            isolation_json = _encode_json(isolation_metadata)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Candidate metadata must be finite JSON: {exc}") from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM improvement_proposals WHERE proposal_id = ?",
                    (proposal_id,),
                ).fetchone() is None:
                    raise KaiError(f"Improvement proposal {proposal_id} does not exist.")
                cursor = connection.execute(
                    "INSERT INTO kai_candidates "
                    "(candidate_id, proposal_id, parent_candidate_id, candidate_version, "
                    "parent_kai_version, source_snapshot_ref, source_snapshot_sha256, "
                    "candidate_specification_json, isolation_metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        candidate_id,
                        proposal_id,
                        parent_candidate_id,
                        candidate_version,
                        parent_kai_version,
                        source_snapshot_ref,
                        source_snapshot_sha256,
                        specification_json,
                        isolation_json,
                        now,
                    ),
                )
                connection.execute(
                    "INSERT INTO candidate_assessments "
                    "(candidate_id, assessed_at, test_status, benchmark_status, "
                    "evaluation_status, evidence_json) "
                    "VALUES (?, ?, 'NOT_RUN', 'NOT_RUN', 'UNKNOWN', '{}')",
                    (candidate_id, now),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.IntegrityError as exc:
                connection.rollback()
                raise KaiError(f"Cannot create isolated candidate: {exc}") from exc
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot create isolated candidate: {exc}") from exc
        return self.get_kai_candidate(candidate_id)

    def get_kai_candidate(self, candidate_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT c.*, a.test_status, a.benchmark_status, a.evaluation_status, "
                "a.assessed_at, a.evidence_json AS assessment_evidence "
                "FROM kai_candidates AS c LEFT JOIN candidate_assessments AS a "
                "ON a.assessment_id = (SELECT MAX(a2.assessment_id) "
                "FROM candidate_assessments AS a2 WHERE a2.candidate_id = c.candidate_id) "
                "WHERE c.candidate_id = ?",
                (candidate_id,),
            ).fetchone()
            if row is None:
                raise KaiError(f"Candidate {candidate_id!r} does not exist.")
            evaluations = connection.execute(
                "SELECT evaluation_id FROM improvement_evaluations "
                "WHERE candidate_id = ? ORDER BY evaluation_id",
                (candidate_id,),
            ).fetchall()
            decisions = connection.execute(
                "SELECT decision_id FROM improvement_decisions "
                "WHERE candidate_id = ? ORDER BY decision_id",
                (candidate_id,),
            ).fetchall()
        return {
            "candidate_id": row["candidate_id"],
            "proposal_id": row["proposal_id"],
            "parent_candidate_id": row["parent_candidate_id"],
            "candidate_version": row["candidate_version"],
            "parent_kai_version": row["parent_kai_version"],
            "source_snapshot_ref": row["source_snapshot_ref"],
            "source_snapshot_sha256": row["source_snapshot_sha256"],
            "candidate_specification": json.loads(row["candidate_specification_json"]),
            "isolation_metadata": json.loads(row["isolation_metadata_json"]),
            "created_at": row["created_at"],
            "test_status": row["test_status"],
            "benchmark_status": row["benchmark_status"],
            "evaluation_status": row["evaluation_status"],
            "last_assessed_at": row["assessed_at"],
            "assessment_evidence": json.loads(row["assessment_evidence"]),
            "deployment_status": row["deployment_status"],
            "evaluation_ids": [item["evaluation_id"] for item in evaluations],
            "decision_ids": [item["decision_id"] for item in decisions],
        }

    def list_kai_candidates(self, limit: int = 50) -> list[dict]:
        _validate_limit(limit, "Candidate")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT candidate_id FROM kai_candidates "
                "ORDER BY created_at DESC, candidate_id LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get_kai_candidate(row["candidate_id"]) for row in rows]

    def register_evaluation_protocol(self, protocol: EvaluationProtocol) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM evaluation_protocols WHERE protocol_id = ? AND version = ?",
                (protocol.protocol_id, protocol.version),
            ).fetchone()
        if row is None:
            raise KaiError(
                f"Evaluation protocol {protocol.protocol_id}@{protocol.version} is not registered."
            )
        if (
            row["definition_sha256"] != protocol.definition_sha256
            or json.loads(row["definition_json"]) != protocol.to_dict()
        ):
            raise KaiError("Persisted evaluation protocol differs from its registered definition.")
        return {
            "protocol_id": row["protocol_id"],
            "version": row["version"],
            "definition_sha256": row["definition_sha256"],
            "definition": json.loads(row["definition_json"]),
            "created_at": row["created_at"],
        }

    def list_evaluation_protocols(self) -> list[dict]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT protocol_id, version, definition_sha256, definition_json, created_at "
                "FROM evaluation_protocols ORDER BY protocol_id, version"
            ).fetchall()
        return [
            {
                "protocol_id": row["protocol_id"],
                "version": row["version"],
                "definition_sha256": row["definition_sha256"],
                "definition": json.loads(row["definition_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def create_candidate_workspace(
        self,
        *,
        workspace_id: str,
        candidate_id: str,
        relative_path: str,
        source_revision: str | None,
        source_tree_sha256: str,
        configuration_sha256: str,
        schema_version: int,
        python_version: str,
        operating_system: str,
        environment_fingerprint: str | None,
        metadata: dict,
    ) -> dict:
        try:
            if str(uuid.UUID(workspace_id)) != workspace_id or relative_path != workspace_id:
                raise ValueError("workspace IDs and relative paths must be canonical UUIDs")
            metadata_json = _encode_json(metadata)
            now = utc_now()
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM kai_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone() is None:
                    raise KaiError(f"Candidate {candidate_id!r} does not exist.")
                connection.execute(
                    "INSERT INTO candidate_workspaces "
                    "(workspace_id, candidate_id, relative_path, created_at, source_revision, "
                    "source_tree_sha256, configuration_sha256, schema_version, python_version, "
                    "operating_system, environment_fingerprint, metadata_json, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ACTIVE')",
                    (
                        workspace_id,
                        candidate_id,
                        relative_path,
                        now,
                        source_revision,
                        source_tree_sha256,
                        configuration_sha256,
                        schema_version,
                        python_version,
                        operating_system,
                        environment_fingerprint,
                        metadata_json,
                    ),
                )
                connection.execute(
                    "INSERT INTO candidate_workspace_events "
                    "(workspace_id, occurred_at, event_type, details_json) "
                    "VALUES (?, ?, 'CREATED', ?)",
                    (
                        workspace_id,
                        now,
                        _encode_json(
                            {
                                "candidate_id": candidate_id,
                                "source_tree_sha256": source_tree_sha256,
                                "source_revision": source_revision,
                            }
                        ),
                    ),
                )
                connection.commit()
        except (ValueError, sqlite3.IntegrityError) as exc:
            raise KaiError(f"Cannot persist candidate workspace: {exc}") from exc
        except sqlite3.Error as exc:
            raise KaiError(f"Cannot persist candidate workspace: {exc}") from exc
        return self.get_candidate_workspace(workspace_id)

    def get_candidate_workspace(self, workspace_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM candidate_workspaces WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchone()
            if row is None:
                raise KaiError(f"Candidate workspace {workspace_id!r} does not exist.")
            events = connection.execute(
                "SELECT workspace_event_id, occurred_at, event_type, details_json "
                "FROM candidate_workspace_events WHERE workspace_id = ? "
                "ORDER BY workspace_event_id",
                (workspace_id,),
            ).fetchall()
        return {
            "workspace_id": row["workspace_id"],
            "candidate_id": row["candidate_id"],
            "relative_path": row["relative_path"],
            "created_at": row["created_at"],
            "source_revision": row["source_revision"],
            "source_revision_status": "OBSERVED" if row["source_revision"] else "UNKNOWN",
            "source_tree_sha256": row["source_tree_sha256"],
            "configuration_sha256": row["configuration_sha256"],
            "schema_version": row["schema_version"],
            "python_version": row["python_version"],
            "operating_system": row["operating_system"],
            "environment_fingerprint": row["environment_fingerprint"],
            "metadata": json.loads(row["metadata_json"]),
            "status": row["status"],
            "events": [
                {
                    "workspace_event_id": event["workspace_event_id"],
                    "occurred_at": event["occurred_at"],
                    "event_type": event["event_type"],
                    "details": json.loads(event["details_json"]),
                }
                for event in events
            ],
        }

    def list_candidate_workspaces(self, limit: int = 50) -> list[dict]:
        _validate_limit(limit, "Candidate workspace")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT workspace_id FROM candidate_workspaces "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get_candidate_workspace(row["workspace_id"]) for row in rows]

    def record_candidate_workspace_event(
        self,
        workspace_id: str,
        event_type: str,
        details: dict,
        timestamp: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,40}", event_type):
            raise KaiError("Candidate workspace event type is invalid.")
        try:
            payload = _encode_json(details)
            now = _utc_timestamp(timestamp) if timestamp else utc_now()
            with self.connection() as connection:
                connection.execute(
                    "INSERT INTO candidate_workspace_events "
                    "(workspace_id, occurred_at, event_type, details_json) "
                    "VALUES (?, ?, ?, ?)",
                    (workspace_id, now, event_type, payload),
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise KaiError(f"Cannot record candidate workspace event: {exc}") from exc
        except sqlite3.Error as exc:
            raise KaiError(f"Cannot record candidate workspace event: {exc}") from exc

    def update_candidate_workspace_status(
        self, workspace_id: str, status: str, details: dict
    ) -> dict:
        allowed = {
            "ACTIVE": {"VALIDATED", "TESTED", "FAILED", "PRESERVED", "CLEANED"},
            "VALIDATED": {"TESTED", "FAILED", "PRESERVED", "CLEANED"},
            "TESTED": {"VALIDATED", "FAILED", "PRESERVED", "CLEANED"},
            "FAILED": {"VALIDATED", "TESTED", "PRESERVED", "CLEANED"},
            "PRESERVED": set(),
            "CLEANED": set(),
        }
        if status not in allowed:
            raise KaiError("Unsupported candidate workspace status.")
        now = utc_now()
        try:
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    "SELECT status FROM candidate_workspaces WHERE workspace_id = ?",
                    (workspace_id,),
                ).fetchone()
                if current is None:
                    raise KaiError(f"Candidate workspace {workspace_id!r} does not exist.")
                if status not in allowed[current["status"]]:
                    if status == current["status"]:
                        connection.rollback()
                        return self.get_candidate_workspace(workspace_id)
                    raise KaiError(
                        f"Invalid workspace status transition {current['status']} -> {status}."
                    )
                connection.execute(
                    "UPDATE candidate_workspaces SET status = ? WHERE workspace_id = ?",
                    (status, workspace_id),
                )
                connection.execute(
                    "INSERT INTO candidate_workspace_events "
                    "(workspace_id, occurred_at, event_type, details_json) "
                    "VALUES (?, ?, ?, ?)",
                    (workspace_id, now, status, _encode_json(details)),
                )
                connection.commit()
        except KaiError:
            raise
        except sqlite3.Error as exc:
            raise KaiError(f"Cannot update candidate workspace status: {exc}") from exc
        return self.get_candidate_workspace(workspace_id)

    def record_candidate_modification(
        self,
        *,
        modification_id: str,
        candidate_id: str,
        workspace_id: str,
        target_file: str,
        operation: str,
        original_sha256: str | None,
        resulting_sha256: str | None,
        patch: str,
        reason: str,
        hypothesis_id: int,
        validation_status: str,
        timestamp: str,
    ) -> dict:
        if operation not in {"ADD", "MODIFY", "DELETE"}:
            raise KaiError("Unsupported candidate modification operation.")
        if validation_status not in {"VALID", "INVALID"}:
            raise KaiError("Unsupported candidate modification validation status.")
        if not target_file.startswith(("kai/", "tests/")) or not target_file.endswith(".py"):
            raise KaiError("Candidate modification target is outside the Python source allowlist.")
        if len(patch.encode("utf-8")) > 2 * 1024 * 1024:
            raise KaiError("Candidate modification patch exceeds the size limit.")
        try:
            when = _utc_timestamp(timestamp)
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                lineage = connection.execute(
                    "SELECT 1 FROM candidate_workspaces AS workspace "
                    "JOIN kai_candidates AS candidate "
                    "ON candidate.candidate_id = workspace.candidate_id "
                    "JOIN improvement_proposals AS proposal "
                    "ON proposal.proposal_id = candidate.proposal_id "
                    "JOIN improvement_hypotheses AS hypothesis "
                    "ON hypothesis.hypothesis_id = proposal.hypothesis_id "
                    "WHERE workspace.workspace_id = ? "
                    "AND workspace.candidate_id = ? "
                    "AND hypothesis.hypothesis_id = ?",
                    (workspace_id, candidate_id, hypothesis_id),
                ).fetchone()
                if lineage is None:
                    raise KaiError(
                        "Candidate modification must match the workspace, candidate, "
                        "and its proposal hypothesis lineage."
                    )
                connection.execute(
                    "INSERT INTO candidate_modifications "
                    "(modification_id, candidate_id, workspace_id, target_file, operation, "
                    "original_sha256, resulting_sha256, patch, reason, hypothesis_id, "
                    "created_at, validation_status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        modification_id,
                        candidate_id,
                        workspace_id,
                        target_file,
                        operation,
                        original_sha256,
                        resulting_sha256,
                        patch,
                        reason,
                        hypothesis_id,
                        when,
                        validation_status,
                    ),
                )
                connection.execute(
                    "INSERT INTO candidate_workspace_events "
                    "(workspace_id, occurred_at, event_type, details_json) "
                    "VALUES (?, ?, 'MODIFIED', ?)",
                    (
                        workspace_id,
                        when,
                        _encode_json({"modification_id": modification_id}),
                    ),
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise KaiError(f"Cannot persist candidate modification: {exc}") from exc
        except sqlite3.Error as exc:
            raise KaiError(f"Cannot persist candidate modification: {exc}") from exc
        return self.get_candidate_modification(modification_id)

    def get_candidate_modification(self, modification_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM candidate_modifications WHERE modification_id = ?",
                (modification_id,),
            ).fetchone()
        if row is None:
            raise KaiError(f"Candidate modification {modification_id!r} does not exist.")
        return {
            "modification_id": row["modification_id"],
            "candidate_id": row["candidate_id"],
            "workspace_id": row["workspace_id"],
            "target_file": row["target_file"],
            "operation": row["operation"],
            "original_sha256": row["original_sha256"],
            "resulting_sha256": row["resulting_sha256"],
            "patch": row["patch"],
            "reason": row["reason"],
            "hypothesis_id": row["hypothesis_id"],
            "created_at": row["created_at"],
            "validation_status": row["validation_status"],
        }

    def list_candidate_modifications(
        self, workspace_id: str, limit: int = 100
    ) -> list[dict]:
        _validate_limit(limit, "Candidate modification")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT modification_id FROM candidate_modifications "
                "WHERE workspace_id = ? ORDER BY created_at, modification_id LIMIT ?",
                (workspace_id, limit),
            ).fetchall()
        return [
            self.get_candidate_modification(row["modification_id"])
            for row in rows
        ]

    def record_candidate_execution(self, execution: dict) -> dict:
        required = {
            "execution_id", "candidate_id", "workspace_id", "procedure_id", "status",
            "started_at", "completed_at", "duration_seconds", "exit_code", "stdout",
            "stderr", "environment", "reason",
        }
        if required - set(execution):
            raise KaiError("Candidate execution record is incomplete.")
        if execution["status"] not in {
            "SUCCESS", "FAILED", "TIMEOUT", "RESOURCE_LIMIT", "INVALID", "CRASHED", "UNKNOWN"
        }:
            raise KaiError("Candidate execution status is invalid.")
        if (
            not isinstance(execution["stdout"], str)
            or not isinstance(execution["stderr"], str)
            or not isinstance(execution["environment"], dict)
            or not isinstance(execution["duration_seconds"], (int, float))
            or isinstance(execution["duration_seconds"], bool)
            or not math.isfinite(execution["duration_seconds"])
            or execution["duration_seconds"] < 0
        ):
            raise KaiError("Candidate execution record contains invalid output or duration.")
        if len(execution["stdout"].encode()) > 128 * 1024 or len(execution["stderr"].encode()) > 128 * 1024:
            raise KaiError("Candidate execution output exceeds its persisted size limit.")
        try:
            started = _utc_timestamp(execution["started_at"])
            completed = _utc_timestamp(execution["completed_at"])
            environment = _encode_json(execution["environment"])
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                workspace = connection.execute(
                    "SELECT candidate_id, status FROM candidate_workspaces "
                    "WHERE workspace_id = ?",
                    (execution["workspace_id"],),
                ).fetchone()
                if (
                    workspace is None
                    or workspace["candidate_id"] != execution["candidate_id"]
                    or workspace["status"] in {"PRESERVED", "CLEANED"}
                ):
                    raise KaiError(
                        "Candidate execution must match an active workspace and candidate."
                    )
                connection.execute(
                    "INSERT INTO candidate_executions "
                    "(execution_id, candidate_id, workspace_id, procedure_id, status, "
                    "started_at, completed_at, duration_seconds, exit_code, stdout, stderr, "
                    "environment_json, reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        execution["execution_id"],
                        execution["candidate_id"],
                        execution["workspace_id"],
                        execution["procedure_id"],
                        execution["status"],
                        started,
                        completed,
                        execution["duration_seconds"],
                        execution["exit_code"],
                        execution["stdout"],
                        execution["stderr"],
                        environment,
                        execution["reason"],
                    ),
                )
                connection.execute(
                    "INSERT INTO candidate_workspace_events "
                    "(workspace_id, occurred_at, event_type, details_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        execution["workspace_id"],
                        completed,
                        "TESTED"
                        if execution["procedure_id"].startswith("unittest:")
                        else "EVALUATED",
                        _encode_json(
                            {
                                "execution_id": execution["execution_id"],
                                "procedure_id": execution["procedure_id"],
                                "status": execution["status"],
                            }
                        ),
                    ),
                )
                connection.commit()
        except sqlite3.IntegrityError as exc:
            raise KaiError(f"Cannot persist candidate execution: {exc}") from exc
        except sqlite3.Error as exc:
            raise KaiError(f"Cannot persist candidate execution: {exc}") from exc
        return self.get_candidate_execution(execution["execution_id"])

    def get_candidate_execution(self, execution_id: str) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM candidate_executions WHERE execution_id = ?",
                (execution_id,),
            ).fetchone()
        if row is None:
            raise KaiError(f"Candidate execution {execution_id!r} does not exist.")
        return {
            "execution_id": row["execution_id"],
            "candidate_id": row["candidate_id"],
            "workspace_id": row["workspace_id"],
            "procedure_id": row["procedure_id"],
            "status": row["status"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "duration_seconds": row["duration_seconds"],
            "exit_code": row["exit_code"],
            "stdout": row["stdout"],
            "stderr": row["stderr"],
            "environment": json.loads(row["environment_json"]),
            "reason": row["reason"],
        }

    def list_candidate_executions(self, candidate_id: str, limit: int = 50) -> list[dict]:
        _validate_limit(limit, "Candidate execution")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT execution_id FROM candidate_executions "
                "WHERE candidate_id = ? ORDER BY started_at DESC LIMIT ?",
                (candidate_id, limit),
            ).fetchall()
        return [self.get_candidate_execution(row["execution_id"]) for row in rows]

    def latest_candidate_decision(self, candidate_id: str) -> dict | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT decision_id FROM improvement_decisions WHERE candidate_id = ? "
                "ORDER BY decision_id DESC LIMIT 1",
                (candidate_id,),
            ).fetchone()
        return (
            self.get_improvement_decision(row["decision_id"]) if row else None
        )

    def record_candidate_evaluation(
        self,
        *,
        candidate_id: str,
        baseline: dict,
        evaluation_status: str,
        baseline_metrics: dict,
        candidate_metrics: dict,
        comparison: dict,
        test_status: str,
        benchmark_status: str,
        regression_count: int,
        failure_count: int,
        evidence_quality: str,
        evidence: dict | list,
        hypothesis_satisfied: bool | None,
        timestamp: str | None = None,
    ) -> dict:
        if evaluation_status not in IMPROVEMENT_EVALUATION_STATUSES:
            raise KaiError("Unsupported improvement evaluation status.")
        if test_status not in {"NOT_RUN", "PASSED", "FAILED"}:
            raise KaiError("Unsupported candidate test status.")
        if benchmark_status not in {"NOT_RUN", "COMPLETED", "FAILED"}:
            raise KaiError("Unsupported candidate benchmark status.")
        if evidence_quality not in {"NONE", "INSUFFICIENT", "PRELIMINARY", "ADEQUATE"}:
            raise KaiError("Unsupported improvement evidence quality.")
        for label, value in (
            ("regression", regression_count),
            ("failure", failure_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise KaiError(f"Candidate {label} count must be a non-negative integer.")
        if hypothesis_satisfied is not None and not isinstance(hypothesis_satisfied, bool):
            raise KaiError("Hypothesis satisfaction must be true, false, or unknown.")
        try:
            baseline_json = _encode_json(baseline_metrics)
            candidate_json = _encode_json(candidate_metrics)
            comparison_json = _encode_json(comparison)
            evidence_json = _encode_json(evidence)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Evaluation data must be finite JSON: {exc}") from exc
        if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
            raise KaiError("Baseline and candidate metrics must be objects.")
        if not isinstance(comparison, dict) or not isinstance(evidence, dict):
            raise KaiError("Evaluation comparison and evidence must be objects.")
        required_baseline_fields = {
            "kai_version",
            "architecture_version",
            "predictive_model_versions",
            "experiment_definition_versions",
            "configuration",
            "environment_context",
            "evaluation_dataset",
            "metrics",
        }
        if not isinstance(baseline, dict) or required_baseline_fields - set(baseline):
            raise KaiError("Candidate evaluation requires a complete baseline snapshot.")
        if baseline["metrics"] != baseline_metrics:
            raise KaiError("Baseline snapshot metrics must match the evaluated metrics.")
        if (
            not isinstance(baseline["kai_version"], str)
            or not baseline["kai_version"].strip()
            or not isinstance(baseline["architecture_version"], str)
            or not baseline["architecture_version"].strip()
            or not isinstance(baseline["predictive_model_versions"], list)
            or not isinstance(baseline["experiment_definition_versions"], list)
            or not isinstance(baseline["configuration"], dict)
            or not isinstance(baseline["environment_context"], dict)
            or not isinstance(baseline["evaluation_dataset"], dict)
        ):
            raise KaiError("Candidate evaluation baseline snapshot is malformed.")
        evidence = {**evidence, "baseline": baseline}
        baseline_snapshot_sha256 = hashlib.sha256(
            _encode_json(baseline).encode("utf-8")
        ).hexdigest()
        try:
            evidence_json = _encode_json(evidence)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Evaluation baseline must be finite JSON: {exc}") from exc
        source_observation_ids = evidence.get("source_observation_ids")
        if (
            not isinstance(source_observation_ids, list)
            or not source_observation_ids
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 1
                for item in source_observation_ids
            )
            or len(set(source_observation_ids)) != len(source_observation_ids)
        ):
            raise KaiError(
                "Candidate evaluation requires unique persisted source observations."
            )
        dataset = baseline["evaluation_dataset"]
        dataset_observation_count = dataset.get("observation_count")
        if (
            dataset.get("observation_ids") != source_observation_ids
            or isinstance(dataset_observation_count, bool)
            or not isinstance(dataset_observation_count, int)
            or dataset_observation_count < 0
            or dataset_observation_count > len(source_observation_ids)
        ):
            raise KaiError(
                "Baseline dataset must reference the evaluation's persisted observations."
            )
        primary_metric = comparison.get("primary_metric")
        metric_direction = comparison.get("metric_direction")
        paired_count = comparison.get(
            "paired_forecast_count", comparison.get("sample_count")
        )
        if paired_count is None:
            paired_count = 0
        if (
            isinstance(paired_count, bool)
            or not isinstance(paired_count, int)
            or paired_count < 0
            or paired_count > len(source_observation_ids)
        ):
            raise KaiError("Candidate evaluation paired sample count is invalid.")
        if dataset.get("paired_forecast_count") != paired_count:
            raise KaiError(
                "Baseline paired sample count conflicts with the comparison."
            )
        protocol_id = None
        protocol_snapshot = baseline.get("evaluation_protocol")
        if protocol_snapshot is not None:
            if not isinstance(protocol_snapshot, dict):
                raise KaiError("Evaluation protocol snapshot is malformed.")
            protocol_id = protocol_snapshot.get("protocol_id")
            protocol_version = protocol_snapshot.get("version")
            protocol_hash = protocol_snapshot.get("definition_sha256")
            protocol_definition = protocol_snapshot.get("definition")
            if (
                not isinstance(protocol_id, str)
                or isinstance(protocol_version, bool)
                or not isinstance(protocol_version, int)
            ):
                raise KaiError("Evaluation protocol identity is malformed.")
            if (
                not isinstance(protocol_definition, dict)
                or protocol_definition.get("protocol_id") != protocol_id
                or protocol_definition.get("version") != protocol_version
            ):
                raise KaiError("Evaluation protocol definition does not match its identity.")
            try:
                persisted_protocol = EvaluationProtocol(**protocol_definition)
                if persisted_protocol.definition_sha256 != protocol_hash:
                    raise KaiError("Evaluation protocol digest does not match its definition.")
                derived_protocol_status = validate_protocol_comparison(
                    comparison, persisted_protocol
                )
            except (TypeError, ValueError, KeyError) as exc:
                raise KaiError(f"Evaluation protocol comparison is invalid: {exc}") from exc
        else:
            derived_protocol_status = None
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM kai_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone() is None:
                    raise KaiError(f"Candidate {candidate_id!r} does not exist.")
                if protocol_id is not None:
                    stored_protocol = connection.execute(
                        "SELECT definition_sha256, definition_json FROM evaluation_protocols "
                        "WHERE protocol_id = ? AND version = ?",
                        (protocol_id, protocol_version),
                    ).fetchone()
                    if (
                        stored_protocol is None
                        or stored_protocol["definition_sha256"] != protocol_hash
                        or json.loads(stored_protocol["definition_json"])
                        != protocol_definition
                    ):
                        raise KaiError(
                            "Evaluation protocol is not the immutable registered definition."
                        )
                source_rows = connection.execute(
                    "SELECT observation_id, epistemic_status, evidence_status "
                    "FROM self_observations WHERE observation_id IN ("
                    + ",".join("?" for _ in source_observation_ids)
                    + ")",
                    source_observation_ids,
                ).fetchall()
                if len(source_rows) != len(source_observation_ids):
                    raise KaiError(
                        "Candidate evaluation references missing self-observations."
                    )
                observed_count = sum(
                    row["epistemic_status"] == "OBSERVED"
                    and row["evidence_status"] in {"observed", "verified"}
                    for row in source_rows
                )
                expected_quality = (
                    "NONE"
                    if observed_count == 0
                    else "INSUFFICIENT"
                    if observed_count < 2 or paired_count < 2
                    else "PRELIMINARY"
                    if paired_count < IMPROVEMENT_EVIDENCE_MINIMUM
                    else "ADEQUATE"
                    if observed_count >= paired_count
                    else "INSUFFICIENT"
                )
                if evidence_quality != expected_quality:
                    raise KaiError(
                        "Candidate evidence quality does not match its persisted "
                        "observations and paired sample count."
                    )
                if derived_protocol_status is not None:
                    if test_status == "FAILED" or benchmark_status == "FAILED" or failure_count:
                        derived_status = "FAILED"
                    elif test_status != "PASSED" or benchmark_status != "COMPLETED":
                        derived_status = "UNKNOWN"
                    else:
                        derived_status = derived_protocol_status
                    derived_hypothesis = (
                        True
                        if derived_status == "IMPROVED"
                        else False
                        if derived_status == "REGRESSED"
                        else None
                    )
                    if (
                        baseline_metrics.get(primary_metric)
                        != comparison["baseline_metrics"]["mae_seconds"]
                        or candidate_metrics.get(primary_metric)
                        != comparison["candidate_metrics"]["mae_seconds"]
                    ):
                        raise KaiError(
                            "Repeated protocol metrics do not match persisted evaluation metrics."
                        )
                else:
                    try:
                        derived_status, derived_hypothesis = classify_candidate_comparison(
                            baseline_metrics=baseline_metrics,
                            candidate_metrics=candidate_metrics,
                            primary_metric=primary_metric,
                            metric_direction=metric_direction,
                            evidence_quality=evidence_quality,
                            test_status=test_status,
                            benchmark_status=benchmark_status,
                            regression_count=regression_count,
                            failure_count=failure_count,
                        )
                    except ValueError as exc:
                        raise KaiError(str(exc)) from exc
                if evaluation_status != derived_status:
                    raise KaiError(
                        f"Evaluation status {evaluation_status} conflicts with "
                        f"the recorded metrics and evidence ({derived_status})."
                    )
                if hypothesis_satisfied is not derived_hypothesis:
                    raise KaiError(
                        "Hypothesis satisfaction conflicts with the measured comparison."
                    )
                if comparison.get("status") not in (None, evaluation_status):
                    raise KaiError(
                        "Comparison summary conflicts with the evaluation status."
                    )
                cursor = connection.execute(
                    "INSERT INTO improvement_evaluations "
                    "(candidate_id, created_at, hypothesis_satisfied, evaluation_status, "
                    "baseline_metrics_json, candidate_metrics_json, comparison_json, "
                    "test_status, benchmark_status, regression_count, failure_count, "
                    "evidence_quality, evidence_json, evaluation_protocol_id, "
                    "baseline_snapshot_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        candidate_id,
                        now,
                        None if hypothesis_satisfied is None else int(hypothesis_satisfied),
                        evaluation_status,
                        baseline_json,
                        candidate_json,
                        comparison_json,
                        test_status,
                        benchmark_status,
                        regression_count,
                        failure_count,
                        evidence_quality,
                        evidence_json,
                        protocol_id,
                        baseline_snapshot_sha256,
                    ),
                )
                evaluation_id = int(cursor.lastrowid)
                connection.execute(
                    "INSERT INTO candidate_assessments "
                    "(candidate_id, assessed_at, test_status, benchmark_status, "
                    "evaluation_status, evidence_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        candidate_id,
                        now,
                        test_status,
                        benchmark_status,
                        evaluation_status,
                        evidence_json,
                    ),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot persist candidate evaluation: {exc}") from exc
        return self.get_improvement_evaluation(evaluation_id)

    def get_improvement_evaluation(self, evaluation_id: int) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM improvement_evaluations WHERE evaluation_id = ?",
                (evaluation_id,),
            ).fetchone()
        if row is None:
            raise KaiError(f"Improvement evaluation {evaluation_id} does not exist.")
        return {
            "evaluation_id": row["evaluation_id"],
            "candidate_id": row["candidate_id"],
            "created_at": row["created_at"],
            "hypothesis_satisfied": (
                None
                if row["hypothesis_satisfied"] is None
                else bool(row["hypothesis_satisfied"])
            ),
            "evaluation_status": row["evaluation_status"],
            "baseline_metrics": json.loads(row["baseline_metrics_json"]),
            "candidate_metrics": json.loads(row["candidate_metrics_json"]),
            "comparison": json.loads(row["comparison_json"]),
            "test_status": row["test_status"],
            "benchmark_status": row["benchmark_status"],
            "regression_count": row["regression_count"],
            "failure_count": row["failure_count"],
            "evidence_quality": row["evidence_quality"],
            "evidence": json.loads(row["evidence_json"]),
            "baseline": json.loads(row["evidence_json"]).get("baseline"),
            "evaluation_protocol_id": row["evaluation_protocol_id"],
            "baseline_snapshot_sha256": row["baseline_snapshot_sha256"],
        }

    def list_improvement_evaluations(self, limit: int = 50) -> list[dict]:
        _validate_limit(limit, "Improvement evaluation")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT evaluation_id FROM improvement_evaluations "
                "ORDER BY evaluation_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get_improvement_evaluation(row["evaluation_id"]) for row in rows]

    def record_improvement_decision(
        self,
        *,
        candidate_id: str,
        evaluation_id: int | None,
        decision: str,
        reason: str,
        evidence: dict | list,
        timestamp: str | None = None,
    ) -> dict:
        if decision not in IMPROVEMENT_DECISIONS:
            raise KaiError("Unsupported improvement decision.")
        if not isinstance(reason, str) or not reason.strip():
            raise KaiError("Improvement decision requires a reason.")
        try:
            evidence_json = _encode_json(evidence)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Decision evidence must be finite JSON: {exc}") from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                candidate = connection.execute(
                    "SELECT * FROM kai_candidates WHERE candidate_id = ?",
                    (candidate_id,),
                ).fetchone()
                if candidate is None:
                    raise KaiError(f"Candidate {candidate_id!r} does not exist.")
                evaluation = None
                if evaluation_id is not None:
                    evaluation_row = connection.execute(
                        "SELECT * FROM improvement_evaluations WHERE evaluation_id = ? "
                        "AND candidate_id = ?",
                        (evaluation_id, candidate_id),
                    ).fetchone()
                    if evaluation_row is None:
                        raise KaiError(
                            "Decision evaluation does not belong to this candidate."
                        )
                    latest_evaluation = connection.execute(
                        "SELECT MAX(evaluation_id) FROM improvement_evaluations "
                        "WHERE candidate_id = ?",
                        (candidate_id,),
                    ).fetchone()[0]
                    if evaluation_id != latest_evaluation:
                        raise KaiError(
                            "A decision must reference the candidate's latest evaluation."
                        )
                    evaluation = {
                        "status": evaluation_row["evaluation_status"],
                        "test_status": evaluation_row["test_status"],
                        "evidence_quality": evaluation_row["evidence_quality"],
                    }
                regression_count = (
                    evaluation_row["regression_count"]
                    if evaluation_id is not None
                    else None
                )
                try:
                    validate_improvement_decision(
                        decision, evaluation, regression_count
                    )
                except ValueError as exc:
                    raise KaiError(str(exc)) from exc
                if decision == "ACCEPT_CANDIDATE" and candidate["deployment_status"] != "NOT_DEPLOYED":
                    raise KaiError("Candidate deployment state is not eligible.")
                cursor = connection.execute(
                    "INSERT INTO improvement_decisions "
                    "(candidate_id, evaluation_id, decided_at, decision, reason, "
                    "evidence_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        candidate_id,
                        evaluation_id,
                        now,
                        decision,
                        reason,
                        evidence_json,
                    ),
                )
                decision_id = int(cursor.lastrowid)
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot persist improvement decision: {exc}") from exc
        return self.get_improvement_decision(decision_id)

    def get_improvement_decision(self, decision_id: int) -> dict:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM improvement_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            raise KaiError(f"Improvement decision {decision_id} does not exist.")
        return {
            "decision_id": row["decision_id"],
            "candidate_id": row["candidate_id"],
            "evaluation_id": row["evaluation_id"],
            "decided_at": row["decided_at"],
            "decision": row["decision"],
            "reason": row["reason"],
            "evidence": json.loads(row["evidence_json"]),
        }

    def list_improvement_decisions(self, limit: int = 50) -> list[dict]:
        _validate_limit(limit, "Improvement decision")
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT decision_id FROM improvement_decisions "
                "ORDER BY decision_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.get_improvement_decision(row["decision_id"]) for row in rows]

    def improvement_history(self, limit: int = 50) -> dict:
        _validate_limit(limit, "Improvement history")
        return {
            "opportunities": self.list_improvement_opportunities(limit),
            "candidates": self.list_kai_candidates(limit),
            "evaluations": self.list_improvement_evaluations(limit),
            "metrics": self.improvement_metrics(),
        }

    def improvement_metrics(self) -> dict:
        with self.connection() as connection:
            opportunities = connection.execute(
                "SELECT status, COUNT(*) AS count FROM improvement_opportunities GROUP BY status"
            ).fetchall()
            candidates = connection.execute(
                "SELECT COUNT(*) FROM kai_candidates"
            ).fetchone()[0]
            evaluation_rows = connection.execute(
                "SELECT evaluation_status, COUNT(*) AS count FROM "
                "improvement_evaluations GROUP BY evaluation_status"
            ).fetchall()
            decisions = connection.execute(
                "SELECT decision, COUNT(*) AS count FROM improvement_decisions GROUP BY decision"
            ).fetchall()
            evaluated = connection.execute(
                "SELECT COUNT(*) FROM improvement_evaluations"
            ).fetchone()[0]
            regressions = connection.execute(
                "SELECT COUNT(*) FROM improvement_evaluations WHERE regression_count > 0 "
                "OR evaluation_status = 'REGRESSED'"
            ).fetchone()[0]
        return {
            "opportunities_by_status": {
                row["status"]: row["count"] for row in opportunities
            },
            "candidate_count": int(candidates),
            "evaluations_by_status": {
                row["evaluation_status"]: row["count"] for row in evaluation_rows
            },
            "decisions_by_status": {row["decision"]: row["count"] for row in decisions},
            "regression_rate": (regressions / evaluated) if evaluated else None,
            "improvement_evaluations_count": int(evaluated),
            "interpretation": (
                "Descriptive ledger counts only; too few observations for claims "
                "about the long-run success rate of self-improvement."
            ),
        }

    @staticmethod
    def _predictive_model_row(row: sqlite3.Row) -> PredictiveModel:
        return {
            "model_id": row["model_id"],
            "model_type": row["model_type"],
            "domain": row["domain"],
            "context_key": row["context_key"],
            "version": row["version"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "parameters": json.loads(row["parameters_json"]),
            "confidence": json.loads(row["confidence_json"]),
            "performance": json.loads(row["performance_json"]),
            "training_evidence_references": json.loads(row["evidence_refs_json"]),
        }

    def get_cpu_baseline_model(self, context: dict) -> dict | None:
        try:
            validate_cpu_baseline_context(context)
        except ValueError as exc:
            raise KaiError(str(exc)) from exc
        key = prediction_context_key(context)
        with self.connection() as connection:
            try:
                row = connection.execute(
                    "SELECT * FROM predictive_models WHERE domain = ? AND context_key = ? "
                    "AND status = 'ACTIVE' ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                    (CPU_BASELINE_DOMAIN, key),
                ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot load predictive model: {exc}") from exc
        return self._predictive_model_row(row) if row else None

    def list_predictive_models(self) -> list[PredictiveModel]:
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM predictive_models "
                    "ORDER BY domain, context_key, updated_at DESC"
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot list predictive models: {exc}") from exc
        return [self._predictive_model_row(row) for row in rows]

    def predictive_model_history(self, model_id: str, limit: int = 20) -> dict:
        _validate_limit(limit, "Model history")
        with self.connection() as connection:
            try:
                versions = connection.execute(
                    "SELECT * FROM predictive_model_versions WHERE model_id = ? "
                    "ORDER BY version DESC LIMIT ?",
                    (model_id, limit),
                ).fetchall()
                updates = connection.execute(
                    "SELECT * FROM predictive_model_updates WHERE model_id = ? "
                    "ORDER BY update_id DESC LIMIT ?",
                    (model_id, limit),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read predictive model history: {exc}") from exc
        return {
            "versions": [
                {
                    "version": row["version"],
                    "created_at": row["created_at"],
                    "parameters": json.loads(row["parameters_json"]),
                    "confidence": json.loads(row["confidence_json"]),
                    "performance": json.loads(row["performance_json"]),
                    "training_evidence_references": json.loads(
                        row["evidence_refs_json"]
                    ),
                    "status": row["status"],
                }
                for row in versions
            ],
            "updates": [
                {
                    "update_id": row["update_id"],
                    "experience_id": row["experience_id"],
                    "prediction_id": row["prediction_id"],
                    "experiment_id": row["experiment_id"],
                    "created_at": row["created_at"],
                    "version_before": row["version_before"],
                    "version_after": row["version_after"],
                    "update_rule": row["update_rule"],
                    "parameters_before": json.loads(row["parameters_before_json"]),
                    "parameters_after": json.loads(row["parameters_after_json"]),
                    "input_evidence": json.loads(row["input_evidence_json"]),
                    "prediction_error": _decode_json(row["prediction_error_json"]),
                    "reverses_update_id": row["reverses_update_id"],
                    "update_event_id": row["update_event_id"],
                    "execution_id": row["execution_id"],
                    "experiment_definition_id": row["experiment_definition_id"],
                    "experiment_version": row["experiment_version"],
                    "update_reason": row["update_reason"],
                }
                for row in updates
            ],
        }

    def prediction_statistics(self, model_id: str | None = None) -> dict:
        with self.connection() as connection:
            try:
                if model_id is None:
                    rows = connection.execute(
                        "SELECT status, confidence, difference_json FROM predictions "
                        "ORDER BY prediction_id"
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT status, confidence, difference_json FROM predictions "
                        "WHERE model_id = ? ORDER BY prediction_id",
                        (model_id,),
                    ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot calculate prediction statistics: {exc}") from exc
        return prediction_statistics(
            [
                {
                    "status": row["status"],
                    "confidence": row["confidence"],
                    "difference": _decode_json(row["difference_json"]),
                }
                for row in rows
            ]
        )

    def prediction_report(self, context: dict | None = None, limit: int = 20) -> dict:
        _validate_limit(limit, "Prediction history")
        models = self.list_predictive_models()
        if context is not None:
            context_key = prediction_context_key(context)
            models = [
                model
                for model in models
                if model["context_key"] == context_key and model["status"] == "ACTIVE"
            ]
        with self.connection() as connection:
            try:
                if models:
                    model_ids = [model["model_id"] for model in models]
                    placeholders = ",".join("?" for _ in model_ids)
                    predictions = connection.execute(
                        "SELECT * FROM predictions WHERE model_id IN "
                        f"({placeholders}) ORDER BY prediction_id DESC LIMIT ?",
                        (*model_ids, limit),
                    ).fetchall()
                else:
                    predictions = []
                experiments = connection.execute(
                    "SELECT * FROM experiment_executions ORDER BY execution_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot build prediction report: {exc}") from exc
        return {
            "models": models,
            "statistics": {
                model["model_id"]: self.prediction_statistics(model["model_id"])
                for model in models
            },
            "predictions": [_prediction_row(row) for row in predictions],
            "experiments": [_experiment_row(row) for row in experiments],
        }

    def register_experiment_definition(self, definition: dict) -> dict:
        try:
            experiment_id = definition["experiment_id"]
            version = definition["version"]
            status = definition["status"]
            definition_key = f"{experiment_id}@{version}"
            timestamp = utc_now()
            with self.connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM experiments WHERE experiment_id = ? AND version = ?",
                    (experiment_id, version),
                ).fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO experiments "
                        "(definition_key, experiment_id, domain, name, description, "
                        "version, status, procedure_id, parameter_specification_json, "
                        "context_specification_json, outcome_specification_json, "
                        "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            definition_key,
                            experiment_id,
                            definition["domain"],
                            definition["name"],
                            definition["description"],
                            version,
                            status,
                            definition["procedure_id"],
                            _encode_json(definition["parameter_specification"]),
                            _encode_json(definition["context_specification"]),
                            _encode_json(definition["outcome_specification"]),
                            timestamp,
                            timestamp,
                        ),
                    )
                else:
                    immutable = (
                        ("domain", definition["domain"]),
                        ("name", definition["name"]),
                        ("description", definition["description"]),
                        ("procedure_id", definition["procedure_id"]),
                        (
                            "parameter_specification_json",
                            _encode_json(definition["parameter_specification"]),
                        ),
                        (
                            "context_specification_json",
                            _encode_json(definition["context_specification"]),
                        ),
                        (
                            "outcome_specification_json",
                            _encode_json(definition["outcome_specification"]),
                        ),
                    )
                    if any(row[column] != value for column, value in immutable):
                        raise KaiError(
                            "An experiment definition cannot change without a new version."
                        )
                connection.commit()
        except KaiError:
            raise
        except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            raise KaiError(f"Cannot persist experiment definition: {exc}") from exc
        stored = self.get_experiment_definition(experiment_id, version)
        if stored is None:
            raise KaiError("Experiment definition was not persisted.")
        return stored

    def get_experiment_definition(
        self, experiment_id: str, version: int | None = None
    ) -> dict | None:
        with self.connection() as connection:
            try:
                if version is None:
                    row = connection.execute(
                        "SELECT * FROM experiments WHERE experiment_id = ? "
                        "ORDER BY version DESC LIMIT 1",
                        (experiment_id,),
                    ).fetchone()
                else:
                    row = connection.execute(
                        "SELECT * FROM experiments WHERE experiment_id = ? AND version = ?",
                        (experiment_id, version),
                    ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot load experiment definition: {exc}") from exc
        return _experiment_definition_row(row) if row else None

    def list_experiment_definitions(self) -> list[dict]:
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM experiments ORDER BY experiment_id, version"
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot list experiment definitions: {exc}") from exc
        return [_experiment_definition_row(row) for row in rows]

    def set_experiment_definition_status(
        self, experiment_id: str, status: str, version: int = 1
    ) -> dict:
        if status not in {"DEFINED", "ENABLED", "DISABLED", "DEPRECATED"}:
            raise KaiError(f"Unsupported experiment status: {status}")
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "UPDATE experiments SET status = ?, updated_at = ? "
                    "WHERE experiment_id = ? AND version = ?",
                    (status, utc_now(), experiment_id, version),
                )
                if cursor.rowcount != 1:
                    raise KaiError(
                        f"Experiment {experiment_id!r} version {version} is not registered."
                    )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot update experiment status: {exc}") from exc
        stored = self.get_experiment_definition(experiment_id, version)
        if stored is None:
            raise KaiError("Experiment definition disappeared after status update.")
        return stored

    def get_experiment_execution(self, execution_id: int) -> dict:
        with self.connection() as connection:
            try:
                row = connection.execute(
                    "SELECT x.*, r.execution_id AS result_execution_id, "
                    "r.outcome_value, r.unit, r.sample_count, r.duration_seconds, "
                    "r.metadata_json, r.measurement_status, "
                    "v.execution_id AS evaluation_execution_id, "
                    "v.prediction_id AS evaluation_prediction_id, v.expected_value, "
                    "v.observed_value, v.error, v.absolute_error, v.relative_error, "
                    "v.evaluation_status, v.evidence_quality, v.evidence_json "
                    "FROM experiment_executions AS x "
                    "LEFT JOIN experiment_results AS r ON r.execution_id = x.execution_id "
                    "LEFT JOIN experiment_evaluations AS v ON v.execution_id = x.execution_id "
                    "WHERE x.execution_id = ?",
                    (execution_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read experiment execution: {exc}") from exc
        if row is None:
            raise KaiError(f"Experiment execution {execution_id} does not exist.")
        execution = _experiment_execution_row(row)
        with self.connection() as connection:
            prediction_row = connection.execute(
                "SELECT prediction_id FROM predictions WHERE execution_id = ? "
                "ORDER BY prediction_id DESC LIMIT 1",
                (execution_id,),
            ).fetchone()
            update_row = connection.execute(
                "SELECT update_id, model_id, version_before, version_after, "
                "update_reason, update_rule FROM predictive_model_updates "
                "WHERE execution_id = ? ORDER BY update_id DESC LIMIT 1",
                (execution_id,),
            ).fetchone()
        execution["prediction"] = (
            self.get_prediction(prediction_row["prediction_id"])
            if prediction_row
            else None
        )
        execution["model_update"] = (
            {
                "update_id": update_row["update_id"],
                "model_id": update_row["model_id"],
                "version_before": update_row["version_before"],
                "version_after": update_row["version_after"],
                "reason": update_row["update_reason"],
                "rule": update_row["update_rule"],
            }
            if update_row
            else None
        )
        if execution["model_update"] is not None:
            history = self.predictive_model_history(
                execution["model_update"]["model_id"], limit=MAX_LIST_LIMIT
            )
            execution["model_update"]["version_record"] = next(
                (
                    item
                    for item in history["versions"]
                    if item["version"] == execution["model_update"]["version_after"]
                ),
                None,
            )
        return execution

    def list_experiment_executions(
        self, limit: int = 20, experiment_id: str | None = None
    ) -> list[dict]:
        _validate_limit(limit, "Experiment execution")
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT x.*, r.execution_id AS result_execution_id, "
                    "r.outcome_value, r.unit, r.sample_count, r.duration_seconds, "
                    "r.metadata_json, r.measurement_status, "
                    "v.execution_id AS evaluation_execution_id, "
                    "v.prediction_id AS evaluation_prediction_id, v.expected_value, "
                    "v.observed_value, v.error, v.absolute_error, v.relative_error, "
                    "v.evaluation_status, v.evidence_quality, v.evidence_json "
                    "FROM experiment_executions AS x "
                    "LEFT JOIN experiment_results AS r ON r.execution_id = x.execution_id "
                    "LEFT JOIN experiment_evaluations AS v ON v.execution_id = x.execution_id "
                    "WHERE (? IS NULL OR x.experiment_definition_id = ?) "
                    "ORDER BY x.execution_id DESC LIMIT ?",
                    (experiment_id, experiment_id, limit),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot list experiment executions: {exc}") from exc
        return [self.get_experiment_execution(row["execution_id"]) for row in rows]

    def list_experiments(self, limit: int = 20) -> list[Experiment]:
        _validate_limit(limit, "Experiment")
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM experiment_executions ORDER BY execution_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot list experiments: {exc}") from exc
        return [_experiment_row(row) for row in rows]

    def begin_cpu_baseline_experiment(
        self, context: dict, timestamp: str | None = None
    ) -> dict:
        try:
            validate_cpu_baseline_context(context)
        except ValueError as exc:
            raise KaiError(str(exc)) from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        context_key = prediction_context_key(context)
        resources = {
            "cpu": "low (bounded fixed 100000-iteration integer loop)",
            "ram": "low",
            "gpu": "none",
            "storage": "low (bounded read-only sample of existing KAI database)",
            "duration": "short, system-load dependent",
        }
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                model_row = connection.execute(
                    "SELECT * FROM predictive_models WHERE domain = ? "
                    "AND context_key = ? AND status = 'ACTIVE' "
                    "ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                    (CPU_BASELINE_DOMAIN, context_key),
                ).fetchone()
                model = self._predictive_model_row(model_row) if model_row else None
                experience_id = None
                prediction_id = None
                if model is not None:
                    pending = connection.execute(
                        "SELECT * FROM predictions WHERE model_id = ? "
                        "AND model_version = ? AND status = 'PENDING' "
                        "ORDER BY prediction_id DESC LIMIT 1",
                        (model["model_id"], model["version"]),
                    ).fetchone()
                    if pending is not None:
                        prediction_id = pending["prediction_id"]
                        experience_id = pending["experience_id"]
                if experience_id is None:
                    cursor = connection.execute(
                        "INSERT INTO experiences "
                        "(created_at, updated_at, status, context_json, "
                        "source_event_ids_json) VALUES (?, ?, 'OPEN', ?, '[]')",
                        (now, now, _encode_json(context)),
                    )
                    experience_id = int(cursor.lastrowid)
                before = (
                    uncertainty(model["parameters"])
                    if model
                    else {
                        "status": "UNKNOWN",
                        "reason": "No prior CPU baseline evidence exists.",
                        "standard_error_seconds": None,
                    }
                )
                cursor = connection.execute(
                    "INSERT INTO experiment_executions "
                    "(experience_id, domain, question, hypothesis, experiment_type, "
                    "resource_estimate_json, status, uncertainty_before_json, started_at, "
                    "experiment_definition_key, experiment_definition_id, "
                    "experiment_version, procedure_id, execution_status, "
                    "environment_snapshot_id, context_profile, parameters_json, context_json) "
                    "VALUES (?, ?, ?, ?, 'cpu_baseline', ?, 'RUNNING', ?, ?, ?, ?, ?, ?, "
                    "'RUNNING', ?, ?, ?, ?)",
                    (
                        experience_id,
                        CPU_BASELINE_DOMAIN,
                        "How long will the fixed CPU baseline take in this environment profile?",
                        (
                            f"Runtime will be near {model['parameters']['mean_seconds']:.6f} seconds."
                            if model
                            else "A baseline observation will initialize a runtime estimate."
                        ),
                        _encode_json(resources),
                        _encode_json(before),
                        now,
                        f"{CPU_BASELINE_EXPERIMENT.experiment_id}@{CPU_BASELINE_EXPERIMENT.version}",
                        CPU_BASELINE_EXPERIMENT.experiment_id,
                        CPU_BASELINE_EXPERIMENT.version,
                        CPU_BASELINE_EXPERIMENT.procedure_id,
                        context.get("environment_snapshot_id"),
                        context["environment_profile"],
                        _encode_json({"iterations": 100_000}),
                        _encode_json(context),
                    ),
                )
                experiment_id = int(cursor.lastrowid)
                registered_definition = connection.execute(
                    "SELECT status FROM experiments WHERE experiment_id = ? AND version = ?",
                    (
                        CPU_BASELINE_EXPERIMENT.experiment_id,
                        CPU_BASELINE_EXPERIMENT.version,
                    ),
                ).fetchone()
                if registered_definition is None:
                    raise KaiError("The CPU baseline experiment definition is not registered.")
                if registered_definition["status"] != "ENABLED":
                    raise KaiError(
                        f"Experiment {CPU_BASELINE_EXPERIMENT.experiment_id!r} "
                        f"is {registered_definition['status'].lower()}."
                    )
                selection_event_id = _insert_event(
                    connection,
                    "EXPERIMENT_SELECTED",
                    "Selected the explicit allowlisted CPU baseline experiment.",
                    {
                        "experiment_id": experiment_id,
                        "domain": CPU_BASELINE_DOMAIN,
                        "uncertainty_before": before,
                        "resource_estimate": resources,
                    },
                    source="kai.experiment_selector",
                    experiment_id=experiment_id,
                    execution_id=experiment_id,
                    experiment_definition_key=(
                        f"{CPU_BASELINE_EXPERIMENT.experiment_id}@"
                        f"{CPU_BASELINE_EXPERIMENT.version}"
                    ),
                    experiment_definition_id=CPU_BASELINE_EXPERIMENT.experiment_id,
                    experiment_version=CPU_BASELINE_EXPERIMENT.version,
                    confidence=None,
                    epistemic_status="UNKNOWN" if model is None else "INFERENCE",
                    occurred_at=now,
                )
                if prediction_id is None and model is not None:
                    estimate = model["parameters"]["mean_seconds"]
                    stats = self.prediction_statistics(model["model_id"])
                    confidence = confidence_from_statistics(stats)
                    cursor = connection.execute(
                        "INSERT INTO predictions "
                        "(experience_id, created_at, context_json, hypothesis, "
                        "expected_outcome_json, confidence, basis, observation_window, "
                        "status, model_id, model_version, experiment_id, execution_id, "
                        "experiment_definition_key, experiment_definition_id, "
                        "experiment_version, environment_snapshot_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            experience_id,
                            now,
                            _encode_json(context),
                            f"CPU baseline runtime will be near {estimate:.6f} seconds.",
                            _encode_json(estimate),
                            confidence["empirical_accuracy"],
                            f"{CPU_BASELINE_MODEL_TYPE} model version {model['version']}",
                            "the next explicit CPU baseline experiment",
                            model["model_id"],
                            model["version"],
                            experiment_id,
                            experiment_id,
                            f"{CPU_BASELINE_EXPERIMENT.experiment_id}@{CPU_BASELINE_EXPERIMENT.version}",
                            CPU_BASELINE_EXPERIMENT.experiment_id,
                            CPU_BASELINE_EXPERIMENT.version,
                            context.get("environment_snapshot_id"),
                        ),
                    )
                    prediction_id = int(cursor.lastrowid)
                    prediction_event_id = _insert_event(
                        connection,
                        "PREDICTION_CREATED",
                        "KAI generated a CPU baseline prediction from its stored model.",
                        {
                            "prediction_id": prediction_id,
                            "model_id": model["model_id"],
                            "model_version": model["version"],
                            "expected_seconds": estimate,
                            "uncertainty": before,
                            "confidence": confidence,
                        },
                        source="kai.predictor",
                        execution_id=experiment_id,
                        experiment_definition_key=(
                            f"{CPU_BASELINE_EXPERIMENT.experiment_id}@"
                            f"{CPU_BASELINE_EXPERIMENT.version}"
                        ),
                        experiment_definition_id=CPU_BASELINE_EXPERIMENT.experiment_id,
                        experiment_version=CPU_BASELINE_EXPERIMENT.version,
                        experiment_id=experiment_id,
                        causal_parent_id=selection_event_id,
                        confidence=confidence["empirical_accuracy"],
                        epistemic_status="HYPOTHESIS",
                        occurred_at=now,
                    )
                    connection.execute(
                        "UPDATE predictions SET created_event_id = ? "
                        "WHERE prediction_id = ?",
                        (prediction_event_id, prediction_id),
                    )
                    source_ids = json.loads(
                        connection.execute(
                            "SELECT source_event_ids_json FROM experiences "
                            "WHERE experience_id = ?",
                            (experience_id,),
                        ).fetchone()[0]
                    )
                    source_ids.append(prediction_event_id)
                    connection.execute(
                        "UPDATE experiences SET prediction_id = ?, experiment_id = ?, "
                        "source_event_ids_json = ?, updated_at = ? "
                        "WHERE experience_id = ?",
                        (
                            prediction_id,
                            experiment_id,
                            _encode_json(source_ids),
                            now,
                            experience_id,
                        ),
                    )
                    connection.execute(
                        "UPDATE predictions SET experiment_id = ? "
                        "WHERE prediction_id = ?",
                        (experiment_id, prediction_id),
                    )
                else:
                    connection.execute(
                        "UPDATE experiences SET experiment_id = ?, updated_at = ? "
                        "WHERE experience_id = ?",
                        (experiment_id, now, experience_id),
                    )
                    if prediction_id is not None:
                        connection.execute(
                            "UPDATE predictions SET experiment_id = ?, execution_id = ?, "
                            "experiment_definition_key = ?, experiment_definition_id = ?, "
                            "experiment_version = ?, environment_snapshot_id = ? "
                            "WHERE prediction_id = ?",
                            (
                                experiment_id,
                                experiment_id,
                                f"{CPU_BASELINE_EXPERIMENT.experiment_id}@"
                                f"{CPU_BASELINE_EXPERIMENT.version}",
                                CPU_BASELINE_EXPERIMENT.experiment_id,
                                CPU_BASELINE_EXPERIMENT.version,
                                context.get("environment_snapshot_id"),
                                prediction_id,
                            ),
                        )
                started_event_id = _insert_event(
                    connection,
                    "EXPERIMENT_STARTED",
                    "The bounded CPU baseline experiment started.",
                    {
                        "experiment_id": experiment_id,
                        "prediction_id": prediction_id,
                        "model_version": model["version"] if model else None,
                        "measurement_allowlist": "fixed in-process integer loop and bounded readonly database read",
                    },
                    source="kai.experiment_runner",
                    experiment_id=experiment_id,
                    execution_id=experiment_id,
                    experiment_definition_key=(
                        f"{CPU_BASELINE_EXPERIMENT.experiment_id}@"
                        f"{CPU_BASELINE_EXPERIMENT.version}"
                    ),
                    experiment_definition_id=CPU_BASELINE_EXPERIMENT.experiment_id,
                    experiment_version=CPU_BASELINE_EXPERIMENT.version,
                    environment_snapshot_id=context.get("environment_snapshot_id"),
                    causal_parent_id=selection_event_id,
                    occurred_at=now,
                )
                source_ids = json.loads(
                    connection.execute(
                        "SELECT source_event_ids_json FROM experiences "
                        "WHERE experience_id = ?",
                        (experience_id,),
                    ).fetchone()[0]
                )
                source_ids.extend([selection_event_id, started_event_id])
                connection.execute(
                    "UPDATE experiences SET source_event_ids_json = ? "
                    "WHERE experience_id = ?",
                    (_encode_json(list(dict.fromkeys(source_ids))), experience_id),
                )
                connection.commit()
            except (KaiError, sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                if isinstance(exc, KaiError):
                    raise
                raise KaiError(f"Cannot start CPU baseline experiment: {exc}") from exc
        return {
            "experiment_id": experiment_id,
            "execution_id": experiment_id,
            "experiment_definition_id": CPU_BASELINE_EXPERIMENT.experiment_id,
            "experiment_version": CPU_BASELINE_EXPERIMENT.version,
            "experience_id": experience_id,
            "prediction_id": prediction_id,
            "model": model,
            "prediction": self.get_prediction(prediction_id) if prediction_id else None,
            "uncertainty_before": before,
            "resource_estimate": resources,
            "started_event_id": started_event_id,
        }

    def complete_cpu_baseline_experiment(
        self,
        experiment_id: int,
        measured_seconds: float,
        timestamp: str | None = None,
        experiment_result: ExperimentResult | dict | None = None,
    ) -> dict:
        result_record = validate_experiment_result(
            experiment_result
            if experiment_result is not None
            else ExperimentResult(
                outcome_value=measured_seconds,
                unit="seconds",
                sample_count=1,
                duration_seconds=measured_seconds,
                metadata={},
                measurement_status="MEASURED",
                raw_result={"measured_runtime_seconds": measured_seconds},
            )
        )
        if (
            result_record.measurement_status != "MEASURED"
            or result_record.outcome_value != measured_seconds
            or result_record.unit != "seconds"
        ):
            raise KaiError("CPU baseline completion requires a measured seconds result.")
        try:
            initial_parameters(measured_seconds)
        except ValueError as exc:
            raise KaiError(str(exc)) from exc
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                experiment = connection.execute(
                    "SELECT * FROM experiment_executions WHERE execution_id = ?",
                    (experiment_id,),
                ).fetchone()
                if experiment is None:
                    raise KaiError(f"Experiment {experiment_id} does not exist.")
                if experiment["status"] != "RUNNING":
                    raise KaiError(f"Experiment {experiment_id} is not running.")
                experience = connection.execute(
                    "SELECT * FROM experiences WHERE experience_id = ?",
                    (experiment["experience_id"],),
                ).fetchone()
                context = json.loads(experience["context_json"])
                key = prediction_context_key(context)
                model_row = connection.execute(
                    "SELECT * FROM predictive_models WHERE domain = ? "
                    "AND context_key = ? AND status = 'ACTIVE' "
                    "ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                    (CPU_BASELINE_DOMAIN, key),
                ).fetchone()
                model = self._predictive_model_row(model_row) if model_row else None
                prediction = None
                if experience["prediction_id"] is not None:
                    prediction = connection.execute(
                        "SELECT * FROM predictions WHERE prediction_id = ?",
                        (experience["prediction_id"],),
                    ).fetchone()
                parameters_before = model["parameters"] if model else {}
                version_before = model["version"] if model else 0
                prediction_error = None
                prediction_status = None
                eval_event_id = None
                if prediction is not None:
                    if prediction["status"] != "PENDING":
                        raise KaiError("The experiment's prediction is no longer pending.")
                    if model is None or prediction["model_id"] != model["model_id"]:
                        raise KaiError("The experiment prediction model is no longer active.")
                    if prediction["model_version"] != model["version"]:
                        raise KaiError("The experiment prediction uses a stale model version.")
                    expected = json.loads(prediction["expected_outcome_json"])
                    signed_error = measured_seconds - expected
                    magnitude = abs(signed_error)
                    tolerance = max(
                        CPU_BASELINE_TOLERANCE_SECONDS,
                        abs(expected) * 0.05,
                    )
                    prediction_status = (
                        "CONFIRMED"
                        if magnitude == 0
                        else "PARTIALLY_CONFIRMED"
                        if magnitude <= tolerance
                        else "REFUTED"
                    )
                    prediction_error = {
                        "expected_seconds": expected,
                        "observed_seconds": measured_seconds,
                        "signed_error_seconds": signed_error,
                        "absolute_error_seconds": magnitude,
                        "relative_error": (
                            signed_error / abs(expected) if expected != 0 else None
                        ),
                        "tolerance_seconds": tolerance,
                        "sign_convention": "observed minus expected",
                    }
                    eval_event_id = _insert_event(
                        connection,
                        "PREDICTION_EVALUATED",
                        "KAI compared its CPU baseline prediction with the measured runtime.",
                        {
                            "prediction_id": prediction["prediction_id"],
                            "status": prediction_status,
                            **prediction_error,
                        },
                        source="kai.predictor",
                        experiment_id=experiment_id,
                        execution_id=experiment_id,
                        experiment_definition_key=experiment["experiment_definition_key"],
                        experiment_definition_id=experiment["experiment_definition_id"],
                        experiment_version=experiment["experiment_version"],
                        causal_parent_id=prediction["created_event_id"],
                        confidence=prediction["confidence"],
                        epistemic_status="OBSERVATION",
                        occurred_at=now,
                    )
                    connection.execute(
                        "UPDATE predictions SET status = ?, evaluated_at = ?, "
                        "observed_outcome_json = ?, difference_json = ?, evaluation = ?, "
                        "evaluated_event_id = ? WHERE prediction_id = ?",
                        (
                            prediction_status,
                            now,
                            _encode_json(measured_seconds),
                            _encode_json(
                                {
                                    "difference": signed_error,
                                    "error_magnitude": magnitude,
                                    "relative_error": (
                                        signed_error / abs(expected)
                                        if expected != 0
                                        else None
                                    ),
                                    "tolerance": tolerance,
                                    "sign_convention": "observed minus expected",
                                }
                            ),
                            f"CPU baseline observed-minus-expected error: {signed_error:.9f} seconds.",
                            eval_event_id,
                            prediction["prediction_id"],
                        ),
                    )
                parameters_after = (
                    update_running_mean(parameters_before, measured_seconds)
                    if model
                    else initial_parameters(measured_seconds)
                )
                model_id = model["model_id"] if model else str(uuid.uuid4())
                version_after = version_before + 1
                update_reason = "INITIAL_SEED" if model is None else "OBSERVATION_UPDATE"
                training_refs = (
                    model["training_evidence_references"] if model else []
                ) + [
                    f"experience:{experiment['experience_id']}",
                    f"execution:{experiment_id}",
                ]
                if prediction is not None:
                    training_refs.append(f"prediction:{prediction['prediction_id']}")
                scored_rows = []
                model_prediction_rows = connection.execute(
                    "SELECT status, confidence, difference_json FROM predictions "
                    "WHERE model_id = ? ORDER BY prediction_id",
                    (model_id,),
                ).fetchall()
                for row in model_prediction_rows:
                    scored_rows.append(
                        {
                            "status": row["status"],
                            "confidence": row["confidence"],
                            "difference": _decode_json(row["difference_json"]),
                        }
                    )
                performance = prediction_statistics(scored_rows)
                confidence = confidence_from_statistics(performance)
                if model is None:
                    connection.execute(
                        "INSERT INTO predictive_models "
                        "(model_id, model_type, domain, context_key, version, status, "
                        "created_at, updated_at, parameters_json, confidence_json, "
                        "performance_json, evidence_refs_json) "
                        "VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?, ?, ?)",
                        (
                            model_id,
                            CPU_BASELINE_MODEL_TYPE,
                            CPU_BASELINE_DOMAIN,
                            key,
                            version_after,
                            now,
                            now,
                            _encode_json(parameters_after),
                            _encode_json(confidence),
                            _encode_json(performance),
                            _encode_json(training_refs),
                        ),
                    )
                else:
                    connection.execute(
                        "UPDATE predictive_models SET version = ?, updated_at = ?, "
                        "parameters_json = ?, confidence_json = ?, performance_json = ?, "
                        "evidence_refs_json = ? WHERE model_id = ? AND version = ? "
                        "AND status = 'ACTIVE'",
                        (
                            version_after,
                            now,
                            _encode_json(parameters_after),
                            _encode_json(confidence),
                            _encode_json(performance),
                            _encode_json(training_refs),
                            model_id,
                            version_before,
                        ),
                    )
                    if connection.execute("SELECT changes()").fetchone()[0] != 1:
                        raise KaiError("Predictive model changed concurrently; update cancelled.")
                after_uncertainty = uncertainty(parameters_after)
                before_uncertainty = json.loads(experiment["uncertainty_before_json"])
                if (
                    before_uncertainty.get("standard_error_seconds") is not None
                    and after_uncertainty.get("standard_error_seconds") is not None
                ):
                    information_gain = {
                        "status": "ESTIMATED",
                        "method": "previous standard error minus updated standard error",
                        "uncertainty_reduction_seconds": (
                            before_uncertainty["standard_error_seconds"]
                            - after_uncertainty["standard_error_seconds"]
                        ),
                        "not_a_bayesian_information_gain": True,
                    }
                else:
                    information_gain = {
                        "status": "NOT QUANTIFIABLE",
                        "reason": "Insufficient repeated measurements to estimate uncertainty.",
                    }
                evaluation_status = (
                    "EVALUATED" if prediction is not None else "NOT_EVALUATED"
                )
                evidence_quality, evidence = classify_evidence_quality(
                    sample_count=parameters_after["observation_count"],
                    context_consistency=True,
                    measurement_status=result_record.measurement_status,
                    prediction_available=prediction is not None,
                    evaluation_available=prediction is not None,
                )
                evidence.update(
                    {
                        "execution_id": experiment_id,
                        "experiment_definition_id": experiment["experiment_definition_id"],
                        "experiment_version": experiment["experiment_version"],
                        "environment_profile": experiment["context_profile"],
                        "environment_snapshot_id": experiment["environment_snapshot_id"],
                        "model_id": model_id,
                        "model_version_before": version_before or None,
                        "model_version_after": version_after,
                        "prediction_id": (
                            prediction["prediction_id"] if prediction else None
                        ),
                        "prediction_error": prediction_error,
                    }
                )
                update_event_id = _insert_event(
                    connection,
                    "PREDICTIVE_MODEL_CREATED" if model is None else "PREDICTIVE_MODEL_UPDATED",
                    (
                        "Initialized KAI's CPU baseline running-mean predictor."
                        if model is None
                        else "Updated KAI's CPU baseline running-mean predictor."
                    ),
                    {
                        "model_id": model_id,
                        "domain": CPU_BASELINE_DOMAIN,
                        "update_reason": update_reason,
                        "experiment_definition_id": experiment["experiment_definition_id"],
                        "experiment_version": experiment["experiment_version"],
                        "execution_id": experiment_id,
                        "version_before": version_before,
                        "version_after": version_after,
                        "update_rule": (
                            "initialize running mean from first observation"
                            if model is None
                            else CPU_BASELINE_UPDATE_RULE
                        ),
                        "parameters_before": parameters_before,
                        "parameters_after": parameters_after,
                        "prediction_error": prediction_error,
                        "information_gain": information_gain,
                    },
                    source="kai.predictor",
                    experiment_id=experiment_id,
                    execution_id=experiment_id,
                    experiment_definition_key=experiment["experiment_definition_key"],
                    experiment_definition_id=experiment["experiment_definition_id"],
                    experiment_version=experiment["experiment_version"],
                    causal_parent_id=eval_event_id,
                    epistemic_status="INFERENCE",
                    occurred_at=now,
                )
                connection.execute(
                    "INSERT INTO predictive_model_versions "
                    "(model_id, version, created_at, parameters_json, confidence_json, "
                    "performance_json, evidence_refs_json, status, source_execution_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)",
                    (
                        model_id,
                        version_after,
                        now,
                        _encode_json(parameters_after),
                        _encode_json(confidence),
                        _encode_json(performance),
                        _encode_json(training_refs),
                        experiment_id,
                    ),
                )
                cursor = connection.execute(
                    "INSERT INTO predictive_model_updates "
                    "(model_id, experience_id, prediction_id, experiment_id, created_at, "
                    "version_before, version_after, update_rule, parameters_before_json, "
                    "parameters_after_json, input_evidence_json, prediction_error_json, "
                    "execution_id, experiment_definition_key, experiment_definition_id, "
                    "experiment_version, update_reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        model_id,
                        experiment["experience_id"],
                        prediction["prediction_id"] if prediction else None,
                        experiment_id,
                        now,
                        version_before,
                        version_after,
                        "initialize running mean from first observation"
                        if model is None
                        else CPU_BASELINE_UPDATE_RULE,
                        _encode_json(parameters_before),
                        _encode_json(parameters_after),
                        _encode_json(
                            {
                                "measured_runtime_seconds": measured_seconds,
                                "execution_id": experiment_id,
                                "experiment_definition_id": experiment["experiment_definition_id"],
                                "experiment_version": experiment["experiment_version"],
                            }
                        ),
                        _json_optional(prediction_error),
                        experiment_id,
                        experiment["experiment_definition_key"],
                        experiment["experiment_definition_id"],
                        experiment["experiment_version"],
                        update_reason,
                    ),
                )
                update_id = int(cursor.lastrowid)
                connection.execute(
                    "UPDATE predictive_model_updates SET update_event_id = ? "
                    "WHERE update_id = ?",
                    (update_event_id, update_id),
                )
                learning_id = None
                if prediction is not None:
                    learning_event_id = _insert_event(
                        connection,
                        "LEARNING_OCCURRED",
                        "Prediction error updated a versioned CPU baseline model.",
                        {
                            "model_id": model_id,
                            "prediction_id": prediction["prediction_id"],
                            "update_id": update_id,
                            "version_before": version_before,
                            "version_after": version_after,
                        },
                        source="kai.learning",
                        experiment_id=experiment_id,
                        execution_id=experiment_id,
                        experiment_definition_key=experiment["experiment_definition_key"],
                        experiment_definition_id=experiment["experiment_definition_id"],
                        experiment_version=experiment["experiment_version"],
                        causal_parent_id=update_event_id,
                        confidence=prediction["confidence"],
                        epistemic_status="INFERENCE",
                        occurred_at=now,
                    )
                    cursor = connection.execute(
                        "INSERT INTO learning_records "
                        "(experience_id, created_at, before_json, happened_json, "
                        "predicted_json, error_json, changed_json, why, evidence_json, "
                        "confidence, source_experiences_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            experiment["experience_id"],
                            now,
                            experience["belief_json"] or "{}",
                            _encode_json(measured_seconds),
                            prediction["expected_outcome_json"],
                            _encode_json(prediction_error),
                            _encode_json(
                                {
                                    "model_version_before": version_before,
                                    "model_version_after": version_after,
                                    "parameters_after": parameters_after,
                                }
                            ),
                            "Observed CPU baseline error updated the running-mean model.",
                            _encode_json(
                                [prediction["evaluated_event_id"], learning_event_id]
                            ),
                            prediction["confidence"],
                            _encode_json([experiment["experience_id"]]),
                        ),
                    )
                    learning_id = int(cursor.lastrowid)
                completed_event_id = _insert_event(
                    connection,
                    "EXPERIMENT_COMPLETED",
                    "The bounded CPU baseline experiment completed.",
                    {
                        "experiment_id": experiment_id,
                        "measured_runtime_seconds": measured_seconds,
                        "prediction_status": prediction_status,
                        "model_id": model_id,
                        "model_version": version_after,
                    },
                    source="kai.experiment_runner",
                    experiment_id=experiment_id,
                    execution_id=experiment_id,
                    experiment_definition_key=experiment["experiment_definition_key"],
                    experiment_definition_id=experiment["experiment_definition_id"],
                    experiment_version=experiment["experiment_version"],
                    causal_parent_id=update_event_id,
                    epistemic_status="OBSERVATION",
                    occurred_at=now,
                )
                connection.execute(
                    "UPDATE experiment_executions SET status = 'COMPLETED', "
                    "execution_status = 'SUCCEEDED', raw_result_json = ?, "
                    "failure_information = NULL, update_reason = ?, "
                    "uncertainty_after_json = ?, information_gain_json = ?, "
                    "completed_at = ?, output_json = ? WHERE execution_id = ?",
                    (
                        _encode_json(result_record.raw_result),
                        update_reason,
                        _encode_json(after_uncertainty),
                        _encode_json(information_gain),
                        now,
                        _encode_json(
                            {
                                "measured_runtime_seconds": measured_seconds,
                                "model_id": model_id,
                                "model_version": version_after,
                                "prediction_status": prediction_status,
                            }
                        ),
                        experiment_id,
                    ),
                )
                connection.execute(
                    "INSERT INTO experiment_results "
                    "(execution_id, outcome_value, unit, sample_count, duration_seconds, "
                    "metadata_json, measurement_status) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        experiment_id,
                        result_record.outcome_value,
                        result_record.unit,
                        result_record.sample_count,
                        result_record.duration_seconds,
                        _encode_json(result_record.metadata),
                        result_record.measurement_status,
                    ),
                )
                connection.execute(
                    "INSERT INTO experiment_evaluations "
                    "(execution_id, prediction_id, expected_value, observed_value, error, "
                    "absolute_error, relative_error, evaluation_status, evidence_quality, "
                    "evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        experiment_id,
                        prediction["prediction_id"] if prediction else None,
                        prediction_error["expected_seconds"] if prediction_error else None,
                        measured_seconds,
                        prediction_error["signed_error_seconds"] if prediction_error else None,
                        prediction_error["absolute_error_seconds"] if prediction_error else None,
                        prediction_error["relative_error"] if prediction_error else None,
                        evaluation_status,
                        evidence_quality,
                        _encode_json(evidence),
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE experiences SET status = 'COMPLETED', "
                    "observation_json = ?, outcome_json = ?, evaluation_json = ?, "
                    "learning_json = ?, source_event_ids_json = ?, updated_at = ? "
                    "WHERE experience_id = ?",
                    (
                        _encode_json(
                            {
                                "task": "CPU_BASELINE",
                                "runtime_seconds": measured_seconds,
                            }
                        ),
                        _encode_json({"runtime_seconds": measured_seconds}),
                        _encode_json(
                            {
                                "prediction_status": prediction_status,
                                "prediction_error": prediction_error,
                                "evaluation_status": evaluation_status,
                                "evidence_quality": evidence_quality,
                            }
                        ),
                        _json_optional({"learning_id": learning_id, "update_id": update_id}),
                        _encode_json(
                            json.loads(experience["source_event_ids_json"])
                            + [update_event_id, completed_event_id]
                        ),
                        now,
                        experiment["experience_id"],
                    ),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot complete CPU baseline experiment: {exc}") from exc
        model_after = self.get_cpu_baseline_model(context)
        next_forecast = self.create_cpu_baseline_forecast(context)
        return {
            "experiment_id": experiment_id,
            "execution_id": experiment_id,
            "experiment_definition_id": experiment["experiment_definition_id"],
            "experiment_version": experiment["experiment_version"],
            "experience_id": experiment["experience_id"],
            "prediction_id": prediction["prediction_id"] if prediction else None,
            "prediction_status": prediction_status,
            "prediction_error": prediction_error,
            "evaluation_status": evaluation_status,
            "evidence_quality": evidence_quality,
            "evidence": evidence,
            "measured_runtime_seconds": measured_seconds,
            "model_update": {
                "update_id": update_id,
                "model_id": model_id,
                "version_before": version_before,
                "version_after": version_after,
                "update_rule": (
                    "initialize running mean from first observation"
                    if model is None
                    else CPU_BASELINE_UPDATE_RULE
                ),
                "parameters_before": parameters_before,
                "parameters_after": parameters_after,
            },
            "model": model_after,
            "uncertainty_before": before_uncertainty,
            "uncertainty_after": after_uncertainty,
            "information_gain": information_gain,
            "learning_id": learning_id,
            "next_prediction": next_forecast,
        }

    def create_cpu_baseline_forecast(self, context: dict) -> dict | None:
        model = self.get_cpu_baseline_model(context)
        if model is None:
            return None
        now = utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT prediction_id FROM predictions WHERE model_id = ? "
                    "AND model_version = ? AND status = 'PENDING' "
                    "ORDER BY prediction_id DESC LIMIT 1",
                    (model["model_id"], model["version"]),
                ).fetchone()
                if existing:
                    connection.commit()
                    return self.get_prediction(existing["prediction_id"])
                cursor = connection.execute(
                    "INSERT INTO experiences "
                    "(created_at, updated_at, status, context_json, source_event_ids_json) "
                    "VALUES (?, ?, 'OPEN', ?, '[]')",
                    (now, now, _encode_json(context)),
                )
                experience_id = int(cursor.lastrowid)
                estimate = model["parameters"]["mean_seconds"]
                confidence = confidence_from_statistics(
                    self.prediction_statistics(model["model_id"])
                )
                cursor = connection.execute(
                    "INSERT INTO predictions "
                    "(experience_id, created_at, context_json, hypothesis, "
                    "expected_outcome_json, confidence, basis, observation_window, status, "
                    "model_id, model_version, experiment_definition_key, "
                    "experiment_definition_id, experiment_version, environment_snapshot_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?)",
                    (
                        experience_id,
                        now,
                        _encode_json(context),
                        f"CPU baseline runtime will be near {estimate:.6f} seconds.",
                        _encode_json(estimate),
                        confidence["empirical_accuracy"],
                        f"{CPU_BASELINE_MODEL_TYPE} model version {model['version']}",
                        "the next explicit CPU baseline experiment",
                        model["model_id"],
                        model["version"],
                        f"{CPU_BASELINE_EXPERIMENT.experiment_id}@"
                        f"{CPU_BASELINE_EXPERIMENT.version}",
                        CPU_BASELINE_EXPERIMENT.experiment_id,
                        CPU_BASELINE_EXPERIMENT.version,
                        context.get("environment_snapshot_id"),
                    ),
                )
                prediction_id = int(cursor.lastrowid)
                event_id = _insert_event(
                    connection,
                    "PREDICTION_CREATED",
                    "KAI generated its next CPU baseline prediction from the updated model.",
                    {
                        "prediction_id": prediction_id,
                        "model_id": model["model_id"],
                        "model_version": model["version"],
                        "expected_seconds": estimate,
                        "uncertainty": uncertainty(model["parameters"]),
                        "confidence": confidence,
                    },
                    source="kai.predictor",
                    experiment_definition_key=(
                        f"{CPU_BASELINE_EXPERIMENT.experiment_id}@"
                        f"{CPU_BASELINE_EXPERIMENT.version}"
                    ),
                    experiment_definition_id=CPU_BASELINE_EXPERIMENT.experiment_id,
                    experiment_version=CPU_BASELINE_EXPERIMENT.version,
                    environment_snapshot_id=context.get("environment_snapshot_id"),
                    causal_parent_id=connection.execute(
                        "SELECT update_event_id FROM predictive_model_updates "
                        "WHERE model_id = ? ORDER BY update_id DESC LIMIT 1",
                        (model["model_id"],),
                    ).fetchone()[0],
                    confidence=confidence["empirical_accuracy"],
                    epistemic_status="HYPOTHESIS",
                    occurred_at=now,
                )
                connection.execute(
                    "UPDATE predictions SET created_event_id = ? WHERE prediction_id = ?",
                    (event_id, prediction_id),
                )
                connection.execute(
                    "UPDATE experiences SET prediction_id = ?, belief_json = ?, "
                    "source_event_ids_json = ? WHERE experience_id = ?",
                    (
                        prediction_id,
                        _encode_json(
                            {
                                "model_id": model["model_id"],
                                "model_version": model["version"],
                                "expected_seconds": estimate,
                            }
                        ),
                        _encode_json([event_id]),
                        experience_id,
                    ),
                )
                connection.commit()
            except (KaiError, sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                if isinstance(exc, KaiError):
                    raise
                raise KaiError(f"Cannot create next CPU baseline forecast: {exc}") from exc
        return self.get_prediction(prediction_id)

    def fail_cpu_baseline_experiment(
        self,
        experiment_id: int,
        error: str,
        *,
        execution_status: str = "FAILED",
        measurement_status: str = "UNKNOWN",
        evaluation_status: str | None = None,
        raw_result: dict | None = None,
    ) -> None:
        if not isinstance(error, str) or not error.strip():
            raise KaiError("Experiment failure requires an error description.")
        if execution_status not in {"FAILED", "UNKNOWN", "ABORTED"}:
            raise KaiError("Unsupported failed-experiment execution status.")
        if measurement_status not in {"UNKNOWN", "INVALID"}:
            raise KaiError("Failed experiment measurement must be UNKNOWN or INVALID.")
        now = utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM experiment_executions WHERE execution_id = ?",
                    (experiment_id,),
                ).fetchone()
                if row is None:
                    raise KaiError(f"Experiment {experiment_id} does not exist.")
                if row["status"] != "RUNNING":
                    raise KaiError(f"Experiment {experiment_id} is not running.")
                event_id = _insert_event(
                    connection,
                    "EXPERIMENT_FAILED",
                    "The allowlisted CPU baseline experiment failed.",
                    {"experiment_id": experiment_id, "error": error},
                    source="kai.experiment_runner",
                    experiment_id=experiment_id,
                    execution_id=experiment_id,
                    experiment_definition_key=row["experiment_definition_key"],
                    experiment_definition_id=row["experiment_definition_id"],
                    experiment_version=row["experiment_version"],
                    epistemic_status="OBSERVATION",
                )
                connection.execute(
                    "UPDATE experiment_executions SET status = 'FAILED', completed_at = ?, "
                    "execution_status = ?, failure_information = ?, raw_result_json = ?, "
                    "output_json = ? WHERE execution_id = ?",
                    (
                        now,
                        execution_status,
                        error,
                        _json_optional(raw_result),
                        _encode_json({"error": error}),
                        experiment_id,
                    ),
                )
                prediction_id = connection.execute(
                    "SELECT prediction_id FROM experiences WHERE experience_id = ?",
                    (row["experience_id"],),
                ).fetchone()["prediction_id"]
                connection.execute(
                    "UPDATE predictions SET status = 'UNKNOWN', evaluated_at = ?, "
                    "evaluation = 'Experiment failed before an outcome was obtained.', "
                    "evaluated_event_id = ? WHERE execution_id = ? AND status = 'PENDING'",
                    (now, event_id, experiment_id),
                )
                if evaluation_status is None:
                    evaluation_status = (
                        "INVALID"
                        if measurement_status == "INVALID"
                        else "UNKNOWN"
                        if prediction_id is not None
                        else "NOT_EVALUATED"
                    )
                quality, evidence = classify_evidence_quality(
                    sample_count=0,
                    context_consistency=False,
                    measurement_status=measurement_status,
                    prediction_available=prediction_id is not None,
                    evaluation_available=False,
                )
                evidence.update(
                    {
                        "execution_id": experiment_id,
                        "failure_information": error,
                    }
                )
                connection.execute(
                    "INSERT INTO experiment_results "
                    "(execution_id, outcome_value, unit, sample_count, duration_seconds, "
                    "metadata_json, measurement_status) VALUES (?, NULL, 'seconds', 0, NULL, ?, ?)",
                    (experiment_id, _encode_json({"failure": error}), measurement_status),
                )
                connection.execute(
                    "INSERT INTO experiment_evaluations "
                    "(execution_id, prediction_id, evaluation_status, evidence_quality, "
                    "evidence_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        experiment_id,
                        prediction_id,
                        evaluation_status,
                        quality,
                        _encode_json(evidence),
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE experiences SET status = 'COMPLETED', "
                    "evaluation_json = ?, updated_at = ? WHERE experience_id = ?",
                    (
                        _encode_json(
                            {
                                "experiment_status": "FAILED",
                                "execution_status": execution_status,
                                "evaluation_status": evaluation_status,
                            }
                        ),
                        now,
                        row["experience_id"],
                    ),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot record CPU experiment failure: {exc}") from exc

    def rollback_predictive_model_update(self, update_id: int) -> dict:
        now = utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                update = connection.execute(
                    "SELECT * FROM predictive_model_updates WHERE update_id = ?",
                    (update_id,),
                ).fetchone()
                if update is None:
                    raise KaiError(f"Model update {update_id} does not exist.")
                model = connection.execute(
                    "SELECT * FROM predictive_models WHERE model_id = ?",
                    (update["model_id"],),
                ).fetchone()
                if (
                    model is None
                    or model["version"] != update["version_after"]
                    or model["status"] != "ACTIVE"
                ):
                    raise KaiError(
                        "Only the latest active model update can be rolled back safely."
                    )
                before = json.loads(update["parameters_before_json"])
                if not before:
                    raise KaiError(
                        "The initial seed update has no prior model state to restore."
                    )
                version_before = model["version"]
                version_after = version_before + 1
                stats = prediction_statistics(
                    [
                        {
                            "status": row["status"],
                            "confidence": row["confidence"],
                            "difference": _decode_json(row["difference_json"]),
                        }
                        for row in connection.execute(
                            "SELECT status, confidence, difference_json FROM predictions "
                            "WHERE model_id = ?",
                            (update["model_id"],),
                        ).fetchall()
                    ]
                )
                confidence = confidence_from_statistics(stats)
                refs = json.loads(model["evidence_refs_json"]) + [
                    f"rollback:{update_id}"
                ]
                connection.execute(
                    "UPDATE predictive_models SET version = ?, updated_at = ?, "
                    "parameters_json = ?, confidence_json = ?, performance_json = ?, "
                    "evidence_refs_json = ? WHERE model_id = ? AND version = ?",
                    (
                        version_after,
                        now,
                        _encode_json(before),
                        _encode_json(confidence),
                        _encode_json(stats),
                        _encode_json(refs),
                        update["model_id"],
                        version_before,
                    ),
                )
                connection.execute(
                    "INSERT INTO predictive_model_versions "
                    "(model_id, version, created_at, parameters_json, confidence_json, "
                    "performance_json, evidence_refs_json, status, source_execution_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE', ?)",
                    (
                        update["model_id"],
                        version_after,
                        now,
                        _encode_json(before),
                        _encode_json(confidence),
                        _encode_json(stats),
                        _encode_json(refs),
                        update["execution_id"],
                    ),
                )
                cursor = connection.execute(
                    "INSERT INTO predictive_model_updates "
                    "(model_id, experience_id, prediction_id, experiment_id, created_at, "
                    "version_before, version_after, update_rule, parameters_before_json, "
                    "parameters_after_json, input_evidence_json, prediction_error_json, "
                    "reverses_update_id, execution_id, experiment_definition_key, "
                    "experiment_definition_id, experiment_version, update_reason) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        update["model_id"],
                        update["experience_id"],
                        update["prediction_id"],
                        update["experiment_id"],
                        now,
                        version_before,
                        version_after,
                        f"rollback: restore parameters from update {update_id}",
                        model["parameters_json"],
                        _encode_json(before),
                        _encode_json({"rollback_of_update_id": update_id}),
                        update["prediction_error_json"],
                        update_id,
                        update["execution_id"],
                        update["experiment_definition_key"],
                        update["experiment_definition_id"],
                        update["experiment_version"],
                        "ROLLBACK",
                    ),
                )
                rollback_event_id = _insert_event(
                    connection,
                    "PREDICTIVE_MODEL_UPDATED",
                    "A model update was explicitly reverted in a new version.",
                    {
                        "model_id": update["model_id"],
                        "reverses_update_id": update_id,
                        "version_before": version_before,
                        "version_after": version_after,
                    },
                    source="kai.predictor.rollback",
                    experiment_id=update["experiment_id"],
                    execution_id=update["execution_id"],
                    experiment_definition_key=update["experiment_definition_key"],
                    experiment_definition_id=update["experiment_definition_id"],
                    experiment_version=update["experiment_version"],
                    causal_parent_id=None,
                    epistemic_status="INFERENCE",
                )
                connection.execute(
                    "UPDATE predictive_model_updates SET update_event_id = ? "
                    "WHERE update_id = ?",
                    (rollback_event_id, cursor.lastrowid),
                )
                connection.execute(
                    "UPDATE predictions SET status = 'EXPIRED', evaluated_at = ?, "
                    "evaluation = 'Forecast superseded by an explicit model rollback.', "
                    "evaluated_event_id = ? WHERE model_id = ? AND status = 'PENDING' "
                    "AND model_version < ?",
                    (now, rollback_event_id, update["model_id"], version_after),
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot roll back predictive model update: {exc}") from exc
        return {
            "model_id": update["model_id"],
            "update_id": int(cursor.lastrowid),
            "reverses_update_id": update_id,
            "rollback_event_id": rollback_event_id,
            "version_before": version_before,
            "version_after": version_after,
            "parameters": before,
        }

    def add_memory(
        self, kind: str, content: str, source: str, confidence: float | None
    ) -> int:
        if kind not in {
            "fact", "observation", "inference", "hypothesis", "speculation", "unknown"
        }:
            raise KaiError(f"Invalid memory kind: {kind}")
        if not content.strip():
            raise KaiError("Memory content cannot be empty.")
        if not source.strip():
            raise KaiError("Memory source cannot be empty.")
        _validate_confidence(confidence, "Memory")
        with self.connection() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO memories (created_at, kind, content, source, confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (utc_now(), kind, content, source, confidence),
                )
                memory_id = int(cursor.lastrowid)
                epistemic_status = {
                    "fact": "HYPOTHESIS",
                    "observation": "OBSERVATION",
                    "inference": "INFERENCE",
                    "hypothesis": "HYPOTHESIS",
                    "speculation": "HYPOTHESIS",
                    "unknown": "UNKNOWN",
                }[kind]
                connection.execute(
                    "INSERT INTO memory_items "
                    "(layer, created_at, content, source, epistemic_status, "
                    "verification_status, confidence, evidence_json, "
                    "source_experiences_json) VALUES "
                    "('semantic', ?, ?, ?, ?, ?, ?, '[]', '[]')",
                    (
                        utc_now(),
                        content,
                        source,
                        epistemic_status,
                        "candidate" if kind == "fact" else "unverified",
                        confidence,
                    ),
                )
                _insert_event(
                    connection,
                    "CLAIM_CREATED",
                    "A labeled memory was stored.",
                    {"memory_id": memory_id, "kind": kind},
                    source=source,
                    confidence=confidence,
                    epistemic_status=epistemic_status,
                )
                connection.commit()
                return memory_id
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot store memory: {exc}") from exc

    def list_memories(self, query: str | None = None, limit: int = 20) -> list[dict]:
        _validate_limit(limit, "Memory")
        with self.connection() as connection:
            try:
                if query:
                    rows = connection.execute(
                        "SELECT memory_id, created_at, kind, content, source, confidence "
                        "FROM memories WHERE instr(lower(content), lower(?)) > 0 "
                        "ORDER BY memory_id DESC LIMIT ?",
                        (query, limit),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT memory_id, created_at, kind, content, source, confidence "
                        "FROM memories ORDER BY memory_id DESC LIMIT ?",
                        (limit,),
                    ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read memory: {exc}") from exc
        return [dict(row) for row in rows]

    def add_goal(
        self, title: str, priority: int, success_criteria: str
    ) -> int:
        if not title.strip():
            raise KaiError("Goal title cannot be empty.")
        if priority < 0:
            raise KaiError("Goal priority must be zero or greater.")
        with self.connection() as connection:
            try:
                cursor = connection.execute(
                    "INSERT INTO goals (created_at, title, priority, success_criteria) "
                    "VALUES (?, ?, ?, ?)",
                    (utc_now(), title, priority, success_criteria),
                )
                goal_id = int(cursor.lastrowid)
                _insert_event(
                    connection,
                    "GOAL_CREATED",
                    "A goal was stored.",
                    {"goal_id": goal_id, "priority": priority},
                    source="kai.cli",
                    epistemic_status="OBSERVATION",
                )
                connection.commit()
                return goal_id
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot store goal: {exc}") from exc

    def list_goals(self) -> list[dict]:
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT goal_id, created_at, title, priority, status, success_criteria "
                    "FROM goals ORDER BY priority DESC, goal_id"
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read goals: {exc}") from exc
        return [dict(row) for row in rows]

    def save_environment_snapshot(
        self,
        captured_at: str,
        fingerprint: str,
        snapshot: dict,
        claims: list[dict],
        measurements: dict | None = None,
        changes: dict | None = None,
        retention: int = 100,
    ) -> dict:
        if not fingerprint:
            raise KaiError("Environment fingerprint cannot be empty.")
        if retention < 2:
            raise KaiError("Snapshot retention must be at least 2.")
        captured_at = _utc_timestamp(captured_at)
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                previous = connection.execute(
                    "SELECT snapshot_id, captured_at, environment_fingerprint, "
                    "snapshot_json, measurements_json FROM environment_snapshots "
                    "ORDER BY snapshot_id DESC LIMIT 1"
                ).fetchone()
                cursor = connection.execute(
                    "INSERT INTO environment_snapshots "
                    "(captured_at, environment_fingerprint, snapshot_json, measurements_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        captured_at,
                        fingerprint,
                        json.dumps(snapshot, sort_keys=True),
                        json.dumps(measurements or {}, sort_keys=True),
                    ),
                )
                snapshot_id = int(cursor.lastrowid)
                for item in claims:
                    self._validate_claim(item)
                    connection.execute(
                        "INSERT INTO epistemic_claims "
                        "(snapshot_id, claim, classification, source, observed_at, "
                        "confidence, evidence_json, verification_status, "
                        "contradictory_evidence_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            snapshot_id,
                            item["claim"],
                            item["classification"],
                            item["source"],
                            item["timestamp"],
                            item["confidence"],
                            json.dumps(item["evidence"], sort_keys=True),
                            item["verification_status"],
                            json.dumps(item.get("contradictory_evidence", []), sort_keys=True),
                        ),
                    )
                observed_event_id = _insert_event(
                    connection,
                    "ENVIRONMENT_OBSERVED",
                    "A structured environment snapshot was captured.",
                    {
                        "environment_fingerprint": fingerprint,
                        "environment_summary": {
                            "os": snapshot.get("operating_system", {}).get("distribution"),
                            "architecture": snapshot.get("operating_system", {}).get(
                                "architecture"
                            ),
                            "gpu_pci_status": snapshot.get("gpu", {})
                            .get("pci_enumeration", {})
                            .get("status"),
                        },
                    },
                    source="kai.environment_model",
                    environment_snapshot_id=snapshot_id,
                    confidence=0.95,
                    occurred_at=captured_at,
                )
                claims_event_id = _insert_event(
                    connection,
                    "CLAIM_CREATED",
                    "Epistemic claims were generated from the environment snapshot.",
                    {
                        "snapshot_id": snapshot_id,
                        "claim_count": len(claims),
                        "classifications": {
                            classification: sum(
                                item["classification"] == classification for item in claims
                            )
                            for classification in (
                                "OBSERVATION",
                                "FACT",
                                "INFERENCE",
                                "HYPOTHESIS",
                                "UNKNOWN",
                            )
                        },
                    },
                    source="kai.environment_model",
                    environment_snapshot_id=snapshot_id,
                    causal_parent_id=observed_event_id,
                    confidence=0.95,
                    epistemic_status="INFERENCE",
                    occurred_at=captured_at,
                )
                experience_cursor = connection.execute(
                    "INSERT INTO experiences "
                    "(created_at, updated_at, status, context_json, observation_json, "
                    "causal_event_id, source_event_ids_json) "
                    "VALUES (?, ?, 'COMPLETED', ?, ?, ?, ?)",
                    (
                        captured_at,
                        captured_at,
                        json.dumps(
                            {
                                "environment_fingerprint": fingerprint,
                                "environment_snapshot_id": snapshot_id,
                            },
                            sort_keys=True,
                        ),
                        json.dumps(
                            {
                                "snapshot_captured": True,
                                "changes": changes or {},
                            },
                            sort_keys=True,
                        ),
                        observed_event_id,
                        json.dumps([observed_event_id, claims_event_id]),
                    ),
                )
                experience_id = int(experience_cursor.lastrowid)
                change_event_id = None
                if changes and any(
                    changes.get(key)
                    for key in (
                        "added_capabilities",
                        "removed_capabilities",
                        "changed_hardware",
                        "changed_software",
                        "changed_resources",
                        "changed_drivers",
                        "changed_runtime_availability",
                        "changed_kernel_visibility",
                        "other_changes",
                    )
                ):
                    change_event_id = _insert_event(
                        connection,
                        "ENVIRONMENT_CHANGED",
                        "The current environment differs from its previous snapshot.",
                        changes,
                        source="kai.environment_model",
                        environment_snapshot_id=snapshot_id,
                        causal_parent_id=observed_event_id,
                        confidence=0.95,
                        epistemic_status="INFERENCE",
                        occurred_at=captured_at,
                    )
                    connection.execute(
                        "UPDATE experiences SET source_event_ids_json = ? "
                        "WHERE experience_id = ?",
                        (
                            json.dumps([observed_event_id, claims_event_id, change_event_id]),
                            experience_id,
                        ),
                    )
                working_context = json.dumps(
                    {
                        "environment_snapshot_id": snapshot_id,
                        "environment_fingerprint": fingerprint,
                        "experience_id": experience_id,
                        "captured_at": captured_at,
                    },
                    sort_keys=True,
                )
                connection.execute(
                    "INSERT INTO working_memory "
                    "(memory_key, value_json, source, updated_at) VALUES "
                    "('current_environment', ?, 'kai.environment_model', ?) "
                    "ON CONFLICT(memory_key) DO UPDATE SET "
                    "value_json = excluded.value_json, source = excluded.source, "
                    "updated_at = excluded.updated_at",
                    (working_context, captured_at),
                )
                connection.execute(
                    "DELETE FROM environment_snapshots WHERE snapshot_id IN "
                    "(SELECT snapshot_id FROM environment_snapshots "
                    "ORDER BY snapshot_id DESC LIMIT -1 OFFSET ?)",
                    (retention,),
                )
                connection.commit()
            except (KaiError, sqlite3.Error, KeyError, TypeError, ValueError) as exc:
                connection.rollback()
                if isinstance(exc, KaiError):
                    raise
                raise KaiError(f"Cannot persist environment snapshot: {exc}") from exc
        return {
            "snapshot_id": snapshot_id,
            "previous": self._snapshot_row(previous) if previous else None,
            "event_id": observed_event_id,
            "change_event_id": change_event_id,
            "experience_id": experience_id,
        }

    @staticmethod
    def _validate_claim(claim: dict) -> None:
        required = (
            "claim",
            "classification",
            "source",
            "timestamp",
            "evidence",
            "verification_status",
        )
        missing = [field for field in required if field not in claim]
        if missing:
            raise KaiError(f"Epistemic claim missing required fields: {', '.join(missing)}")
        if claim["classification"] not in {
            "OBSERVATION", "FACT", "INFERENCE", "HYPOTHESIS", "UNKNOWN"
        }:
            raise KaiError(f"Invalid epistemic classification: {claim['classification']}")
        if claim["verification_status"] not in {
            "observed", "verified", "unverified", "contradicted"
        }:
            raise KaiError(f"Invalid claim verification status: {claim['verification_status']}")
        confidence = claim.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(confidence)
            or not 0.0 <= confidence <= 1.0
        ):
            raise KaiError("Claim confidence must be between 0.0 and 1.0.")
        if claim["classification"] == "UNKNOWN" and claim["verification_status"] != "unverified":
            raise KaiError("UNKNOWN claims must have unverified verification status.")
        if claim["classification"] == "FACT" and (
            claim["verification_status"] != "verified" or not claim["evidence"]
        ):
            raise KaiError("FACT claims require evidence and verified status.")
        if claim["verification_status"] == "contradicted" and not claim.get(
            "contradictory_evidence"
        ):
            raise KaiError("Contradicted claims require contradictory evidence.")
        if any(not isinstance(claim[field], str) or not claim[field].strip()
               for field in ("claim", "source", "timestamp")):
            raise KaiError("Claim text, source, and timestamp must be non-empty strings.")

    @staticmethod
    def _snapshot_row(row: sqlite3.Row) -> dict:
        return {
            "snapshot_id": row["snapshot_id"],
            "captured_at": row["captured_at"],
            "environment_fingerprint": row["environment_fingerprint"],
            "environment": json.loads(row["snapshot_json"]),
            "measurements": json.loads(row["measurements_json"]),
        }

    def latest_environment_snapshot(self) -> dict | None:
        with self.connection() as connection:
            try:
                row = connection.execute(
                    "SELECT snapshot_id, captured_at, environment_fingerprint, "
                    "snapshot_json, measurements_json FROM environment_snapshots "
                    "ORDER BY snapshot_id DESC LIMIT 1"
                ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read latest environment snapshot: {exc}") from exc
        return self._snapshot_row(row) if row else None

    def environment_snapshot_count(self) -> int:
        with self.connection() as connection:
            try:
                return int(
                    connection.execute("SELECT COUNT(*) FROM environment_snapshots").fetchone()[0]
                )
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot count environment snapshots: {exc}") from exc

    def claims_for_snapshot(self, snapshot_id: int) -> list[dict]:
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT claim, classification, source, observed_at, confidence, "
                    "evidence_json, verification_status, contradictory_evidence_json "
                    "FROM epistemic_claims WHERE snapshot_id = ? ORDER BY claim_id",
                    (snapshot_id,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read epistemic claims: {exc}") from exc
        return [
            {
                "claim": row["claim"],
                "classification": row["classification"],
                "source": row["source"],
                "timestamp": row["observed_at"],
                "confidence": row["confidence"],
                "evidence": json.loads(row["evidence_json"]),
                "verification_status": row["verification_status"],
                "contradictory_evidence": json.loads(row["contradictory_evidence_json"]),
            }
            for row in rows
        ]

    def create_experience(
        self,
        context: dict,
        *,
        observation: object | None = None,
        belief: object | None = None,
        action: object | None = None,
        outcome: object | None = None,
        evaluation: object | None = None,
        learning: object | None = None,
        causal_event_id: int | None = None,
        source_event_ids: list[int] | None = None,
        timestamp: str | None = None,
    ) -> int:
        if not isinstance(context, dict):
            raise KaiError("Experience context must be an object.")
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        values = (observation, belief, action, outcome, evaluation, learning)
        status = "COMPLETED" if any(value is not None for value in values[3:]) else "OPEN"
        source_ids = list(source_event_ids or [])
        if any(not isinstance(item, int) or item < 1 for item in source_ids):
            raise KaiError("Experience source event IDs must be positive integers.")
        with self.connection() as connection:
            try:
                _validate_references(connection, "events", "event_id", source_ids)
                cursor = connection.execute(
                    "INSERT INTO experiences "
                    "(created_at, updated_at, status, context_json, observation_json, "
                    "belief_json, action_json, outcome_json, evaluation_json, learning_json, "
                    "causal_event_id, source_event_ids_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        now,
                        now,
                        status,
                        _encode_json(context),
                        _json_optional(observation),
                        _json_optional(belief),
                        _json_optional(action),
                        _json_optional(outcome),
                        _json_optional(evaluation),
                        _json_optional(learning),
                        causal_event_id,
                        json.dumps(source_ids),
                    ),
                )
                experience_id = int(cursor.lastrowid)
                experience_event_id = _insert_event(
                    connection,
                    "EXPERIENCE_RECORDED",
                    "A structured experience was recorded.",
                    {"experience_id": experience_id, "status": status},
                    source="kai.experience",
                    causal_parent_id=causal_event_id,
                    occurred_at=now,
                )
                if experience_event_id not in source_ids:
                    source_ids.append(experience_event_id)
                    connection.execute(
                        "UPDATE experiences SET source_event_ids_json = ? "
                        "WHERE experience_id = ?",
                        (_encode_json(source_ids), experience_id),
                    )
                connection.commit()
                return experience_id
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot create experience: {exc}") from exc

    def list_experiences(self, limit: int = 20) -> list[Experience]:
        _validate_limit(limit, "Experience")
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM experiences ORDER BY experience_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read experiences: {exc}") from exc
        return [_experience_row(row) for row in rows]

    def get_experience(self, experience_id: int) -> Experience:
        with self.connection() as connection:
            try:
                row = connection.execute(
                    "SELECT * FROM experiences WHERE experience_id = ?",
                    (experience_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read experience: {exc}") from exc
        if row is None:
            raise KaiError(f"Experience {experience_id} does not exist.")
        return _experience_row(row)

    def create_prediction(
        self,
        hypothesis: str,
        expected_outcome: object,
        *,
        context: dict,
        confidence: float | None,
        basis: str,
        observation_window: str | None = None,
        experience_id: int | None = None,
        timestamp: str | None = None,
    ) -> int:
        if (
            not isinstance(hypothesis, str)
            or not isinstance(basis, str)
            or not hypothesis.strip()
            or not basis.strip()
        ):
            raise KaiError("Prediction hypothesis and basis must be non-empty.")
        if not isinstance(context, dict):
            raise KaiError("Prediction context must be an object.")
        _validate_confidence(confidence, "Prediction")
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                if experience_id is None:
                    cursor = connection.execute(
                        "INSERT INTO experiences "
                        "(created_at, updated_at, status, context_json, source_event_ids_json) "
                        "VALUES (?, ?, 'OPEN', ?, '[]')",
                        (now, now, _encode_json(context)),
                    )
                    experience_id = int(cursor.lastrowid)
                cursor = connection.execute(
                    "INSERT INTO predictions "
                    "(experience_id, created_at, context_json, hypothesis, "
                    "expected_outcome_json, confidence, basis, observation_window, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')",
                    (
                        experience_id,
                        now,
                        _encode_json(context),
                        hypothesis,
                        _encode_json(expected_outcome),
                        confidence,
                        basis,
                        observation_window,
                    ),
                )
                prediction_id = int(cursor.lastrowid)
                event_id = _insert_event(
                    connection,
                    "PREDICTION_CREATED",
                    "A testable prediction was recorded.",
                    {"prediction_id": prediction_id, "hypothesis": hypothesis},
                    source="kai.prediction",
                    causal_parent_id=(
                        connection.execute(
                            "SELECT causal_event_id FROM experiences WHERE experience_id = ?",
                            (experience_id,),
                        ).fetchone()[0]
                    ),
                    confidence=confidence,
                    epistemic_status="HYPOTHESIS",
                    occurred_at=now,
                )
                connection.execute(
                    "UPDATE predictions SET created_event_id = ? WHERE prediction_id = ?",
                    (event_id, prediction_id),
                )
                experience = connection.execute(
                    "SELECT experience_id, source_event_ids_json FROM experiences "
                    "WHERE experience_id = ?",
                    (experience_id,),
                ).fetchone()
                if experience is None:
                    raise KaiError(f"Experience {experience_id} does not exist.")
                source_ids = json.loads(experience["source_event_ids_json"])
                source_ids.append(event_id)
                connection.execute(
                    "UPDATE experiences SET prediction_id = ?, "
                    "belief_json = coalesce(belief_json, ?), "
                    "source_event_ids_json = ?, updated_at = ? WHERE experience_id = ?",
                    (
                        prediction_id,
                        json.dumps(
                            {"hypothesis": hypothesis, "basis": basis},
                            sort_keys=True,
                        ),
                        json.dumps(source_ids),
                        now,
                        experience_id,
                    ),
                )
                connection.commit()
                return prediction_id
            except KaiError:
                connection.rollback()
                raise
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot create prediction: {exc}") from exc

    def evaluate_prediction(
        self,
        prediction_id: int,
        observed_outcome: object,
        *,
        outcome_available: bool = True,
        timestamp: str | None = None,
        tolerance: float = 0.0,
    ) -> dict:
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance < 0
        ):
            raise KaiError("Prediction tolerance must be a finite non-negative number.")
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                prediction = connection.execute(
                    "SELECT * FROM predictions WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
                if prediction is None:
                    raise KaiError(f"Prediction {prediction_id} does not exist.")
                if prediction["status"] != "PENDING":
                    raise KaiError(
                        f"Prediction {prediction_id} has already been evaluated."
                    )
                expected = json.loads(prediction["expected_outcome_json"])
                difference = None
                error_magnitude = None
                if not outcome_available:
                    status = "UNKNOWN"
                    evaluation = "No observed outcome was provided."
                elif (
                    isinstance(expected, (int, float))
                    and not isinstance(expected, bool)
                    and isinstance(observed_outcome, (int, float))
                    and not isinstance(observed_outcome, bool)
                ):
                    difference = observed_outcome - expected
                    error_magnitude = abs(difference)
                    status = (
                        "CONFIRMED"
                        if error_magnitude == 0
                        else "PARTIALLY_CONFIRMED"
                        if tolerance > 0 and error_magnitude <= tolerance
                        else "REFUTED"
                    )
                    evaluation = (
                        "Observed numeric outcome compared with the expected value "
                        f"using tolerance {tolerance}."
                    )
                else:
                    status = (
                        "CONFIRMED"
                        if _encode_json(observed_outcome) == _encode_json(expected)
                        else "REFUTED"
                    )
                    evaluation = "Observed outcome compared with the expected value."
                difference_value = {
                    "difference": difference,
                    "error_magnitude": error_magnitude,
                    "tolerance": tolerance,
                } if error_magnitude is not None else None
                event_id = _insert_event(
                    connection,
                    "PREDICTION_EVALUATED",
                    "A prediction was compared with an observed outcome.",
                    {
                        "prediction_id": prediction_id,
                        "status": status,
                        "expected": expected,
                        "observed": observed_outcome,
                        "difference": difference_value,
                    },
                    source="kai.prediction",
                    environment_snapshot_id=None,
                    causal_parent_id=prediction["created_event_id"],
                    confidence=prediction["confidence"],
                    epistemic_status="OBSERVATION" if outcome_available else "UNKNOWN",
                    occurred_at=now,
                )
                connection.execute(
                    "UPDATE predictions SET status = ?, evaluated_at = ?, "
                    "observed_outcome_json = ?, difference_json = ?, evaluation = ?, "
                    "evaluated_event_id = ? WHERE prediction_id = ?",
                    (
                        status,
                        now,
                        _encode_json(observed_outcome) if outcome_available else None,
                        _json_optional(difference_value),
                        evaluation,
                        event_id,
                        prediction_id,
                    ),
                )
                learning_id = None
                experience_id = prediction["experience_id"]
                if experience_id is not None and status != "UNKNOWN":
                    experience = connection.execute(
                        "SELECT belief_json, source_event_ids_json FROM experiences "
                        "WHERE experience_id = ?",
                        (experience_id,),
                    ).fetchone()
                    learning_event_id = _insert_event(
                        connection,
                        "LEARNING_OCCURRED",
                        "Prediction error and outcome were recorded for later learning.",
                        {
                            "prediction_id": prediction_id,
                            "experience_id": experience_id,
                            "status": status,
                            "error": difference_value,
                        },
                        source="kai.learning",
                        causal_parent_id=event_id,
                        confidence=prediction["confidence"],
                        epistemic_status="INFERENCE",
                        occurred_at=now,
                    )
                    cursor = connection.execute(
                        "INSERT INTO learning_records "
                        "(experience_id, created_at, before_json, happened_json, "
                        "predicted_json, error_json, changed_json, why, evidence_json, "
                        "confidence, source_experiences_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            experience_id,
                            now,
                            experience["belief_json"] or "{}",
                            _encode_json(observed_outcome),
                            prediction["expected_outcome_json"],
                            _encode_json(difference_value),
                            _encode_json(
                                {"prediction_status": status},
                            ),
                            evaluation,
                            _encode_json([event_id, learning_event_id]),
                            prediction["confidence"],
                            _encode_json([experience_id]),
                        ),
                    )
                    learning_id = int(cursor.lastrowid)
                    source_event_ids = json.loads(experience["source_event_ids_json"])
                    source_event_ids.extend(
                        item
                        for item in (event_id, learning_event_id)
                        if item not in source_event_ids
                    )
                else:
                    source_event_ids = None
                if experience_id is not None:
                    connection.execute(
                        "UPDATE experiences SET status = ?, outcome_json = ?, "
                        "evaluation_json = ?, learning_json = ?, source_event_ids_json = "
                        "coalesce(?, source_event_ids_json), updated_at = ? "
                        "WHERE experience_id = ?",
                        (
                            "OPEN" if status == "UNKNOWN" else "COMPLETED",
                            _encode_json(observed_outcome) if outcome_available else None,
                            _encode_json(
                                {"prediction_status": status, "evaluation": evaluation}
                            ),
                            _json_optional(
                                {"learning_id": learning_id}
                                if learning_id is not None
                                else None
                            ),
                            _json_optional(source_event_ids),
                            now,
                            experience_id,
                        ),
                    )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot evaluate prediction: {exc}") from exc
        return {
            "prediction_id": prediction_id,
            "experience_id": experience_id,
            "status": status,
            "expected": expected,
            "observed": observed_outcome,
            "difference": difference_value,
            "evaluation": evaluation,
            "learning_id": learning_id,
        }

    def list_predictions(self, limit: int = 20) -> list[Prediction]:
        _validate_limit(limit, "Prediction")
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM predictions ORDER BY prediction_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read predictions: {exc}") from exc
        return [_prediction_row(row) for row in rows]

    def get_prediction(self, prediction_id: int) -> Prediction:
        with self.connection() as connection:
            try:
                row = connection.execute(
                    "SELECT * FROM predictions WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read prediction: {exc}") from exc
        if row is None:
            raise KaiError(f"Prediction {prediction_id} does not exist.")
        return _prediction_row(row)

    def expire_prediction(self, prediction_id: int, timestamp: str | None = None) -> None:
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT created_event_id, experience_id, status FROM predictions "
                    "WHERE prediction_id = ?",
                    (prediction_id,),
                ).fetchone()
                if row is None:
                    raise KaiError(f"Prediction {prediction_id} does not exist.")
                if row["status"] != "PENDING":
                    raise KaiError(f"Prediction {prediction_id} is not pending.")
                event_id = _insert_event(
                    connection,
                    "PREDICTION_EVALUATED",
                    "A prediction's observation window expired without an outcome.",
                    {"prediction_id": prediction_id, "status": "EXPIRED"},
                    source="kai.prediction",
                    causal_parent_id=row["created_event_id"],
                    epistemic_status="UNKNOWN",
                    occurred_at=now,
                )
                connection.execute(
                    "UPDATE predictions SET status = 'EXPIRED', evaluated_at = ?, "
                    "evaluation = ?, evaluated_event_id = ? WHERE prediction_id = ?",
                    (
                        now,
                        "Observation window ended without sufficient evidence.",
                        event_id,
                        prediction_id,
                    ),
                )
                if row["experience_id"] is not None:
                    connection.execute(
                        "UPDATE experiences SET evaluation_json = ?, updated_at = ? "
                        "WHERE experience_id = ?",
                        (
                            _encode_json({"prediction_status": "EXPIRED"}),
                            now,
                            row["experience_id"],
                        ),
                    )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot expire prediction: {exc}") from exc

    def record_learning(
        self,
        *,
        before: object,
        happened: object,
        predicted: object,
        error: object,
        changed: object,
        why: str,
        evidence: list[object],
        confidence: float | None,
        source_experiences: list[int],
        experience_id: int | None = None,
        timestamp: str | None = None,
    ) -> int:
        if not why.strip() or not evidence:
            raise KaiError("Learning records require a reason and evidence.")
        _validate_confidence(confidence, "Learning")
        source_experiences = list(source_experiences)
        if not source_experiences or any(
            not isinstance(item, int) or item < 1 for item in source_experiences
        ):
            raise KaiError("Learning records require positive source experience IDs.")
        if experience_id is not None and experience_id not in source_experiences:
            source_experiences = [*source_experiences, experience_id]
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        with self.connection() as connection:
            try:
                _validate_references(
                    connection, "experiences", "experience_id", source_experiences
                )
                cursor = connection.execute(
                    "INSERT INTO learning_records "
                    "(experience_id, created_at, before_json, happened_json, "
                    "predicted_json, error_json, changed_json, why, evidence_json, "
                    "confidence, source_experiences_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        experience_id,
                        now,
                        _encode_json(before),
                        _encode_json(happened),
                        _encode_json(predicted),
                        _encode_json(error),
                        _encode_json(changed),
                        why,
                        json.dumps(evidence, sort_keys=True),
                        confidence,
                        json.dumps(source_experiences),
                    ),
                )
                learning_id = int(cursor.lastrowid)
                learning_event_id = _insert_event(
                    connection,
                    "LEARNING_OCCURRED",
                    "An auditable learning record was stored.",
                    {"learning_id": learning_id, "experience_id": experience_id},
                    source="kai.learning",
                    causal_parent_id=_latest_experience_event_id(
                        connection, source_experiences
                    ),
                    confidence=confidence,
                    epistemic_status="INFERENCE",
                    occurred_at=now,
                )
                if experience_id is not None:
                    experience_row = connection.execute(
                        "SELECT source_event_ids_json FROM experiences "
                        "WHERE experience_id = ?",
                        (experience_id,),
                    ).fetchone()
                    source_event_ids = json.loads(
                        experience_row["source_event_ids_json"]
                    )
                    if learning_event_id not in source_event_ids:
                        source_event_ids.append(learning_event_id)
                    connection.execute(
                        "UPDATE experiences SET status = 'COMPLETED', learning_json = ?, "
                        "source_event_ids_json = ?, "
                        "updated_at = ? WHERE experience_id = ?",
                        (
                            json.dumps({"learning_id": learning_id}, sort_keys=True),
                            _encode_json(source_event_ids),
                            now,
                            experience_id,
                        ),
                    )
                connection.commit()
                return learning_id
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot store learning record: {exc}") from exc

    def list_learning_records(self, limit: int = 20) -> list[LearningRecord]:
        _validate_limit(limit, "Learning record")
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM learning_records ORDER BY learning_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read learning records: {exc}") from exc
        return [_learning_row(row) for row in rows]

    def set_working_memory(
        self, key: str, value: object, source: str, timestamp: str | None = None
    ) -> None:
        if not key.strip() or not source.strip():
            raise KaiError("Working-memory key and source must be non-empty.")
        now = _utc_timestamp(timestamp) if timestamp else utc_now()
        try:
            encoded = _encode_json(value)
        except (TypeError, ValueError) as exc:
            raise KaiError(f"Working-memory value is not JSON serializable: {exc}") from exc
        with self.connection() as connection:
            try:
                connection.execute(
                    "INSERT INTO working_memory (memory_key, value_json, source, updated_at) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(memory_key) DO UPDATE SET "
                    "value_json = excluded.value_json, source = excluded.source, "
                    "updated_at = excluded.updated_at",
                    (key, encoded, source, now),
                )
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot update working memory: {exc}") from exc

    def working_memory(self) -> list[dict]:
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT memory_key, value_json, source, updated_at "
                    "FROM working_memory ORDER BY memory_key"
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read working memory: {exc}") from exc
        return [
            {
                "key": row["memory_key"],
                "value": json.loads(row["value_json"]),
                "source": row["source"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def add_layered_memory(
        self,
        layer: str,
        content: str,
        source: str,
        *,
        epistemic_status: str = "HYPOTHESIS",
        verification_status: str = "unverified",
        confidence: float | None = None,
        evidence: list[object] | None = None,
        source_experiences: list[int] | None = None,
    ) -> int:
        if layer not in {"semantic", "procedural"}:
            raise KaiError("Persistent layered memory must be semantic or procedural.")
        if (
            not isinstance(content, str)
            or not isinstance(source, str)
            or not content.strip()
            or not source.strip()
        ):
            raise KaiError("Layered memory content and source must be non-empty.")
        if epistemic_status not in {
            "OBSERVATION", "FACT", "INFERENCE", "HYPOTHESIS", "UNKNOWN"
        }:
            raise KaiError(f"Invalid memory epistemic status: {epistemic_status}")
        if verification_status not in {
            "candidate", "unverified", "verified", "contradicted"
        }:
            raise KaiError(f"Invalid memory verification status: {verification_status}")
        if epistemic_status == "UNKNOWN" and verification_status == "verified":
            raise KaiError("UNKNOWN memory cannot be marked verified.")
        if verification_status == "verified" and not evidence:
            raise KaiError("Verified memory requires evidence.")
        if verification_status == "verified" and epistemic_status != "FACT":
            raise KaiError("Only an explicitly verified FACT can be marked verified.")
        if epistemic_status == "FACT" and verification_status != "verified":
            raise KaiError("FACT memory requires verified status and evidence.")
        if verification_status == "contradicted" and not evidence:
            raise KaiError("Contradicted memory requires evidence.")
        evidence_items = list(evidence or [])
        experience_ids = list(source_experiences or [])
        if any(
            not isinstance(item, int) or item < 1 for item in experience_ids
        ):
            raise KaiError("Source experience IDs must be positive integers.")
        _validate_confidence(confidence, "Memory")
        now = utc_now()
        with self.connection() as connection:
            try:
                _validate_references(
                    connection, "experiences", "experience_id", experience_ids
                )
                cursor = connection.execute(
                    "INSERT INTO memory_items "
                    "(layer, created_at, content, source, epistemic_status, "
                    "verification_status, confidence, evidence_json, "
                    "source_experiences_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        layer,
                        now,
                        content,
                        source,
                        epistemic_status,
                        verification_status,
                        confidence,
                        _encode_json(evidence_items),
                        _encode_json(experience_ids),
                    ),
                )
                memory_id = int(cursor.lastrowid)
                event_type = (
                    "CLAIM_CREATED" if layer == "semantic" else "PROCEDURE_RECORDED"
                )
                _insert_event(
                    connection,
                    event_type,
                    f"A {layer} memory item was stored.",
                    {"memory_id": memory_id, "epistemic_status": epistemic_status},
                    source=source,
                    causal_parent_id=_latest_experience_event_id(
                        connection, experience_ids
                    ),
                    confidence=confidence,
                    epistemic_status=epistemic_status,
                )
                connection.commit()
                return memory_id
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot store layered memory: {exc}") from exc

    def verify_semantic_memory(
        self, memory_id: int, evidence: list[object], source: str
    ) -> None:
        if not evidence or not source.strip():
            raise KaiError("Verification requires evidence and a source.")
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM memory_items WHERE memory_id = ? AND layer = 'semantic'",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    raise KaiError(f"Semantic memory {memory_id} does not exist.")
                combined = json.loads(row["evidence_json"]) + evidence
                before = {
                    "epistemic_status": row["epistemic_status"],
                    "verification_status": row["verification_status"],
                }
                after = {
                    "epistemic_status": "FACT",
                    "verification_status": "verified",
                }
                connection.execute(
                    "UPDATE memory_items SET epistemic_status = 'FACT', "
                    "verification_status = 'verified', evidence_json = ? "
                    "WHERE memory_id = ?",
                    (_encode_json(combined), memory_id),
                )
                _insert_event(
                    connection,
                    "CLAIM_VERIFIED",
                    "A semantic memory claim was explicitly verified.",
                    {
                        "memory_id": memory_id,
                        "before": before,
                        "after": after,
                        "evidence": evidence,
                    },
                    source=source,
                    causal_parent_id=_latest_experience_event_id(
                        connection,
                        json.loads(row["source_experiences_json"]),
                    ),
                    epistemic_status="FACT",
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot verify semantic memory: {exc}") from exc

    def contradict_semantic_memory(
        self, memory_id: int, evidence: list[object], source: str
    ) -> None:
        if not evidence or not source.strip():
            raise KaiError("Contradiction requires evidence and a source.")
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT * FROM memory_items WHERE memory_id = ? AND layer = 'semantic'",
                    (memory_id,),
                ).fetchone()
                if row is None:
                    raise KaiError(f"Semantic memory {memory_id} does not exist.")
                prior_evidence = json.loads(row["evidence_json"])
                combined = prior_evidence + evidence
                after_epistemic_status = (
                    "HYPOTHESIS" if row["epistemic_status"] == "FACT"
                    else row["epistemic_status"]
                )
                connection.execute(
                    "UPDATE memory_items SET verification_status = 'contradicted', "
                    "epistemic_status = ?, evidence_json = ? "
                    "WHERE memory_id = ?",
                    (
                        after_epistemic_status,
                        _encode_json(combined),
                        memory_id,
                    ),
                )
                _insert_event(
                    connection,
                    "CLAIM_CONTRADICTED",
                    "Evidence contradicting a semantic memory claim was recorded.",
                    {
                        "memory_id": memory_id,
                        "claim": row["content"],
                        "prior_epistemic_status": row["epistemic_status"],
                        "prior_verification_status": row["verification_status"],
                        "new_epistemic_status": after_epistemic_status,
                        "new_verification_status": "contradicted",
                        "contradictory_evidence": evidence,
                    },
                    source=source,
                    causal_parent_id=_latest_experience_event_id(
                        connection,
                        json.loads(row["source_experiences_json"]),
                    ),
                    confidence=row["confidence"],
                    epistemic_status="OBSERVATION",
                )
                connection.commit()
            except KaiError:
                connection.rollback()
                raise
            except (sqlite3.Error, TypeError, ValueError) as exc:
                connection.rollback()
                raise KaiError(f"Cannot record claim contradiction: {exc}") from exc

    def list_layered_memory(self, layer: str, limit: int = 20) -> list[dict]:
        if layer not in {"semantic", "procedural"}:
            raise KaiError(f"Invalid persistent memory layer: {layer}")
        _validate_limit(limit, "Layered memory")
        with self.connection() as connection:
            try:
                rows = connection.execute(
                    "SELECT * FROM memory_items WHERE layer = ? "
                    "ORDER BY memory_id DESC LIMIT ?",
                    (layer, limit),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot read {layer} memory: {exc}") from exc
        return [_memory_item_row(row) for row in rows]

    def search_memory(self, query: str, limit: int = 20) -> dict:
        if not query.strip():
            raise KaiError("Memory search query cannot be empty.")
        _validate_limit(limit, "Memory search")
        with self.connection() as connection:
            try:
                legacy = connection.execute(
                    "SELECT memory_id, created_at, kind, content, source, confidence "
                    "FROM memories WHERE instr(lower(content), lower(?)) > 0 "
                    "ORDER BY memory_id DESC LIMIT ?",
                    (query, limit),
                ).fetchall()
                layered = connection.execute(
                    "SELECT * FROM memory_items "
                    "WHERE instr(lower(content), lower(?)) > 0 "
                    "ORDER BY memory_id DESC LIMIT ?",
                    (query, limit),
                ).fetchall()
                episodes = connection.execute(
                    "SELECT * FROM experiences WHERE "
                    "instr(lower(coalesce(observation_json, '')), lower(?)) > 0 OR "
                    "instr(lower(coalesce(belief_json, '')), lower(?)) > 0 OR "
                    "instr(lower(coalesce(outcome_json, '')), lower(?)) > 0 "
                    "ORDER BY experience_id DESC LIMIT ?",
                    (query, query, query, limit),
                ).fetchall()
            except sqlite3.Error as exc:
                raise KaiError(f"Cannot search memory: {exc}") from exc
        return {
            "legacy_entries": [dict(row) for row in legacy],
            "semantic_and_procedural": [_memory_item_row(row) for row in layered],
            "episodes": [_experience_row(row) for row in episodes],
        }

    def consolidate_observations(self, minimum_occurrences: int = 3) -> list[int]:
        if minimum_occurrences < 2:
            raise KaiError("Consolidation requires at least two source episodes.")
        created = []
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                episodes = connection.execute(
                    "SELECT experience_id, observation_json, source_event_ids_json "
                    "FROM experiences "
                    "WHERE observation_json IS NOT NULL "
                    "ORDER BY experience_id DESC LIMIT 1000"
                ).fetchall()
                candidates = connection.execute(
                    "SELECT source_experiences_json FROM memory_items "
                    "WHERE layer = 'semantic' AND source_experiences_json != '[]'"
                ).fetchall()
                existing_sources = {
                    tuple(json.loads(row["source_experiences_json"])) for row in candidates
                }
                groups: dict[str, list[int]] = {}
                source_events: dict[int, list[int]] = {}
                for episode in episodes:
                    observation = json.loads(episode["observation_json"])
                    key = _encode_json(observation)
                    groups.setdefault(key, []).append(episode["experience_id"])
                    source_events[episode["experience_id"]] = json.loads(
                        episode["source_event_ids_json"]
                    )
                for observed, episode_ids in groups.items():
                    if len(episode_ids) < minimum_occurrences:
                        continue
                    source_ids = sorted(episode_ids)
                    source_signature = tuple(source_ids)
                    if source_signature in existing_sources:
                        continue
                    evidence = [f"experience:{item}" for item in source_ids]
                    content = f"Repeated observation candidate: {observed}"
                    cursor = connection.execute(
                        "INSERT INTO memory_items "
                        "(layer, created_at, content, source, epistemic_status, "
                        "verification_status, confidence, evidence_json, "
                        "source_experiences_json) VALUES "
                        "('semantic', ?, ?, 'deterministic episodic consolidation', "
                        "'HYPOTHESIS', 'candidate', NULL, ?, ?)",
                        (
                            utc_now(),
                            content,
                            _encode_json(evidence),
                            _encode_json(source_ids),
                        ),
                    )
                    memory_id = int(cursor.lastrowid)
                    causal_parent_id = (
                        source_events[source_ids[-1]][-1]
                        if source_events[source_ids[-1]]
                        else None
                    )
                    _insert_event(
                        connection,
                        "CLAIM_CREATED",
                        "Repeated episode observations produced an unverified candidate.",
                        {
                            "memory_id": memory_id,
                            "source_experiences": source_ids,
                            "verification_status": "candidate",
                        },
                        source="deterministic episodic consolidation",
                        causal_parent_id=causal_parent_id,
                        epistemic_status="HYPOTHESIS",
                    )
                    existing_sources.add(source_signature)
                    created.append(memory_id)
                connection.commit()
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot consolidate episodic observations: {exc}") from exc
        return created

    def memory_report(self, limit: int = 10) -> dict:
        return {
            "working": self.working_memory(),
            "episodes": self.list_experiences(limit),
            "semantic": self.list_layered_memory("semantic", limit),
            "procedural": self.list_layered_memory("procedural", limit),
            "learning": self.list_learning_records(limit),
        }

    def recover_interrupted_operations(self) -> list[int]:
        with self.connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")
                rows = connection.execute(
                    "SELECT started.event_id, started.occurred_at, started.experiment_id, "
                    "execution.experience_id, execution.experiment_definition_key, "
                    "execution.experiment_definition_id, execution.experiment_version "
                    "FROM events AS started LEFT JOIN experiment_executions AS execution "
                    "ON execution.execution_id = started.experiment_id "
                    "WHERE (started.event_type = 'MEASUREMENT_STARTED' OR "
                    "(started.event_type = 'EXPERIMENT_STARTED' "
                    "AND execution.status = 'RUNNING')) "
                    "AND NOT EXISTS (SELECT 1 FROM events AS finished "
                    "WHERE finished.causal_parent_id = started.event_id "
                    "AND finished.event_type IN "
                    "('MEASUREMENT_COMPLETED', 'MEASUREMENT_FAILED', 'EXPERIMENT_COMPLETED', "
                    "'EXPERIMENT_FAILED')) "
                    "AND NOT EXISTS (SELECT 1 FROM events AS recovery "
                    "WHERE recovery.causal_parent_id = started.event_id "
                    "AND recovery.event_type = 'RECOVERY') "
                    "ORDER BY started.event_id"
                ).fetchall()
                recovery_ids = []
                for row in rows:
                    recovery_id = _insert_event(
                            connection,
                            "RECOVERY",
                            "An experiment or measurement was interrupted by an earlier process.",
                            {
                                "interrupted_event_id": row["event_id"],
                                "recovered_at": utc_now(),
                                "recovery_action": "recorded interruption; no repair attempted",
                            },
                            source="kai.recovery",
                            causal_parent_id=row["event_id"],
                            experiment_id=row["experiment_id"],
                            execution_id=row["experiment_id"],
                            experiment_definition_key=row["experiment_definition_key"],
                            experiment_definition_id=row["experiment_definition_id"],
                            experiment_version=row["experiment_version"],
                            epistemic_status="INFERENCE",
                        )
                    recovery_ids.append(recovery_id)
                    if row["experiment_id"] is not None:
                        connection.execute(
                            "UPDATE experiment_executions SET status = 'INTERRUPTED', "
                            "execution_status = 'UNKNOWN', failure_information = ?, "
                            "completed_at = ?, output_json = ? "
                            "WHERE execution_id = ? AND status = 'RUNNING'",
                            (
                                "Process ended before a trustworthy result was persisted.",
                                utc_now(),
                                _encode_json(
                                    {
                                        "interrupted_event_id": row["event_id"],
                                        "recovery_event_id": recovery_id,
                                    }
                                ),
                                row["experiment_id"],
                            ),
                        )
                        prediction_id = connection.execute(
                            "SELECT prediction_id FROM experiences WHERE experience_id = ?",
                            (row["experience_id"],),
                        ).fetchone()["prediction_id"]
                        connection.execute(
                            "UPDATE predictions SET status = 'UNKNOWN', evaluated_at = ?, "
                            "evaluation = 'Experiment was interrupted before an outcome was stored.', "
                            "evaluated_event_id = ? WHERE experiment_id = ? AND status = 'PENDING'",
                            (
                                utc_now(),
                                recovery_id,
                                row["experiment_id"],
                            ),
                        )
                        evidence = {
                            "sample_count": 0,
                            "context_consistency": False,
                            "measurement_status": "UNKNOWN",
                            "prediction_available": prediction_id is not None,
                            "evaluation_available": False,
                            "execution_id": row["experiment_id"],
                            "recovery_event_id": recovery_id,
                        }
                        connection.execute(
                            "INSERT OR IGNORE INTO experiment_results "
                            "(execution_id, outcome_value, unit, sample_count, duration_seconds, "
                            "metadata_json, measurement_status) "
                            "VALUES (?, NULL, 'seconds', 0, NULL, ?, 'UNKNOWN')",
                            (
                                row["experiment_id"],
                                _encode_json({"reason": "interrupted"}),
                            ),
                        )
                        connection.execute(
                            "INSERT OR IGNORE INTO experiment_evaluations "
                            "(execution_id, prediction_id, evaluation_status, evidence_quality, "
                            "evidence_json, created_at) VALUES (?, ?, ?, 'NONE', ?, ?)",
                            (
                                row["experiment_id"],
                                prediction_id,
                                "UNKNOWN" if prediction_id is not None else "NOT_EVALUATED",
                                _encode_json(evidence),
                                utc_now(),
                            ),
                        )
                        connection.execute(
                            "UPDATE experiences SET evaluation_json = ?, updated_at = ? "
                            "WHERE experience_id = (SELECT experience_id FROM experiment_executions "
                            "WHERE execution_id = ?)",
                            (
                                _encode_json(
                                    {
                                        "experiment_status": "INTERRUPTED",
                                        "recovery_event_id": recovery_id,
                                    }
                                ),
                                utc_now(),
                                row["experiment_id"],
                            ),
                        )
                connection.commit()
                return recovery_ids
            except sqlite3.Error as exc:
                connection.rollback()
                raise KaiError(f"Cannot recover interrupted operations: {exc}") from exc
