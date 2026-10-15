"""Model-independent persistent experience and learning record shapes."""

from typing import TypedDict

EVENT_TYPES = (
    "SYSTEM_BOOT",
    "ENVIRONMENT_OBSERVED",
    "ENVIRONMENT_CHANGED",
    "CLAIM_CREATED",
    "CLAIM_VERIFIED",
    "CLAIM_CONTRADICTED",
    "MEASUREMENT_STARTED",
    "MEASUREMENT_COMPLETED",
    "MEASUREMENT_FAILED",
    "ACTION_REQUESTED",
    "ACTION_COMPLETED",
    "ACTION_FAILED",
    "PREDICTION_CREATED",
    "PREDICTION_EVALUATED",
    "GOAL_CREATED",
    "GOAL_UPDATED",
    "LEARNING_OCCURRED",
    "ERROR",
    "RECOVERY",
    "EXPERIMENT_SELECTED",
    "EXPERIMENT_STARTED",
    "EXPERIMENT_COMPLETED",
    "EXPERIMENT_FAILED",
    "PREDICTIVE_MODEL_CREATED",
    "PREDICTIVE_MODEL_UPDATED",
)

PREDICTION_STATUSES = (
    "PENDING",
    "CONFIRMED",
    "PARTIALLY_CONFIRMED",
    "REFUTED",
    "EXPIRED",
    "UNKNOWN",
)

MEMORY_LAYERS = ("working", "episodic", "semantic", "procedural")


class Event(TypedDict):
    event_id: int
    occurred_at: str
    timestamp: str
    event_type: str
    summary: str
    source: str
    environment_snapshot_id: int | None
    causal_parent_id: int | None
    experiment_id: int | None
    execution_id: int | None
    experiment_definition_id: str | None
    experiment_version: int | None
    details: dict
    payload: dict
    confidence: float | None
    epistemic_status: str


class Experience(TypedDict):
    experience_id: int
    created_at: str
    updated_at: str
    status: str
    context: dict
    observation: object | None
    belief: object | None
    prediction_id: int | None
    action: object | None
    outcome: object | None
    evaluation: object | None
    learning: object | None
    causal_event_id: int | None
    source_event_ids: list[int]


class Prediction(TypedDict):
    prediction_id: int
    experience_id: int | None
    created_at: str
    context: dict
    hypothesis: str
    expected_outcome: object
    confidence: float | None
    basis: str
    observation_window: str | None
    status: str
    evaluated_at: str | None
    observed_outcome: object | None
    difference: object | None
    evaluation: str | None
    created_event_id: int | None
    evaluated_event_id: int | None
    model_id: str | None
    model_version: int | None
    experiment_id: int | None
    execution_id: int | None
    experiment_definition_id: str | None
    experiment_version: int | None
    environment_snapshot_id: int | None


class LearningRecord(TypedDict):
    learning_id: int
    experience_id: int | None
    created_at: str
    before: object
    happened: object
    predicted: object
    error: object
    changed: object
    why: str
    evidence: list[object]
    confidence: float | None
    source_experiences: list[int]


class ExperimentDefinition(TypedDict):
    definition_key: str
    experiment_id: str
    domain: str
    name: str
    description: str
    version: int
    status: str
    procedure_id: str
    parameter_specification: dict
    context_specification: dict
    outcome_specification: dict
    created_at: str
    updated_at: str


class ExperimentExecution(TypedDict):
    execution_id: int
    experiment_id: str
    experiment_version: int
    procedure_id: str
    started_at: str
    completed_at: str | None
    execution_status: str
    legacy_status: str
    environment_snapshot_id: int | None
    context_profile: str | None
    context: dict
    parameters: dict
    raw_result: dict | None
    failure_information: str | None
    result: dict | None
    evaluation: dict | None
