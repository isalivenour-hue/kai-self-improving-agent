import argparse
import json
import logging
import sys
from typing import Sequence

from kai.database import IDENTITY_NAME
from kai.errors import KaiError
from kai.local_llm import TASK_CATEGORIES
from kai.runtime import KaiRuntime

MEMORY_KINDS = (
    "fact",
    "observation",
    "inference",
    "hypothesis",
    "speculation",
    "unknown",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kai",
        description=(
            "KAI 0.007: multi-model gateway with isolated engineering evaluation."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("boot", help="initialize persistent state and record a boot")
    commands.add_parser("status", help="show current self-report and known state")
    commands.add_parser("report", help="print a complete JSON self-report")
    commands.add_parser("inspect", help="inspect OS, software, resources, and network")
    commands.add_parser("hardware", help="show layered discovered hardware and driver/runtime state")
    self_model = commands.add_parser("self", help="inspect KAI's persistent engineering self-model")
    self_actions = self_model.add_subparsers(dest="self_action")
    self_actions.add_parser("status", help="show the current versioned self-model")
    self_actions.add_parser(
        "doctor", help="report isolated candidate-development and sandbox readiness"
    )
    self_observations = self_actions.add_parser(
        "observations", help="show provenance-bearing self-observations"
    )
    self_observations.add_argument("--limit", type=int, default=50)
    self_observations.add_argument("--subject")
    self_observations.add_argument(
        "--epistemic-status", choices=("OBSERVED", "INFERRED", "UNKNOWN")
    )
    self_actions.add_parser("capabilities", help="show observed and unknown capabilities")
    self_actions.add_parser("weaknesses", help="show known limitations and unknown weaknesses")
    self_actions.add_parser("failures", help="show persisted experiment failures")
    self_actions.add_parser("metrics", help="show self-model and improvement metrics")

    llm = commands.add_parser(
        "llm", help="discover, select, route, and query configured model providers"
    )
    llm_actions = llm.add_subparsers(dest="llm_action", required=True)
    llm_actions.add_parser("status", help="check active provider and model readiness")
    llm_actions.add_parser("providers", help="list provider availability and capabilities")
    llm_actions.add_parser("models", help="discover models available from configured providers")
    llm_actions.add_parser("active", help="show active model, routes, and fallbacks")
    llm_use = llm_actions.add_parser("use", help="validate and select provider/model")
    llm_use.add_argument("target", help="provider/model-id")
    llm_actions.add_parser("test", help="send one explicit bounded connection-test request")
    llm_generate = llm_actions.add_parser(
        "generate", help="generate one response using the configured model gateway"
    )
    llm_generate.add_argument("--task", choices=TASK_CATEGORIES, default="general")
    llm_generate.add_argument("--model", help="explicit provider/model-id override")
    llm_generate.add_argument(
        "--local-only",
        action="store_true",
        help="refuse cloud models and cloud fallbacks for this request",
    )
    llm_generate.add_argument("prompt")
    llm_history = llm_actions.add_parser(
        "history", help="inspect privacy-safe model request metadata"
    )
    llm_history.add_argument("--limit", type=int, default=50)
    llm_fallback = llm_actions.add_parser(
        "fallback", help="configure a bounded ordered list of fallback models"
    )
    llm_fallback_actions = llm_fallback.add_subparsers(
        dest="llm_fallback_action", required=True
    )
    llm_fallback_actions.add_parser("list", help="list configured model fallbacks")
    llm_fallback_add = llm_fallback_actions.add_parser(
        "add", help="append one validated fallback model (maximum three)"
    )
    llm_fallback_add.add_argument("target", help="provider/model-id")
    llm_fallback_actions.add_parser("clear", help="remove configured model fallbacks")
    llm_route = llm_actions.add_parser(
        "route", help="inspect or configure per-task model selection"
    )
    llm_route_actions = llm_route.add_subparsers(
        dest="llm_route_action", required=True
    )
    llm_route_actions.add_parser("list", help="list task-specific model routes")
    llm_route_set = llm_route_actions.add_parser(
        "set", help="validate and set one task-specific model"
    )
    llm_route_set.add_argument("task", choices=TASK_CATEGORIES)
    llm_route_set.add_argument("target", help="provider/model-id")
    llm_route_clear = llm_route_actions.add_parser(
        "clear", help="remove one task-specific route"
    )
    llm_route_clear.add_argument("task", choices=TASK_CATEGORIES)

    engineer = commands.add_parser(
        "engineer",
        help="inspect, plan, and run bounded selected-model engineering tasks",
    )
    engineer_actions = engineer.add_subparsers(dest="engineer_action", required=True)
    engineer_actions.add_parser("status", help="check model, sandbox, and task readiness")
    engineer_actions.add_parser("inspect", help="inspect source and machine capabilities without a model")
    engineer_plan = engineer_actions.add_parser(
        "plan", help="ask the selected model for a validated task plan"
    )
    engineer_plan.add_argument("objective")
    engineer_run = engineer_actions.add_parser(
        "run", help="run an isolated task with fixed tests and bounded repairs"
    )
    engineer_run.add_argument("objective")
    engineer_run.add_argument(
        "--repair-attempts",
        type=int,
        choices=range(0, 3),
        default=1,
        help="maximum repair attempts for this task (0-2, default: 1)",
    )
    engineer_tasks = engineer_actions.add_parser(
        "tasks", help="list persisted engineering tasks"
    )
    engineer_tasks.add_argument("--limit", type=int, default=50)
    engineer_task = engineer_actions.add_parser(
        "task", help="inspect one task and its append-only stage history"
    )
    engineer_task.add_argument("task_id")

    memory = commands.add_parser("memory", help="inspect or add epistemically labeled memory")
    memory_actions = memory.add_subparsers(dest="memory_action")
    memory_list = memory_actions.add_parser("list", help="summarize current and persistent memory")
    memory_list.add_argument("--limit", type=int, default=10)
    add_memory = memory_actions.add_parser("add", help="store one labeled memory")
    add_memory.add_argument("--kind", choices=MEMORY_KINDS, required=True)
    add_memory.add_argument("--content", required=True)
    add_memory.add_argument("--source", default="user-provided")
    add_memory.add_argument("--confidence", type=float)
    search_memory = memory_actions.add_parser("search", help="search memory text")
    search_memory.add_argument("query")
    consolidate = memory_actions.add_parser(
        "consolidate", help="create unverified semantic candidates from repeated episodes"
    )
    consolidate.add_argument("--minimum-occurrences", type=int, default=3)
    verify_memory = memory_actions.add_parser(
        "verify", help="explicitly verify a semantic claim with supplied evidence"
    )
    verify_memory.add_argument("memory_id", type=int)
    verify_memory.add_argument("--evidence", action="append", required=True)
    verify_memory.add_argument("--source", required=True)
    contradict_memory = memory_actions.add_parser(
        "contradict", help="record evidence against a semantic claim"
    )
    contradict_memory.add_argument("memory_id", type=int)
    contradict_memory.add_argument("--evidence", action="append", required=True)
    contradict_memory.add_argument("--source", required=True)

    goals = commands.add_parser("goals", help="list or add simple persistent goals")
    goal_actions = goals.add_subparsers(dest="goal_action")
    goal_actions.add_parser("list", help="list stored goals")
    add_goal = goal_actions.add_parser("add", help="add a goal")
    add_goal.add_argument("title")
    add_goal.add_argument("--priority", type=int, default=0)
    add_goal.add_argument("--success-criteria", default="")

    events = commands.add_parser("events", help="show recent system events")
    events.add_argument("--limit", type=int, default=20)
    events.add_argument("--type", dest="event_type")
    events.add_argument("--since", help="ISO-8601 timestamp including timezone")
    events.add_argument("--types", action="store_true", help="list known and observed event types")
    experiences = commands.add_parser("experiences", help="show recent linked experiences")
    experiences.add_argument("--limit", type=int, default=20)
    predictions = commands.add_parser(
        "predict", help="create, evaluate, and inspect deterministic predictions"
    )
    prediction_actions = predictions.add_subparsers(dest="prediction_action", required=True)
    prediction_actions.add_parser("models", help="list stored predictive models")
    prediction_actions.add_parser("status", help="show model and accuracy status")
    prediction_history = prediction_actions.add_parser("history", help="show prediction and model-update history")
    prediction_history.add_argument("--limit", type=int, default=20)
    prediction_history.add_argument("--model-id")
    prediction_actions.add_parser("next", help="ask KAI to generate its next CPU baseline prediction")
    create_prediction = prediction_actions.add_parser("create", help="record a testable prediction")
    create_prediction.add_argument("--hypothesis", required=True)
    create_prediction.add_argument("--expected-outcome", required=True, help="JSON value")
    create_prediction.add_argument("--context", default="{}", help="JSON object")
    create_prediction.add_argument("--basis", required=True)
    create_prediction.add_argument("--confidence", type=float)
    create_prediction.add_argument("--observation-window")
    create_prediction.add_argument("--experience-id", type=int)
    evaluate_prediction = prediction_actions.add_parser("evaluate", help="compare prediction with outcome")
    evaluate_prediction.add_argument("prediction_id", type=int)
    outcome = evaluate_prediction.add_mutually_exclusive_group(required=True)
    outcome.add_argument("--observed-outcome", help="observed JSON value, including null")
    outcome.add_argument(
        "--unknown",
        action="store_true",
        help="record that the outcome could not be observed",
    )
    evaluate_prediction.add_argument("--tolerance", type=float, default=0.0)
    expire_prediction = prediction_actions.add_parser(
        "expire", help="mark a pending prediction's observation window expired"
    )
    expire_prediction.add_argument("prediction_id", type=int)
    rollback_prediction = prediction_actions.add_parser(
        "rollback", help="reversibly restore the parameters before a model update"
    )
    rollback_prediction.add_argument("update_id", type=int)
    experiments = commands.add_parser(
        "experiments", help="inspect bounded experiment options and recent runs"
    )
    experiments.add_argument("--limit", type=int, default=10)
    experiments.add_argument("--run", choices=("cpu-baseline",))
    experiment_actions = experiments.add_subparsers(dest="experiment_action")
    experiment_show = experiment_actions.add_parser(
        "show", help="show an allowlisted experiment definition"
    )
    experiment_show.add_argument("experiment_id")
    experiment_show.add_argument("--version", type=int)
    experiment_run = experiment_actions.add_parser(
        "run", help="execute one explicitly selected allowlisted experiment"
    )
    experiment_run.add_argument("experiment_id")
    experiment_run.add_argument("--version", type=int)
    experiment_history = experiment_actions.add_parser(
        "history", help="inspect experiment execution, result, evaluation, and evidence"
    )
    experiment_history.add_argument("--limit", type=int, default=20)
    experiment_history.add_argument("--experiment")
    execution_inspect = experiment_actions.add_parser(
        "inspect-execution", help="inspect a complete experiment execution record"
    )
    execution_inspect.add_argument("execution_id", type=int)
    learn = commands.add_parser(
        "learn", help="run one explicitly requested allowlisted adaptive experiment"
    )
    learn.add_argument("experiment", choices=("cpu-baseline",))
    improvements = commands.add_parser(
        "improvements", help="inspect the auditable improvement ledger"
    )
    improvement_actions = improvements.add_subparsers(
        dest="improvement_action", required=True
    )
    opportunity_list = improvement_actions.add_parser(
        "list", help="list improvement opportunities"
    )
    opportunity_list.add_argument("--limit", type=int, default=50)
    opportunity_list.add_argument("--status", choices=(
        "OPEN", "INVESTIGATING", "PROPOSED", "EXPERIMENTING",
        "ACCEPTED", "REJECTED", "DEFERRED", "INVALIDATED",
    ))
    opportunity_show = improvement_actions.add_parser(
        "show", help="inspect an opportunity and its linked lineage"
    )
    opportunity_show.add_argument("opportunity_id", type=int)
    improvement_history = improvement_actions.add_parser(
        "history", help="show improvement attempts and process metrics"
    )
    improvement_history.add_argument("--limit", type=int, default=50)
    improvement_actions.add_parser(
        "demo-cpu",
        help="run the isolated, repeated-protocol CPU predictor demonstration",
    )
    improvement = commands.add_parser(
        "improvement", help="inspect one opportunity and its linked improvement lineage"
    )
    improvement_actions_single = improvement.add_subparsers(
        dest="improvement_single_action", required=True
    )
    improvement_inspect = improvement_actions_single.add_parser(
        "inspect", help="inspect an improvement opportunity"
    )
    improvement_inspect.add_argument("opportunity_id", type=int)
    candidates = commands.add_parser("candidates", help="inspect isolated KAI candidates")
    candidate_actions = candidates.add_subparsers(dest="candidate_action", required=True)
    candidate_list = candidate_actions.add_parser("list", help="list isolated candidates")
    candidate_list.add_argument("--limit", type=int, default=50)
    candidate_show = candidate_actions.add_parser("show", help="show candidate lineage and evaluation")
    candidate_show.add_argument("candidate_id")
    candidate_diff = candidate_actions.add_parser(
        "diff", help="show audited workspace changes for a candidate"
    )
    candidate_diff.add_argument("candidate_id")
    candidate_evaluate = candidate_actions.add_parser(
        "evaluate", help="inspect a candidate's recorded evaluation and decision"
    )
    candidate_evaluate.add_argument("candidate_id")
    evaluate = commands.add_parser("evaluate", help="inspect a candidate's persisted evaluation")
    evaluate.add_argument("candidate_id")
    evaluations = commands.add_parser(
        "evaluations", help="list or inspect immutable candidate evaluations"
    )
    evaluation_actions = evaluations.add_subparsers(dest="evaluation_action")
    evaluation_list = evaluation_actions.add_parser(
        "list", help="list recent candidate evaluations"
    )
    evaluation_list.add_argument("--limit", type=int, default=50)
    evaluation_show = evaluation_actions.add_parser(
        "show", help="show one complete candidate evaluation and baseline"
    )
    evaluation_show.add_argument("evaluation_id", type=int)
    workspaces = commands.add_parser(
        "workspaces", help="inspect and operate on isolated candidate source snapshots"
    )
    workspace_actions = workspaces.add_subparsers(
        dest="workspace_action", required=True
    )
    workspace_list = workspace_actions.add_parser("list", help="list candidate workspaces")
    workspace_list.add_argument("--limit", type=int, default=50)
    workspace_create = workspace_actions.add_parser(
        "create", help="create a source snapshot for an existing candidate"
    )
    workspace_create.add_argument("candidate_id")
    workspace_create.add_argument(
        "--dry-run", action="store_true", help="validate snapshot inputs without persisting a workspace"
    )
    workspace_show = workspace_actions.add_parser("show", help="show workspace provenance")
    workspace_show.add_argument("workspace_id")
    workspace_validate = workspace_actions.add_parser(
        "validate", help="validate candidate source and workspace boundaries"
    )
    workspace_validate.add_argument("workspace_id")
    workspace_diff = workspace_actions.add_parser(
        "diff", help="show audited candidate modifications"
    )
    workspace_diff.add_argument("workspace_id")
    workspace_edit = workspace_actions.add_parser(
        "edit", help="apply one explicit audited Python source change"
    )
    workspace_edit.add_argument("workspace_id")
    workspace_edit.add_argument("operation", choices=("ADD", "MODIFY", "DELETE"))
    workspace_edit.add_argument("target_file")
    workspace_edit.add_argument("--content")
    workspace_edit.add_argument("--reason", required=True)
    workspace_edit.add_argument("--hypothesis-id", type=int, required=True)
    workspace_test = workspace_actions.add_parser(
        "test", help="run a fixed unittest procedure in the bubblewrap sandbox"
    )
    workspace_test.add_argument("workspace_id")
    workspace_test.add_argument("--module")
    workspace_cleanup = workspace_actions.add_parser(
        "cleanup", help="remove a disposable workspace or explicitly preserve it"
    )
    workspace_cleanup.add_argument("workspace_id")
    workspace_cleanup.add_argument("--preserve", action="store_true")
    learning = commands.add_parser("learning", help="inspect recent learning records and models")
    learning.add_argument("--limit", type=int, default=20)
    commands.add_parser("doctor", help="check state database and runtime health")
    benchmark = commands.add_parser("benchmark", help="run local or evidence-driven benchmarks")
    benchmark.add_argument("--rounds", type=int, default=10)
    benchmark_actions = benchmark.add_subparsers(dest="benchmark_action")
    benchmark_actions.add_parser(
        "cpu-predictor", help="compare the running-mean predictor with the fixed candidate"
    )
    commands.add_parser(
        "measure",
        help="run a safe, small CPU and read-only disk baseline; no GPU workload",
    )
    return parser


def _json(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False))


def _dispatch(arguments: argparse.Namespace, runtime: KaiRuntime) -> int:
    command = arguments.command
    if command == "boot":
        _json(runtime.boot())
    elif command == "status" or command == "report":
        _json(runtime.status())
    elif command == "inspect":
        report = runtime.status()
        _json(
            {
                "environment_fingerprint": report["environment_fingerprint"],
                "environment": report["environment"],
                "claims": report["claims"],
                "changes": report["changes"],
                "self_model": report["self_model"],
            }
        )
    elif command == "hardware":
        report = runtime.status()
        environment = report["environment"]
        _json(
            {
                "environment_fingerprint": report["environment_fingerprint"],
                "cpu": environment["cpu"],
                "memory": environment["memory"],
                "gpu": environment["gpu"],
                "pci_devices": environment["pci_devices"],
                "storage": environment["storage"],
                "claims": [
                    claim for claim in report["claims"]
                    if any(
                        word in claim["claim"].lower()
                        for word in ("cpu", "gpu", "pci", "driver", "cuda", "ram", "storage")
                    )
                ],
                "changes": report["changes"],
                "self_model": report["self_model"],
            }
        )
    elif command == "llm":
        if arguments.llm_action == "status":
            _json(runtime.local_llm_status())
        elif arguments.llm_action == "providers":
            _json(runtime.llm_providers())
        elif arguments.llm_action == "models":
            _json({"models": runtime.llm_models()})
        elif arguments.llm_action == "active":
            _json(runtime.llm_active())
        elif arguments.llm_action == "use":
            _json(runtime.llm_use(arguments.target))
        elif arguments.llm_action == "test":
            _json(runtime.llm_test())
        elif arguments.llm_action == "generate":
            _json(
                runtime.generate_local_response(
                    arguments.prompt,
                    task_category=arguments.task,
                    model_override=arguments.model,
                    local_only=arguments.local_only,
                )
            )
        elif arguments.llm_action == "history":
            if arguments.limit < 1:
                raise KaiError("LLM request history limit must be at least 1.")
            _json(runtime.llm_history(arguments.limit))
        elif arguments.llm_action == "fallback":
            if arguments.llm_fallback_action == "list":
                _json(runtime.llm_fallbacks())
            elif arguments.llm_fallback_action == "add":
                _json(runtime.llm_fallback_add(arguments.target))
            else:
                _json(runtime.llm_fallback_clear())
        elif arguments.llm_route_action == "list":
            _json(runtime.llm_routes())
        elif arguments.llm_route_action == "set":
            _json(runtime.llm_route_set(arguments.task, arguments.target))
        else:
            _json(runtime.llm_route_clear(arguments.task))
    elif command == "engineer":
        if arguments.engineer_action == "status":
            _json(runtime.engineer_status())
        elif arguments.engineer_action == "inspect":
            _json(runtime.engineer_inspect())
        elif arguments.engineer_action == "plan":
            _json(runtime.engineer_plan(arguments.objective))
        elif arguments.engineer_action == "run":
            _json(
                runtime.engineer_run(
                    arguments.objective, arguments.repair_attempts
                )
            )
        elif arguments.engineer_action == "tasks":
            if arguments.limit < 1:
                raise KaiError("Engineering task limit must be at least 1.")
            _json(runtime.list_engineering_tasks(arguments.limit))
        else:
            _json(runtime.get_engineering_task(arguments.task_id))
    elif command == "self":
        action = getattr(arguments, "self_action", None) or "status"
        if action == "status":
            _json(runtime.self_status())
        elif action == "doctor":
            _json(runtime.self_doctor())
        elif action == "observations":
            if arguments.limit < 1:
                raise KaiError("Self-observation limit must be at least 1.")
            _json(
                runtime.database.list_self_observations(
                    arguments.limit,
                    subject=arguments.subject,
                    epistemic_status=arguments.epistemic_status,
                )
            )
        elif action == "capabilities":
            _json(runtime.self_capabilities())
        elif action == "weaknesses":
            _json(runtime.self_weaknesses())
        elif action == "failures":
            _json(runtime.self_failures())
        elif action == "metrics":
            _json(runtime.self_metrics())
    elif command == "memory":
        action = arguments.memory_action or "list"
        if action == "add":
            memory_id = runtime.database.add_memory(
                arguments.kind,
                arguments.content,
                arguments.source,
                arguments.confidence,
            )
            _json({"memory_id": memory_id, "kind": arguments.kind})
        elif action == "consolidate":
            _json(
                {
                    "created_candidate_memory_ids": runtime.database.consolidate_observations(
                        arguments.minimum_occurrences
                    ),
                    "epistemic_status": "HYPOTHESIS",
                    "verification_status": "candidate",
                }
            )
        elif action == "verify":
            runtime.database.verify_semantic_memory(
                arguments.memory_id, arguments.evidence, arguments.source
            )
            _json({"memory_id": arguments.memory_id, "verification_status": "verified"})
        elif action == "contradict":
            runtime.database.contradict_semantic_memory(
                arguments.memory_id, arguments.evidence, arguments.source
            )
            _json(
                {
                    "memory_id": arguments.memory_id,
                    "verification_status": "contradicted",
                }
            )
        elif action == "list":
            limit = getattr(arguments, "limit", 10)
            if limit < 1:
                raise KaiError("Memory limit must be at least 1.")
            _json(
                {
                    **runtime.database.memory_report(limit),
                    "legacy_entries": runtime.database.list_memories(limit=limit),
                }
            )
        else:
            _json(runtime.database.search_memory(arguments.query))
    elif command == "goals":
        action = arguments.goal_action or "list"
        if action == "add":
            goal_id = runtime.database.add_goal(
                arguments.title,
                arguments.priority,
                arguments.success_criteria,
            )
            _json({"goal_id": goal_id, "status": "active"})
        else:
            _json(runtime.database.list_goals())
    elif command == "events":
        if arguments.types:
            _json(runtime.database.event_types())
        elif arguments.limit < 1:
            raise KaiError("Event limit must be at least 1.")
        else:
            _json(
                runtime.database.recent_events(
                    limit=arguments.limit,
                    event_type=arguments.event_type,
                    since=arguments.since,
                )
            )
    elif command == "experiences":
        if arguments.limit < 1:
            raise KaiError("Experience limit must be at least 1.")
        _json(runtime.database.list_experiences(limit=arguments.limit))
    elif command == "predict":
        if arguments.prediction_action == "create":
            try:
                context = json.loads(arguments.context)
                expected = json.loads(arguments.expected_outcome)
            except json.JSONDecodeError as exc:
                raise KaiError(f"Prediction inputs must be valid JSON: {exc}") from exc
            prediction_id = runtime.database.create_prediction(
                arguments.hypothesis,
                expected,
                context=context,
                confidence=arguments.confidence,
                basis=arguments.basis,
                observation_window=arguments.observation_window,
                experience_id=arguments.experience_id,
            )
            _json(
                {
                    "prediction_id": prediction_id,
                    "prediction": runtime.database.get_prediction(prediction_id),
                }
            )
        elif arguments.prediction_action == "evaluate":
            if arguments.unknown:
                observed = None
            else:
                try:
                    observed = json.loads(arguments.observed_outcome)
                except json.JSONDecodeError as exc:
                    raise KaiError(f"Observed outcome must be valid JSON: {exc}") from exc
            _json(
                runtime.database.evaluate_prediction(
                    arguments.prediction_id,
                    observed,
                    outcome_available=not arguments.unknown,
                    tolerance=arguments.tolerance,
                )
            )
        elif arguments.prediction_action == "expire":
            runtime.database.expire_prediction(arguments.prediction_id)
            _json(runtime.database.get_prediction(arguments.prediction_id))
        elif arguments.prediction_action == "models":
            _json(runtime.database.list_predictive_models())
        elif arguments.prediction_action == "status":
            _json(runtime.predictor_status())
        elif arguments.prediction_action == "history":
            if arguments.limit < 1:
                raise KaiError("Prediction history limit must be at least 1.")
            report = runtime.predictor_status()
            model_ids = (
                [arguments.model_id]
                if arguments.model_id
                else [model["model_id"] for model in report["models"]]
            )
            report["model_history"] = {
                model_id: runtime.database.predictive_model_history(
                    model_id, arguments.limit
                )
                for model_id in model_ids
            }
            report["predictions"] = report["predictions"][-arguments.limit:]
            report["experiments"] = report["experiments"][-arguments.limit:]
            _json(report)
        elif arguments.prediction_action == "next":
            _json(runtime.predict_cpu_baseline())
        else:
            _json(
                runtime.database.rollback_predictive_model_update(
                    arguments.update_id
                )
            )
    elif command == "experiments":
        action = getattr(arguments, "experiment_action", None)
        if action == "show":
            _json(
                runtime.show_experiment_definition(
                    arguments.experiment_id,
                    arguments.version,
                )
            )
        elif action == "run":
            _json(
                runtime.run_experiment(
                    arguments.experiment_id,
                    arguments.version,
                )
            )
        elif action == "history":
            if arguments.limit < 1:
                raise KaiError("Experiment history limit must be at least 1.")
            _json(
                runtime.database.list_experiment_executions(
                    arguments.limit,
                    arguments.experiment,
                )
            )
        elif action == "inspect-execution":
            _json(runtime.database.get_experiment_execution(arguments.execution_id))
        elif getattr(arguments, "limit", 10) < 1:
            raise KaiError("Experiment limit must be at least 1.")
        elif arguments.run:
            _json(runtime.run_cpu_baseline_experiment())
        else:
            definitions = runtime.list_experiment_definitions()
            _json(
                {
                    "registered_experiments": definitions,
                    "allowlisted_experiments": definitions,
                    "recent": runtime.database.list_experiments(arguments.limit),
                    "executions": runtime.database.list_experiment_executions(
                        arguments.limit
                    ),
                }
            )
    elif command == "learn":
        if arguments.experiment == "cpu-baseline":
            _json(runtime.run_experiment("cpu-baseline"))
    elif command == "learning":
        if arguments.limit < 1:
            raise KaiError("Learning history limit must be at least 1.")
        result = runtime.learning_history(arguments.limit)
        result["records"] = result["records"][:arguments.limit]
        _json(result)
    elif command == "improvements":
        if arguments.improvement_action == "list":
            if arguments.limit < 1:
                raise KaiError("Improvement limit must be at least 1.")
            _json(
                runtime.database.list_improvement_opportunities(
                    arguments.limit, arguments.status
                )
            )
        elif arguments.improvement_action == "show":
            _json(runtime.inspect_improvement(arguments.opportunity_id))
        elif arguments.improvement_action == "history":
            if arguments.limit < 1:
                raise KaiError("Improvement history limit must be at least 1.")
            _json(runtime.database.improvement_history(arguments.limit))
        else:
            _json(runtime.demonstrate_cpu_candidate_workspace())
    elif command == "improvement":
        _json(runtime.inspect_improvement(arguments.opportunity_id))
    elif command == "candidates":
        if arguments.candidate_action == "list":
            if arguments.limit < 1:
                raise KaiError("Candidate limit must be at least 1.")
            _json(runtime.database.list_kai_candidates(arguments.limit))
        elif arguments.candidate_action == "diff":
            _json(runtime.candidate_diff(arguments.candidate_id))
        else:
            _json(runtime.inspect_candidate(arguments.candidate_id))
    elif command == "evaluate":
        _json(runtime.evaluate_candidate(arguments.candidate_id))
    elif command == "evaluations":
        action = getattr(arguments, "evaluation_action", None) or "list"
        if action == "show":
            _json(
                runtime.database.get_improvement_evaluation(
                    arguments.evaluation_id
                )
            )
        else:
            limit = getattr(arguments, "limit", 50)
            if limit < 1:
                raise KaiError("Evaluation limit must be at least 1.")
            _json(runtime.database.list_improvement_evaluations(limit))
    elif command == "workspaces":
        action = arguments.workspace_action
        if action == "list":
            if arguments.limit < 1:
                raise KaiError("Workspace limit must be at least 1.")
            _json(runtime.list_candidate_workspaces(arguments.limit))
        elif action == "create":
            _json(
                runtime.preview_candidate_workspace(arguments.candidate_id)
                if arguments.dry_run
                else runtime.create_candidate_workspace(arguments.candidate_id)
            )
        elif action == "show":
            _json(runtime.get_candidate_workspace(arguments.workspace_id))
        elif action == "validate":
            _json(runtime.validate_candidate_workspace(arguments.workspace_id))
        elif action == "diff":
            _json(runtime.candidate_workspace_diff(arguments.workspace_id))
        elif action == "edit":
            if arguments.operation != "DELETE" and arguments.content is None:
                raise KaiError("ADD and MODIFY require --content.")
            if arguments.operation == "DELETE" and arguments.content is not None:
                raise KaiError("DELETE does not accept --content.")
            _json(
                runtime.edit_candidate_workspace(
                    workspace_id=arguments.workspace_id,
                    target_file=arguments.target_file,
                    operation=arguments.operation,
                    content=arguments.content,
                    reason=arguments.reason,
                    hypothesis_id=arguments.hypothesis_id,
                )
            )
        elif action == "test":
            _json(
                runtime.test_candidate_workspace(
                    arguments.workspace_id, module=arguments.module
                )
            )
        else:
            _json(
                runtime.cleanup_candidate_workspace(
                    arguments.workspace_id, preserve=arguments.preserve
                )
            )
    elif command == "doctor":
        runtime.database.initialize()
        result = {
            "status": "healthy",
            "database": str(runtime.config.database_path),
            "schema_version": runtime.database.schema_version(),
            "integrity_check": runtime.database.integrity_check(),
            "log_file": str(runtime.config.log_path),
            "identity": runtime.database.identity()["identity_id"],
        }
        _json(result)
    elif command == "benchmark":
        if arguments.benchmark_action == "cpu-predictor":
            _json(runtime.run_cpu_predictor_benchmark())
            return 0
        if arguments.rounds < 1 or arguments.rounds > 1000:
            raise KaiError("Benchmark rounds must be between 1 and 1000.")
        _json(runtime.benchmark(rounds=arguments.rounds))
    elif command == "measure":
        _json(runtime.measure())
    else:
        raise KaiError(f"Unknown command: {command}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    arguments = _parser().parse_args(argv)
    try:
        read_only_inspection = (
            arguments.command == "engineer"
            and arguments.engineer_action == "inspect"
        )
        runtime = KaiRuntime(configure_logging=not read_only_inspection)
        if arguments.command == "boot":
            result = runtime.boot()
            _json(
                {
                    "message": f"{IDENTITY_NAME} boot completed.",
                    "identity": result["identity"],
                    "version": result["version"],
                    "event_id": result["event_id"],
                    "environment_event_id": result["environment_event_id"],
                    "experience_id": result["experience_id"],
                    "recovered_event_ids": result["recovered_event_ids"],
                    "capabilities": result["capabilities"],
                }
            )
            return 0
        if not read_only_inspection:
            runtime.database.initialize()
        return _dispatch(arguments, runtime)
    except KaiError as exc:
        print(f"kai: error: {exc}", file=sys.stderr)
        return 1
