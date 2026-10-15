# KAI 0.007

KAI extends a small local foundation for a persistent artificial cognitive
architecture. It maintains a stable identity, inspects its current environment,
records events and epistemically labeled memory, stores simple goals, and
provides a deterministic, auditable experiment framework around its bounded
adaptive prediction loop. KAI 0.004 established a generated persistent
self-model and evidence ledger. KAI 0.005 added disposable source workspaces
and sandboxed repeated evaluation for improvement candidates. KAI 0.006 added
an explicitly invoked local-model engineering workflow. KAI 0.007 unifies
model access behind a provider gateway, supports installed Ollama model
selection, and adds a strictly opt-in OpenAI adapter and privacy-safe request
history.

This milestone is not a general intelligence, autonomous agent, or claim of
consciousness. Models are used only after explicit commands; generated changes
remain disposable candidates and are never automatically deployed. Cloud
requests are disabled unless explicitly enabled and authorized.
Machine and network measurements are best-effort; unavailable values are
reported as unknown rather than guessed.

## Requirements

- Python 3.10 or newer
- SQLite support in the Python standard library
- Linux bubblewrap (`bwrap`) for candidate test and benchmark execution

No third-party Python dependencies are required.

## Run

From this directory:

```sh
python3 -m kai boot
python3 -m kai status
python3 -m kai doctor
python3 -m kai measure
python3 -m kai predict next
python3 -m kai predict status
python3 -m kai experiments
python3 -m kai learn cpu-baseline
python3 -m kai predict history
python3 -m kai self status
python3 -m kai self doctor
python3 -m kai self capabilities
python3 -m kai improvements list
python3 -m kai improvements demo-cpu
python3 -m kai workspaces list
python3 -m kai improvements history
python3 -m kai candidates list
python3 -m kai engineer status
python3 -m kai engineer inspect
python3 -m kai llm status
python3 -m kai llm providers
python3 -m kai llm models
python3 -m kai llm active
```

By default, persistent data is stored under `~/.local/share/kai`. Set `KAI_HOME`
to use another state directory, for example:

```sh
KAI_HOME=/tmp/kai-test python3 -m kai boot
```

## Optional local language model

KAI connects only to a locally running Ollama service by default. The default
selection is `ollama/qwen2.5:0.5b`, a deliberately small CPU-friendly model
choice for this machine class; it is not proof that the model is installed.
Use `kai llm models` to discover the actual installed models. Responses may be
slow or limited in quality. This host has about 3 GiB of RAM, so close other
memory-heavy applications before inference.

Install Ollama using its official instructions, then explicitly download the
model and check readiness:

```sh
ollama pull qwen2.5:0.5b
ollama list
python3 -m kai llm status
python3 -m kai llm models
python3 -m kai llm use ollama/qwen2.5:0.5b
python3 -m kai llm generate "Explain what KAI currently does."
```

If Ollama is not already serving requests, start it separately with
`ollama serve`. KAI contacts only `http://127.0.0.1:11434`; it does not install
Ollama, download models, call cloud APIs, include KAI's stored memory, or grant
the model tools/actions. Model downloads are manual Ollama operations; KAI
never pulls a model. Selection is persisted in KAI's SQLite state only after
the provider confirms that the model is available. A failed switch leaves the
prior selection intact. Use `KAI_LLM_PROVIDER=disabled` to disable model access.

`kai llm generate` prompts and responses are not persisted. KAI records request
IDs, task category, selected/actual provider and model, timestamps, duration,
success/failure, token counts (when supplied), error category, and fallback
reason—but never request/response text or secrets. `kai llm history` shows
these records.

### Model routing and fallback

Task routes and fallbacks are opt-in, persistent, and inspectable. No fallback
is configured initially. KAI tries a configured list in order, at most once
per model, and reports which model actually answered. Cloud models are never
used unless explicitly enabled; code-related cloud routes additionally require
the explicit source-sharing authorization shown below.

```sh
python3 -m kai llm active
python3 -m kai llm route list
python3 -m kai llm route set coding ollama/qwen2.5:0.5b
python3 -m kai llm generate --task coding "Review kai/runtime.py" --local-only
python3 -m kai llm fallback add ollama/qwen2.5:1.5b
python3 -m kai llm fallback list
python3 -m kai llm history --limit 20
python3 -m kai llm test
```

Fallbacks are limited to three and are not automatically discovered or added.
`--local-only` rejects a cloud primary and filters cloud fallbacks. Ollama
requests are serialized within KAI, also serialized across KAI processes using
a private state-directory lock, and sent with `keep_alive=0` so KAI asks Ollama
to unload the model after each request. This cannot control unrelated programs
using the Ollama service. RAM warnings compare installed weight size with
available RAM only; inference memory requirements and GPU inference usability
remain unknown unless independently verified.

### Optional OpenAI API (disabled unless explicitly authorized)

Local operation needs no OpenAI account, key, SDK, or API request. To prepare
the optional adapter later:

1. Create an API key in your OpenAI account. Do not paste it into KAI, source
   files, SQLite, or a committed `.env` file.
2. Provide it to the KAI process through a secure environment/secret manager.
   For an interactive private terminal, one option that avoids shell history
   and echo is: `read -rsp "OpenAI API key: " OPENAI_API_KEY; export OPENAI_API_KEY; printf '\n'`.
3. Explicitly enable the provider in that process:
   `export KAI_OPENAI_ENABLED=true`. This setting alone does not activate a
   request; the key must also be present.
4. Install the optional `openai` SDK only in the Python environment running
   KAI, if it is not already installed. KAI does not install it automatically.
5. Select a model ID returned by the account's model listing:
   `python3 -m kai llm providers`, then
   `python3 -m kai llm models`, then
   `python3 -m kai llm use openai/<model-id>`.
   KAI does not assume a model ID is available to your account. `use` checks
   the account's available IDs before persisting the selection.
6. `python3 -m kai llm status` checks local configuration without querying
   OpenAI. Explicit `python3 -m kai llm models` discovers account-accessible
   models, and `python3 -m kai llm use openai/<model-id>` verifies and persists
   a selection. `python3 -m kai llm test` makes a bounded generation request
   and can incur API usage.

Set account/project spending limits and monitor usage in OpenAI's dashboard.
KAI limits each generation's output and disables automatic SDK retries, but
does not control provider pricing or impose a monetary budget. OpenAI requests
use the fixed official API endpoint and are not made during install or tests.
Source code is never sent to a cloud model for `coding` or `code_review`
without the separate, explicit `KAI_OPENAI_ALLOW_SOURCE=true` setting.
Cloud fallbacks are also skipped unless the provider is enabled and any
task-specific data-sharing policy is satisfied.

## Bounded engineering tasks

`kai engineer inspect` reports source inventory, machine resources, local model,
and sandbox readiness without initializing or writing KAI state. `plan` asks
the ready local model for a constrained plan; `run` creates a persisted task,
generates one audited candidate change, requests an advisory model review, then
runs the selected existing unittest module against baseline and candidate in
bubblewrap. Each repaired candidate is reviewed again; review never replaces
static validation or tests and cannot approve deployment. The run permits at
most two explicitly bounded repair attempts. Candidate files stay outside
production source and are never deployed. Task stages and outcomes are
inspectable after restart:

```sh
python3 -m kai engineer status
python3 -m kai engineer inspect
python3 -m kai engineer plan "Add a small tested utility"
python3 -m kai engineer run "Add a small tested utility" --repair-attempts 1
python3 -m kai engineer tasks
python3 -m kai engineer task <task-id>
python3 -m kai candidates diff <candidate-id>
```

The workflow does not yet have a generic benchmark protocol. Passing a targeted
test therefore does not prove the objective improved; evaluation reports
`UNKNOWN`/`INCONCLUSIVE` and requests more evidence rather than claiming a
successful improvement. A missing local model blocks planning and execution;
a missing sandbox never causes host-side test execution.

Commands:

```text
python3 -m kai boot
python3 -m kai status
python3 -m kai llm status
python3 -m kai llm providers
python3 -m kai llm models
python3 -m kai llm active
python3 -m kai llm use ollama/qwen2.5:0.5b
python3 -m kai llm generate "Your prompt"
python3 -m kai llm generate --task coding --local-only "Review kai/runtime.py"
python3 -m kai llm test
python3 -m kai llm route list
python3 -m kai llm fallback list
python3 -m kai llm history
python3 -m kai engineer status
python3 -m kai engineer inspect
python3 -m kai engineer plan "Your objective"
python3 -m kai engineer run "Your objective"
python3 -m kai engineer tasks
python3 -m kai inspect
python3 -m kai hardware
python3 -m kai memory list
python3 -m kai memory add --kind observation --content "A recorded observation"
python3 -m kai memory search observation
python3 -m kai memory consolidate --minimum-occurrences 3
python3 -m kai memory verify 1 --evidence "Independent check succeeded" --source "operator"
python3 -m kai memory contradict 1 --evidence "Later result disagreed" --source "measurement"
python3 -m kai goals list
python3 -m kai goals add "Build a tested memory subsystem" --priority 2
python3 -m kai events --type ENVIRONMENT_CHANGED --since 2026-10-01T00:00:00Z
python3 -m kai experiences --limit 20
python3 -m kai predict create --hypothesis "The next baseline is near one second" \
  --expected-outcome 1.0 --context '{"iterations": 100000}' \
  --basis "prior baseline measurement" --confidence 0.6
python3 -m kai predict evaluate 1 --observed-outcome 0.9 --tolerance 0.2
python3 -m kai predict models
python3 -m kai predict next
python3 -m kai predict status
python3 -m kai predict history
python3 -m kai predict rollback 1
python3 -m kai experiments --limit 10
python3 -m kai experiments show cpu-baseline --version 1
python3 -m kai experiments run cpu-baseline --version 1
python3 -m kai experiments history --experiment cpu-baseline --limit 20
python3 -m kai experiments inspect-execution 1
python3 -m kai learn cpu-baseline
python3 -m kai self observations --limit 20
python3 -m kai self doctor
python3 -m kai self weaknesses
python3 -m kai self failures
python3 -m kai self metrics
python3 -m kai improvements show 1
python3 -m kai candidates show cpu-baseline-last-observation-v1
python3 -m kai evaluate cpu-baseline-last-observation-v1
python3 -m kai evaluations
python3 -m kai evaluations show 1
python3 -m kai workspaces list
python3 -m kai workspaces create <candidate-id> --dry-run
python3 -m kai workspaces show <workspace-id>
python3 -m kai workspaces validate <workspace-id>
python3 -m kai workspaces diff <workspace-id>
python3 -m kai workspaces test <workspace-id> --module tests.test_candidate
python3 -m kai workspaces cleanup <workspace-id>
python3 -m kai learning
python3 -m kai report
python3 -m kai doctor
python3 -m kai benchmark --rounds 10
python3 -m kai measure
```

`status`, the general `inspect`, and `hardware` commands persist a timestamped environment snapshot,
its evidence-backed claims, a privacy-preserving environment-profile fingerprint,
and a comparison with the previous snapshot. `measure` explicitly runs a small
CPU integer-loop timing and a bounded read-only sample from KAI's existing
database; neither is an ML benchmark. No GPU workload is run. Status and
environment snapshots create linked observation events and experiences; KAI
stores an updated, versioned self-model from runtime, environment, database,
and static source inventory. Source inspection parses Python AST without
importing or executing project code. Its test-function count is inventory only,
not a claim that the test suite passed.

The environment report distinguishes PCI enumeration, kernel driver binding,
device-node visibility, runtime-library presence, usable capability, and measured
performance. None implies the next. Tool discovery reports PATH visibility and
does not execute discovered programs. Claims include an epistemic class, source,
timestamp, confidence, evidence, verification state, and contradictory evidence.

`kai predict next` generates a forecast from a persisted per-environment-profile
running-mean model when evidence exists. With no evidence it reports
`INSUFFICIENT DATA`; the first explicitly requested `learn cpu-baseline` or
`experiments run cpu-baseline` run is a seed observation, not a fabricated
forecast. Later runs predict, measure, score signed error as observed minus
expected, update a new model version, and create the next forecast. The only
executable experiment is an allowlisted fixed
100,000-iteration in-process CPU loop with a bounded read-only sample of KAI's
existing database; it is opt-in and does not use a GPU or execute discovered
tools. The versioned definition is stored separately from each execution;
executions have structured result and evaluation records linked to procedure,
prediction, model, and evidence provenance. Result validation rejects
incompatible units, malformed values, non-finite numbers, and mismatched
sample counts. Resource estimates are qualitative and do not promise a
duration.

The procedure is fixed, but its elapsed wall-clock outcome is not reproducible:
host load, scheduling, thermal state, and other conditions vary. KAI records
that observation rather than treating procedure reproducibility as runtime
determinism. Evidence quality is reported as `NONE`, `INSUFFICIENT`,
`PRELIMINARY`, or `ADEQUATE`; `ADEQUATE` requires at least 20 same-context
observations and an available prediction evaluation. This is an engineering
reporting tier, not statistical significance or calibrated confidence. Failed
measurements remain `UNKNOWN`; malformed measurements are `INVALID` and do not
update the model.

The model uses Welford's running mean and sample variance. Standard error is
`UNKNOWN` for one observation; with repeated observations it is descriptive,
not a calibrated prediction interval. Prediction accuracy excludes
`PENDING`, `UNKNOWN`, and `EXPIRED`; empirical accuracy is withheld until five
scored predictions, and confidence-versus-accuracy remains `INSUFFICIENT DATA`
until 20. Neither empirical hit rate nor the descriptive calibration bins are
claimed as calibrated probabilities. Updates are transactionally
versioned and audited. `predict rollback <update-id>` restores the previous
parameters in a new version and expires forecasts tied to superseded versions.

`kai self status` persists and returns the generated self-model version and its
provenance-linked observations. Self-observation reports KAI's identity,
version, schema, active models, registered experiments, capabilities, known
limitations, runtime errors, recent experiment outcomes, source tree,
dependencies, and improvement history. Capabilities and test execution that
are not established by safe observation remain `UNKNOWN`. Self-model history
is append-only; it is a point-in-time report, not independent introspection.

`kai improvements demo-cpu` is a separate explicit measurement action. It
creates a disposable filesystem snapshot outside the production source tree,
adds a fixed candidate predictor and contract tests through the audited editor,
and runs those tests under bubblewrap before measuring. The repeated protocol
collects 30 bounded CPU baseline timings in three predeclared sequences and
compares 24 paired prequential forecasts after two warm-up observations per
sequence. Lower MAE is the declared metric; the engineering decision requires
a 10% aggregate change and directionally consistent sequence results. This is
not a statistical significance test or general performance claim. Candidate
acceptance requires adequate persisted evidence, passing candidate contract
checks, zero reported regressions, and the latest matching evaluation. The
full project suite remains a separate release gate and is not run by the demo.
Candidate execution requires Linux bubblewrap, disables networking, exposes
the snapshot read-only, uses ephemeral mounts, and applies resource limits. If
the sandbox or a complete measurement sequence is unavailable, the result
remains `UNKNOWN`; no unsandboxed fallback is used. The demo changes neither
production source nor predictor parameters, and no candidate can be deployed.
Every evaluation preserves an immutable baseline
snapshot: KAI and architecture versions, relevant model and experiment
definition versions, configuration, environment context, dataset references,
and metrics. Inspect it with `kai evaluations show <id>`.

`kai events` filters by event type and timezone-aware ISO-8601 `--since`; event
IDs preserve committed ordering. `kai memory list` summarizes working,
episodic, semantic, procedural, and learning records. `kai memory consolidate`
creates only unverified semantic candidates from repeated episode observations.
`kai predict create/evaluate` still supports explicit user-authored hypotheses
and comparisons; numeric error is recorded, and missing outcomes remain
UNKNOWN. The bounded CPU predictor is a separate, narrowly-scoped adaptive
model and does not rewrite semantic knowledge.

Memory kinds are `fact`, `observation`, `inference`, `hypothesis`,
`speculation`, and `unknown`. The kind and source are retained as metadata;
an unverified legacy `fact` label is only a candidate hypothesis in semantic
memory until explicitly verified with evidence. Storing a hypothesis does not
promote it to a fact. Events, experiences, memory, knowledge, and truth have
separate meanings; see [ADR-0002](./docs/adr/0002-separate-events-experiences-and-knowledge.md).

Read [ARCHITECTURE.md](./ARCHITECTURE.md), [ROADMAP.md](./ROADMAP.md),
[DEVELOPMENT.md](./DEVELOPMENT.md), [SECURITY.md](./SECURITY.md), and
[DECISIONS.md](./DECISIONS.md) for the current model and boundaries.

## Persistent state and recovery

Identity, goals, events, layered memory, experiences, predictions, learning
records, predictive models/versions/updates, experiment definitions, executions,
results, evaluations, environment snapshots, versioned self-models,
self-observations, the improvement ledger, protocols, workspace provenance,
audited edits, sandbox executions, engineering task histories, model
selection/routing, and privacy-safe model request metadata are stored in
`kai.sqlite3`. API keys and request/response content are not stored.
Schema v9 migrates prior schemas while
preserving KAI's UUID; `kai doctor` runs SQLite's `quick_check`.
Initialization fails explicitly on corruption or an unsupported future schema.
Back up the state by stopping KAI and copying the entire data directory; SQLite's
database file is the source of truth. Operational messages are also written to
`kai.log`.

## Tests

```sh
PYTHONPATH=. python3 -m unittest discover -s tests -v
```

## Current boundaries

- Environment discovery uses local OS interfaces and Linux procfs/sysfs. Internet
  connectivity is not probed.
- GPU runtime and workload usability may remain unknown even when a PCI GPU,
  driver, device node, or runtime library is visible.
- Goals are records, not an automated planner.
- Memory is explicit SQLite-backed structured state, not neural/semantic
  retrieval. Consolidation is exact-pattern candidate generation, not
  generalized intelligence or a complete model of a person.
- The running-mean CPU model is deterministic and domain-bounded, not a neural
  network or general predictive system. It has no evidence before its first
  explicit experiment, and one observation cannot quantify uncertainty.
- Candidate workspaces permit explicit, audited edits only to `.py` files
  beneath `kai/` and `tests/`. The CLI does not generate code autonomously.
  Sandboxed execution is restricted to fixed unittest and CPU-protocol
  procedures; a candidate decision is not deployment.
- There is no neural learner, discovered-tool execution, autonomous action
  executor, or production self-modifying code. Ollama generation is local and
  explicit. The optional OpenAI adapter remains disabled unless enabled with
  both a key and provider flag; sending source code additionally requires its
  explicit authorization flag. No cloud fallback is configured by default.

KAI's persistent UUID is deliberately independent of the environment
fingerprint, hardware inventory, and model versions. The environment
fingerprint identifies a similar observed profile, not KAI or a unique physical
machine. Epistemic labels (`OBSERVATION`, `FACT`, `INFERENCE`, `HYPOTHESIS`,
`UNKNOWN`) describe the support for claims and remain distinct from identity,
hardware, capabilities, and measured experiment results.

The next steps should be driven by tests and measured experiments, not by
assuming that persistent state or repeated execution alone constitutes learning.
