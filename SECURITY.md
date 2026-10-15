# Security Model

## Current permissions

KAI performs read-only machine discovery and writes only its own
application state under `KAI_HOME` or `~/.local/share/kai`. It does not install
packages, change drivers or permissions, edit system configuration, modify boot
settings, or invoke arbitrary discovered commands.

Tool availability is established through PATH lookup. GPU and hardware discovery
uses read-only sysfs/procfs/device-node metadata. The opt-in `measure` command performs a small CPU loop and opens the existing
KAI database read-only for one bounded read; it does not write benchmark files
or exercise a GPU. The separate `learn cpu-baseline` command explicitly runs
that same fixed, bounded workload as the sole allowlisted experiment. The
versioned experiment registry binds a static procedure identifier to its
in-process implementation; CLI inputs cannot supply code, shell commands, or
mutable workload parameters. It persists the observation, prediction
evaluation (when available), model update, and subsequent forecast only in
KAI's SQLite state. Environment discovery remains read-only; changing KAI's
own experiment registry status affects only application state.
Event, experience, prediction, and memory commands mutate only KAI's SQLite
state. Generic predictions compare explicitly provided observations; the
adaptive CPU predictor runs only its fixed workload after explicit user
invocation. Neither path triggers arbitrary actions or external probes.
The schema-v6 self-model reads persisted KAI state and statically parses
`kai/*.py` and `tests/test_*.py` using Python's AST; it does not import or
execute project files, run tests, or execute discovered dependencies. Its
source digest covers those source files only and intentionally omits raw source
contents from the self-report. The explicitly invoked 0.005 CPU demonstration
copies only `kai/` and `tests/` into a KAI-owned state directory outside
production source, applies audited Python edits to that copy, collects 30
bounded measurements, and evaluates the source candidate only in a disposable
sandboxed workspace. It never edits production source or predictor state.
`ACTION_REQUESTED`, `ACTION_COMPLETED`, and
`ACTION_FAILED` are event types for future audit records only; no action
executor exists. Restart recovery records incomplete measurements but does not
retry them. Interrupted adaptive experiments are marked interrupted, their
outcome remains UNKNOWN, and any associated pending forecast is resolved as
UNKNOWN rather than being reused as evidence.

## Trust and uncertainty

The environment can be virtualized, containerized, permission-restricted, or
otherwise filtered. “Not visible” describes the current OS namespace and is not
proof that the physical host lacks a device. PCI, driver, runtime, and usable
capability are separately reported. No network connectivity check is performed.

Events preserve their source, optional snapshot and causal parent, payload,
confidence, and epistemic status. Experiences and learning records link events
to evidence. Deterministic consolidation creates unverified hypotheses and
does not establish truth. Claims preserve sources and contradictory evidence.
Environment fingerprints omit machine serials, MAC addresses, hostnames, and
user identity; they are profile identifiers, not unique hardware attestations
or credentials.

## State protection and future actions

Persistent state is stored under `KAI_HOME` or `~/.local/share/kai`; filesystem
permissions are inherited from the user account. SQLite integrity checks detect
some corruption but are not tamper-proof. Do not store secrets in memory or
claim evidence files.

There is no action executor, package installer, privilege broker, autonomous
code generator, production self-modification, or deployment facility. Explicit
`kai llm` and `kai engineer` commands may contact only either the fixed loopback
Ollama API (`http://127.0.0.1:11434`) or, if explicitly enabled with both
`KAI_OPENAI_ENABLED=true` and `OPENAI_API_KEY`, the fixed official OpenAI API
endpoint. No API key is read from KAI state, source files, or candidate
workspaces; it is never logged or included in request history. OpenAI model
discovery requires an explicit provider/model command and a configured key.
Engineering prompts may contain the explicit objective and bounded source
context, but never KAI's stored memory or tools. `coding` and `code_review`
requests cannot send source to the cloud unless `KAI_OPENAI_ALLOW_SOURCE=true`
is separately set. One-shot requests/responses are not stored in request
history; safe metadata only is persisted. Engineering plans, patches, task
objectives, and test/evaluation events are persisted for audit. Ollama
installation and model downloads are performed separately by the user, outside
KAI. Generated candidate edits are isolated, path-allowlisted, and audited;
candidate code execution is not production action execution.

Candidate test and benchmark execution requires Linux bubblewrap. The runner
binds the root filesystem read-only, overlays sensitive and writable system
locations with ephemeral mounts, then mounts the candidate snapshot read-only
over `/usr/share`; it uses unshared network, PID, IPC, and UTS namespaces, a
minimal device/proc view, sanitized environment, and limits on CPU time,
address space, process/thread count, file size, open files, temporary storage,
wall time, input, and captured output.
Procedures are fixed unittest selection or the fixed repeated CPU predictor
worker; no shell string or arbitrary procedure is accepted. Bubblewrap absence
or startup failure is reported as `UNKNOWN`, never as successful or
unsandboxed execution.
`kai self doctor` explicitly runs one fixed, no-op Python command inside the
same bubblewrap namespace to establish whether the sandbox can start; it does
not execute candidate source or any discovered third-party program.

`kai engineer inspect` performs source AST inventory, environment discovery,
local provider readiness, and a no-op sandbox readiness probe without creating
KAI state or a log file. `kai engineer plan/run` require a ready, explicitly
selected model and any applicable cloud/data-sharing authorization; they
persist the objective, validated plan, candidate lineage, edits, and bounded
test/evaluation events. A run is limited to one Python target under `kai/` or
`tests/`, one existing unittest module, and at most two repair attempts. The
task runner executes only the selected test procedure through bubblewrap; a
missing or failed sandbox blocks test execution and never triggers host-side
fallback. There is currently no generic benchmark protocol, so passing those
tests is not reported as proof of improvement; the evaluation remains
`UNKNOWN`/`INCONCLUSIVE` and requests more evidence.

The provider registry describes support explicitly: text and structured
generation plus token counts where returned are supported; streaming, tools,
and cancellation are not implemented. Ollama generations have bounded output,
are serialized across KAI processes, and request model unload after each call.
At most three configured fallbacks are attempted once each; there is no
implicit cloud fallback. A configured cloud fallback still requires cloud
enablement and any task-specific data-sharing authorization. Request history
is append-only and excludes prompts, generated text, keys, headers, and raw
provider error bodies.

The workspace editor rejects traversal, symlinks, secret-like paths/content,
oversized source, invalid Python, and protected runner infrastructure. The
database verifies that an edit's workspace, candidate, and hypothesis belong
to one proposal lineage. Candidate source is mounted read-only during
execution. Workspace and edit/test/evaluation records retain provenance;
explicit cleanup removes only the exact managed workspace path, while
preservation keeps it available for inspection. This boundary assumes the
local KAI account and state directory are not concurrently controlled by a
hostile process; bubblewrap is not a VM boundary against kernel vulnerabilities.

The current running-mean predictor is not a general-purpose benchmark runner:
workload parameters are fixed in code, no model-selected command is executed,
and rollback changes only the persisted model state by appending an audited
version. Resource estimates are qualitative; no GPU or ML performance is
claimed.

Experiment definitions are declarations; they do not themselves prove a
capability. Each run is a separate execution record with validated result and
evaluation records. The trusted procedure is reproducible, but wall-clock
runtime is affected by load and scheduling. KAI preserves unavailable outcomes
as `UNKNOWN` and malformed results as `INVALID`; neither is accepted as a
successful model-training observation. Evidence-quality thresholds are
descriptive engineering tiers, not security attestations, calibrated
confidence, or statistical-significance claims. The environment fingerprint
is not a unique hardware identity or security credential, and KAI's persistent
identity remains separate from it.

The improvement ledger validates acceptance against persisted evaluation
metrics, test/benchmark state, regressions, evidence quality, and linked
observations. Each evaluation preserves baseline versions, configuration,
environment context, dataset references, and metrics; inconsistent dataset
links or paired sample counts are rejected. An accepted candidate remains an application-level decision
record only; deployment is rejected because no deployment mechanism exists.
The repeated candidate protocol uses 30 bounded timings in three fixed
sequences, with 24 paired forecasts after warmups. Its directional criterion
is an engineering rule, not statistical significance. Evaluation rows retain
the protocol digest and baseline snapshot digest; a stored status that does
not match paired errors is rejected. Do not treat a favorable result from one
measured CPU timing sequence as a general performance guarantee.
