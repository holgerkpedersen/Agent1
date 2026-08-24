# Agent1 — A Self-Improving Python AI Coding Agent

> An agentic coding assistant that runs **local models through LM Studio** and **hosted models through opencode-go**, executes real tools (files, shell, git, tests), verifies its own writes, traces every step it takes, and closes the loop by diagnosing its own failures — all from one interactive REPL.

|                        |                                                                                    |
| ---------------------- | ---------------------------------------------------------------------------------- |
| **REPL commands**      | 24 registered commands                                                             |
| **Native LLM tools**   | 12 schema-declared, dispatcher-executed tools                                      |
| **Test suite**         | **1,311 passed / 2 skipped** (`tests/` + `agent_core/tests/`, verified 2026-08-22) |
| **LLM providers**      | LM Studio (local, OpenAI-compatible) · opencode-go (hosted, native tool calling)   |
| **Parallel multi-LLM** | Simultaneous dispatch to different models with per-model expert roles              |
| **Self-improvement**   | Trace → diagnose → repair → gated acceptance loop (HarnessFix)                     |
| **Language / Runtime** | Python 3.12+ · asyncio-native                                                      |

---

## Table of Contents

1. [What Makes It Different](#what-makes-it-different)
2. [Architecture Overview](#architecture-overview)
3. [Natural-Language Agent with Native Tool Calling](#natural-language-agent-with-native-tool-calling)
4. [The 24 REPL Commands](#the-24-repl-commands)
5. [Workflow Pipeline: Analysis → Implementation](#workflow-pipeline-analysis--implementation)
6. [Safety & Verification Layers](#safety--verification-layers)
7. [Multi-Provider LLM Layer](#multi-provider-llm-layer)
8. [Parallel Multi-LLM Dispatch (`multillm`)](#parallel-multi-llm-dispatch-multillm)
9. [Vision Input](#vision-input)
10. [Memory, Context & Design Decisions](#memory-context--design-decisions)
11. [Observability: Tracing, HTIR & Dashboards](#observability-tracing-htir--dashboards)
12. [HarnessFix: The Self-Improvement Loop](#harnessfix-the-self-improvement-loop)
13. [Human Review Gate & Regression Export](#human-review-gate--regression-export)
14. [Security Primitives](#security-primitives)
15. [Multi-Agent Framework & Swarm Orchestration](#multi-agent-framework--swarm-orchestration)
16. [Quick Start](#quick-start)
17. [Configuration](#configuration)
18. [Supported Models](#supported-models)
19. [Testing & Quality](#testing--quality)
20. [Project Structure](#project-structure)

---

## What Makes It Different

Most agent demos stop at "the LLM called a tool." Agent1 is built around what happens *after* that:

- **Actions, not descriptions** — native OpenAI-format tool calling forces the model to actually execute. It cannot say *"I will read the file"* without reading it. Every call and result is printed live.
- **Verified writes** — every `write`/`edit` runs through `py_compile`; the result is appended to the tool output so the model reports *verified* changes, not hoped-for ones.
- **Errors feed back** — a failed tool returns its error text into the conversation so the model can retry with a different approach instead of stalling.
- **Convergence-based progress guards** — long tasks are not killed for "not mutating files": read-only audits run to completion because progress means *discovering something new*, not just writing.
- **Never ends without an answer** — forced synthesis with retry plus a concrete fallback (loop statistics, last action, termination reason) when a large context produces an empty response.
- **A closed improvement loop** — failed runs leave JSONL traces; those traces are compiled, diagnosed, repaired, and only accepted if the full test suite, security gates, and benchmarks pass.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                       agent.py (REPL)                        │
│                                                              │
│   user input ──► CommandRegistry (24 commands)               │
│        │                                                     │
│        └──► anything else ──► chat_nlp()                     │
│                                  │                           │
│                   ┌──────────────▼───────────────┐           │
│                   │   ToolLoopRunner             │           │
│                   │   (native tool_calls loop)   │           │
│                   │   convergence-based guards   │           │
│                   └───────────────┬──────────────┘           │
│                                   │                          │
│            ┌──────────────────────▼───────────────────────┐  │
│            │  ToolDispatcher ── 12 handlers on the Agent  │  │
│            │  read · write · edit · search · list_files   │  │
│            │  run · git · diff · tests · fix · analyze    │  │
│            │  web_search                                  │  │
│            └──────────────────────┬───────────────────────┘  │
└───────────────────────────────────┼──────────────────────────┘
                                    │
                 ┌──────────────────▼──────────────────┐
                 │      agent_core/llm/provider.py     │
                 │   build_provider(model prefix)      │
                 ├─────────────────┬───────────────────┤
                 ▼                 ▼                   ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐
        │ LMStudioProvider│ │ OpencodeProvider│ │ run_parallel()   │
        │ localhost:1234  │ │ hosted API      │ │ N providers,     │
        │ OpenAI-compat   │ │ native tools    │ │ asyncio.gather   │
        └─────────────────┘ └─────────────────┘ └──────────────────┘
```

Supporting subsystems: `harnessfix/` (trace corpus, diagnosis, repairs, gates), `agent_core/security/` (path sandboxing, sanitizers, secret manager, cleanup-aware shutdown), `src/agent1/` (multi-agent framework), `agent1/swarm/` (thread-pool swarm orchestrator).

---

## Natural-Language Agent with Native Tool Calling

Any REPL input that doesn't match a command goes to the LLM with the full persisted conversation history. The model receives JSON schemas for exactly the tools the dispatcher can execute — the schema set and executable set **cannot drift apart** because both derive from `agent_core/tool_schemas.py`.

```
> What safety guards does the implement command use?
  [tool] read(path=agent_core/commands/implement_cmd.py)
  [result] ... (file contents fed back to the model)
The implement command guards against stdlib shadowing, class-name
collisions, and unwired modules (see agent_core/commands/implement_cmd.py).
```

- **Visible execution** — every call prints `[tool] name(args)` and every result prints `[result] ...`
- **Verified writes** — `py_compile` summary appended after `write`/`edit`
- **Error recovery** — tool errors return to the model for a different approach
- **Conversation persistence** — history survives restarts in `chat_history.json`

## The 24 REPL Commands

All commands are registered in `agent.py` against `CommandRegistry` (`agent_core/commands/registry.py`):

| Category                | Commands                                                                                                                              |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **File operations**     | `read` · `write` · `search` · `cleanup`                                                                                               |
| **Code intelligence**   | `analyze` (follows imports, `--deep`, `--stdin`) · `display`                                                                          |
| **Generation pipeline** | `plan` · `entities` · `taskplan` · `implement` · `workflow`                                                                           |
| **Repair**              | `fix` (traceback cascade analysis or on-demand `--desc`) · `self_heal` (patch-and-test until green, auto-revert non-improving rounds) |
| **Optimization**        | `optimize` (mechanical perf/memory fixes with side-by-side diff preview) · `perf` (per-command timing dashboard, `--html`)            |
| **Models**              | `model` (list/load/unload/reload/profile across both providers)                                                                       |
| **Multi-LLM**           | `multillm` (simultaneous multi-model dispatch with roles & tools)                                                                     |
| **Memory & knowledge**  | `clear` (stats + reset) · `decide` (record/search/check/resolve design decisions)                                                     |
| **Quality process**     | `review` (human gate over failed task traces) · `reconstruct` (rebuild files from JSONL traces)                                       |
| **Direct execution**    | `run` (LLM-free shell execution)                                                                                                      |
| **Multimodal**          | `paste_image` (clipboard/file image for vision-capable LLMs) · `paste`                                                                |

Unmatched input falls through to natural language — the REPL never dead-ends on syntax.

## Workflow Pipeline: Analysis → Implementation

`workflow` chains `analyze → plan → entities → taskplan → implement` with three entry modes:

| Mode       | Flag                                    | Behavior                                                                                                                                                |
| ---------- | --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Greenfield | `--desc` / `--stdin` / `--from spec.md` | Spec analysis first (scope, assumptions, risks, dependencies, threat model, blockers) before planning; halts with clarifying questions unless `--force` |
| Brownfield | `workflow .`                            | Five-dimension audit: CODE QUALITY, COMPLETENESS, ARCHITECTURE, INNOVATION, PRODUCTION (+`--brainstorm` sixth dimension)                                |
| Autonomous | `--auto`                                | Skips confirmations, runs implement inline; `--continue` resumes the newest run under `.docs/<ts>/`                                                     |

Pipeline hardening:

- **Code-claim verification** — every file path, symbol, line number, and snippet in generated analyses is checked against the real workspace; unverifiable claims are flagged in a `## Verification Report`
- **Reasoning-leak stripping** — chain-of-thought markers, thinking tags, and self-correction artifacts are removed before any file is written
- **Smart context truncation** — per-file excerpts around filename mentions instead of repeating full documents (~50K → ~4K tokens per batch)

## Safety & Verification Layers

**`implement` — six defense layers**

1. Prompt-level prevention of bare root-level filenames
2. Stdlib-shadowing detection (`logging/`, `json/`, `types/`…) with auto-redirect to safe alternatives
3. Collision warnings at taskplan time (existing class/function/filename scan)
4. Post-write rejection of class-name conflicts (auto-delete, no manual cleanup)
5. Static auto-review after every run (conflicts, collisions, unwired modules)
6. SOLID enforcement: SRP prompts, 150-line cap on new files, composition over inheritance

**Patch discipline**

- Minimal `[PATCH:]` diffs preferred over wholesale rewrites — changed lines shown, `y/N` before applying, old lines must exist, result must compile
- Anchored fallback matcher relocates wrong `@@` hunks by matching actual file content in a ±60-line window (absorbs fence-wrapped diffs, fused headers, `N |` numbered contexts)
- `--modify` mode merges into compile-OK modules via reviewed unified diff; similarity ≥ 0.5 required for rewrites, `--allow-rewrite` to opt in

**`fix` — traceback cascade analysis**

- Targets the *first user file* in the import chain (skips stdlib and `<frozen>` frames)
- Detects stdlib shadowing and refuses writes under the Python installation directory
- On-demand mode scores candidate files by keyword relevance, sends top 5; the model requests more with `[READ: path]` across up to 3 rounds

**`self_heal`** — proposes an LLM patch for failing tests, re-runs the suite, and **auto-reverts** any round that does not reduce the failure count, chaining until green or round budget exhausted.

## Multi-Provider LLM Layer

| Provider                     | Transport                                          | Notes                                                                    |
| ---------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------ |
| **LM Studio** (`lmstudio`)   | OpenAI-compatible `/chat/completions` on localhost | Local models, zero cloud dependency                                      |
| **opencode-go** (`opencode`) | Hosted API, direct mode + serve mode               | Native tool calling; key from `OPENCODE_API_KEY`, env var, or OS keyring |

- Provider selection persists in `model.json`; model switches rebuild the provider
- **Model profiles** (`agent_core/llm/model_profiles.py`): temperature/max-tokens/thinking presets per task type (`fast-codegen`, `deep-analysis`, …)
- **Meta-policy evolution** (`meta_policy.py`): profile-selection weights adapt from tracked performance data with decay
- **Retry policy** with exponential backoff; provider errors surface the gateway body, never a silent failure
- **Safe thinking-disable**: `reasoning: "off"` plus only per-model `chat_template_kwargs` validated against known jinja templates — aggressive knob sets were removed after probing showed they burn the entire output budget on `reasoning_content`

## Parallel Multi-LLM Dispatch (`multillm`)

Fire the same question at multiple models **simultaneously** — one provider instance per model, all calls concurrent via `asyncio.gather`:

```
multillm "review agent.py, and synthesize the two models' responses" \
  --models laguna-s-2.1,opencode-go/deepseek-v4-flash \
  --role "laguna-s-2.1:You are a security auditor..." \
  --role "opencode-go/deepseek-v4-flash:You are a performance engineer..."
```

- **Real parallelism, regression-tested** — blocking HTTP is dispatched via `asyncio.to_thread`; a test drives the actual REPL parse path with two 0.2 s-latency fake providers and asserts wall time < 0.35 s (it fails at ~0.47 s if the calls serialize)
- **Per-model roles** — each model gets its own system prompt (repeatable `--role`, or `--role-file roles.json`)
- **Tools for every model** — each model runs the same `ToolLoopRunner` the agent uses, with its own loop instance (no cross-model state contamination): they read/search/list files and run tests instead of asking you to paste code
- **Error isolation** — providers return error strings rather than raising; `return_exceptions=True` means one dead server cannot abort the other answers
- **Consensus wiring** — every verdict is recorded under a template id; approval-ratio quorum exposed via `ParallelRun.quorum_reached()` (ConsensusVoter / RefinementVoter machinery finally has a real producer)

## Vision Input

`paste_image` grabs an image from the clipboard (Pillow `ImageGrab`) or a file path, base64-encodes it as a data URL, and sends it to vision-capable models as an OpenAI-format multimodal content block alongside an optional prompt. Image blobs are **never persisted** — they are stripped from saved history while mixed text+image turns keep their text.

## Memory, Context & Design Decisions

- **Persistent conversation** across sessions with orphan-tool-message trimming (strict gateways reject assistant `tool_calls` whose results were cut — Agent1 drops the orphans at trim time *and* in a provider-level sanitizer for old histories)
- **Stale-file detection** — files read are tracked with mtimes; external changes invalidate cached knowledge and the REPL warns when imported modules changed on disk mid-session (`freshness`)
- **Working memory, semantic index, knowledge graph** — inspectable via `clear stats`
- **Decision tracking** (`decide`) — record design decisions with rationale/tags/files; past decisions are injected as **hard constraints** into LLM prompts; LLM-powered contradiction checking (`decide check`) and resolution (`decide resolve`); auto-extraction from workflow/implement/fix runs

## Observability: Tracing, HTIR & Dashboards

- **JSONL tracing** — every loop event (task begin, tool call, tool result, LLM response, forced synthesis) is traced under a correlation ID (`contextvars`-based propagation) into `reports/traces/*.jsonl`
- **HTIR** — traces compile into a hierarchical intermediate representation for diagnosis
- **Trace dashboard** — `python -m harnessfix.dashboard` summarizes recent traces and diagnoses
- **Performance dashboard** — `perf --detail|--reset|--html` reports per-command timing (HTML output included in repo root)
- **Reconstruction** — `reconstruct` replays write/edit operations from trace logs in timestamp order to rebuild lost files (`--dry-run`, `--search`, workspace targeting)

## HarnessFix: The Self-Improvement Loop

`python -m harnessfix.loop` runs the closed loop end to end:

```
1. Collect traces          reports/traces/
2. Compile + diagnose      HTIR → layer attribution → reports/harnessfix/diagnoses
3. Propose repair          highest-frequency diagnosed layer → catalog repair
4. Human review gate       fail-closed headless; --approve to proceed
5. Apply + run gates       pytest suite + security-primitive checks + benchmark pass-rate
6. Accept iff everything passes   benchmark must not regress
7. Verdict                 reports/harnessfix/summary.json
```

Gate and scorer code is never modified by a repair run. Evolution metrics use sliding windows — sustained low performance triggers an evolution decision.

## Human Review Gate & Regression Export

`review` turns failed traces into a curated, labeled ledger:

| Subcommand                                         | Purpose                                                               |
| -------------------------------------------------- | --------------------------------------------------------------------- |
| `review refresh`                                   | Rebuild the ledger from the trace corpus                              |
| `review list` / `show <task>`                      | Table view / full record (prompt, model, effects, diagnosis, outcome) |
| `review label <task> <bug\|regression\|noise\|ok>` | Human classification                                                  |
| `review label <task> auto` / `review auto`         | Evidence-based agent labeling when a human can't decide               |
| `review export <task>`                             | Emit a **diagnosis-pinning pytest regression file**                   |

Labels feed back into the improvement loop — bugs become pinned regression tests, noise stops polluting the corpus.

## Security Primitives

Located in `agent_core/security/`:

- **`path_utils`** — deterministic workspace-boundary resolution; traversal (`..`) and absolute escapes blocked (regression-tested)
- **`sanitizer`** — payload hygiene for provider requests (orphan tool messages, role ordering)
- **`allowlist`** — explicit permit lists for dangerous operations
- **`secrets`** — tiered secret manager: OS keyring (Windows Credential Manager / macOS Keychain) with Fernet-encrypted file fallback — replaces plaintext `.env` API keys
- **`shutdown`** — cleanup-aware SIGBREAK handler: best-effort memory save + trace flush/close before exit (each hook isolated so one failure can't mask shutdown)
- **Destructive-command screening** for the `run` tool, plus `benchmarks/security_benchmarks.py` wired into the HarnessFix acceptance gates

## Multi-Agent Framework & Swarm Orchestration

Beyond the single REPL agent, the multi-agent framework now lives inside
`agent_core/` (the retired `src/agent1/` and root `agent1/` packages were
merged in and deleted):

| Module                     | Contents                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------- |
| `orchestration`            | AgentMessage, TaskNode, DependencyGraph (networkx + fallback), dependency-aware TaskScheduler |
| `memory`                   | MemoryStore with caching, SQLiteStorage backend, EmbeddingService + VectorDatabase      |
| `plugins`                  | BasePlugin, PluginRegistry, PluginManager with lifecycle management                     |
| `monitoring`               | MetricsCollector, DashboardAPIServer (live web dashboard), AlertSystem                  |
| `swarm_orchestrator`       | thread-pool swarm coordinator (up to 20 workers) dispatching tasks across agents        |
| `evolution_metrics`        | sliding-window execution scoring with threshold-triggered evolution decisions           |
| `llm/async_provider`       | async LLM provider ABC with rate limiting and exponential-backoff retries               |

Integration tests cover multi-agent memory isolation, broadcast, and message routing; performance tests cover scaling.

## Quick Start

**Prerequisites:** Python 3.12+ and either LM Studio running locally or an opencode-go API key.

```bash
git clone <repo-url>
cd Agent1
pip install -r requirements.txt
python agent.py
```

> `requirements.txt` mirrors the runtime dependencies declared in
> `pyproject.toml`; for development use `pip install -e .[dev]` instead.

```
> Summarize this repository and list its five most complex modules
  [tool] list_files(path=.)
  [tool] read(path=agent.py)
  ...
Agent1 is a Python AI coding agent with 24 commands...

> switch to the hosted model
  model opencode-go/deepseek-v4-flash
```

Try the pipeline on any target workspace:

```
workflow .                      # brownfield 5-dimension audit
workflow --desc "a CLI todo app with sync"   # greenfield from description
workflow --from spec.md --auto  # autonomous from a specification
```

## Configuration

Environment variables (or `.env` — see `.env.example`):

```env
AGENT_MODEL=laguna-s-2.1
AGENT_LLM_PROVIDER=lmstudio        # or opencode ("opencode-go/..." model prefix wins)
LMSTUDIO_URL=http://localhost:1234/v1
OPENCODE_API_KEY=sk-...            # hosted opencode-go API (also read from OS keyring)
OPENCODE_TIMEOUT=600
AGENT1_SECRETS_DIR=                # redirect encrypted-secret store (tests/portable)
```

## Supported Models

| Model                           | Provider | Description                                              |
| ------------------------------- | -------- | -------------------------------------------------------- |
| `laguna-s-2.1`                  | lmstudio | Laguna S 2.1 MoE A8B — fast agentic coding               |
| `qwen3.6-27b-mtp`               | lmstudio | Qwen 3.6 27B — chat, codegen, large context              |
| `qwen/qwen3.8-27b`              | lmstudio | Qwen 3.8 27B — reasoning (safe minimal thinking-disable) |
| `google/gemma-4-31b`            | lmstudio | Gemma 4 31B — chat, reasoning, fast generation           |
| `opencode-go/deepseek-v4-flash` | opencode | Hosted default with native tool calling                  |

Switch at runtime: `model <name>` · `model list` · `model reload`. The active choice persists in `model.json`.

## Testing & Quality

```bash
pytest tests/ -v --no-cov          # main suite
pytest agent_core/tests -q --no-cov  # entry-point package tests
python -m pytest tests/ agent_core/tests -q --no-cov   # everything
```

Current status (**verified 2026-08-22**, full run ≈ 105 s):

```
1311 passed, 2 skipped in 104.47s
```

Coverage highlights: parallel-dispatch wall-clock regression tests, provider payload sanitization, orphan-tool-message handling, path traversal blocking, stdlib-shadowing protection, patch-application edge cases (strict / anchored / deletion-only / `N |` prefixes), forced-synthesis fallbacks, timeout process-tree kills, multi-agent isolation, and scaling benchmarks.

Static typing is enforced with mypy strict mode (`pyproject.toml`: `strict = true`, `disallow_untyped_defs`).

## Project Structure

```
Agent1/
├── agent.py                  # Main agent + REPL (24 commands registered)
├── tool_router.py            # Pydantic-validated tool routing & function-call parsing
├── benchmark.py              # Benchmarking harness
├── agent_core/
│   ├── llm/                  # Providers, tool loop, parallel dispatch, profiles,
│   │                         #   meta-policy evolution, retry, consensus voters
│   ├── commands/             # 24 command implementations + registry
│   ├── security/             # Path sandbox, sanitizer, allowlist, secrets, shutdown
│   ├── routing/              # Async-safe message bus
│   ├── diff/                 # Semantic diff parsing
│   ├── state_machine/        # Snapshot & transition validation
│   ├── tool_schemas.py       # Single source of truth: schemas == executable set
│   ├── tool_dispatcher.py    # Registry-based tool dispatch
│   ├── path_utils.py         # Workspace-aware path normalization
│   ├── file_system.py        # File I/O
│   ├── file_searcher.py      # Deterministic pure-Python walker
│   └── context_management.py # Correlation-ID propagation
├── harnessfix/               # Self-improvement loop: tracing, HTIR, diagnose,
│                             #   repairs, gates, review, evolution metrics, dashboard
├── fixcommand/               # Enhanced executor: structured parser, prompt generator
├── src/agent1/               # Multi-agent framework (core/memory/orchestration/
│                             #   plugins/monitoring)
├── agent1/                   # Swarm orchestrator, evolution metrics, async provider
├── docs/                     # AGENT_ARCHITECTURE, HARNESSFIX_SPEC/PLAN, REVIEW_GUIDE
├── tests/                    # 49 test files (unit/integration/performance)
└── USAGE.md                  # User manual
```

## Documentation

| Document                                                   | Contents                                    |
| ---------------------------------------------------------- | ------------------------------------------- |
| [`README.md`](README.md)                                   | Feature reference and command documentation |
| [`USAGE.md`](USAGE.md)                                     | User manual with examples                   |
| [`Architecture.md`](Architecture.md)                       | Layer-by-layer architecture walkthrough     |
| [`docs/AGENT_ARCHITECTURE.md`](docs/AGENT_ARCHITECTURE.md) | System diagrams and entry-point internals   |
| [`docs/HARNESSFIX_SPEC.md`](docs/HARNESSFIX_SPEC.md)       | Self-improvement loop specification         |
| [`docs/REVIEW_GUIDE.md`](docs/REVIEW_GUIDE.md)             | Trace-labeling methodology                  |
| [`CHANGES.md`](CHANGES.md)                                 | Full change log with rationale per entry    |

---

*All statistics on this page (command count, tool count, test results) were verified against the working tree on 2026-08-22.*
