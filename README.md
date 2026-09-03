# Agent1

A REPL-based AI coding agent. Ask questions in natural language or use explicit commands — the agent reads, writes, searches, and edits your code with verified tool calls.

```
> What does the ToolRouter class do?

  [tool] read(path=agent_core/llm/tool_router.py)
  [result] ... (file contents)

The ToolRouter class routes natural language prompts to registered tools
using a two-phase strategy: regex patterns for argument extraction, then
intent keyword matching as fallback.
```

---

## Quick Start

### 1. Install

```bash
git clone <repo-url> && cd Agent1
pip install -e ".[dev]"
```

Requires **Python 3.12+**.

### 2. Choose a provider

You need one LLM backend. Pick whichever is convenient:

| Provider | Setup | Best for |
|---|---|---|
| **Cloud (zero-install)** | Get a free API key from [opencode.ai](https://opencode.ai) — no server needed | Fastest start, no local GPU |
| **OpenRouter** | Get a key from [openrouter.ai](https://openrouter.ai) — choose from hundreds of models | Model variety |
| **LM Studio (local)** | Install [LM Studio](https://lmstudio.ai), download a model, start the server on port 1234 | Privacy, offline use |

### 3. Configure

Copy the example config and fill in your chosen provider:

```bash
cp .env.example .env
```

For the **cloud option** (easiest), edit `.env` and set just one key:

```
OPENCODE_API_KEY=sk-your-key-here
```

For **LM Studio**, no `.env` changes are needed — just start LM Studio with a model loaded on `http://localhost:1234`.

### 4. Run

```bash
python agent.py
```

You'll get an interactive prompt. Try asking something:

```
> hello — what can you do?
> read agent.py
> search ToolRouter
```

Type `quit` or `exit` to leave.

---

## What the Agent Can Do

**Talk to it naturally** — anything that doesn't match a command goes to the LLM with full tool access:

```
> Find all places where ConfigurationError is raised
> How does the implement command prevent file collisions?
> Add a docstring to the _handle_execute method
```

**Use commands** for direct operations:

| Command | What it does |
|---|---|
| `read <path>` | Read a file |
| `write <path> <text>` | Write content to a file |
| `search <query>` | Search all files for a string |
| `analyze <file>` | AI code review with import following |
| `fix <traceback>` | Paste an error → get an auto-fix |
| `fix <file> --desc "issue"` | Describe a problem → agent fixes it |
| `workflow <target>` | Full pipeline: analyze → plan → implement |
| `implement <taskplan>` | Generate files from a task plan |
| `model list` | Show available LLM models |
| `model <name>` | Switch model |
| `mode plan` | Read-only research mode (no writes) |
| `mode build` | Full access mode (default) |
| `clear` | View and clear memory |

Run `help` inside the REPL for the full list.

---

## Provider Details

The agent supports multiple LLM providers with automatic failover. The provider is selected by model name prefix, `.env` setting, or the last `model` command.

### Cloud (opencode-go)

```
OPENCODE_API_KEY=sk-...
```

Default model: `opencode-go/deepseek-v4-flash`. The free tier (`opencode-zen/hy3-free`) works without an API key — set `AGENT_LLM_PROVIDER=opencode` and the agent auto-selects a free model.

### OpenRouter

```
OPENROUTER_API_KEY=sk-or-...
```

Use any model: `openrouter/anthropic/claude-3.5-haiku`, `openrouter/meta-llama/llama-3.1-8b-instruct:free`, etc.

### LM Studio (local)

Start LM Studio, load any model, and ensure the local server runs on `http://localhost:1234`. Default model: `laguna-s-2.1`.

### llama.cpp (local)

Point to your llama-server instance:

```
AGENT_LLM_PROVIDER=llama
```

Default endpoint: `http://localhost:8080/v1`.

### Failover chain

The agent automatically retries on a different provider when the active one is unreachable. Configure the chain in `.env`:

```
AGENT_LLM_PROVIDERS=opencode:opencode-zen/hy3-free,opencode:opencode-go/deepseek-v4-flash,lmstudio,llama
```

---

## Working with the Agent

### Natural language

The agent uses native OpenAI-format tool calling — the LLM must emit a structured tool call or a text answer. It cannot just *say* "I'll read the file":

```
> What safety guards does the implement command use?

  [tool] read(path=agent_core/commands/implement_cmd.py)
  [result] ... (1487 lines returned)

The implement command includes 6 layers of file safety: stdlib shadowing
detection, class-name collision prevention, post-write rejection...
```

- **Verified writes** — after every `write`/`edit`, a `py_compile` check confirms the file is valid Python.
- **Error recovery** — tool errors feed back to the model, which can retry with a different approach.
- **Conversation history** — follow-up questions build on previous answers.

### Read-only research mode

Switch to plan mode to prevent any file modifications while exploring a codebase:

```
> mode plan
Plan mode — read-only tools active (search, read, list_files, diff, web_search)

> How would you add retry logic to the provider?
  [tool] search(query="retry")
  ... (agent researches without touching files)

> mode build
Build mode — full toolset restored.
```

---

## Web Dashboard

Start the live metrics dashboard alongside the REPL:

```bash
python agent.py --dashboard     # REPL + dashboard on http://localhost:8081
python agent.py --serve         # dashboard only
```

Seed it with demo data (no LLM needed):

```
> demo_data                 # 5 activities → all views populated
> demo_data --count 20      # more data
> demo_data --clear         # wipe
```

---

## Development

### Project layout

```
agent.py                      # REPL + entry point
agent_core/
  llm/
    lmstudio.py               # LM Studio provider
    opencode_provider.py      # Cloud opencode-go provider
    openrouter_provider.py    # OpenRouter provider
    llama_provider.py         # llama.cpp provider
    provider.py               # Provider protocol + failover
    tool_loop.py              # Agentic tool loop runner
    model_profiles.py         # Temperature/max-tokens presets
  tool_schemas.py             # Tool schemas sent to the LLM (single source of truth)
  tool_dispatcher.py          # Schema-based tool dispatch
  commands/                   # All REPL commands (read, write, search, workflow, ...)
  constants.py                # Model catalog, defaults, settings
harnessfix/                   # Self-improvement loop (traces, diagnostics, repairs)
tests/                        # ~1448 tests (pytest)
```

### Testing

```bash
pytest tests/ -v              # full suite
pytest tests/ -x -q --no-cov  # fast, stop on first failure
python -m pytest agent_core/tests -q --no-cov  # entry-point package tests
```

### Configuration

All config lives in `.env` (see `.env.example` for the full annotated list) and `model.json` (persisted model/provider selection). Key variables:

```
AGENT_LLM_PROVIDER=opencode        # Provider: opencode | lmstudio | openrouter | llama
OPENCODE_API_KEY=sk-...            # Cloud API key (or use LM Studio locally)
AGENT_DISPLAY_MODE=clean           # Tool output: verbose | clean | quiet
LOG_LEVEL=INFO                     # Logging: DEBUG | INFO | WARNING | ERROR
```

---

## License

MIT
