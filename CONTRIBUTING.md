# Contributing to Agent1

Thanks for your interest in contributing! Agent1 is in early alpha — your feedback and contributions are invaluable.

---

## Development Setup

### Prerequisites

- **Python 3.12+** (3.10+ technically supported, but 3.12 recommended)
- **Git**
- An LLM backend (see [USAGE.md](USAGE.md) for options)

### Clone and install

```bash
git clone <repo-url> && cd Agent1
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `[dev]` extra installs all development tools: `ruff`, `mypy`, `pytest`, `pytest-cov`, `pytest-testmon`, `psutil`, and type stubs.

---

## Code Style

Agent1 enforces style through tooling — not manual review.

| Tool | Purpose | Command |
|---|---|---|
| **ruff** | Linting + import sorting | `ruff check .` |
| **ruff** | Auto-fix safe issues | `ruff check --fix .` |
| **mypy** | Static type checking | `mypy --strict` |

### Configuration

All settings live in `pyproject.toml`:

- **Line length**: 88 characters (`[tool.ruff] line-length = 88`)
- **Target**: Python 3.9+ (`[tool.ruff] target-version = "py39"`)
- **Lint rules**: `E`, `F`, `W` with `E741` (ambiguous variable names) suppressed globally
- **mypy**: `strict = true` — all functions must have type annotations, no implicit `Optional`, unreachable code flagged

### Before you commit

```bash
ruff check .                # lint
mypy --strict               # types
pytest                      # tests
```

All three must pass before submitting a PR.

---

## Running Tests

Agent1 uses **pytest** with coverage and integration markers.

| Command | What it runs |
|---|---|
| `pytest` | Full suite with coverage report |
| `pytest -q` | Quick run, minimal output |
| `pytest -m integration` | Integration tests only (require a local LLM server) |
| `pytest --testmon` | Smart mode — only tests affected by your changes |
| `pytest tests/test_agent.py` | Single file |

### Test markers

- `integration` — end-to-end tests that need a real `llama-server` binary on `PATH`; auto-skipped when none is found
- `harnessfix_self_test` — loop/repair self-tests that mutate tool internals; excluded from the autonomous test gate

### Coverage

Coverage is collected automatically (`--cov=agent_core --cov-report=term-missing`). The project targets meaningful coverage of the `agent_core` package.

---

## Pull Request Process

1. **Branch from `master`** — create a feature branch with a descriptive name (`feat/tool-xyz`, `fix/memory-leak`).
2. **Keep PRs focused** — one logical change per PR. Small, reviewable PRs are preferred over large sweeps.
3. **Write tests** — bug fixes should include a regression test; new features should include unit tests.
4. **Ensure CI passes** — `ruff check`, `mypy --strict`, and `pytest` must all be green.
5. **Update documentation** — if you change public behavior, update `USAGE.md` and/or `Architecture.md`. Add a `CHANGES.md` entry describing what changed and why.
6. **Request review** — open the PR against `master` and describe what changed, why, and how to test it.

### Commit messages

Follow the existing convention in `CHANGES.md`:

```
type: short imperative summary

Longer description of *why* the change was made.

Files: list of key files changed. Verified: test suite green.
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`.

---

## Reporting Issues

Open a [GitHub Issue](../../issues) with:

- **Steps to reproduce** the problem
- **Expected vs. actual behavior**
- **Environment** (OS, Python version, LLM provider/model)
- **Logs or error output** if applicable

For security vulnerabilities, please see [SECURITY.md](SECURITY.md).

---

## Architecture Overview

Before modifying core code, read [Architecture.md](Architecture.md) for an overview of the system's layers:

- **LLM Layer** (`agent_core/llm/`) — provider abstraction, tool loop runner, retry policy
- **Agent Core** (`agent.py`) — REPL, memory stores, NLP tool-call loop
- **Command System** (`agent_core/commands/`) — registry-based REPL commands
- **Tools** (`agent_core/tools/`) — NLP-callable tools (read, write, search, run, git, etc.)
- **Modes** (`agent_core/modes.py`) — session modes (`build`, `plan` read-only)

Key invariants are documented in `AGENTS.md` — read it if you're working on the core loop, tool dispatch, or memory system.

---

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
