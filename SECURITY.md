# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.0 (Alpha) | ✅ Yes |

Agent1 is currently in **early alpha**. Security features are being actively developed. Expect limitations and report issues generously.

---

## Reporting a Vulnerability

**Do not open a public GitHub Issue for security vulnerabilities.**

### Options

1. **GitHub Security Advisories** (preferred): Use the "Report a vulnerability" button on the [Security tab](../../security/advisories/new) for private disclosure.

2. **Email**: Contact the maintainers at the address listed in the repository's `pyproject.toml` or GitHub profile. Include:
   - A description of the vulnerability
   - Steps to reproduce
   - Potential impact assessment
   - Any suggested fixes (if you have them)

### What to expect

- **Acknowledgement** within 72 hours of your report.
- **Status update** within 7 days while we triage.
- We will coordinate disclosure timing with you before making any public announcement.

We welcome reports on all versions — since only 0.1.0 exists, there's no need to check.

---

## Security Best Practices for Users

### API Key Management

Agent1 requires API keys for LLM providers. Handle them carefully:

- **Use `.env`** — copy `.env.example` to `.env` and fill in your keys. The `.env` file is already in `.gitignore` and will not be committed.
- **Never commit API keys** — not in code, not in commit messages, not in issue titles.
- **Use environment-specific keys** — don't share production keys with development environments.
- **Rotate keys regularly** — especially if you suspect exposure.
- **Prefer restricted keys** — if your provider supports scope-limited keys, use the most restrictive set that works.

### Workspace Sandboxing

Agent1 operates on a configurable workspace directory (`WORKSPACE_ROOT` in `.env`):

- **Point it at a project directory**, not your home directory or system root.
- **Use version control** — keep your workspace under git so you can audit and revert changes.
- **Back up before large operations** — commands like `implement`, `fix`, and `workflow` modify multiple files.
- **Review changes** — run `git diff` after agent operations to inspect what changed.

### Trust Boundaries

The agent's tool-calling loop mediates between the LLM (which generates intentions) and the filesystem/network (which executes them):

- **The LLM decides what to do; the tool layer does it.** Verify tool schemas match your expectations (defined in `agent_core/tool_schemas.py`).
- **Plan mode is read-only** — use `mode plan` when you want the agent to analyze without modifying files (enforced in `agent_core/modes.py`).
- **Speculative branches are sandboxed** — `speculate` branches run read-only tools only (`_BRANCH_TOOLS` allowlist in `speculate_cmd.py`).

---

## Known Security Considerations

Agent1 is a powerful tool with elevated capabilities. Be aware of these inherent risks:

### Code Execution

- The `run` tool executes shell commands in the workspace directory. Commands are run via `subprocess` with configurable timeouts.
- The agent can propose arbitrary commands through its tool-call loop — always review before approving in an interactive session.

### File System Access

- The agent reads, writes, edits, and deletes files within the configured workspace.
- Path traversal is checked (paths must resolve within `WORKSPACE_ROOT`), but the workspace itself is fully mutable.

### External API Calls

- LLM calls go to configured providers (LM Studio, opencode, OpenRouter, or llama.cpp).
- The `web_search` tool (when enabled) makes external HTTP requests.
- Network requests use `httpx` with configurable timeouts (`HTTP_TIMEOUT_CONNECT`, `HTTP_TIMEOUT_READ` in `.env`).

### Memory and Persistence

- `chat_history.json` stores full conversation history (including tool results).
- `agent_memory.json` stores semantic indices and knowledge graphs.
- Both are gitignored but contain sensitive context about your codebase.

---

## Dependency Security

### Pinned with lower bounds

All dependencies in `pyproject.toml` use `>=` constraints (e.g., `pydantic>=2.0`, `httpx>=0.24`). This means:

- You get the **latest compatible version** at install time.
- Version ranges are **not pinned to exact versions**, so run `pip list` or use a lockfile tool (`pip-compile`, `uv lock`) to capture your actual dependency versions.
- **Audit regularly** — run `pip-audit` or `safety check` periodically.

### Key dependencies

| Package | Purpose | Min Version |
|---|---|---|
| `pydantic` | Config validation, data models | ≥2.0 |
| `httpx` | HTTP client for LLM APIs | ≥0.24 |
| `openai` | OpenAI-compatible API client | ≥1.0 |
| `networkx` | Dependency graphs | ≥3.1 |
| `numpy` | Vector math for memory | ≥1.24 |
| `cryptography` | Optional secrets encryption | ≥41.0 |

### Reporting dependency vulnerabilities

If you discover a vulnerability in a transitive dependency, report it through the channels above. We will evaluate whether the vulnerability is exploitable in Agent1's usage context and update or mitigate as needed.

---

## Status

This policy applies to Agent1 **v0.1.0 (Alpha)**. As the project matures, this document will be updated with additional details on security guarantees, formal audit results, and hardened defaults.
