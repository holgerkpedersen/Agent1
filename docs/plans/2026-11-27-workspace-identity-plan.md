# Plan — Make the workspace explicit (one source of truth)

Status: **T1-T5 implemented** (2026-11-27)
Date: 2026-11-27

## Problem

The workspace was implicit at three layers and never stated to the model:

1. **The default came from the code, not the caller.** `AgentCore.DEFAULT_WORKSPACE`
   was `dirname(dirname(abspath(__file__)))` — Agent1's *own* repo. Started in
   `C:\Dev\Strata`, the agent silently operated on `C:\Dev\Agent1`.
2. **No flag and no switch.** `build_arg_parser()` offered only session and
   dashboard flags. The only override was `agent._nlp_workspace`, set solely by
   `paste --workspace`.
3. **The model was never told.** `_refresh_system_message` assembled goal,
   constraints, skills, habits, KG, semantic memory, file history, mode and
   language blocks — no workspace block. The model had to infer the root from
   tool results, which is how edits and tests get aimed at the wrong repo.
4. **Two sources disagreed.** `config.py` derived `workspace_root` from
   `AGENT_WORKSPACE_ROOT`, but `Agent.__init__` never read it — a documented
   setting that did nothing.

## The precedence decision

One chain, in `agent_core/workspace.py:83` (`resolve_workspace`):

| # | Source | Why in this position |
|---|---|---|
| 1 | `--workspace PATH` | The most explicit human intent available; must win over ambient state. |
| 2 | `AGENT_WORKSPACE_ROOT` | Per-environment intent; outranks guesses about the directory. |
| 3 | `cwd`, when it looks like a project (`.git/`, `pyproject.toml`) | Starting inside a checkout means you mean that checkout. |
| 4 | repository root | The old behaviour, kept last so no existing setup changes silently. |

Two deliberate properties:

- **A non-existent override is ignored with a warning, never silently replaced
  by the default.** Falling back quietly is exactly the failure mode this module
  exists to kill, so the override is dropped loudly instead.
- **The function never raises** and always returns an absolute path. It is
  called at startup and from `config.py`, where an exception would be fatal.

## Tasks

- **T1 — one resolver** (`agent_core/workspace.py`): `resolve_workspace()`,
  `looks_like_repo()`, `describe_workspace_mismatch()`.
  `AgentCore.DEFAULT_WORKSPACE` delegates to it; `config.py:587-589` calls the
  same function, so `AGENT_WORKSPACE_ROOT` is no longer a lie.
- **T2 — startup flag** (`agent_core/session/cli.py`): `--workspace PATH` in
  `build_arg_parser()`; `resolve_startup` passes the hint through
  `resolve_workspace`, and `_recorded_workspace()` makes a resumed session
  return to its own root instead of snapping back to the repo.
- **T3 — tell the model** (`agent_core/prompts/context_blocks.py`,
  `agent_core/run_utils.py`): `_workspace_block()` reports the absolute path,
  the git branch (or detached HEAD), and the rule *"All relative paths resolve
  against this root. Use an absolute path to touch anything else."*
  `WORKSPACE_MARKER` is registered in the `markers` tuple — mandatory, or the
  block accumulates one stale copy per turn. Empty for the default root, so a
  single-repo prompt stays byte-identical.
- **T4 — `workspace` command** (`agent_core/commands/workspace_cmd.py`): show /
  switch / reset. The switch sets `agent._nlp_workspace` (already honoured by
  every tool via `_effective_ws_dir`) and rebuilds the system prompt so the
  model learns the new root immediately.
- **T5 — guardrails**: warn once when the workspace differs from the shell's
  `cwd` (`describe_workspace_mismatch`), and when `implement` / `workflow`
  receive a `--workspace` different from the agent's
  (`describe_target_divergence`) — today that silently forks the command's
  world from the conversation's.

## Non-goals

- Changing the default. The fallback stays the repository root; the change makes
  the root *visible and overridable*, not different.
- Making the five per-command `--workspace` parsers into one. T5 makes the
  divergence visible; collapsing the parsers is separate work.

## Notes

- `build/lib/...` copies are build artifacts and must not be edited.

## Verification

- `tests/test_workspace_resolution.py` — precedence matrix, the
  "never returns the package dir when cwd is another repo" regression, the
  WORKSPACE block contract (one copy after two rebuilds, absent for default).
- `tests/test_workspace_command.py` — show/switch/reset, `_effective_ws_dir()`
  follows the switch, bad path falls back, mismatch and divergence guards.
- `python -m pytest tests/test_workspace_command.py tests/test_workspace_resolution.py tests/test_session_cli.py -q --no-cov`
  -> `52 passed`.
