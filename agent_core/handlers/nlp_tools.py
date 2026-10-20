"""NLP tool handlers extracted from Agent.

Each former ``Agent._nlp_*`` method is now a plain module-level ``async def``
that takes the owning :class:`Agent` instance as its first parameter.  Names
that still live in ``agent.py`` (helpers not yet extracted) are resolved
through the ``_stub_missing`` pattern below: agent-instance methods are
forwarded via ``getattr(agent, name)`` at call time (so monkeypatching and
subclass overrides keep working), and module-level names from ``agent.py``
are resolved lazily from the ``agent`` module; anything missing raises
``NotImplementedError`` and is replaced with a proper import once the full
chain of extractions is done.
"""
from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, Callable

from agent_core.shell_utils import run_subprocess_captured as _shell_run_subprocess_captured
from agent_core.pytest_cmd_policy import pytest_flag_conflict as _pytest_flag_conflict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent

logger = logging.getLogger(__name__)

# Type alias for the NLP tool handler callables (mirrors agent.NlpToolHandler).
NlpToolHandler = Callable[..., Any]


# ── Stub layer ──────────────────────────────────────────────────────────────

def _stub_missing(name: str) -> NotImplementedError:
    """Build the error raised by an unresolved cross-reference stub."""
    return NotImplementedError(f"stub: {name}")


def _agent_attr(agent: "Agent", name: str) -> Any:
    """Resolve an Agent-instance helper from agent.py (lazy, patch-friendly).

    For methods still owned by ``Agent`` (e.g. ``_resolve_nlp_path``,
    ``_note_effect``, ``_save_verify_note``, ``spawn_subagent``): dispatch
    through the instance so overrides/monkeypatches are honoured.  Raises
    ``NotImplementedError`` if the attribute does not exist yet.
    """
    attr = getattr(agent, name, None)
    if attr is None:
        raise _stub_missing(name)
    return attr


def _agent_module_attr(name: str) -> Any:
    """Resolve a module-level name from agent.py lazily (circular-import safe)."""
    import agent as _agent_mod  # late import: agent.py imports this module

    attr = getattr(_agent_mod, name, None)
    if attr is None:
        raise _stub_missing(name)
    return attr


# ── agent.py module-level constants (resolved lazily) ───────────────────────

def _const(name: str) -> Any:
    return _agent_module_attr(name)


def _MAX_ACTIVE_SUBAGENTS() -> Any:
    return _const("_MAX_ACTIVE_SUBAGENTS")


def _DELEGATE_TIMEOUT_S() -> Any:
    return _const("_DELEGATE_TIMEOUT_S")


def _MAX_READ_FILES() -> Any:
    return _const("_MAX_READ_FILES")


def _MAX_READ_LINES() -> Any:
    return _const("_MAX_READ_LINES")


def _CONTEXT_AST_THRESHOLD_KB() -> Any:
    return _const("_CONTEXT_AST_THRESHOLD_KB")


def _MAX_CONSECUTIVE_READS() -> Any:
    return _const("_MAX_CONSECUTIVE_READS")


def _READ_LOOP_NOTE() -> Any:
    return _const("_READ_LOOP_NOTE")


def _MAX_RUN_TIMEOUT_S() -> Any:
    return _const("_MAX_RUN_TIMEOUT_S")


def DEFAULT_PAGE_LINES() -> Any:
    return _const("DEFAULT_PAGE_LINES")


# ── agent.py module-level functions (resolved lazily) ───────────────────────

def _truncate_output(text: str, limit: int = 5000) -> str:
    return _agent_module_attr("_truncate_output")(text, max_lines=limit)


def _is_pytest_command(cmd: str) -> bool:
    return _agent_module_attr("_is_pytest_command")(cmd)


def _is_full_pytest_command(cmd: str) -> bool:
    return _agent_module_attr("_is_full_pytest_command")(cmd)


def _is_subset_pytest_command(cmd: str) -> bool:
    return _agent_module_attr("_is_subset_pytest_command")(cmd)


def _pytest_full_suite_timeout() -> Any:
    return _agent_module_attr("_pytest_full_suite_timeout")()


def _blocked_shell_command(cmd: str) -> Any:
    return _agent_module_attr("_blocked_shell_command")(cmd)


def _run_shell_command(cmd: str) -> list[str]:
    # The run tool's shell selector: argv that executes *cmd* in the shell the
    # system prompt advertises (see agent_core/shell_utils.run_shell_command).
    return _agent_module_attr("_run_shell_command")(cmd)


def _parse_ps_exit_marker(result: str) -> tuple[str, Any]:
    # Strip the PowerShell exit-code marker line from captured stdout and
    # return (cleaned, code) — see shell_utils.parse_ps_exit_marker.
    return _agent_module_attr("_parse_ps_exit_marker")(result)


def _shape_run_stderr(err: Any, output: Any, returncode: Any) -> str:
    # The agent-module-visible seam: combines output + cleaned stderr and
    # appends the [EXIT CODE: N] marker for non-zero returncodes.
    return _agent_module_attr("_shape_run_stderr")(err, output, returncode)


def _run_subprocess_captured(
    cmd: list[str],
    cwd: str = ".",
    timeout: int = 60,
    label: str = "Run",
):
    """Shared subprocess capture used by the nlp git/tests/diff handlers.

    This is the agent-module-visible seam (``agent._run_subprocess_captured``):
    it takes the handler *label* and returns ``(output, error_message)`` —
    on success *output* carries the captured stdout with ``[STDERR]`` lines
    tagged in, an empty run rendered as ``(no output)`` and the whole thing
    truncated; on failure *error_message* carries ``<label> timed out after
    Ns`` or ``<label> error: ...`` (the strings
    tests/test_agent_dry_refactor.py asserts on).
    """
    code, output, error = _shell_run_subprocess_captured(
        cmd, cwd=cwd, timeout=timeout
    )
    if code == -1:
        if error and error.lower().startswith("timed out"):
            return f"{label} timed out after {timeout}s", None
        return "", f"{label} error: {error}"
    output = output or ""
    if error:
        output = (output + "\n" if output else "") + f"[STDERR] {error}"
    if not output.strip():
        output = "(no output)"
    return _agent_module_attr("_truncate_output")(output), None


def _kill_process_tree(proc: Any) -> None:
    _agent_module_attr("_kill_process_tree")(proc)


def collect_definitions(source: str, filename: str = "") -> str:
    return _agent_module_attr("collect_definitions")(source, filename=filename)


def collect_references(symbol: str, ws_dir: str, max_results: int) -> str:
    return _agent_module_attr("collect_references")(symbol, ws_dir, max_results)


def read_text_preserving(path: str):
    return _agent_module_attr("read_text_preserving")(path)


def encode_in_encoding(text: str, encoding: str, path: str) -> None:
    _agent_module_attr("encode_in_encoding")(text, encoding, path)


def NonUtf8TextError() -> Any:  # noqa: N802 - exception class proxy
    return _agent_module_attr("NonUtf8TextError")


def UnrepresentableTextError() -> Any:  # noqa: N802 - exception class proxy
    return _agent_module_attr("UnrepresentableTextError")


def read_skill(ws_dir: str, name: str, offset: int, limit: int):
    return _agent_module_attr("read_skill")(ws_dir, name, offset=offset, limit=limit)


def SkillError() -> Any:  # noqa: N802 - exception class proxy
    return _agent_module_attr("SkillError")


def format_skill_page(page: Any) -> str:
    return _agent_module_attr("format_skill_page")(page)


def to_windows_path(path: str) -> str:
    return _agent_module_attr("to_windows_path")(path)


def record_full_run_seconds(seconds: float, completed: bool = True) -> None:
    _agent_module_attr("record_full_run_seconds")(seconds, completed=completed)


# ── Extracted handlers ──────────────────────────────────────────────────────

async def _nlp_delegate(agent: "Agent", args: dict[str, Any]) -> str:
    """Run one task in a role subagent and return its final answer.

    The child gets an isolated history (parent context stays small) and a
    restricted toolset; its tool calls flow through this parent's
    executor with ``_delegating`` set so child reads never inflate the
    parent turn's read-loop streak.  Concurrency is capped at
    ``_MAX_ACTIVE_SUBAGENTS``; a hung child surfaces as an error after
    ``_DELEGATE_TIMEOUT_S`` instead of holding the turn hostage.
    """

    from agent_core.modes import MODE_PLAN, is_plan_mode
    from agent_core.subagent_roles import get_role, role_names

    role = str(args.get("role", "")).strip().strip('"').strip("'")
    task = str(args.get("task", "")).strip()
    if not role or not task:
        return "Error: delegate requires 'role' and 'task'."

    spec = get_role(role)
    if spec is None:
        return (
            f"Error: unknown role '{role}'. "
            f"Available: {', '.join(role_names())}"
        )

    if len(agent._active_subagents) >= _MAX_ACTIVE_SUBAGENTS():
        return (
            f"Error: {_MAX_ACTIVE_SUBAGENTS()} subagents already running — "
            "wait for one to finish before delegating again."
        )

    agent._delegate_counter += 1
    name = f"{spec.name}-{agent._delegate_counter}"
    try:
        sub = agent.spawn_subagent(name, role=role)
    except Exception as exc:  # defensive: a broken spawn must not kill the turn
        logger.exception("Delegated subagent %s failed to spawn", name)
        return f"[delegate:{name}] failed to spawn: {exc}"
    agent._active_subagents.add(name)
    prev_delegating = agent._delegating
    agent._delegating = True
    try:
        result = await asyncio.wait_for(
            sub.respond(task), timeout=_DELEGATE_TIMEOUT_S(),
        )
    except asyncio.TimeoutError:
        result = (
            f"[delegate:{name}] timed out after {_DELEGATE_TIMEOUT_S():.0f}s "
            "— the task did not finish. Partial context preserved in the "
            "subagent; retry with a smaller task."
        )
    except Exception as exc:  # defensive: a broken child must not kill the turn
        logger.exception("Delegated subagent %s failed", name)
        result = f"[delegate:{name}] failed: {exc}"
    finally:
        agent._delegating = prev_delegating
        agent._active_subagents.discard(name)

    summary = sub.get_context_summary(max_messages=3)
    reason = getattr(sub, "last_termination_reason", None)
    reason_note = (
        "" if reason == "answer"
        else f", stopped early: {reason or 'unknown'}"
    )
    mode_note = ""
    if is_plan_mode(agent.mode):
        mode_note = (
            f"\n[plan mode] The '{spec.name}' child was capped to read-only "
            "(plan-mode parent); it reported findings only."
            if spec.mode != MODE_PLAN
            else ""
        )
    return (
        f"{result}\n\n{summary}\n"
        f"[delegate:{name}] finished ({spec.title}, {sub.mode} mode"
        f"{reason_note}).{mode_note}"
    )


async def _nlp_delegate_batch(agent: "Agent", args: dict[str, Any]) -> str:
    """Fan one task out to several roles in parallel and merge reports.

    Each child still runs alone in its own context; the parent only pays
    for the merged summaries.  The per-call cap is
    ``min(len(roles), _MAX_ACTIVE_SUBAGENTS)``.
    """
    from agent_core.subagent_roles import get_role, role_names

    raw_roles = args.get("roles") or []
    if not isinstance(raw_roles, list):
        return "Error: 'roles' must be a list of role names."
    roles = [str(r).strip() for r in raw_roles if str(r).strip()]
    task = str(args.get("task", "")).strip()
    if not roles or not task:
        return "Error: delegate_batch requires 'roles' (list) and 'task'."
    unknown = [r for r in roles if get_role(r) is None]
    if unknown:
        return (
            f"Error: unknown role(s): {', '.join(unknown)}. "
            f"Available: {', '.join(role_names())}"
        )
    # Dedupe (preserve order), then cap concurrency. Using `seen.add(r)`
    # inside a boolean `or` is a fragile idiom (set.add returns None), so we
    # spell the dedupe out explicitly.
    seen: set[str] = set()
    unique_roles: list[str] = []
    for r in roles:
        if r not in seen:
            seen.add(r)
            unique_roles.append(r)
    unique_roles = unique_roles[:_MAX_ACTIVE_SUBAGENTS()]
    dropped = len(roles) - len(unique_roles)

    agent._delegate_counter += 1
    batch_id = agent._delegate_counter

    async def run_one(role_name: str, idx: int) -> tuple[str, str]:
        name = f"{role_name}-b{batch_id}-{idx}"
        try:
            sub = agent.spawn_subagent(name, role=role_name)
        except Exception as exc:
            logger.exception("Batch subagent %s failed to spawn", name)
            return role_name, f"### {role_name}\nfailed to spawn: {exc}"
        prev = agent._delegating
        agent._delegating = True
        agent._active_subagents.add(name)
        try:
            res = await asyncio.wait_for(
                sub.respond(task), timeout=_DELEGATE_TIMEOUT_S(),
            )
        except asyncio.TimeoutError:
            res = (f"timed out after {_DELEGATE_TIMEOUT_S():.0f}s — no result.")
        except Exception as exc:
            logger.exception("Batch subagent %s failed", name)
            res = f"failed: {exc}"
        finally:
            agent._delegating = prev
            agent._active_subagents.discard(name)
        reason = getattr(sub, "last_termination_reason", None)
        reason_note = (
            "" if reason == "answer" else f" [stopped early: {reason}]"
        )
        return role_name, (
            f"### {role_name}{reason_note}\n"
            f"{res}\n\n{sub.get_context_summary(max_messages=2)}"
        )

    results = await asyncio.gather(
        *(run_one(r, i) for i, r in enumerate(unique_roles))
    )
    header = (
        f"[delegate_batch:{batch_id}] {len(unique_roles)} role(s) on: "
        f"{task[:120]}{'...' if len(task) > 120 else ''}"
    )
    if dropped:
        header += f" ({dropped} extra role(s) dropped — concurrency cap)"
    body = "\n\n".join(report for _role, report in results)
    return f"{header}\n\n{body}"


async def _run_command_quietly(
    agent: "Agent", command: Any, args: list[str], done_msg: str,
) -> str:
    """Run a registered command with stdout captured and return its output.

    Shared by the NLP ``fix`` and ``analyze`` tool handlers so the
    "capture → execute → fall back when empty" pattern lives in one place.
    """
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await command.execute(args, agent)
    return buf.getvalue() or done_msg


def _effective_ws_dir(agent: "Agent") -> str:
    """Resolve the subprocess cwd for NLP tools.

    The workspace override (``paste --workspace``) wins when it exists;
    a bad override (e.g. a Git-Bash /c/... path) falls back to the real
    workspace because subprocess cwd must be an existing directory.
    """
    ws_dir = os.path.abspath(
        to_windows_path(agent._nlp_workspace or agent.workspace)
    )
    if not os.path.isdir(ws_dir):
        ws_dir = agent.workspace
    return ws_dir


async def _nlp_create_subagent(agent: "Agent", args: dict[str, Any]) -> str:
    """Create a persistent subagent for the LLM to reuse."""
    from agent_core.subagent_roles import get_role, role_names
    name = str(args.get("name", "")).strip().strip('"').strip("'")
    role = str(args.get("role", "")).strip().strip('"').strip("'")
    workspace = args.get("workspace")
    if not name or not role:
        return "Error: create_subagent requires 'name' and 'role'."
    spec = get_role(role)
    if spec is None:
        return (
            f"Error: unknown role '{role}'. "
            f"Available: {', '.join(role_names())}"
        )
    # ``AgentCore.__init__`` owns ``_subagents`` (declared there, not
    # lazy-bootstrapped here: PEP 526 forbids annotated assignment to a
    # non-self target — ``agent._subagents: dict[...] = {}`` was a mypy [misc]
    # error and left the attribute undeclared on the class).
    if name in agent._subagents:
        return (
            f"Error: subagent '{name}' already exists. "
            "Use a different name or reset it first."
        )
    sub = agent.spawn_subagent(name, workspace=workspace, role=role)
    agent._subagents[name] = sub
    return (
        f"Created subagent '{name}' with role '{spec.name}' "
        f"({spec.title})."
    )


async def _nlp_run_subagent_task(agent: "Agent", args: dict[str, Any]) -> str:
    """Send a task to an existing named subagent."""
    name = str(args.get("name", "")).strip().strip('"').strip("'")
    task = str(args.get("task", "")).strip()
    if not name or not task:
        return "Error: run_subagent_task requires 'name' and 'task'."
    subs = getattr(agent, "_subagents", {})
    sub = subs.get(name)
    if sub is None:
        return (
            f"Error: subagent '{name}' not found. "
            "Create it with create_subagent first."
        )
    result = await sub.respond(task)
    summary = sub.get_context_summary(max_messages=3)
    return f"{result}\n\n{summary}"


def _nlp_tool_handlers(agent: "Agent") -> dict[str, NlpToolHandler]:
    """Name → handler map for every entry in :data:`NLP_TOOL_NAMES`.

    Built once per instance and cached (decision 120): the map holds
    plain ``self``-bound methods that read workspace state at call time,
    so it cannot go stale after ``_nlp_workspace`` changes, and no caller
    mutates the returned mapping.
    """
    cache: dict[str, NlpToolHandler] | None = getattr(
        agent, "_handlers_cache", None
    )
    if cache is None:
        cache = {
            "analyze": lambda a: _nlp_analyze(agent, a),
            "definitions": lambda a: _nlp_definitions(agent, a),
            "diff": lambda a: _nlp_diff(agent, a),
            "edit": lambda a: _nlp_edit(agent, a),
            "fix": lambda a: _nlp_fix(agent, a),
            "git": lambda a: _nlp_git(agent, a),
            "jev_decide": lambda a: _nlp_jev_decide(agent, a),
            "list_files": lambda a: _nlp_list_files(agent, a),
            "merge": lambda a: _nlp_merge(agent, a),
            # Late-bound through the Agent attribute so tests (and any
            # subclass) that patch Agent._nlp_read intercept this route.
            # The module-level function reference bypassed the patch and
            # the dispatcher's exception containment (test_handler_exception_is_contained).
            "read": lambda a: agent._nlp_read(a),
            "read_skill": lambda a: _nlp_read_skill(agent, a),
            "references": lambda a: _nlp_references(agent, a),
            "run": lambda a: _nlp_run(agent, a),
            "search": lambda a: _nlp_search(agent, a),
            "speculate": lambda a: _nlp_speculate(agent, a),
            "tests": lambda a: _nlp_tests(agent, a),
            "web_search": lambda a: _nlp_web_search(agent, a),
            "write": lambda a: _nlp_write(agent, a),
            "delegate": lambda a: _nlp_delegate(agent, a),
            "delegate_batch": lambda a: _nlp_delegate_batch(agent, a),
            "create_subagent": lambda a: _nlp_create_subagent(agent, a),
            "run_subagent_task": lambda a: _nlp_run_subagent_task(agent, a),
            "mcp_tools": lambda a: _nlp_mcp_tools(agent, a),
            "mcp_call": lambda a: _nlp_mcp_call(agent, a),
            "get_current_datetime": lambda a: _nlp_get_current_datetime(agent, a),
            "hue_control": lambda a: _nlp_hue_control(agent, a),
            "plan_status": lambda a: _nlp_plan_status(agent, a),
            "plan_start": lambda a: _nlp_plan_start(agent, a),
            "plan_step": lambda a: _nlp_plan_step(agent, a),
            "plan_finish": lambda a: _nlp_plan_finish(agent, a),
        }
        agent._handlers_cache = cache
    return cache


async def _nlp_mcp_tools(agent: "Agent", args: dict[str, Any]) -> str:
    """List MCP tools the user exposed to the LLM (opt-in per server)."""
    from agent_core.mcp.manager import get_manager
    catalog = get_manager().llm_catalog()
    if not catalog:
        return (
            "No MCP tools available: either no server is connected or none "
            "is exposed to you. The user opts a server in with "
            "'mcp expose <name> on'."
        )
    lines = ["Available MCP tools (server.tool):"]
    for server, tools in sorted(catalog.items()):
        for tool in tools:
            lines.append(f"  - {server}.{tool}")
    return "\n".join(lines)


async def _nlp_mcp_call(agent: "Agent", args: dict[str, Any]) -> str:
    """Invoke an MCP tool on an LLM-exposed server.

    The expose_to_llm gate is re-checked HERE (not just in llm_catalog)
    so a hallucinated server name can never reach a non-exposed one.
    Arguments are schema-validated inside the client before anything
    leaves the process; results are size-capped.
    """
    from agent_core.mcp.config import McpConfigError
    from agent_core.mcp.manager import McpManagerError, get_manager
    server = str(args.get("server", "")).strip().strip('"').strip("'")
    tool = str(args.get("tool", "")).strip().strip('"').strip("'")
    if not server or not tool:
        return "Error: mcp_call needs both 'server' and 'tool'."
    manager = get_manager()
    try:
        exposed = manager.llm_catalog()
    except Exception as e:
        return f"mcp error: {e}"
    if server not in exposed:
        return (
            f"Error: MCP server '{server}' is not exposed to you "
            f"(exposed: {', '.join(sorted(exposed)) or 'none'}). The user "
            f"controls exposure with 'mcp expose <name> on'."
        )
    arguments = args.get("arguments")
    if arguments is not None and not isinstance(arguments, dict):
        return "Error: 'arguments' must be a JSON object."
    try:
        # require_exposed=True re-validates the opt-in flag under the
        # manager lock - no TOCTOU window between gate check and call.
        return manager.call_tool(server, tool, arguments, require_exposed=True)
    except McpManagerError as e:
        return f"mcp error: {e}"
    except McpConfigError as e:
        return f"mcp config error: {e}"
    except Exception as e:
        return f"mcp error calling {server}.{tool}: {type(e).__name__}: {e}"


async def _nlp_definitions(agent: "Agent", args: dict[str, Any]) -> str:
    """Index a Python file's classes/functions via AST (plan item B-#8)."""
    path = _agent_attr(agent, "_resolve_nlp_path")(
        str(args.get("path", "")).strip('"').strip("'")
    )
    if not path.endswith(".py"):
        return f"Error: definitions needs a .py file, got: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except OSError as e:
        return f"Definitions error: {e}"
    _agent_attr(agent, "_note_effect")(path)
    return collect_definitions(source, filename=path)


async def _nlp_references(agent: "Agent", args: dict[str, Any]) -> str:
    """Locate a symbol across workspace .py files (plan item B-#8)."""
    symbol = str(args.get("symbol", "")).strip().strip('"').strip("'")
    if not symbol:
        return "Error: references requires a symbol name."
    try:
        max_results = max(1, min(int(args.get("max_results") or 60), 200))
    except (TypeError, ValueError):
        max_results = 60
    _agent_attr(agent, "_note_effect")(agent.workspace)
    return await asyncio.to_thread(
        collect_references, symbol, _effective_ws_dir(agent), max_results,
    )


async def _nlp_jev_decide(agent: "Agent", args: dict[str, Any]) -> str:
    """Typed probabilistic decision from the dedicated small Jev model.

    The engine is built from ``settings.jev_model`` (never ``self.llm``), so
    a fast local small model makes yes/no, choice and rubric-score calls
    without involving the main model.  Failures return an error string; the
    tool is read-only and therefore allowed in plan mode too.
    """
    from agent_core.jev_engine import JevQuestion, build_jev_engine

    kind = str(args.get("kind") or "yesno").strip().lower()
    text = str(args.get("question") or "").strip()
    if not text:
        return "Error: jev_decide requires 'question'."
    raw_options = args.get("options") or []
    if isinstance(raw_options, str):
        raw_options = [o.strip() for o in raw_options.split("|")]
    options = tuple(str(o).strip() for o in raw_options if str(o).strip())
    rubric = str(args.get("rubric") or "")
    state = str(args.get("state") or "")
    raw_threshold = args.get("threshold")
    if raw_threshold is None:
        # B4: calibrated default — once >= 10 labeled decisions exist,
        # the threshold that best separates correct/incorrect replaces
        # the hardcoded 0.7 guess; fallback stays 0.7 (never raises).
        threshold = 0.7
        try:
            from harnessfix.jev_telemetry import load_suggested_threshold

            suggestion = load_suggested_threshold(
                workspace=getattr(agent, "workspace", None),
                kind=kind,
                min_samples=10,
            )
            if suggestion:
                threshold = float(suggestion["threshold"])
        except Exception:  # noqa: BLE001 - calibration must not break the tool
            threshold = 0.7
    else:
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            threshold = 0.7
    threshold = min(1.0, max(0.0, threshold))
    try:
        question = JevQuestion(
            kind=kind, text=text, options=options, rubric=rubric,
        )
    except ValueError as exc:
        return f"Error: {exc}"
    try:
        engine = build_jev_engine(
            threshold=threshold,
            source="tool",
            log_workspace=getattr(agent, "workspace", None),
        )
    except Exception as exc:  # noqa: BLE001 - surface a clear setup error
        return f"jev_decide error: {exc}"
    try:
        result = await engine.decide(question, state)
    except Exception as exc:  # noqa: BLE001 - never kill the turn
        return f"jev_decide error: {exc}"
    return result.summary()


async def _nlp_speculate(agent: "Agent", args: dict[str, Any]) -> str:
    """On-demand speculative deliberation for the LLM.

    Runs the SAME pipeline as the REPL ``speculate`` command — N parallel
    branches (each carrying this agent's live system prompt and a bounded
    READ-ONLY tool loop), judge scoring, COMMIT/REFUSE gate — and returns
    its printed verdict as the tool result, so the main model can get a
    fast, independently corroborated answer instead of one long reasoning
    chain.  Branches are hard-capped to ``_BRANCH_TOOLS`` (read-only),
    which is why this tool is plan-mode safe too: it cannot mutate files.

    Argument validation happens HERE (clear errors for the model); the
    command's own parsing guards cover the rest (threshold range, branch
    count).  Output is captured via ``redirect_stdout`` so nothing leaks
    to the REPL — the verdict exists only in this tool result.
    """
    import contextlib
    import io

    from agent_core.commands.speculate_cmd import SpeculateCommand

    question = str(args.get("question") or "").strip().strip('"').strip("'")
    if not question:
        return "Error: speculate requires 'question'."
    if question.startswith("--"):
        return (
            "Error: 'question' must not start with '--' (it would be "
            "parsed as a command flag)."
        )

    parts = [question]

    branches = args.get("branches")
    if branches is not None:
        try:
            n = max(1, int(branches))
        except (TypeError, ValueError):
            return "Error: 'branches' must be a positive integer."
        parts += ["--branches", str(n)]

    judge = str(args.get("judge") or "llm").strip().lower()
    if judge not in ("llm", "jev", "both"):
        return "Error: 'judge' must be one of: llm, jev, both."
    parts += ["--judge", judge]

    branch_model = str(args.get("branch_model") or "chat").strip().lower()
    if branch_model not in ("chat", "jev"):
        return "Error: 'branch_model' must be one of: chat, jev."
    if branch_model != "chat":
        parts += ["--branch-model", branch_model]

    threshold = args.get("threshold")
    if threshold is not None:
        try:
            t = float(threshold)
        except (TypeError, ValueError):
            return "Error: 'threshold' must be a number in [0.0, 1.0]."
        if not 0.0 <= t <= 1.0:
            return "Error: 'threshold' must be within [0.0, 1.0]."
        parts += ["--threshold", repr(t)]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        await SpeculateCommand().execute(parts, agent)
    text = buf.getvalue().strip()
    if not text:
        return "speculate produced no output."
    # The committed answer is the point of this tool — give it more room
    # than a 5k shell dump before truncating.
    return _truncate_output(text, limit=8000)


async def _nlp_search(agent: "Agent", args: dict[str, Any]) -> str:
    """Search workspace files for text (first 30 matches)."""
    query = str(args.get("query", ""))
    search_path = _agent_attr(agent, "_resolve_nlp_path")(
        str(args.get("path") or ".")
    )
    try:
        found = await agent.searcher.search(query, search_path)
    except Exception as e:
        return f"Search error: {e}"
    if not found or found == "No matches found":
        return "No files found matching that query."
    _agent_attr(agent, "_note_effect")(search_path)
    lines = found.splitlines()
    return "\n".join(f"  {line}" for line in lines[:30])


async def _nlp_read(agent: "Agent", args: dict[str, Any]) -> str:
    """Read one file (``path``) or several files (``paths``) in one call.

    The ``paths`` array is what lets a model satisfy "read these two files"
    in a SINGLE tool call instead of refusing ("I can only read one file at
    a time") or asking the user to re-specify.
    """
    raw_paths = args.get("paths")
    if isinstance(raw_paths, list) and raw_paths:
        return await _nlp_read_many(agent, raw_paths, args)
    if not str(args.get("path", "")).strip():
        return "Read error: provide 'path' (one file) or 'paths' (a list)."
    path = _agent_attr(agent, "_resolve_nlp_path")(
        str(args.get("path", "")).strip('"').strip("'")
    )
    return await _read_page(agent, path, args, original=args.get("path"))


async def _nlp_read_many(
    agent: "Agent", raw_paths: list[Any], args: dict[str, Any]
) -> str:
    """Read several files in one call (the ``paths`` form of ``read``).

    Each file is paged with the same offset/limit as a single read and
    returned under a ``=== <path> ===`` header.  The call is capped at
    ``_MAX_READ_FILES`` distinct paths so a runaway model cannot pull the
    whole workspace into context in one shot.
    """
    cleaned: list[str] = []
    for raw in raw_paths:
        value = str(raw).strip().strip('"').strip("'")
        if value and value not in cleaned:
            cleaned.append(value)
    if not cleaned:
        return "Read error: 'paths' must be a non-empty list of file paths."
    extra = ""
    if len(cleaned) > _MAX_READ_FILES():
        extra = (
            f"\n\n[read capped at {_MAX_READ_FILES()} files per call; "
            f"{len(cleaned) - _MAX_READ_FILES()} skipped]"
        )
        cleaned = cleaned[:_MAX_READ_FILES()]
    blocks: list[str] = []
    for raw in cleaned:
        path = _agent_attr(agent, "_resolve_nlp_path")(raw)
        text = await _read_page(agent, path, args, original=raw)
        blocks.append(f"=== {raw} ===\n{text}")
    return "\n\n".join(blocks) + extra


async def _read_page(
    agent: "Agent", path: str, args: dict[str, Any], original: Any
) -> str:
    """Read *limit* lines of *path* starting at 1-based *offset* (shared
    by single- and multi-file reads)."""
    if os.path.isdir(path):
        return (
            f"Not a file (directory): {original} — read takes a "
            "FILE path; use list_files or search for directories."
        )
    try:
        # Paging is LINE-BASED (1-indexed): models and callers pass the
        # starting line number and a line count, not character offsets —
        # char offsets kept landing every read in the same short
        # docstring, causing repeated-read loops.  *limit* is capped so a
        # single call cannot pull an entire large file into context (the
        # history trimmer would have to clean up after it next turn).
        offset = max(1, int(args.get("offset") or 1))
        limit = max(1, min(int(args.get("limit") or 100), _MAX_READ_LINES()))
    except (TypeError, ValueError):
        return "Read error: offset/limit must be integers."
    # AST strategy: a BARE read (no explicit offset/limit) of a large .py
    # file returns a definition summary instead of full content (saves
    # context tokens).  An explicit page is ALWAYS honoured — the summary
    # used to be returned for every offset too, so paging a large file
    # yielded the same definitions each time and the model looped forever.
    page_requested = (
        args.get("offset") is not None or args.get("limit") is not None
    )
    if (
        not page_requested
        and _CONTEXT_AST_THRESHOLD_KB() > 0
        and path.endswith(".py")
        and os.path.isfile(path)
        and os.path.getsize(path) > _CONTEXT_AST_THRESHOLD_KB() * 1024
    ):
        content = await agent.read_file(path, track_read=False)
        if content.startswith("File not found") or content.startswith("Error"):
            return content
        _agent_attr(agent, "_note_effect")(path)
        return collect_definitions(content, filename=path) + (
            "\n\n[Large file: showing a definitions summary. "
            "Call read with offset=<line> to fetch source lines.]"
        )
    content = await agent.read_file(path, track_read=False)
    if content.startswith("File not found") or content.startswith("Error"):
        return content
    _agent_attr(agent, "_note_effect")(path)
    if not agent._delegating:
        agent._read_streak += 1
    lines = content.splitlines()
    if offset > len(lines):
        return f"Offset {offset} is beyond the end of {path} ({len(lines)} lines)."
    chunk_lines = lines[offset - 1: offset - 1 + limit]
    chunk = "\n".join(chunk_lines)
    result = chunk
    if offset - 1 + limit < len(lines):
        result = (
            f"{chunk}\n[truncated — use read with offset="
            f"{offset + limit} to continue]"
        )
    if agent._read_streak >= _MAX_CONSECUTIVE_READS():
        result += _READ_LOOP_NOTE().format(n=agent._read_streak)
    return result


async def _nlp_read_skill(agent: "Agent", args: dict[str, Any]) -> str:
    """Load one page of a workspace skill body (``read_skill`` tool).

    Reads ``skills/<name>/SKILL.md`` from the effective workspace for a
    name listed in the system-prompt SKILLS index.  Paging mirrors
    ``read`` (1-based ``offset``, ``limit`` lines).  Every failure mode —
    bad/unknown name, unreadable or invalid file, out-of-range offset — is
    returned as a plain string so a bad call can never break the loop.
    """
    name = str(args.get("name", "")).strip()
    try:
        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or DEFAULT_PAGE_LINES())
    except (TypeError, ValueError):
        return "Skill error: offset/limit must be integers."
    try:
        page = read_skill(
            _effective_ws_dir(agent), name, offset=offset, limit=limit,
        )
    except SkillError() as exc:
        return f"Skill error: {exc}"
    except Exception:
        logger.exception('read_skill failed:\n')
        return f"Skill error: could not load skill {name!r}."
    _agent_attr(agent, "_note_effect")(str(page.skill.path))
    return format_skill_page(page)


async def _nlp_list_files(agent: "Agent", args: dict[str, Any]) -> str:
    """List up to 50 directory entries, directories marked with ``/``."""
    raw_path = str(args.get("path") or ".").strip('"').strip("'")
    path = _agent_attr(agent, "_resolve_nlp_path")(raw_path)
    abs_path = path if os.path.isabs(path) else os.path.abspath(path)
    try:
        entries = sorted(os.listdir(abs_path))[:50]
    except NotADirectoryError:
        return f"Not a directory: {path}"
    except FileNotFoundError:
        return f"Directory not found: {path}"
    except OSError as e:
        return f"List error: {e}"
    _agent_attr(agent, "_note_effect")(path)
    lines = []
    for entry in entries:
        full = os.path.join(abs_path, entry)
        suffix = "/" if os.path.isdir(full) else ""
        lines.append(f"  {entry}{suffix}")
    return "\n".join(lines)


async def _nlp_fix(agent: "Agent", args: dict[str, Any]) -> str:
    """Run the ``fix`` REPL command on a resolved target file."""
    from agent_core.commands.fix_cmd import FixCommand

    resolved_args = [str(a) for a in args.get("args", [])]
    target = (
        resolved_args[0] if resolved_args and not resolved_args[0].startswith("--")
        else None
    )
    if target:
        resolved_args[0] = _agent_attr(agent, "_resolve_nlp_path")(target)
    output = await _run_command_quietly(
        agent, FixCommand(), resolved_args, "Fix executed. Check the file for changes.",
    )
    if resolved_args and not resolved_args[0].startswith("--"):
        _agent_attr(agent, "_note_effect")(resolved_args[0])
    return output


async def _nlp_write(agent: "Agent", args: dict[str, Any]) -> str:
    """Write *content* to *path*; py_compile-verify Python files."""
    path = _agent_attr(agent, "_resolve_nlp_path")(str(args.get("path", "")))
    content = str(args.get("content", ""))
    try:
        return await _agent_attr(agent, "_save_verify_note")(
            path, content, f"Written {path} ({len(content)} bytes)",
        )
    except Exception as e:
        return f"Write error: {e}"


async def _nlp_run(agent: "Agent", args: dict[str, Any]) -> str:
    """Execute one shell command with destructive-pattern blocking,
    whole-tree timeout kill, and Unix-ism hints."""
    cmd_to_run = str(args.get("command", "")).strip()
    # Upper bound: an unbounded model-supplied timeout could stall a turn
    # for hours; 10 min is the hard ceiling (longer jobs should be split).
    timeout = max(1, min(int(args.get("timeout") or 120), _MAX_RUN_TIMEOUT_S()))
    if not cmd_to_run:
        return "Error: run requires a command."
    is_pytest = _is_pytest_command(cmd_to_run)
    is_full_run = _is_full_pytest_command(cmd_to_run)
    if is_full_run:
        # A full run costs ~2 min and must be EARNED (a real source file
        # changed) and RARE (one per session by default) - so it happens at
        # the END of real work, never after a _tmp_* scratch probe.
        refusal = agent._full_run_gate.check()
        if refusal:
            return refusal
    # The suite budget bounds ANY pytest run: a subset selector such as
    # ``--lf`` can select most of the suite and used to be killed at
    # whatever timeout the model guessed (typically 300s).  Targeted path
    # runs and non-pytest commands stay capped by _MAX_RUN_TIMEOUT_S.
    # This escape hatch is deliberately NOT capped by _MAX_RUN_TIMEOUT_S.
    if is_full_run or (is_pytest and _is_subset_pytest_command(cmd_to_run)):
        timeout = max(timeout, int(_pytest_full_suite_timeout()))
    blocked = _blocked_shell_command(cmd_to_run)
    if blocked:
        return f"Error: Dangerous command blocked ({blocked}): {cmd_to_run}"
    # Known-bad pytest flag combos crash pytest itself (INTERNALERROR) rather
    # than producing a usage error the model could self-correct from.  Refuse
    # with an actionable message instead of burning a turn on a stack trace.
    if is_pytest:
        conflict = _pytest_flag_conflict(cmd_to_run)
        if conflict:
            return f"Error: {conflict}"
    # Windows: start the shell in its own process group so a timeout
    # can kill the WHOLE tree. Killing only cmd.exe leaves orphaned
    # children (e.g. a harnessfix.loop python process) running and
    # holding the captured pipes open — the caller then waits far
    # beyond the timeout while the "killed" command keeps working.
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    started = time.monotonic()
    # Run the command in the shell the system prompt ADVERTISES, not whatever
    # subprocess's shell=True picks (on Windows that is always cmd.exe — so a
    # model told SHELL_NAME=powershell got cmd.exe and its PowerShell cmdlets
    # died as 'is not recognized').  See shell_utils.run_shell_command.
    run_argv = _run_shell_command(cmd_to_run)
    try:
        proc = subprocess.Popen(
            run_argv,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Pin UTF-8 explicitly: without it the child's stdout/stderr
            # are decoded with the locale codepage (cp1252 on Windows), so
            # a command emitting UTF-8 (CJK, emoji, git output) comes back
            # as mojibake or raises UnicodeDecodeError out of communicate()
            # (decision #114 — every text=True child call must pin this).
            encoding="utf-8",
            errors="replace",
            cwd=_effective_ws_dir(agent),
            creationflags=creationflags,
        )
        try:
            output, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc)
            output, err = proc.communicate()
            if is_full_run:
                # A killed run proves it lasted AT LEAST this long.  Record
                # that lower bound so the next budget grows past the point
                # that killed this one, instead of losing the same run again.
                record_full_run_seconds(
                    time.monotonic() - started, completed=False,
                )
            hint = (
                f" Measured full-suite budget: "
                f"{int(_pytest_full_suite_timeout())}s (pytest)."
                if is_pytest
                else ""
            )
            return (
                f"Command timed out after {timeout}s — process tree killed. "
                f"For long-running jobs, pass a larger timeout.{hint}"
            )
    except Exception as e:
        return f"Error: {e}"
    if is_full_run:
        agent._full_run_gate.record_full_run()
        # Measure, don't guess: the next run's timeout is 1.5x this.
        record_full_run_seconds(time.monotonic() - started)
    # PowerShell's -Command runs a statement list, so the process exit code is
    # the LAST statement's (the marker), not the user command's.  The marker
    # carries the real code; when it is present it wins over proc.returncode.
    exit_code = proc.returncode
    output, marker_code = _parse_ps_exit_marker(output)
    if marker_code is not None:
        exit_code = marker_code
    return _truncate_output(
        _shape_run_stderr(err, output, exit_code)
    )


async def _nlp_git(agent: "Agent", args: dict[str, Any]) -> str:
    """Run a whitelisted git subcommand via arg-list execution (no shell)."""
    subcmd = str(args.get("subcommand") or "status").lower()
    git_args = str(args.get("args") or "")
    git_cmds = {
        "status": ["status"],
        "diff": ["diff", "--no-color"],
        "log": ["log", "--oneline", "-15"],
        "add": ["add"],
        "commit": ["commit"],
        "push": ["push"],
        "pull": ["pull"],
        "branch": ["branch"],
        "checkout": ["checkout"],
        "stash": ["stash"],
        "show": ["show"],
        "blame": ["blame"],
        "remote": ["remote", "-v"],
        "branches": ["branch", "-a"],
    }
    base = git_cmds.get(subcmd)
    if not base:
        return (
            f"Unknown git command: {subcmd}. "
            f"Available: {', '.join(git_cmds.keys())}"
        )
    try:
        extra = shlex.split(git_args)
    except ValueError as e:
        return f"Git error: invalid arguments: {e}"
    # `git add` needs real paths: a bare "-" (or no args) is a pathspec
    # error that wastes a call, so answer with the correct form instead.
    if subcmd == "add" and (not extra or extra == ["-"]):
        return (
            "Git error: `git add` needs paths. Use args=\"-A\" to stage all "
            "changes, or args=\"<path>\" for specific files (a bare '-' is "
            "not a valid pathspec)."
        )
    # A message-less `git commit` opens an editor and blocks until the
    # 30s timeout (the harness sets no GIT_EDITOR), so answer with the
    # correct form instead of paying for a hang. Editor-free message
    # sources: -m/-F, reuse -C/-c, --fixup/--squash, and `--amend
    # --no-edit`. A bare `--amend` is NOT accepted (it opens the editor).
    has_message = any(
        a in ("-m", "--message", "-F", "--file")
        or a.startswith(("--message=", "-m", "--file="))
        or a in ("-C", "--reuse-message", "--fixup", "--squash")
        for a in extra
    )
    amend_no_edit = "--amend" in extra and "--no-edit" in extra
    if subcmd == "commit" and not (extra and (has_message or amend_no_edit)):
        return (
            "Git guidance: `git commit` needs a message or it opens an "
            "editor and hangs. Use subcommand='commit', args='-m \"<message>\"'."
        )
    # `git push` only sends committed work: if changes are staged but not
    # committed, a push leaves them behind — the 2026-09-28 "commit
    # changes" run pushed an older commit and silently skipped the staged
    # work. Stop and point at the missing commit instead.
    if subcmd == "push":
        staged, _staged_err = _run_subprocess_captured(
            ["git", "diff", "--cached", "--name-only"],
            _effective_ws_dir(agent), 30, "Git",
        )
        staged_files = [
            ln.strip()
            for ln in staged.splitlines()
            if ln.strip() and ln.strip() != "(no output)"
            and not ln.strip().startswith("[STDERR]")
        ]
        if staged_files:
            listing = ", ".join(staged_files[:5])
            more = (
                "" if len(staged_files) <= 5
                else f" (+{len(staged_files) - 5} more)"
            )
            return (
                f"Git guidance: {len(staged_files)} staged change(s) are "
                f"NOT committed yet ({listing}{more}) — `git push` would "
                "leave them behind. Commit first with subcommand='commit', "
                "args='-m \"<message>\"', then push again."
            )
    # Arg-list execution (no shell): shell metacharacters in
    # model-supplied args become literal git arguments.
    output, error = _run_subprocess_captured(
        ["git"] + base + extra, _effective_ws_dir(agent), 30, "Git",
    )
    return error or output


async def _nlp_merge(agent: "Agent", args: dict[str, Any]) -> str:
    """Run a merge through the state-machine module (never interactive).

    ``git merge --continue`` blocks forever on an editor when no
    ``GIT_EDITOR`` is set, and rejects every argument
    (``--continue expects no arguments``), so merges are NOT routed
    through the generic ``git`` tool: :mod:`agent_core.tools.git_merge`
    owns the state machine, disables the editor, and answers no-op
    requests with guidance instead of a wasted git call.
    """
    from agent_core.tools import git_merge

    return await asyncio.to_thread(
        git_merge.run_merge, _effective_ws_dir(agent), args,
    )


async def _nlp_diff(agent: "Agent", args: dict[str, Any]) -> str:
    """Show ``git diff --no-color`` for one file (optionally against another)."""
    file1 = _agent_attr(agent, "_resolve_nlp_path")(str(args.get("file1", "")))
    file2 = args.get("file2")
    if not file1:
        return "Error: diff requires at least one file."
    cmd = ["git", "diff", "--no-color", "--"]
    if file2:
        cmd += [file1, _agent_attr(agent, "_resolve_nlp_path")(str(file2))]
    else:
        cmd += [file1]
    output, error = _run_subprocess_captured(
        cmd, _effective_ws_dir(agent), 30, "Diff",
    )
    if error:
        return error
    return output if output != "(no output)" else "(no differences found)"


async def _nlp_edit(agent: "Agent", args: dict[str, Any]) -> str:
    """Replace the first exact occurrence of *old_text* in a file."""
    path = _agent_attr(agent, "_resolve_nlp_path")(str(args.get("path", "")))
    old_text = str(args.get("old_text", ""))
    new_text = str(args.get("new_text", ""))
    try:
        # Read in the file's own encoding (UTF-8, or legacy cp1252). A file
        # that cannot be decoded and round-tripped is refused, never
        # rewritten with U+FFFD substituted for the original bytes.
        content, encoding = read_text_preserving(path)
    except FileNotFoundError:
        return f"File not found: {path}"
    except NonUtf8TextError() as e:
        return f"Edit refused: {e}"
    except OSError as e:
        return f"Edit error: {e}"
    if old_text not in content:
        return (
            f"Text not found in {path}. Make sure old text matches "
            "exactly (including whitespace)."
        )
    new_content = content.replace(old_text, new_text, 1)
    try:
        # Refuse before writing if the new text cannot be encoded in the
        # file's existing encoding, so a legacy file is never converted.
        encode_in_encoding(new_content, encoding, path)
    except UnrepresentableTextError() as e:
        return f"Edit refused: {e}"
    if encoding == "utf-8":
        try:
            return await _agent_attr(agent, "_save_verify_note")(
                path, new_content, f"Edited {path}",
            )
        except Exception as e:
            return f"Edit error: {e}"
    # Legacy encoding: write in the same encoding, same newline handling
    # as save_file_py (text mode), so only the encoding is preserved.
    try:
        with open(path, "w", encoding=encoding) as f:
            f.write(new_content)
    except OSError as e:
        return f"Edit error: {e}"
    verify = (
        await _agent_attr(agent, "_verify_file")(path)
        if path.endswith(".py") else ""
    )
    _agent_attr(agent, "_note_effect")(path)
    return f"Edited {path} (kept {encoding})\n{verify}".strip()


async def _nlp_tests(agent: "Agent", args: dict[str, Any]) -> str:
    """Run pytest/unittest on *path* (bounded by the measured suite budget)."""
    test_path = _agent_attr(agent, "_resolve_nlp_path")(str(args.get("path") or "."))
    framework = str(args.get("framework") or "pytest")
    #: The full suite takes ~15 minutes here (measured 861.5s), not the
    #: 2.5 minutes this used to assume - 300s killed whole-suite runs.
    timeout = 300
    is_full = (
        os.path.abspath(test_path) == os.path.abspath(_effective_ws_dir(agent))
    )
    if is_full:
        # Whole-workspace run: same "earned and rare" gate as the run tool -
        # no real change means a full pass verifies nothing.  Check BEFORE
        # running, but spend the budget only AFTER (see below): recording
        # first burned the one-shot budget even when the run then timed out
        # or failed to launch, unlike _nlp_run which records on completion.
        refusal = agent._full_run_gate.check()
        if refusal:
            return refusal
    if framework == "pytest":
        # A whole-workspace run gets the measured suite budget; a targeted
        # path keeps the plain cap so nested/single-file runs stay bounded.
        if is_full:
            timeout = max(timeout, int(_pytest_full_suite_timeout()))
        cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
    else:
        cmd = [sys.executable, "-m", "unittest", test_path, "-v"]
    started = time.monotonic()
    output, error = _run_subprocess_captured(
        cmd, _effective_ws_dir(agent), timeout, "Tests",
    )
    if is_full and error is None:
        timed_out = "timed out after" in (output or "")
        if not timed_out:
            # The run completed - only now is the one-shot budget spent
            # (parity with _nlp_run, which records after communicate()).
            agent._full_run_gate.record_full_run()
        # A killed run still proves it lasted AT LEAST this long.
        record_full_run_seconds(
            time.monotonic() - started, completed=not timed_out,
        )
    return error or output


async def _nlp_analyze(agent: "Agent", args: dict[str, Any]) -> str:
    """Run the ``analyze`` REPL command (AI analysis) on an optional path.

    The path may be a FILE or a DIRECTORY (a folder is summarized from a
    bounded sample of its files); omitting it analyzes the whole workspace.
    """
    analyze_args: list[str] = []
    if args.get("path"):
        analyze_args = [_agent_attr(agent, "_resolve_nlp_path")(str(args["path"]))]
        _agent_attr(agent, "_note_effect")(analyze_args[0])
    try:
        from agent_core.commands.analyze_cmd import AnalyzeCommand
        return await _run_command_quietly(
            agent, AnalyzeCommand(), analyze_args, "Analysis complete."
        )
    except Exception as e:
        return f"Analyze error: {e}"


async def _nlp_web_search(agent: "Agent", args: dict[str, Any]) -> str:
    """DuckDuckGo search; results are UNTRUSTED (marked by format_results)."""
    query = str(args.get("query", "")).strip()
    if not query:
        return "Error: web_search requires a query."
    try:
        from agent_core.tools.web_search import (
            MAX_RESULTS_LIMIT,
            format_results,
            sanitize_query,
            search_ddg,
        )
        max_results = max(
            1, min(int(args.get("max_results") or 5), MAX_RESULTS_LIMIT))
        clean_query = sanitize_query(query)
        hits = search_ddg(clean_query, max_results=max_results)
        output = format_results(clean_query, hits)
        return output[:5000]
    except Exception as e:
        return f"Web search error: {e}"


async def _nlp_get_current_datetime(agent: "Agent", args: dict[str, Any]) -> str:
    """Return the current date/time, optionally in a given timezone."""
    from datetime import datetime
    import zoneinfo
    tz_name = args.get("timezone")
    if tz_name:
        try:
            tz = zoneinfo.ZoneInfo(str(tz_name))
        except (ValueError, zoneinfo.ZoneInfoNotFoundError):
            return f"Error: unknown timezone '{tz_name}'. Use IANA names like 'UTC' or 'America/New_York'."
        return datetime.now(tz).isoformat()
    return datetime.now().isoformat()


async def _nlp_hue_control(agent: "Agent", args: dict[str, Any]) -> str:
    """Handle hue_control NLP tool with V1 and V2 API support.

    Supports:
      - list_lights: List all lights (V1)
      - get_light: Get detailed info for a light (V1)
      - set_light: Set on/off/brightness/CT (V1)
      - color_capabilities: Get RGB capabilities via V2 API
      - set_color: Set RGB/HSV/XY colors via V2 API
      - set_color_named: Set color by name (red, blue, etc.) or HSV spec

    For V2 API actions, at least one of r/g/b/x/y/h/s/v must be provided.
    """
    from agent_core.hue.bridge import HueBridge, HueBridgeError

    action = args.get("action", "list_lights")
    try:
        bridge = HueBridge.from_env()
    except HueBridgeError as exc:
        return f"Hue Bridge error: {exc}"
    try:
        if action == "list_lights":
            lights = await bridge.list_lights()
            if not lights:
                return "No lights found on the Hue Bridge."
            lines = []
            for ln in lights:
                state = "ON" if ln.get("on") else "OFF"
                bri = f"{ln.get('brightness'):.1f}%" if ln.get("brightness") is not None else "?"
                lines.append(f"  [{ln['id']}] {ln['name']} -- {state}, brightness={bri}")
            return f"Found {len(lights)} light(s):\n" + "\n".join(lines)

        elif action == "get_light":
            light_id = args.get("light_id", "")
            if not light_id:
                return "Error: light_id is required for get_light."
            info = await bridge.get_light(light_id)
            return json.dumps(info, indent=2, default=str)

        elif action == "set_light":
            light_id = args.get("light_id", "")
            if not light_id:
                return "Error: light_id is required for set_light."
            light_state: dict[str, Any] = {}
            if "on" in args:
                light_state["on"] = bool(args["on"])
            if "brightness" in args:
                light_state["brightness"] = float(args["brightness"])
            if "mirek" in args:
                light_state["mirek"] = int(args["mirek"])
            if not light_state:
                return "Error: no state changes specified (on, brightness, mirek)."
            await bridge.set_light(light_id, **light_state)
            return f"Light {light_id} updated: {light_state}"

        elif action == "color_capabilities":
            light_id = args.get("light_id", "")
            if not light_id:
                return "Error: light_id is required for color_capabilities."
            caps = await bridge.get_color_capabilities(light_id)
            return json.dumps(caps, indent=2, default=str)

        elif action == "set_color":
            light_id = args.get("light_id", "")
            if not light_id:
                return "Error: light_id is required for set_color."

            # Check that at least one RGB/HSV/XY parameter is provided
            has_rgb = "r" in args and "g" in args and "b" in args
            has_xy = "x" in args or "y" in args
            has_hsv = "h" in args or "s" in args or "v" in args

            if not has_rgb and not has_xy and not has_hsv:
                return "Error: set_color requires at least one color parameter (r/g/b, x/y, or h/s/v)."

            kwargs: dict[str, Any] = {}
            if "r" in args:
                kwargs["r"] = int(args["r"])
            if "g" in args:
                kwargs["g"] = int(args["g"])
            if "b" in args:
                kwargs["b"] = int(args["b"])
            if "x" in args:
                kwargs["x"] = float(args["x"])
            if "y" in args:
                kwargs["y"] = float(args["y"])
            if "h" in args:
                kwargs["h"] = int(args["h"])
            if "s" in args:
                kwargs["s"] = float(args["s"])
            if "v" in args:
                kwargs["v"] = float(args["v"])

            await bridge.set_light_color(light_id, **kwargs)
            return f"Light {light_id} color updated."

        elif action == "set_color_named":
            light_id = args.get("light_id", "")
            if not light_id:
                return "Error: light_id is required for set_color_named."

            name = args.get("name", "")
            brightness_val = float(args.get("brightness", 100.0))

            await bridge.set_light_color_named(light_id, name, brightness=brightness_val)
            return f"Light {light_id} color set to '{name}' at {brightness_val}%."

        else:
            return f"Error: unknown action '{action}'. Use list_lights, get_light, " \
                   f"set_light (V1), or color_capabilities/set_color/set_color_named (V2)."

    except HueBridgeError as exc:
        return f"Hue Bridge error: {exc}"


# ── Plan workflow NLP tools ─────────────────────────────────────────────────

async def _get_plan_state(agent: "Agent") -> dict[str, Any]:
    """Detect current plan status + tasks from the filesystem."""
    from agent_core.commands.doc_paths import latest_run_dir
    from agent_core.plan_execution.parser import parse_plan_tasks

    plan_dir = latest_run_dir(agent.workspace)
    if plan_dir is None:
        return {"status": "none", "error": "No .docs run directory found"}

    plan_file = None
    status = "none"
    for candidate, st in [
        (plan_dir / "plan_executing.md", "executing"),
        (plan_dir / "plan_proposed.md", "proposed"),
    ]:
        if candidate.is_file():
            plan_file = candidate
            status = st
            break

    if plan_file is None:
        for f in plan_dir.iterdir():
            if f.name.startswith("plan_executed_"):
                status = "executed"
                plan_file = f
                break
            elif f.name.startswith("plan_failed_"):
                status = "failed"
                plan_file = f
                break

    if plan_file is None:
        return {"status": "none", "error": f"No plan file in {plan_dir}"}

    plan_text = plan_file.read_text(encoding="utf-8")
    tasks = parse_plan_tasks(plan_text, fmt="md")

    return {
        "status": status,
        "plan_dir": str(plan_dir),
        "plan_path": str(plan_file),
        "plan_text": plan_text,
        "tasks": tasks,
    }


async def _nlp_plan_status(agent: "Agent", args: dict[str, Any]) -> str:
    """Query the current plan's lifecycle status and parsed task list."""
    state = await _get_plan_state(agent)
    if state.get("error"):
        return f"Plan status: {state['status']}. {state.get('error', '')}"

    tasks = state.get("tasks", [])
    lines = [
        f"Plan status: {state['status']}",
        f"Plan file: {state['plan_path']}",
        f"Tasks: {len(tasks)}",
    ]
    for t in tasks:
        dep_str = f"  deps: {t.depends_on}" if t.depends_on else ""
        lines.append(f"  - [{t.id}] ({t.role}) {t.description}{dep_str}")
    if state["status"] == "executing":
        lines.append("\nUse plan_step to advance, or plan_finish to complete.")
    elif state["status"] == "proposed":
        lines.append("\nUse plan_start to begin execution.")
    return "\n".join(lines)


async def _nlp_plan_start(agent: "Agent", args: dict[str, Any]) -> str:
    """Start plan execution: proposed -> executing, validate, run tasks."""
    if agent.is_plan_mode():
        return "Error: plan mode is read-only. Run 'mode build' first."

    state = await _get_plan_state(agent)
    if state.get("error"):
        return f"Error: {state['error']}"
    if state["status"] not in ("proposed",):
        return f"Error: plan is already '{state['status']}', cannot start."
    if not state["tasks"]:
        return "Error: no tasks found in the plan."

    from pathlib import Path

    from agent_core.plan_execution.runner import build_and_validate_graph
    from agent_core.orchestration.dependency_graph import CycleError
    from agent_core.subagent_roles import get_role
    from agent_core.commands.plan_lifecycle import PlanLifecycleManager
    from agent_core.commands.plan_dry_run import PlanDryRunner
    from agent_core.commands.plan_decision_gate import PlanDecisionGate

    tasks = state["tasks"]

    for t in tasks:
        if get_role(t.role) is None:
            return f"Error: unknown role '{t.role}' for task '{t.id}'"

    try:
        graph, order = build_and_validate_graph(tasks)
    except CycleError as exc:
        return f"Error: dependency cycle detected: {exc.cycle}"
    except ValueError as exc:
        return f"Error: {exc}"

    dry_runner = PlanDryRunner()
    dry_result = dry_runner.validate(state["plan_text"])
    if not dry_result.valid:
        return "Dry-run safety gate failed:\n" + "\n".join(
            f"  - {e}" for e in dry_result.errors
        )

    decision_gate = PlanDecisionGate(Path(agent.workspace))
    gate_result = decision_gate.validate(state["plan_text"])
    if not gate_result.passed:
        return "Decision gate failed - plan violates constraints:\n" + "\n".join(
            f"  - {v}" for v in gate_result.violations
        )

    # Consensus review gate (opt-in, default off): every configured model
    # votes on the plan with a structured VERDICT line; only a quorum of
    # APPROVEs lets it start. Model disagreement blocks — an honest "no"
    # from models that answered is a real signal. Infrastructure failure
    # fails OPEN (the note below says so) — a dead server must never
    # silently block the workflow. See consensus_gate.py.
    consensus_note = ""
    from agent_core.plan_execution.consensus_gate import gate_enabled as _gate_on
    if _gate_on(agent.workspace):
        from agent_core.plan_execution.consensus_gate import review_plan
        outcome = await review_plan(agent, state["plan_text"])
        note_text = outcome.consensus or outcome.skipped_reason
        consensus_note = f"\nConsensus gate: {note_text}" if note_text else ""
        if not outcome.approved:
            return (
                "Plan blocked by the consensus gate." + consensus_note
                + "\nThe plan stays proposed. Address the reviewers'"
                  " concerns and propose again."
            )

    dry_run = args.get("dry_run", False)

    plan_dir = Path(state["plan_dir"])
    lifecycle = PlanLifecycleManager(plan_dir, Path(agent.workspace))
    try:
        lifecycle.start_plan()
    except Exception as exc:
        return f"Error during lifecycle transition: {exc}"

    if dry_run:
        return (
            f"Plan transitioned to executing (dry-run mode). "
            f"{len(tasks)} tasks validated.{consensus_note}"
        )

    from agent_core.plan_execution.runner import run_plan

    snapshot = await run_plan(agent, tasks, plan_dir=str(plan_dir))

    report_lines = [f"Plan execution complete ({len(tasks)} tasks):"]
    failed = []
    for t in snapshot["tasks"]:
        icon = "+" if t["status"] == "completed" else "X"
        report_lines.append(f"  {icon} [{t['task_id']}] {t['status']}")
        if t["status"] != "completed":
            failed.append(t["task_id"])

    if failed:
        report_lines.append(
            f"\n{len(failed)} task(s) failed: {', '.join(failed)}"
        )
        report_lines.append(
            "Use plan_finish with success=false to mark the plan as failed."
        )
    else:
        report_lines.append("\nAll tasks completed. Use plan_finish to finalize.")

    if consensus_note:
        report_lines.append(consensus_note.strip())

    return "\n".join(report_lines)


async def _nlp_plan_step(agent: "Agent", args: dict[str, Any]) -> str:
    """Advance the plan by executing the next uncompleted task."""
    if agent.is_plan_mode():
        return "Error: plan mode is read-only. Run 'mode build' first."

    state = await _get_plan_state(agent)
    if state.get("error"):
        return f"Error: {state['error']}"
    if state["status"] == "proposed":
        return (
            "Error: plan not yet started. "
            "Use plan_start first."
        )
    if state["status"] != "executing":
        return f"Error: plan is '{state['status']}', cannot step."

    tasks = state["tasks"]
    if not tasks:
        return "Error: no tasks in plan."

    from agent_core.plan_execution.runner import build_and_validate_graph
    from agent_core.orchestration.dependency_graph import CycleError

    try:
        graph, order = build_and_validate_graph(tasks)
    except (CycleError, ValueError) as exc:
        return f"Error: {exc}"

    import re
    from pathlib import Path

    completed_ids: set[str] = set()
    report_path = Path(state["plan_dir"]) / "plan_execution_report.md"
    if report_path.exists():
        report_text = report_path.read_text(encoding="utf-8")
        for m in re.finditer(r"\[(\S+)\]\s+completed", report_text):
            completed_ids.add(m.group(1))

    target_id = args.get("task_id")
    next_task = None
    for tid in order:
        if tid in completed_ids:
            continue
        if target_id and tid != target_id:
            continue
        next_task = next(t for t in tasks if t.id == tid)
        break

    if next_task is None:
        return "No uncompleted tasks remaining. Use plan_finish to finalize."

    unmet = [d for d in next_task.depends_on if d not in completed_ids]
    if unmet:
        return (
            f"Error: task '{next_task.id}' has unmet dependencies: "
            f"{', '.join(unmet)}"
        )

    # Collect upstream dependency results so the subagent can use them.
    dep_results: dict[str, Any] = {}
    for dep_id in next_task.depends_on:
        dep_path = Path(state["plan_dir"]) / f"dep_{dep_id}_result.json"
        if dep_path.exists():
            try:
                dep_results[dep_id] = json.loads(
                    dep_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

    sub = agent.spawn_subagent(
        name=f"exec-{next_task.id}", role=next_task.role
    )
    desc = next_task.description
    if dep_results:
        parts = []
        for dep_id, res in dep_results.items():
            summary = (
                res.get("summary") or res.get("result") or ""
            ) if isinstance(res, dict) else str(res)
            parts.append(f"- {dep_id}: {summary}")
        desc += "\n\nUpstream results from dependencies:\n" + "\n".join(parts)
    result = await sub.respond(desc)

    # Determine whether the subagent actually succeeded.
    _TURN_CAP_SENTINEL = "turn cap reached"
    task_failed = (
        _TURN_CAP_SENTINEL in result
        or not result.strip()
    )
    status_label = "failed" if task_failed else "completed"
    report_entry = (
        f"- [{next_task.id}] {status_label} (role: {next_task.role})"
    )
    try:
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(report_entry + "\n")
    except Exception:
        pass

    remaining = [
        tid for tid in order
        if tid not in completed_ids and tid != next_task.id
    ]

    preview = result[:200] + ("..." if len(result) > 200 else "")
    lines = [
        f"Task [{next_task.id}] {status_label} (role: {next_task.role}).",
        f"Result preview: {preview}",
    ]
    if remaining:
        lines.append(f"\n{len(remaining)} task(s) remaining: {', '.join(remaining)}")
        lines.append("Call plan_step again for the next task.")
    else:
        lines.append("\nAll tasks done. Call plan_finish to finalize.")

    return "\n".join(lines)


async def _nlp_plan_finish(agent: "Agent", args: dict[str, Any]) -> str:
    """Complete or fail the current plan."""
    state = await _get_plan_state(agent)
    if state.get("error"):
        return f"Error: {state['error']}"
    if state["status"] != "executing":
        return (
            f"Error: plan is '{state['status']}', "
            f"can only finish an executing plan."
        )

    success = args.get("success", True)
    reason = args.get("reason", "")
    from pathlib import Path
    from agent_core.commands.plan_lifecycle import PlanLifecycleManager
    plan_dir = Path(state["plan_dir"])
    lifecycle = PlanLifecycleManager(plan_dir, Path(agent.workspace))

    try:
        if success:
            dst = lifecycle.finish_plan()
        else:
            dst = lifecycle.fail_plan()
    except Exception as exc:
        return f"Error during lifecycle transition: {exc}"

    final_status = "executed" if success else "failed"
    lines = [f"Plan marked as {final_status}.", f"File: {dst}"]
    if reason:
        lines.append(f"Reason: {reason}")
    return "\n".join(lines)
