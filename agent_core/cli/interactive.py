"""REPL / CLI layer extracted from agent.py (refactor step r07).

The interactive loop, the command-registry builders, the shutdown reminder,
the chat prompt, and the two turn-quality heuristics.  ``Agent`` is imported
lazily inside :func:`run_interactive` (agent.py imports this module, so a
module-level import would be circular).  agent.py re-exports every public
name so ``python -m agent``, the banner and the monkeypatch tests keep
working.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any

from agent_core.bootstrap import (
    AnalyzeCommand,
    CleanupCommand,
    ClearCommand,
    CommandRegistry,
    DecideCommand,
    DemoDataCommand,
    DisplayCommand,
    EntitiesCommand,
    FixCommand,
    FlowStopped,
    GitCommand,
    ImplementCommand,
    ModelCommand,
    MultiLlmCommand,
    OptimizeCommand,
    PasteCommand,
    PasteImageCommand,
    PerfCommand,
    PerfTracker,
    PlanCommand,
    ReadCommand,
    ReconstructCommand,
    ReviewCommand,
    RunCommand,
    SearchCommand,
    SelfHealCommand,
    TaskplanCommand,
    WorkflowCommand,
    WriteCommand,
    chat_stoppable,
    clear_stop,
    save_meta_policy_weights,
)
from agent_dashboard import record_command_metrics
from agent_core.colors import blue, cyan, gray, green, yellow
from agent_core.line_editing import (
    enable as _enable_line_editing,
    save_history as _save_line_history,
)
from agent_core.signal_handlers import (
    current_git_branch as _current_git_branch,
    install as _install_signal_handlers,
    interruptible_input as _interruptible_input,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent

logger = logging.getLogger(__name__)


def _agent_mod() -> Any:
    """The ``agent`` module, resolved lazily (agent.py imports this module).

    ``main`` and the entrypoint helpers resolve their collaborators through it
    so the dispatch tests' monkeypatches on the ``agent`` module keep working.
    """
    import agent as _agent_mod

    return _agent_mod


def _start_line_editing(agent: Any = None) -> bool:
    """Turn on terminal line editing (arrow keys, command recall).

    Without ``readline`` the terminal stays in canonical mode: the up arrow
    does not recall the previous command and the raw escape bytes end up in
    the command text.  The history is stored in ``<workspace>/.agent_history``.

    Returns True when line editing is active.
    """
    workspace = getattr(agent, "workspace", None)
    try:
        return _enable_line_editing(workspace=workspace)
    except Exception:  # pragma: no cover - readline must never break the REPL
        logger.debug("line editing unavailable", exc_info=True)
        return False


_INCOMPLETE_MARKERS = (
    "budget",
    "tool budget",
    "not finished",
    "could not",
    "couldn't",
    "would need",
    "needs one more",
    "needs 1 more",
    "needs 2 more",
    "remaining",
    "unfinished",
    "incomplete",
    "exhausted",
    "ran out",
    "out of tool",
    "more tool call",
    "cannot complete",
    "still needs",
    "not covered",
    "follow-up",
)
_COMPLETE_MARKERS = (
    "all done",
    "is done",
    "done.",
    "completed",
    "finished",
    "is complete",
    "fully resolved",
    "all fixed",
    "task is complete",
)
_ACTION_VERB_RE = re.compile("\\b(commit|push|stage)\\b", re.I)
_QUESTION_LEAD_RE = re.compile(
    "^\\s*(what|why|how|when|where|who|which|is|are|was|were|does|did|do|can|could|should|would|will|has|have|am|explain|tell|show)\\b",
    re.I,
)
CHAT_INPUT_PROMPT = "> "


def _looks_incomplete(text: str) -> bool:
    """Heuristic: did the model's final answer signal unfinished work?

    Strong completion signals (``all done``, ``is complete``, ...) override
    weak markers — "The remaining issue is fixed — all done." is finished even
    though it contains the word "remaining".
    """
    low = text.lower()
    if any((marker in low for marker in _COMPLETE_MARKERS)):
        return False
    return any((marker in low for marker in _INCOMPLETE_MARKERS))


def _unverified_action_claim(user_input: str, tool_calls_made: int) -> bool:
    """Did the model assert a repository action it never actually performed?

    A turn that used no tool at all cannot know the state of the repository, so
    any answer claiming that work was committed, staged or pushed is a guess.
    The 2026-09-28 failure is the canonical case: the user typed ``git status``
    (a registry command whose output goes to stdout and never into the chat
    history) and then "commit changes", and the model answered "Everything is
    already committed and pushed" without a single tool call.

    Returns ``True`` only for an *imperative* request — a question such as
    "is my work already committed?" is asking for an explanation and is left
    alone, because the loop cannot verify the answer to an open question.
    """
    text = (user_input or "").strip()
    if not text or tool_calls_made > 0:
        return False
    if not _ACTION_VERB_RE.search(text):
        return False
    if "?" in text or _QUESTION_LEAD_RE.match(text):
        return False
    return True


def _register_commands(registry: CommandRegistry) -> None:
    """Instantiate every REPL command into *registry* (single registration
    point — the banner listing and dispatch both read from it)."""
    registry.register(ReadCommand())
    registry.register(WriteCommand())
    registry.register(SearchCommand())
    registry.register(ClearCommand())
    registry.register(ModelCommand())
    registry.register(AnalyzeCommand())
    registry.register(PlanCommand())
    registry.register(EntitiesCommand())
    registry.register(TaskplanCommand())
    registry.register(CleanupCommand())
    registry.register(GitCommand())
    registry.register(ImplementCommand())
    registry.register(FixCommand())
    registry.register(WorkflowCommand())
    registry.register(OptimizeCommand())
    registry.register(PerfCommand())
    from agent_core.commands.habits_cmd import HabitsCommand

    registry.register(HabitsCommand())
    registry.register(PasteCommand())
    registry.register(PasteImageCommand())
    registry.register(DisplayCommand())
    registry.register(DecideCommand())
    registry.register(ReviewCommand())
    from agent_core.commands.issue_cmd import IssueCommand

    registry.register(IssueCommand())
    registry.register(RunCommand())
    registry.register(SelfHealCommand())
    registry.register(ReconstructCommand())
    registry.register(MultiLlmCommand())
    registry.register(DemoDataCommand())
    from agent_core.commands.mode_cmd import ModeCommand

    registry.register(ModeCommand())
    from agent_core.commands.subagent_cmd import SubAgentCommand

    registry.register(SubAgentCommand())
    from agent_core.commands.mcp_cmd import MCPCommand

    registry.register(MCPCommand())
    from agent_core.commands.propose_cmd import ProposeCommand

    registry.register(ProposeCommand())
    from agent_core.commands.speculate_cmd import SpeculateCommand

    registry.register(SpeculateCommand())
    from agent_core.commands.jev_cmd import JevCommand

    registry.register(JevCommand())
    from agent_core.commands.memory_cmd import MemoryCommand

    registry.register(MemoryCommand())
    from agent_core.commands.help_cmd import HelpCommand

    registry.register(HelpCommand())


def _build_registry() -> CommandRegistry:
    """Return a fresh CommandRegistry with every command registered."""
    registry = CommandRegistry()
    _register_commands(registry)
    return registry


def _warn_uncommitted(agent: "Agent") -> None:
    """Print a reminder when the session leaves uncommitted changes.

    AGENTS.md invariant #4: "Commit after every session. Uncommitted work is
    unrecoverable" — the deleted-agent.py incident (decision #058) lost
    written files because they were never committed.  This runs on every
    REPL shutdown path (quit / stdin end / EOF) as a last-line reminder.
    Best-effort and silent outside a git repo; never blocks exit.
    """
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=agent.workspace,
            timeout=10,
        )
        if r.returncode != 0:
            return
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if not lines:
            return
        print(
            yellow(
                f"\n  [git] {len(lines)} uncommitted change(s) in the workspace "
                f"— commit them now (AGENTS.md invariant #4: uncommitted work is "
                f"unrecoverable). Top offenders:"
            )
        )
        for ln in lines[:5]:
            print(f"    {ln}")
        if len(lines) > 5:
            print(gray(f"    ... and {len(lines) - 5} more"))
    except Exception as e:
        logger.debug("Uncommitted-changes check skipped: %s", e)


def _build_chat_prompt(agent: "Agent", branch: str, now: datetime) -> str:
    """Build the interactive chat prompt, colored by session mode.

    The prompt reads ``[build]`` in green (full mutating toolset) or
    ``[plan]`` in blue (read-only research mode), so the active mode is
    unmistakable at a glance.  Falls back to green/build styling when the
    agent has no ``is_plan_mode`` (defensive — the prompt must always render).
    """
    plan = getattr(agent, "is_plan_mode", None)
    plan = bool(plan() if callable(plan) else False)
    mode_tag = "[plan] " if plan else "[build] "
    prompt_color = blue if plan else green
    return f"\n[{now:%Y-%m-%d %H:%M}] {branch} {prompt_color(mode_tag.rstrip())}"


async def run_interactive() -> None:
    """Interactive mode - allows user to input commands."""
    from agent import Agent

    agent = Agent(workspace=Agent.DEFAULT_WORKSPACE)
    banner = blue("=" * 50)
    print(banner)
    print(blue("Agent Interactive Mode with LM Studio"))
    print(f"Workspace: {cyan(Agent.DEFAULT_WORKSPACE)}")
    model_label = agent.llm.model_name
    profile_part = (
        f"  |  Profile: {agent.llm._profile_name}" if agent.llm._profile_name else ""
    )
    print(f"Model: {cyan(model_label)}{gray(profile_part)}")
    try:
        from agent_core.llm.lmstudio import get_models_status

        models = get_models_status()
        loaded = [m for m in models if m.get("loaded")]
        status = (
            f"LM Studio: online ({len(loaded)}/{len(models)} models loaded)"
            if models
            else "LM Studio: online"
        )
        print(green(status) if models else yellow(status))
    except Exception:
        print(yellow("LM Studio: offline"))
    print(blue("=" * 50))
    registry = _build_registry()
    for name in sorted(registry.names()):
        cmd = registry.get(name)
        synopsis = cmd.help_text.splitlines()[0].strip() if cmd else name
        print(f"  {cyan(synopsis)}")
    print(
        gray(
            "  Tip: `propose <taskplan.md>` (or `implement --propose` / "
            "`fix --mypy --propose`) generates a reviewed diff bundle; "
            "it never writes the working tree."
        )
    )
    print(f"  {cyan('quit')} - {gray('Exit')}")
    print(blue("=" * 50))
    _register_commands(registry)
    from agent_core.commands.freshness import (
        diff_snapshots,
        format_stale_warning,
        loaded_module_fingerprints,
    )

    _code_snapshot = loaded_module_fingerprints(__file__)
    _install_signal_handlers(agent)
    _start_line_editing(agent)
    while True:
        try:
            branch = _current_git_branch() or "?"
            print(_build_chat_prompt(agent, branch, datetime.now()))
            user_input = _interruptible_input(CHAT_INPUT_PROMPT)
            if user_input is None:
                agent._save_memory()
                save_meta_policy_weights()
                _save_line_history(workspace=getattr(agent, "workspace", None))
                _warn_uncommitted(agent)
                break
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                agent._save_memory()
                save_meta_policy_weights()
                _save_line_history(workspace=getattr(agent, "workspace", None))
                _warn_uncommitted(agent)
                print(green("Goodbye!"))
                break
            _stale_files = diff_snapshots(_code_snapshot)
            if _stale_files:
                print(yellow(format_stale_warning(_stale_files)))
                _code_snapshot = loaded_module_fingerprints(__file__)
            try:
                parts = shlex.split(user_input, posix=False)
            except ValueError:
                parts = user_input.split(maxsplit=20)
            command = parts[0].lower()
            if command in registry.names():
                import time as _time

                _start = _time.perf_counter()
                print(f"  [cmd] {cyan(command)} running...")
                clear_stop()
                _llm = getattr(agent, "llm", None)
                _chat = getattr(_llm, "chat", None)
                if _llm is not None and _chat is not None:
                    _llm.chat = chat_stoppable(_chat)
                try:
                    try:
                        await registry.execute(command, parts[1:], agent)
                    except FlowStopped:
                        print(yellow("  Flow stopped by user."))
                finally:
                    if _llm is not None and _chat is not None:
                        _llm.chat = _chat
                elapsed = _time.perf_counter() - _start
                print(f"  [cmd] {cyan(command)} done in {green(f'{elapsed:.2f}s')}")
                PerfTracker.record(command, elapsed, user_input)
                record_command_metrics(command, elapsed)
                continue
            else:
                await agent.chat_nlp(user_input)
        except KeyboardInterrupt:
            print(
                yellow(
                    "\nInterrupted — the current run was stopped. Use 'quit' to exit."
                )
            )
        except EOFError:
            if not sys.stdin.isatty():
                print(yellow("\n[stdin] Input stream ended — shutting down."))
            agent._save_memory()
            save_meta_policy_weights()
            _save_line_history(workspace=getattr(agent, "workspace", None))
            _warn_uncommitted(agent)
            break


async def main() -> None:
    """Main entry point - runs interactive mode or the web dashboard."""
    mod = _agent_mod()
    mod._start_kanban_inbound()
    if "--serve" in sys.argv:
        mod.run_dashboard_server()
        return
    if "--dashboard" in sys.argv:
        mod.start_dashboard_thread(mod._dashboard_port())
    await mod.run_interactive()


def cli() -> None:
    """Synchronous console-script entry point (``[project.scripts] agent``).

    ``setuptools`` console scripts call a plain callable; :func:`main` is a
    coroutine, so this wrapper owns the event loop exactly like the
    ``python agent.py`` shim does (KeyboardInterrupt -> clean exit).
    """
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
