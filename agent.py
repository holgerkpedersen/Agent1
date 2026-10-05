#!/usr/bin/env python3
"""Agent implementation with workspace management and tool execution.."""

from collections.abc import Iterator

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, cast

import asyncio
import concurrent.futures
import contextlib
import io
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime

from agent_core import to_windows_path
from agent_core.path_utils import resolve_path, safe_path
from agent_core.pytest_gate import FullRunGate
from agent_core.colors import cyan, green, yellow, blue, magenta, gray, red
from agent_core.constants import (  # noqa: F401
    resolve_model,
    CHAT_HISTORY_JSON_PATH,
    CHAT_HISTORY_TMP_PATH,
    AGENT_MEMORY_JSON_PATH,
    AGENT_MEMORY_TMP_PATH,
    LOOP_NOTE_TAG_KEY,
    TURN_LOG_PATH,
)
from agent_core.config import load_agent_settings, AgentDisplayMode
from agent_core.file_system import FileSystem
from agent_core.file_searcher import FileSearcher
from agent_core.tool_dispatcher import ToolDispatcher
from agent_core.tool_schemas import NLP_TOOL_SCHEMAS, NLP_TOOL_NAMES
from agent_core.modes import (
    MODE_BUILD,
    check_tool_allowed,
    filter_tool_schemas,
    is_plan_mode,
    plan_mode_system_suffix,
    plan_mode_turn_note,
)
from agent_core.subagent_roles import get_role, role_names
from agent_core.llm.provider import (
    get_last_metrics,
    is_connection_failure,
    provider_supports_streaming,
)
from agent_core.llm.tool_loop import ToolLoopRunner
from agent_core.llm.learning import (
    record_turn_outcome,
    recommend_profile,
    save_weights as save_meta_policy_weights,
)
from agent_core.context_management import CORRELATION_ID_CTX, CorrelationIdContext
from agent_core.hue.bridge import HueBridge, HueBridgeError
try:
    from harnessfix.tracing import TraceWriter, trace_enabled
except Exception:  # pragma: no cover - tracing degrades gracefully if unavailable
    TraceWriter = None  # type: ignore[assignment, misc]

    def trace_enabled() -> bool:
        return False

from agent_core.commands.base import (
    Command, FlowStopped, chat_stoppable, clear_stop, save_file_py
)
from agent_core.commands.registry import CommandRegistry
from agent_core.decisions import decisions_as_system_prompt
from agent_core.skills import (
    DEFAULT_PAGE_LINES,
    SKILL_INDEX_MARKER,
    SkillError,
    discover_skills,
    format_skill_page,
    load_skill_index,
    match_skills_for_input,
    read_skill,
    skill_hint_block,
)
from agent_core.habits import HABITS_MARKER, habits_block, load_habits
from agent_core.memory import (
    SEMANTIC_MEMORY_MARKER,
    load_semantic_memory,
    semantic_memory_block,
)
from agent_core.symbol_intel import collect_definitions, collect_references
from agent_core.commands.read_cmd import ReadCommand
from agent_core.commands.write_cmd import WriteCommand
from agent_core.commands.search_cmd import SearchCommand
from agent_core.commands.clear_cmd import ClearCommand
from agent_core.commands.model_cmd import ModelCommand
from agent_core.commands.analyze_cmd import AnalyzeCommand
from agent_core.commands.plan_cmd import PlanCommand
from agent_core.commands.entities_cmd import EntitiesCommand
from agent_core.commands.taskplan_cmd import TaskplanCommand
from agent_core.commands.cleanup_cmd import CleanupCommand
from agent_core.commands.git_cmd import GitCommand
from agent_core.commands.implement_cmd import ImplementCommand
from agent_core.commands.fix_cmd import FixCommand
from agent_core.commands.workflow_cmd import WorkflowCommand
from agent_core.commands.optimize_cmd import OptimizeCommand
from agent_core.commands.paste_cmd import PasteCommand
from agent_core.commands.paste_image_cmd import PasteImageCommand
from agent_core.commands.perf_cmd import PerfCommand, PerfTracker
from agent_core.commands.display_cmd import DisplayCommand
from agent_core.commands.decide_cmd import DecideCommand
from agent_core.commands.review_cmd import ReviewCommand
from agent_core.commands.run_cmd import RunCommand
from agent_core.commands.self_heal_cmd import SelfHealCommand
from agent_core.commands.reconstruct_cmd import ReconstructCommand
from agent_core.commands.multillm_cmd import MultiLlmCommand
from agent_core.commands.demo_data_cmd import DemoDataCommand

if TYPE_CHECKING:
    from agent_core.monitoring import MetricsCollector
    from agent_core.subagent import SubAgent

# Restrict OpenBLAS to a single thread before any numpy import; moving this
# after the import block is safe because none of the imports above pull numpy
# in at module-import time.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# Windows console default codec is cp1252, which cannot encode many Unicode
# glyphs the UI prints (box-drawing chars, arrows). Reconfigure stdout/stderr
# to UTF-8 at startup so no print() crashes on non-cp1252 characters.
for _stream in (sys.stdout, sys.stderr):
    if _stream is not None:
        try:
            _stream.reconfigure(
                encoding="utf-8", errors="replace"
            )  # type: ignore[union-attr]
        except (ValueError, OSError, AttributeError):
            pass

logger = logging.getLogger(__name__)


def _escape_json_string_controls(text: str) -> str:
    """Escape raw control characters that appear INSIDE JSON string literals.

    A whole-text ``str.replace("\\n", "\\\\n")`` cannot repair these payloads:
    pretty-printed model output has legitimate newlines BETWEEN tokens, and the
    same blind replace mangles those (or still leaves an unescaped newline in a
    value), so the parse fails and the call's arguments get wiped to ``{}``.
    A small state machine escapes only where control chars are actually illegal
    — inside string values — leaving inter-token whitespace untouched.
    """
    out: list[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            continue
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = False
            out.append(ch)
            continue
        code = ord(ch)
        if code < 0x20:
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(ch, f"\\u{code:04x}"))
            continue
        out.append(ch)
    return "".join(out)


def _repair_tool_arguments(args_str: str) -> dict[str, Any]:
    """Parse a tool-call ``arguments`` string, repairing common model typos.

    Models (LM Studio in particular) occasionally emit raw control characters —
    unescaped newlines/tabs inside JSON string values.  We escape those ONLY
    inside string literals and retry; anything still unparseable degrades to
    ``{}`` so a malformed payload never aborts the loop (the model sees an
    empty-args result note instead of a crash).
    """
    if not isinstance(args_str, str) or not args_str.strip():
        return {}
    try:
        parsed = json.loads(args_str)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = json.loads(_escape_json_string_controls(args_str))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


@contextlib.contextmanager
def _suppress_and_log(label: str) -> Iterator[None]:
    """Run the block, logging (not propagating) any exception.

    Centralises the "best-effort: log and continue" pattern so call sites have
    a single, duplication-proof handler instead of an inline ``except Exception``
    that an edit can accidentally clone.
    """
    try:
        yield
    except Exception:  # noqa: BLE001 - deliberate best-effort swallow
        logger.warning("%s\n%s", label, traceback.format_exc())


#: Handler signature for the NLP tool dispatch table: each handler receives the
#: parsed JSON arguments and returns a string result for the model.
NlpToolHandler = Callable[[dict[str, Any]], Awaitable[str]]

# ── Interruptible input (so background processes can be killed) ──────────
# On Windows, input() is a C-level blocking call that can't be interrupted by
# signals.  A daemon thread reads stdin while the main thread polls with a
# timeout, allowing SIGBREAK / shutdown flags to be checked periodically.
_SHUTDOWN_FLAG = False
_input_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _signal_break(sig: int, frame: Any) -> None:
    """SIGBREAK handler — set flag so the REPL loop can exit."""
    global _SHUTDOWN_FLAG
    _SHUTDOWN_FLAG = True
    # Force-exit if the REPL is stuck in input() — the flag alone isn't enough
    # because input() won't return until the next keypress on Windows.
    os._exit(1)


# Shutdown-cleanup callbacks — set by _install_signal_handlers when an Agent
# instance is available, so the SIGBREAK handler can persist memory and flush
# traces before exiting (decision #049 / workflow 2026-08-21).
_shutdown_save_memory = None
_shutdown_trace_close = None


def _signal_break_with_cleanup(sig: int, frame: Any) -> None:
    """SIGBREAK handler that performs best-effort cleanup before exit.

    Saves agent memory (cross-session state) and flushes/closes the active
    trace writer so in-flight tool-call effects are not lost from
    ``reports/traces/*.jsonl``.  Each step is guarded — a failing hook logs
    and continues rather than masking the shutdown itself.
    """
    global _SHUTDOWN_FLAG
    _SHUTDOWN_FLAG = True
    from agent_core.security.shutdown import safe_signal_break_handler
    safe_signal_break_handler(
        memory_path=str(AGENT_MEMORY_JSON_PATH),
        trace_writer_close=_shutdown_trace_close,
        save_memory_fn=_shutdown_save_memory,
    )


def _install_signal_handlers(agent: Optional["Agent"] = None) -> None:
    """Register SIGBREAK (Windows Ctrl+Break / taskkill) for clean shutdown.

    When an *agent* instance is provided, the handler performs cleanup
    (memory save + trace flush) before exiting — replacing the bare
    ``os._exit(1)`` that previously skipped these hooks.
    """
    global _shutdown_save_memory, _shutdown_trace_close
    if agent is not None:
        _shutdown_save_memory = agent._save_memory
        # trace_writer_close is set dynamically per-run in chat_nlp;
        # we use a closure that closes whichever writer is active.
        def _close_active_trace() -> None:
            tw = getattr(agent, "_active_trace_writer", None)
            if tw is not None:
                tw.close()
        _shutdown_trace_close = _close_active_trace
        handler = _signal_break_with_cleanup
    else:
        handler = _signal_break
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handler)


def _current_git_branch() -> str | None:
    """Return the current git branch name, or ``None`` if not a git
    repo / unavailable."""
    try:
        head = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".git", "HEAD")
        if os.path.exists(head):
            with open(head, encoding="utf-8", errors="ignore") as fh:
                ref = fh.read().strip()
            if ref.startswith("ref:"):
                return ref.split("/", 2)[-1]
        return None
    except Exception:
        return None


def _interruptible_input(prompt: str) -> str | None:
    """Read a line from stdin using a background thread.

    Returns the stripped input string, or ``None`` when stdin is exhausted or
    shutdown was requested.  On interactive terminals this behaves identically
    to ``input()``; for piped / background processes it allows the main thread
    to break out when a signal arrives.
    """
    global _input_executor
    if _input_executor is None:
        _input_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    future = _input_executor.submit(input, prompt)
    while not _SHUTDOWN_FLAG:
        try:
            return future.result(timeout=0.5).strip()
        except concurrent.futures.TimeoutError:
            continue
        except (EOFError, KeyboardInterrupt):
            return None
    return None


class LLMClient:
    """Thin wrapper around an LLM provider for backward compatibility.

    Delegates to the provider selected by :func:`build_provider` (decision
    #007 — one abstraction; LM Studio default, opencode-go selectable).
    """

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self._model_name: str = resolve_model(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        from agent_core.config import load_agent_settings
        from agent_core.llm.provider import build_provider
        try:
            settings = load_agent_settings()
        except Exception:
            settings = None
        self._provider = build_provider(settings, self._model_name)
        self._profile_name: str | None = None
        #: True once the user pins a profile this session (`model profile
        #: use`) or one is restored from model.json — suppresses the
        #: meta-policy profile suggestion (plan B2; a restored profile
        #: counts as pinned).
        self._profile_pinned: bool = False
        # Restore active profile from model.json on startup
        try:
            from agent_core.constants import load_model_json
            data = load_model_json()
            prof_name = data.get("profile")
            if prof_name:
                from agent_core.llm.model_profiles import get_profile
                profile = get_profile(prof_name)
                self._profile_name = prof_name
                self._profile_pinned = True
                self._provider.apply_profile(
                    prof_name, profile.temperature, profile.max_tokens,
                )
        except Exception as _prof_err:
            logger.exception('Failed to restore active profile from model.json:\n')
            self._profile_name = None
        # Make the llama-server actually serve the requested model (router
        # load/unload, or relaunch with --models-dir).  This is what lets
        # `model llama/... -p llama` (and a persisted llama model at startup)
        # use the Bonsai/whatever model directly instead of silently falling
        # back to whatever GGUF the pre-started server happened to load.
        self._reconcile_llama_model(settings)

    def _reconcile_llama_model(self, settings: Any) -> None:
        """Best-effort: ensure the running llama-server serves self._model_name.

        No-op unless the active provider is llama and the provider is a
        LlamaProvider.  On any failure we log and continue — chat still works
        against whatever the server serves (it self-heals the request ``model``
        id), we just can't guarantee it's the *requested* model.
        """
        try:
            from agent_core.llm.provider import provider_for
            from agent_core.constants import load_model_json
        except Exception:
            return
        if not self._model_name.startswith("llama/"):
            return
        persisted = load_model_json()
        persisted_provider = str(persisted.get("provider") or "")
        provider_setting = getattr(settings, "llm_provider", "lmstudio") if settings else "lmstudio"
        if provider_for(self._model_name, provider_setting, persisted_provider) != "llama":
            return
        provider = self._provider
        # A FailoverProvider may wrap the LlamaProvider when several
        # llm_providers are configured — unwrap to reach the concrete one.
        if type(provider).__name__ == "FailoverProvider":
            for wrapped in getattr(provider, "providers", []):
                if type(wrapped).__name__ == "LlamaProvider":
                    provider = wrapped
                    break
        if type(provider).__name__ != "LlamaProvider":
            return
        with _suppress_and_log("llama model reconciliation failed"):
            from agent_core.llm import llama_server
            api_url = getattr(provider, "api_url", None)
            if not api_url:
                return
            print(f"  [llama] ensuring '{self._model_name}' is served by llama-server...")
            # list_served_models returns bare server IDs (e.g. "Bonsai-27B-Q1_0")
            # while self._model_name may carry the routing prefix ("llama/…").
            bare_name = self._model_name.removeprefix("llama/")
            served = llama_server.list_served_models(api_url)
            if served:
                current_served_model = served[0]
                if current_served_model == bare_name:
                    print(f"  [llama] Model '{bare_name}' already served by server.")
                    provider._cached_server_model_id = current_served_model
                else:
                    print(f"  [llama] Server serving '{current_served_model}', attempting to ensure '{bare_name}' is served...")
                    ok, msg = llama_server.ensure_model_served(api_url, self._model_name)
                    if ok:
                        # Re-check after ensuring it's served
                        served_after = llama_server.list_served_models(api_url)
                        if served_after and served_after[0] == bare_name:
                            provider._cached_server_model_id = served_after[0]
                            print(f"  [llama] Successfully ensured '{bare_name}' is served.")
                        else:
                            print(f"  [llama] WARNING: ensure_model_served reported success, but model not found in list: {msg}")
                    else:
                        print(f"  [llama] WARNING: could not ensure model served: {msg}")
            else:
                print(f"  [llama] No model currently served by the server. Attempting to ensure '{self._model_name}' is served.")
                ok, msg = llama_server.ensure_model_served(api_url, self._model_name)
                if ok:
                    print(f"  [llama] Successfully ensured model served: {msg}")
                else:
                    print(f"  [llama] WARNING: could not ensure model served: {msg}")

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value
        self._provider.model_name = value

    async def chat(
        self, messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None, **kwargs: Any,
    ) -> str:
        """Send chat request to LLM via LM Studio (pass-through wrapper)."""
        return await self._provider.chat(messages, tools, **kwargs)

    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """Chat with real-time token streaming to console."""
        return await self._provider.chat_stream(messages)

    async def chat_with_continuation(
        self, messages: list[dict[str, Any]],
        max_continues: int = 3, max_tokens: int | None = None,
    ) -> str:
        """Chat with auto-resume if response gets truncated at token limit."""
        full_response = ""
        current_messages = [dict(m) for m in messages]

        for i in range(max_continues):
            result = await self.chat(current_messages, max_tokens=max_tokens)

            if result.startswith("[Error") or result.startswith("[LM Studio"):
                return full_response or result

            full_response += result

            stripped = full_response.rstrip()
            if stripped and not stripped.endswith(
                ('```', '}', ')', ']', '"', "'", '.', '\n')
            ):
                print(magenta(
                    f"\n[auto-resume] Truncated ({len(result)} chars), "
                    f"continuing ({i+1}/{max_continues})..."
                ))
                current_messages.append({"role": "assistant", "content": result})
                current_messages.append(
                    {"role": "user", "content":
                     "Continue exactly where you stopped. Output the "
                     "remaining code without repeating anything."}
                )
                continue
            else:
                break

        return full_response

    async def analyze_code(self, code: str) -> str:
        """Analyze code using LLM."""
        return await self._provider.analyze_code(code)


class Agent:
    """Main agent class with workspace management and tool execution."""

    #: Real filesystem path (not a Git-Bash /c/... path) — subprocess cwd must
    #: be a valid Windows directory, so derive it from this file's location.
    DEFAULT_WORKSPACE = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, workspace: str | None = None, model_name: str | None = None):
        # Translate Git-Bash-style paths (/c/Dev/...) to Windows paths so that
        # subprocess cwd and filesystem tools always see a valid directory.
        self.workspace = os.path.abspath(
            to_windows_path(workspace or self.DEFAULT_WORKSPACE)
        )
        self.model_name = resolve_model(model_name)

        self._semantic_index: dict[str, set[int]] = defaultdict(set)
        self._files_read: set[str] = set()
        self._file_mtimes: dict[str, float] = {}
        #: Keeps FULL pytest runs rare and late: a full run is refused unless a
        #: real source file changed since the baseline, and at most
        #: AGENT_MAX_FULL_PYTEST_RUNS (default 1) per session may be spent.
        self._full_run_gate = FullRunGate(self.workspace)
        #: Per-tool-call file effects accumulated while a trace sink is active
        #: (self-improvement files-affected recording; decision #048 — this is
        #: None except during a traced chat_nlp loop, so untraced runs are
        #: byte-identical to before).
        self._pending_effects: list[str] | None = None
        self._knowledge_graph: dict[str, Any] = {}
        self._working_memory: list[Any] = []
        self._history: list[Any] = []
        #: NLP conversation context — persisted to chat_history.json so a new
        #: session continues where the previous one left off.
        self._chat_history: list[dict[str, Any]] = self._load_chat_history()
        #: Index in ``_chat_history`` where the current turn's messages begin
        #: (reset at the top of every :meth:`chat_nlp` call, before the user
        #: message is appended).  A restored session's history contains tool
        #: results from PREVIOUS sessions; per-turn scans such as
        #: :meth:`_mutating_files_this_turn` must never treat those as files
        #: changed by the live turn.
        self._turn_start_index: int = len(self._chat_history)
        #: Run id of the current turn (the turn's correlation id, decision
        #: #050).  Every experience row written this turn carries it in the
        #: llm_decisions provenance log, so one turn's decisions are linkable
        #: as a unit.  ``None`` outside a turn (direct calls, tests).
        self._turn_run_id: str | None = None
        #: Consecutive ``read`` tool calls in the current turn (read-loop
        #: guard).  Reset at turn start and by every non-read tool call; when
        #: it crosses ``_MAX_CONSECUTIVE_READS`` a steering note is appended
        #: to read results telling the model to stop paging and act.
        self._read_streak: int = 0
        #: True while a subagent is executing a tool through this parent's
        #: executor.  Child reads must not inflate the PARENT turn's
        #: read-loop streak (the steering note would leak into the child's
        #: tool result); nested delegation restores the previous value.
        self._delegating: bool = False
        #: Names of currently-running delegated subagents (concurrency cap
        #: for the ``delegate`` tool — children exist to keep contexts small;
        #: a pile of running subagents re-creates the pressure they avoid).
        self._active_subagents: set[str] = set()
        self._delegate_counter: int = 0
        self._nlp_workspace: str | None = None  # workspace override for
        # NLP tools (set by paste --workspace)
        #: Session mode ("build" | "plan", see :mod:`agent_core.modes`).
        #: Plan mode restricts the NLP tool loop to read-only tools so no
        #: file changes while researching; switched via the ``mode`` command.
        self.mode: str = MODE_BUILD

        #: Cross-session memory (files read, semantic index, knowledge graph,
        #: working memory) — restored from agent_memory.json so work done in a
        #: previous session is not forgotten.
        self._load_memory()

        # Initialize LLM client for AI analysis (LM Studio)
        self.llm = LLMClient(model_name=self.model_name)
        # Route the client's transport through this Agent's construction seam.
        # ``build_transport`` is the documented injection point — its docstring
        # says it "returns the provider object that becomes client._provider",
        # and tests patch it to swap in a FakeTransport.  Default behaviour is
        # unchanged: the default build_transport returns exactly what LLMClient
        # built itself, so this is a no-op on real runs; only a patched seam
        # changes anything.  Re-apply any restored profile after swapping so
        # replacing the transport never silently drops the active model profile.
        try:
            _seamed = self.build_transport(self.llm)
            if _seamed is not self.llm._provider:
                prof_name = self.llm._profile_name
                prof = None
                if prof_name:
                    from agent_core.llm.model_profiles import get_profile
                    prof = get_profile(prof_name)
                self.llm._provider = _seamed
                if prof_name and prof is not None:
                    self.llm._provider.apply_profile(
                        prof_name, prof.temperature, prof.max_tokens,
                    )
        except Exception as _seam_err:  # best-effort — never crash construction
            logger.exception("build_transport seam wiring failed:\n")

        # Initialize extracted components
        self.fs = FileSystem(self.workspace)
        self.searcher = FileSearcher(self.workspace)
        self.dispatcher = ToolDispatcher(on_tool=_emit_tool_metrics)
        self._register_tool_handlers()

    def _register_tool_handlers(self) -> None:
        """Register tool handlers with the dispatcher."""
        self.dispatcher.register("read_file", lambda args: self._tool_read_file(**args))
        self.dispatcher.register(
            "write_file", lambda args: self._tool_write_file(**args))
        self.dispatcher.register(
            "apply_patch", lambda args: self._tool_apply_patch(**args))
        self.dispatcher.register("edit_file", lambda args: self._tool_edit_file(**args))
        self.dispatcher.register("search", lambda args: self._tool_search(**args))
        self.dispatcher.register("search_file", lambda args: self._tool_search(**args))
        self.dispatcher.register(
            "list_files", lambda args: self._tool_list_files(**args))
        self.dispatcher.register(
            "delete_file", lambda args: self._tool_delete_file(**args))
        self.dispatcher.register(
            "analyze_file", lambda args: self._tool_analyze_file(**args))
        self.dispatcher.register(
            "llm_analyze", lambda args: self._tool_llm_analyze(**args))

    def build_transport(self, client: "LLMClient") -> Any:
        """Build the transport (provider) that drives this agent's LLM client.

        Public construction seam: returns the provider object that becomes
        ``client._provider`` — the object whose ``chat(messages, tools, ...)``
        answers every turn of a tool loop.  Callers may patch or replace it
        (e.g. agentic_bench injects offline fake transports in tests) without
        touching private attributes; the default builds the production provider
        from settings and the resolved model name.
        """
        from agent_core.config import load_agent_settings
        from agent_core.llm.provider import build_provider
        try:
            settings = load_agent_settings()
        except Exception:
            settings = None
        return build_provider(settings, client.model_name)

    # ── Sub-agent support ───────────────────────────────────────────────
    def spawn_subagent(
        self, name: str, workspace: str | None = None,
        role: str | None = None,
    ) -> "SubAgent":
        """Create a child :class:`SubAgent` sharing this agent's workspace.

        The subagent gets its own conversation history so work done there does
        not pollute the parent's context.  It inherits the parent's model name
        and filesystem access by default; pass *workspace* to isolate it to a
        different directory within the same project tree.

        Pass *role* (see :mod:`agent_core.subagent_roles`) to give the child a
        persona, a tool whitelist and a turn cap; a plan-mode parent caps every
        child at read-only regardless of the role's own mode.
        """
        from agent_core.subagent import SubAgent
        return SubAgent(parent=self, name=name, workspace=workspace, role=role)

    def run_parallel_tasks(
        self, tasks: list[Callable[[], str]], max_workers: int = 10
    ) -> list[str]:
        """Run *tasks* in parallel threads and collect their string results.

        Each task is typically a closure that spawns a :meth:`spawn_subagent`,
        runs work, and returns its result.  The parent can then inspect or
        merge the subagent outputs.
        """
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(task) for task in tasks]
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:  # pragma: no cover - defensive
                    results.append(f"[task-error] {exc}")
        return results

    # ── Dashboard data feed (shared MetricsCollector bridge) ────────────
    def record_demo_activity(
        self, activity: str = "analyze", latency_ms: float = 120.0
    ) -> dict[str, float]:
        """Record one synthetic activity into the shared metrics collector.

        Mirrors exactly what ``record_command_metrics()`` writes for a real
        REPL command, so the dashboard treats demo data identically:
        ``command.<name>.count`` counter + elapsed-seconds histogram sample
        + ``last.command.seconds`` gauge. Returns the written values.
        """
        elapsed_s = max(0.001, latency_ms / 1000.0)
        # Same three writes as a real command (record_command_metrics) so the
        # dashboard cannot tell demo data from live data.
        _emit_command_metrics(activity, elapsed_s)
        collector = self.get_metrics_collector()
        return {
            "events": 3,
            "counter": collector.get_counter_value(f"command.{activity}.count"),
            "elapsed_s": elapsed_s,
        }

    def get_metrics_collector(self) -> "MetricsCollector":
        """Instance delegate for the process-wide collector (test seam)."""
        return get_metrics_collector()

    def _resolve_nlp_path(self, path: str) -> str:
        """Resolve a path for NLP tool use, honouring _nlp_workspace if set.

        Relative paths are resolved against the effective workspace (never the
        process CWD), so tools stay scoped to the agent's workspace.
        """
        import os as _os
        base = self._nlp_workspace or self.workspace
        if not _os.path.isabs(path):
            return _os.path.normpath(_os.path.join(base, path))
        return path

    async def _save_verify_note(self, path: str, content: str, ok_msg: str) -> str:
        """Persist *content*, py_compile-verify .py files, note the effect.

        Shared tail of the NLP ``write`` and ``edit`` tool handlers so both
        produce byte-identical success output (message + verification line)
        and identical trace-effect bookkeeping.  Returns the "Skipped …
        (no changes)" message when ``save_file_py`` finds nothing to do.
        """
        if not save_file_py(path, content, auto_yes=True):
            return f"Skipped {path} (no changes)"
        verify = await self._verify_file(path) if path.endswith(".py") else ""
        self._note_effect(path)
        return f"{ok_msg}\n{verify}".strip()

    async def _verify_file(self, path: str) -> str:
        """Run py_compile on *path* and return a short verification summary."""
        try:
            cwd = os.path.abspath(
                to_windows_path(self._nlp_workspace or self.workspace))
            if not os.path.isdir(cwd):
                cwd = self.workspace
            r = subprocess.run(
                [sys.executable, "-m", "py_compile", path],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                cwd=cwd,
            )
        except OSError as e:
            return f"[verify] py_compile could not run: {e}"
        if r.returncode == 0:
            return "[verify] py_compile ✓"
        return f"[verify] py_compile ✗: {r.stderr.strip()[:300]}"

    def _note_effect(self, path: str) -> None:
        """Record one file affected by the current tool call.

        Only active while ``_pending_effects`` is armed (i.e. inside a traced
        chat_nlp run, decision #048) — otherwise this is a no-op cost-free
        guard, so untraced agent behaviour is unchanged.
        """
        if self._pending_effects is None:
            return
        abs_path = os.path.abspath(to_windows_path(str(path)))
        if abs_path not in self._pending_effects:
            self._pending_effects.append(abs_path)

    def _take_trace_effects(self, name: str, args: dict[str, Any]) -> list[str]:
        """``effects_fn`` for ToolLoopRunner: drain this tool call's effects.

        Returns the files affected by the just-executed tool call and resets
        the buffer for the next one.  Never raises (guarded by the loop too).
        """
        effects = list(self._pending_effects or [])
        if self._pending_effects is not None:
            self._pending_effects.clear()
        return effects

    def set_mode(self, mode: str) -> None:
        """Switch the session mode (``build`` | ``plan``).

        Unknown tags raise ``ValueError`` — the ``mode`` command validates
        before calling, so this is a programming-error guard, not a silent
        fallback to a mode the user did not ask for.
        """
        if mode not in (MODE_BUILD, "plan"):
            raise ValueError(f"unknown mode: {mode!r}")
        self.mode = mode

    def is_plan_mode(self) -> bool:
        """True while plan mode (read-only toolset) is active."""
        return is_plan_mode(self.mode)

    async def _execute_tool_call(self, name: str, args: dict[str, Any]) -> str:
        """Execute a native tool call from the NLP conversation.

        *name* must be one of :data:`NLP_TOOL_NAMES`; *args* is the parsed JSON
        arguments dict.  Writing tools (write/edit) append a py_compile
        verification summary so the model can report verified results.

        Plan mode is enforced HERE — the single choke point every agentic
        loop goes through (``chat_nlp`` and ``multillm`` both pass
        ``self._execute_tool_call`` as ``execute_tool_fn``) — so a mutating
        call is rejected even when its schema was still advertised.

        Dispatch goes through ``self._nlp_tool_handlers`` — one small handler
        per tool instead of one monolithic if-chain.  Unknown tools return an
        error string naming the available set; handler exceptions are caught
        here (the tool loop treats them as tool errors) so a single bad call
        cannot take down the whole turn.
        """
        name = name.lower()
        if name != "read":
            self._read_streak = 0
        rejection = check_tool_allowed(name, self.mode)
        if rejection:
            return rejection
        handler = self._nlp_tool_handlers().get(name)
        if handler is None:
            return (
                f"Unknown tool: {name}. "
                f"Available: {', '.join(sorted(NLP_TOOL_NAMES))}"
            )
        try:
            return await handler(args)
        except Exception as e:
            logger.exception("NLP tool %r failed", name)
            return f"{name} error: {e}"

    async def _nlp_delegate(self, args: dict[str, Any]) -> str:
        """Run one task in a role subagent and return its final answer.

        The child gets an isolated history (parent context stays small) and a
        restricted toolset; its tool calls flow through this parent's
        executor with ``_delegating`` set so child reads never inflate the
        parent turn's read-loop streak.  Concurrency is capped at
        ``_MAX_ACTIVE_SUBAGENTS``; a hung child surfaces as an error after
        ``_DELEGATE_TIMEOUT_S`` instead of holding the turn hostage.
        """

        from agent_core.modes import MODE_PLAN, is_plan_mode

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

        if len(self._active_subagents) >= _MAX_ACTIVE_SUBAGENTS:
            return (
                f"Error: {_MAX_ACTIVE_SUBAGENTS} subagents already running — "
                "wait for one to finish before delegating again."
            )

        self._delegate_counter += 1
        name = f"{spec.name}-{self._delegate_counter}"
        try:
            sub = self.spawn_subagent(name, role=role)
        except Exception as exc:  # defensive: a broken spawn must not kill the turn
            logger.exception("Delegated subagent %s failed to spawn", name)
            return f"[delegate:{name}] failed to spawn: {exc}"
        self._active_subagents.add(name)
        prev_delegating = self._delegating
        self._delegating = True
        try:
            result = await asyncio.wait_for(
                sub.respond(task), timeout=_DELEGATE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            result = (
                f"[delegate:{name}] timed out after {_DELEGATE_TIMEOUT_S:.0f}s "
                "— the task did not finish. Partial context preserved in the "
                "subagent; retry with a smaller task."
            )
        except Exception as exc:  # defensive: a broken child must not kill the turn
            logger.exception("Delegated subagent %s failed", name)
            result = f"[delegate:{name}] failed: {exc}"
        finally:
            self._delegating = prev_delegating
            self._active_subagents.discard(name)

        summary = sub.get_context_summary(max_messages=3)
        reason = getattr(sub, "last_termination_reason", None)
        reason_note = (
            "" if reason == "answer"
            else f", stopped early: {reason or 'unknown'}"
        )
        mode_note = ""
        if is_plan_mode(self.mode):
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

    async def _nlp_delegate_batch(self, args: dict[str, Any]) -> str:
        """Fan one task out to several roles in parallel and merge reports.

        Each child still runs alone in its own context; the parent only pays
        for the merged summaries.  The per-call cap is
        ``min(len(roles), _MAX_ACTIVE_SUBAGENTS)``.
        """
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
        unique_roles = unique_roles[:_MAX_ACTIVE_SUBAGENTS]
        dropped = len(roles) - len(unique_roles)

        self._delegate_counter += 1
        batch_id = self._delegate_counter

        async def run_one(role_name: str, idx: int) -> tuple[str, str]:
            name = f"{role_name}-b{batch_id}-{idx}"
            try:
                sub = self.spawn_subagent(name, role=role_name)
            except Exception as exc:
                logger.exception("Batch subagent %s failed to spawn", name)
                return role_name, f"### {role_name}\nfailed to spawn: {exc}"
            prev = self._delegating
            self._delegating = True
            self._active_subagents.add(name)
            try:
                res = await asyncio.wait_for(
                    sub.respond(task), timeout=_DELEGATE_TIMEOUT_S,
                )
            except asyncio.TimeoutError:
                res = (f"timed out after {_DELEGATE_TIMEOUT_S:.0f}s — no result.")
            except Exception as exc:
                logger.exception("Batch subagent %s failed", name)
                res = f"failed: {exc}"
            finally:
                self._delegating = prev
                self._active_subagents.discard(name)
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
        self, command: Command, args: list[str], done_msg: str,
    ) -> str:
        """Run a registered command with stdout captured and return its output.

        Shared by the NLP ``fix`` and ``analyze`` tool handlers so the
        "capture → execute → fall back when empty" pattern lives in one place.
        """
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            await command.execute(args, self)
        return buf.getvalue() or done_msg

    def _effective_ws_dir(self) -> str:
        """Resolve the subprocess cwd for NLP tools.

        The workspace override (``paste --workspace``) wins when it exists;
        a bad override (e.g. a Git-Bash /c/... path) falls back to the real
        workspace because subprocess cwd must be an existing directory.
        """
        ws_dir = os.path.abspath(
            to_windows_path(self._nlp_workspace or self.workspace)
        )
        if not os.path.isdir(ws_dir):
            ws_dir = self.workspace
        return ws_dir

    async def _nlp_create_subagent(self, args: dict[str, Any]) -> str:
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
        if not hasattr(self, "_subagents"):
            self._subagents: dict[str, object] = {}
        if name in self._subagents:
            return (
                f"Error: subagent '{name}' already exists. "
                "Use a different name or reset it first."
            )
        sub = self.spawn_subagent(name, workspace=workspace, role=role)
        self._subagents[name] = sub
        return (
            f"Created subagent '{name}' with role '{spec.name}' "
            f"({spec.title})."
        )

    async def _nlp_run_subagent_task(self, args: dict[str, Any]) -> str:
        """Send a task to an existing named subagent."""
        name = str(args.get("name", "")).strip().strip('"').strip("'")
        task = str(args.get("task", "")).strip()
        if not name or not task:
            return "Error: run_subagent_task requires 'name' and 'task'."
        subs = getattr(self, "_subagents", {})
        sub = subs.get(name)
        if sub is None:
            return (
                f"Error: subagent '{name}' not found. "
                "Create it with create_subagent first."
            )
        result = await sub.respond(task)
        summary = sub.get_context_summary(max_messages=3)
        return f"{result}\n\n{summary}"

    def _nlp_tool_handlers(self) -> dict[str, NlpToolHandler]:
        """Name → handler map for every entry in :data:`NLP_TOOL_NAMES`.

        Built per call from bound methods (cheap; keeps handlers testable and
        avoids stale-closure bugs after ``_nlp_workspace`` changes).
        """
        return {
            "analyze": self._nlp_analyze,
            "definitions": self._nlp_definitions,
            "diff": self._nlp_diff,
            "edit": self._nlp_edit,
            "fix": self._nlp_fix,
            "git": self._nlp_git,
            "jev_decide": self._nlp_jev_decide,
            "list_files": self._nlp_list_files,
            "merge": self._nlp_merge,
            "read": self._nlp_read,
            "read_skill": self._nlp_read_skill,
            "references": self._nlp_references,
            "run": self._nlp_run,
            "search": self._nlp_search,
            "speculate": self._nlp_speculate,
            "tests": self._nlp_tests,
            "web_search": self._nlp_web_search,
            "write": self._nlp_write,
            "delegate": self._nlp_delegate,
            "delegate_batch": self._nlp_delegate_batch,
            "create_subagent": self._nlp_create_subagent,
            "run_subagent_task": self._nlp_run_subagent_task,
            "mcp_tools": self._nlp_mcp_tools,
            "mcp_call": self._nlp_mcp_call,
            "get_current_datetime": self._nlp_get_current_datetime,
            "hue_control": self._nlp_hue_control,
            "plan_status": self._nlp_plan_status,
            "plan_start": self._nlp_plan_start,
            "plan_step": self._nlp_plan_step,
            "plan_finish": self._nlp_plan_finish,
        }

    async def _nlp_mcp_tools(self, args: dict[str, Any]) -> str:
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

    async def _nlp_mcp_call(self, args: dict[str, Any]) -> str:
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

    async def _nlp_definitions(self, args: dict[str, Any]) -> str:
        """Index a Python file's classes/functions via AST (plan item B-#8)."""
        path = self._resolve_nlp_path(str(args.get("path", "")).strip('"').strip("'"))
        if not path.endswith(".py"):
            return f"Error: definitions needs a .py file, got: {path}"
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except FileNotFoundError:
            return f"File not found: {path}"
        except OSError as e:
            return f"Definitions error: {e}"
        self._note_effect(path)
        return collect_definitions(source, filename=path)

    async def _nlp_references(self, args: dict[str, Any]) -> str:
        """Locate a symbol across workspace .py files (plan item B-#8)."""
        symbol = str(args.get("symbol", "")).strip().strip('"').strip("'")
        if not symbol:
            return "Error: references requires a symbol name."
        try:
            max_results = max(1, min(int(args.get("max_results") or 60), 200))
        except (TypeError, ValueError):
            max_results = 60
        self._note_effect(self.workspace)
        return await asyncio.to_thread(
            collect_references, symbol, self._effective_ws_dir(), max_results,
        )

    async def _nlp_jev_decide(self, args: dict[str, Any]) -> str:
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
                    workspace=getattr(self, "workspace", None),
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
                log_workspace=getattr(self, "workspace", None),
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear setup error
            return f"jev_decide error: {exc}"
        try:
            result = await engine.decide(question, state)
        except Exception as exc:  # noqa: BLE001 - never kill the turn
            return f"jev_decide error: {exc}"
        return result.summary()

    async def _nlp_speculate(self, args: dict[str, Any]) -> str:
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
            await SpeculateCommand().execute(parts, self)
        text = buf.getvalue().strip()
        if not text:
            return "speculate produced no output."
        # The committed answer is the point of this tool — give it more room
        # than a 5k shell dump before truncating.
        return _truncate_output(text, limit=8000)

    async def _nlp_search(self, args: dict[str, Any]) -> str:
        """Search workspace files for text (first 30 matches)."""
        query = str(args.get("query", ""))
        search_path = self._resolve_nlp_path(str(args.get("path") or "."))
        try:
            found = await self.searcher.search(query, search_path)
        except Exception as e:
            return f"Search error: {e}"
        if not found or found == "No matches found":
            return "No files found matching that query."
        self._note_effect(search_path)
        lines = found.splitlines()
        return "\n".join(f"  {line}" for line in lines[:30])

    async def _nlp_read(self, args: dict[str, Any]) -> str:
        """Read one file (``path``) or several files (``paths``) in one call.

        The ``paths`` array is what lets a model satisfy "read these two files"
        in a SINGLE tool call instead of refusing ("I can only read one file at
        a time") or asking the user to re-specify.
        """
        raw_paths = args.get("paths")
        if isinstance(raw_paths, list) and raw_paths:
            return await self._nlp_read_many(raw_paths, args)
        if not str(args.get("path", "")).strip():
            return "Read error: provide 'path' (one file) or 'paths' (a list)."
        path = self._resolve_nlp_path(str(args.get("path", "")).strip('"').strip("'"))
        return await self._read_page(path, args, original=args.get("path"))

    async def _nlp_read_many(self, raw_paths: list[Any], args: dict[str, Any]) -> str:
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
        if len(cleaned) > _MAX_READ_FILES:
            extra = (
                f"\n\n[read capped at {_MAX_READ_FILES} files per call; "
                f"{len(cleaned) - _MAX_READ_FILES} skipped]"
            )
            cleaned = cleaned[:_MAX_READ_FILES]
        blocks: list[str] = []
        for raw in cleaned:
            path = self._resolve_nlp_path(raw)
            text = await self._read_page(path, args, original=raw)
            blocks.append(f"=== {raw} ===\n{text}")
        return "\n\n".join(blocks) + extra

    async def _read_page(self, path: str, args: dict[str, Any], original: Any) -> str:
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
            limit = max(1, min(int(args.get("limit") or 100), _MAX_READ_LINES))
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
            and _CONTEXT_AST_THRESHOLD_KB > 0
            and path.endswith(".py")
            and os.path.isfile(path)
            and os.path.getsize(path) > _CONTEXT_AST_THRESHOLD_KB * 1024
        ):
            content = await self.read_file(path, track_read=False)
            if content.startswith("File not found") or content.startswith("Error"):
                return content
            self._note_effect(path)
            return collect_definitions(content, filename=path) + (
                "\n\n[Large file: showing a definitions summary. "
                "Call read with offset=<line> to fetch source lines.]"
            )
        content = await self.read_file(path, track_read=False)
        if content.startswith("File not found") or content.startswith("Error"):
            return content
        self._note_effect(path)
        if not self._delegating:
            self._read_streak += 1
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
        if self._read_streak >= _MAX_CONSECUTIVE_READS:
            result += _READ_LOOP_NOTE.format(n=self._read_streak)
        return result

    async def _nlp_read_skill(self, args: dict[str, Any]) -> str:
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
            limit = int(args.get("limit") or DEFAULT_PAGE_LINES)
        except (TypeError, ValueError):
            return "Skill error: offset/limit must be integers."
        try:
            page = read_skill(
                self._effective_ws_dir(), name, offset=offset, limit=limit,
            )
        except SkillError as exc:
            return f"Skill error: {exc}"
        except Exception:
            logger.exception('read_skill failed:\n')
            return f"Skill error: could not load skill {name!r}."
        self._note_effect(str(page.skill.path))
        return format_skill_page(page)

    async def _nlp_list_files(self, args: dict[str, Any]) -> str:
        """List up to 50 directory entries, directories marked with ``/``."""
        raw_path = str(args.get("path") or ".").strip('"').strip("'")
        path = self._resolve_nlp_path(raw_path)
        abs_path = path if os.path.isabs(path) else os.path.abspath(path)
        try:
            entries = sorted(os.listdir(abs_path))[:50]
        except NotADirectoryError:
            return f"Not a directory: {path}"
        except FileNotFoundError:
            return f"Directory not found: {path}"
        except OSError as e:
            return f"List error: {e}"
        self._note_effect(path)
        lines = []
        for entry in entries:
            full = os.path.join(abs_path, entry)
            suffix = "/" if os.path.isdir(full) else ""
            lines.append(f"  {entry}{suffix}")
        return "\n".join(lines)

    async def _nlp_fix(self, args: dict[str, Any]) -> str:
        """Run the ``fix`` REPL command on a resolved target file."""
        resolved_args = [str(a) for a in args.get("args", [])]
        target = (
            resolved_args[0] if resolved_args and not resolved_args[0].startswith("--")
            else None
        )
        if target:
            resolved_args[0] = self._resolve_nlp_path(target)
        output = await self._run_command_quietly(
            FixCommand(), resolved_args, "Fix executed. Check the file for changes.",
        )
        if resolved_args and not resolved_args[0].startswith("--"):
            self._note_effect(resolved_args[0])
        return output

    async def _nlp_write(self, args: dict[str, Any]) -> str:
        """Write *content* to *path*; py_compile-verify Python files."""
        path = self._resolve_nlp_path(str(args.get("path", "")))
        content = str(args.get("content", ""))
        try:
            return await self._save_verify_note(
                path, content, f"Written {path} ({len(content)} bytes)",
            )
        except Exception as e:
            return f"Write error: {e}"

    async def _nlp_run(self, args: dict[str, Any]) -> str:
        """Execute one shell command with destructive-pattern blocking,
        whole-tree timeout kill, and Unix-ism hints."""
        cmd_to_run = str(args.get("command", "")).strip()
        # Upper bound: an unbounded model-supplied timeout could stall a turn
        # for hours; 10 min is the hard ceiling (longer jobs should be split).
        timeout = max(1, min(int(args.get("timeout") or 120), _MAX_RUN_TIMEOUT_S))
        if not cmd_to_run:
            return "Error: run requires a command."
        is_full_run = _is_full_pytest_command(cmd_to_run)
        if is_full_run:
            # A full run costs ~2 min and must be EARNED (a real source file
            # changed) and RARE (one per session by default) - so it happens at
            # the END of real work, never after a _tmp_* scratch probe.
            refusal = self._full_run_gate.check()
            if refusal:
                return refusal
            # The suite's own budget (PYTEST_FULL_SUITE_TIMEOUT) is the source
            # of truth here; a model-guessed 600s would kill the run mid-suite.
            # This escape hatch is deliberately NOT capped by _MAX_RUN_TIMEOUT_S.
            timeout = max(timeout, int(_pytest_full_suite_timeout()))
        blocked = _blocked_shell_command(cmd_to_run)
        if blocked:
            return f"Error: Dangerous command blocked ({blocked}): {cmd_to_run}"
        # Windows: start the shell in its own process group so a timeout
        # can kill the WHOLE tree. Killing only cmd.exe leaves orphaned
        # children (e.g. a harnessfix.loop python process) running and
        # holding the captured pipes open — the caller then waits far
        # beyond the timeout while the "killed" command keeps working.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            proc = subprocess.Popen(
                cmd_to_run,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self._effective_ws_dir(),
                creationflags=creationflags,
            )
            try:
                output, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_process_tree(proc)
                output, err = proc.communicate()
                return (
                    f"Command timed out after {timeout}s — process tree killed. "
                    "For long-running jobs, pass a larger timeout."
                )
        except Exception as e:
            return f"Error: {e}"
        if is_full_run:
            self._full_run_gate.record_full_run()
        return _truncate_output(_shape_run_stderr(err, output, proc.returncode))

    async def _nlp_git(self, args: dict[str, Any]) -> str:
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
                self._effective_ws_dir(), 30, "Git",
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
            ["git"] + base + extra, self._effective_ws_dir(), 30, "Git",
        )
        return error or output

    async def _nlp_merge(self, args: dict[str, Any]) -> str:
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
            git_merge.run_merge, self._effective_ws_dir(), args,
        )

    async def _nlp_diff(self, args: dict[str, Any]) -> str:
        """Show ``git diff --no-color`` for one file (optionally against another)."""
        file1 = self._resolve_nlp_path(str(args.get("file1", "")))
        file2 = args.get("file2")
        if not file1:
            return "Error: diff requires at least one file."
        cmd = ["git", "diff", "--no-color", "--"]
        if file2:
            cmd += [file1, self._resolve_nlp_path(str(file2))]
        else:
            cmd += [file1]
        output, error = _run_subprocess_captured(
            cmd, self._effective_ws_dir(), 30, "Diff",
        )
        if error:
            return error
        return output if output != "(no output)" else "(no differences found)"

    async def _nlp_edit(self, args: dict[str, Any]) -> str:
        """Replace the first exact occurrence of *old_text* in a file."""
        path = self._resolve_nlp_path(str(args.get("path", "")))
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        try:
            # Context manager guarantees the handle closes even if
            # save_file_py raises below.
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except FileNotFoundError:
            return f"File not found: {path}"
        except OSError as e:
            return f"Edit error: {e}"
        if old_text not in content:
            return (
                f"Text not found in {path}. Make sure old text matches "
                "exactly (including whitespace)."
            )
        new_content = content.replace(old_text, new_text, 1)
        try:
            return await self._save_verify_note(
                path, new_content, f"Edited {path}",
            )
        except Exception as e:
            return f"Edit error: {e}"

    async def _nlp_tests(self, args: dict[str, Any]) -> str:
        """Run pytest/unittest on *path* (300s cap covers the full suite)."""
        test_path = self._resolve_nlp_path(str(args.get("path") or "."))
        framework = str(args.get("framework") or "pytest")
        #: The full suite takes ~2.5 minutes, so 120s is too short and made
        #: the agent split runs into subsets. 300s covers whole-suite runs.
        timeout = 300
        if os.path.abspath(test_path) == os.path.abspath(self._effective_ws_dir()):
            # Whole-workspace run: same "earned and rare" gate as the run tool -
            # no real change means a full pass verifies nothing.
            refusal = self._full_run_gate.check()
            if refusal:
                return refusal
            # Honour the same budget as the watchdog.
            timeout = max(timeout, int(_pytest_full_suite_timeout()))
            self._full_run_gate.record_full_run()
        if framework == "pytest":
            cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
        else:
            cmd = [sys.executable, "-m", "unittest", test_path, "-v"]
        # The full suite takes ~2.5 minutes; 300s covers whole-suite runs.
        output, error = _run_subprocess_captured(
            cmd, self._effective_ws_dir(), timeout, "Tests",
        )
        return error or output

    async def _nlp_analyze(self, args: dict[str, Any]) -> str:
        """Run the ``analyze`` REPL command (AI analysis) on an optional path.

        The path may be a FILE or a DIRECTORY (a folder is summarized from a
        bounded sample of its files); omitting it analyzes the whole workspace.
        """
        analyze_args: list[str] = []
        if args.get("path"):
            analyze_args = [self._resolve_nlp_path(str(args["path"]))]
            self._note_effect(analyze_args[0])
        try:
            from agent_core.commands.analyze_cmd import AnalyzeCommand
            return await self._run_command_quietly(
                AnalyzeCommand(), analyze_args, "Analysis complete."
            )
        except Exception as e:
            return f"Analyze error: {e}"

    async def _nlp_web_search(self, args: dict[str, Any]) -> str:
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

    async def _nlp_get_current_datetime(self, args: dict[str, Any]) -> str:
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

    async def _nlp_hue_control(self, args: dict[str, Any]) -> str:
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
                for l in lights:
                    state = "ON" if l.get("on") else "OFF"
                    bri = f"{l.get('brightness'):.1f}%" if l.get("brightness") is not None else "?"
                    lines.append(f"  [{l['id']}] {l['name']} -- {state}, brightness={bri}")
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

    # ── Plan workflow NLP tools ─────────────────────────────────────

    async def _get_plan_state(self) -> dict[str, Any]:
        """Detect current plan status + tasks from the filesystem."""
        from agent_core.commands.doc_paths import latest_run_dir
        from agent_core.plan_execution.parser import parse_plan_tasks

        plan_dir = latest_run_dir(self.workspace)
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

    async def _nlp_plan_status(self, args: dict[str, Any]) -> str:
        """Query the current plan's lifecycle status and parsed task list."""
        state = await self._get_plan_state()
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

    async def _nlp_plan_start(self, args: dict[str, Any]) -> str:
        """Start plan execution: proposed -> executing, validate, run tasks."""
        if self.is_plan_mode():
            return "Error: plan mode is read-only. Run 'mode build' first."

        state = await self._get_plan_state()
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

        decision_gate = PlanDecisionGate(Path(self.workspace))
        gate_result = decision_gate.validate(state["plan_text"])
        if not gate_result.passed:
            return "Decision gate failed - plan violates constraints:\n" + "\n".join(
                f"  - {v}" for v in gate_result.violations
            )

        dry_run = args.get("dry_run", False)

        plan_dir = Path(state["plan_dir"])
        lifecycle = PlanLifecycleManager(plan_dir, Path(self.workspace))
        try:
            lifecycle.start_plan()
        except Exception as exc:
            return f"Error during lifecycle transition: {exc}"

        if dry_run:
            return (
                f"Plan transitioned to executing (dry-run mode). "
                f"{len(tasks)} tasks validated."
            )

        from agent_core.plan_execution.runner import run_plan

        snapshot = await run_plan(self, tasks, plan_dir=str(plan_dir))

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

        return "\n".join(report_lines)

    async def _nlp_plan_step(self, args: dict[str, Any]) -> str:
        """Advance the plan by executing the next uncompleted task."""
        if self.is_plan_mode():
            return "Error: plan mode is read-only. Run 'mode build' first."

        state = await self._get_plan_state()
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

        sub = self.spawn_subagent(
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

    async def _nlp_plan_finish(self, args: dict[str, Any]) -> str:
        """Complete or fail the current plan."""
        state = await self._get_plan_state()
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
        lifecycle = PlanLifecycleManager(plan_dir, Path(self.workspace))

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

    # ── Internal file-operation helpers ──────────────────────────────

    async def _tool_read_file(self, path: str, **kwargs: Any) -> str:
        result = await self.fs.read(path)
        if not result.startswith("File not found") and not result.startswith("Error"):
            safe = self.fs.safe_path(path)
            self._files_read.add(safe)
            try:
                self._file_mtimes[safe] = os.path.getmtime(safe)
            except OSError as e:
                logger.warning("Could not stat %s: %s", safe, e)
        return result

    async def _tool_write_file(self, path: str, content: str, **kwargs: Any) -> str:
        return await self.fs.write(path, content)

    async def _tool_apply_patch(
        self, path: str, find: str, replace: str, **kwargs: Any,
    ) -> str:
        return await self.fs.apply_patch(path, find, replace)

    async def _tool_edit_file(self, path: str, content: str, **kwargs: Any) -> str:
        return await self.fs.edit(path, content)

    async def _tool_search(self, query: str, path: str = ".", **kwargs: Any) -> str:
        search_path = self._resolve_nlp_path(path or ".")
        return await self.searcher.search(query, search_path)

    async def _tool_list_files(
        self, path: str = ".", pattern: str = "*", **kwargs: Any,
    ) -> str:
        return await self.fs.list_files(path, pattern)

    async def _tool_delete_file(self, path: str, **kwargs: Any) -> str:
        return await self.fs.delete(path)

    async def _tool_analyze_file(self, path: str, **kwargs: Any) -> str:
        return cast(
            str, await self.llm.analyze_code(await self.read_file(path))
        )  # type: ignore[redundant-cast]

    async def _tool_llm_analyze(self, path: str, **kwargs: Any) -> str:
        file_content = await self.read_file(path, track_read=False)
        if file_content.startswith("File not found:") or (
                file_content.startswith("Error reading file:")
        ):
            return f"Could not analyze: {file_content}"
        return cast(
            str, await self.llm.analyze_code(file_content)
        )  # type: ignore[redundant-cast]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name using the dispatcher."""
        return await self.dispatcher.execute(tool_name, arguments)

    def _normalize_path(self, path: str) -> str:
        """Normalize and validate paths with security checks."""
        return resolve_path(path)

    def _safe_path(self, path: str) -> str:
        """Validate and normalize path in one step."""
        return safe_path(path)

    async def read_file(self, path: str, track_read: bool = True) -> str:
        local_path = self._safe_path(path)

        # A directory is a common mistake (analyze(path="agent_core/")): on
        # Windows open() raises PermissionError, which surfaced as an opaque
        # "Error reading file: [Errno 13]".  Say what is actually wrong.
        if os.path.isdir(local_path):
            return (
                f"Not a file (directory): {path} — read/analyze take a FILE "
                "path; omit the path to analyze the whole workspace, or use "
                "list_files/search to inspect a directory."
            )

        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if track_read:
                self._files_read.add(local_path)
                try:
                    self._file_mtimes[local_path] = os.path.getmtime(local_path)
                except OSError as e:
                    logger.warning("Could not stat %s: %s", local_path, e)

            return content

        except IsADirectoryError:
            return f"Not a file (directory): {path}"
        except PermissionError as e:
            return f"Permission denied reading {path}: {e}"
        except FileNotFoundError:
            return f"File not found: {path}"
        except Exception as e:
            return f"Error reading file: {e}"

    async def write_file(self, path: str, content: str) -> str:
        local_path = self._safe_path(path)

        try:
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)

            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully wrote to {path}"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    async def apply_patch(self, path: str, find: str, replace: str) -> str:
        local_path = self._safe_path(path)

        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if find not in content:
                return "Pattern not found in file"

            count = content.count(find)
            if count > 1:
                return (
                    f"Error: find text matches {count} locations. "
                    "Add more context to make it unique."
                )

            new_content = content.replace(find, replace, 1)

            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            return "Patch applied successfully"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error applying patch: {e}"

    async def edit_file(self, path: str, content: str) -> str:
        local_path = self._safe_path(path)
        try:
            if save_file_py(local_path, content, auto_yes=True):
                return f"Successfully edited {path}"
            return f"Skipped {path} (no changes)"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error editing file: {e}"

    def _build_semantic_index(self, words: list[str], idx: int) -> None:
        """Build semantic index with memory management."""
        MAX_INDEX_SIZE = 10000

        if len(self._semantic_index) > MAX_INDEX_SIZE:
            self._cleanup_semantic_index()

        for word in words:
            normalized_word = word.lower()
            self._semantic_index[normalized_word].add(idx)

    def _cleanup_semantic_index(self) -> None:
        """Remove oldest entries from semantic index."""
        if not self._semantic_index:
            return

        items = list(self._semantic_index.items())
        keep_count = max(100, len(items) - 500)

        sorted_items = sorted(items, key=lambda x: len(x[1]), reverse=True)

        self._semantic_index.clear()
        for word, idx_set in sorted_items[:keep_count]:
            self._semantic_index[word] = idx_set

    async def search_file(self, query: str, path: str | None = None) -> str:
        """Search workspace files for text.

        Delegates to the shared :class:`FileSearcher` (excludes git-ignored
        state/cache/binary files, returns ``path:lineno: content`` lines), so
        the REPL ``search`` command and the NLP ``search`` tool behave alike.
        """
        results = await self.searcher.search(query, path or self.workspace)
        if not results or results == "No matches found":
            return "No matches found"
        return results

    # ------------------------------------------------------------------
    # Turn-pipeline helpers.  chat_nlp delegates to these so each phase —
    # system-prompt refresh, user-turn append, chained tool loop, finish —
    # can be read (and tested) in isolation.  Behaviour is unchanged.
    # ------------------------------------------------------------------

    def _refresh_system_message(self) -> None:
        """Ensure history starts with a system message and rebuild its dynamic
        blocks (decision-constraints / skill index / plan-mode suffix).

        The stored BASE prompt (position 0 minus previously injected dynamic
        blocks) is preserved — only the dynamic blocks are rebuilt, so nothing
        accumulates and a restored session's prompt is never clobbered.  A
        long-lived session must see the CURRENT decision ledger and the CURRENT
        skill index (a runbook added to ``skills/`` mid-session is advertised on
        the next turn), not the snapshot taken on the first turn.
        """
        if not self._chat_history:
            self._chat_history.append({
                "role": "system",
                "content": _SYSTEM_PROMPT + (
                    plan_mode_system_suffix() if self.is_plan_mode() else ""
                ),
            })
        self._chat_history[0] = {
            "role": "system",
            "content": _strip_dynamic_system_blocks(
                str(self._chat_history[0].get("content") or _SYSTEM_PROMPT)
            )
            + self._decision_constraints_block()
            + self._skill_index_block()
            + self._habits_block()
            + self._semantic_memory_block()
            + (plan_mode_system_suffix() if self.is_plan_mode() else ""),
        }

    def _append_user_turn(
        self, user_input: str, images: list[str] | None,
    ) -> None:
        """Append this turn's user message (multimodal when *images* given).

        Plan mode (read-only) prepends a steering note telling the model to
        research and end with a plan instead of promising changes it is not
        allowed to make (mutating tools are additionally blocked in the
        executor — see ``_execute_tool_call``).

        Also sets ``_turn_start_index``: everything appended after this line
        belongs to THIS turn — the boundary per-turn scans such as
        ``_mutating_files_this_turn`` use, so tool results restored from a
        previous session's chat_history.json are never rescanned.
        """
        if self.is_plan_mode() and not images:
            # Same visibility contract as every other status print:
            # suppressed in QUIET mode, which promises only the final answer.
            if _resolve_display_mode() != AgentDisplayMode.QUIET:
                print(yellow(
                    "  [plan mode] Read-only research — mutating tools blocked. "
                    "Switch back with 'mode build' to apply changes."
                ))
            user_input = f"{plan_mode_turn_note()}\n\n{user_input}"
        hint = "" if images else self._skill_hint_block(user_input)
        if hint:
            user_input = f"{hint}\n\n{user_input}"
        self._turn_start_index = len(self._chat_history)
        if images:
            #: OpenAI-format content array: text plus one image_url block per
            #: base64 data URL, so vision-capable models can see the image(s).
            #: Blocks are never persisted (see :func:`_strip_image_blocks`).
            content: Any = [
                {"type": "text", "text": user_input or "(see the attached image)"},
            ]
            for img in images:
                content.append({"type": "image_url", "image_url": {"url": img}})
            self._chat_history.append({"role": "user", "content": content})
        else:
            self._chat_history.append({"role": "user", "content": user_input})

    async def _run_chained_tool_loop(
        self,
        user_input: str,
        messages: list[dict[str, Any]],
        display_mode: AgentDisplayMode,
        seen_calls: dict[tuple[str, str], int],
    ) -> tuple[str, list[dict[str, Any]], str | None, ToolLoopRunner]:
        """Run :class:`ToolLoopRunner`, auto-continuing unfinished tasks.

        Returns ``(final_text, final_messages, llm_error, loop)``.  *llm_error*
        carries a provider-level failure (reasoning-budget exhaustion, HTTP
        error, unreachable server, ...) — such a failure must never be mistaken
        for a model answer, fed into auto-continue chaining, or printed as the
        green final answer.  *loop* exposes observability stats for the
        final-answer fallback.

        Auto-continue rules: a "cap" verdict (budget ran out while
        progressing), an "answer" that signals unfinished work, or a
        "no_progress" verdict whose forced answer STILL signals unfinished
        work justify another run (up to ``_MAX_CHAINED_RUNS``).  Plain
        "stuck"/"no_progress" verdicts mean the model is not making progress —
        continuing would only re-enter the same loop.  A byte-identical
        repeated answer stops chaining: the model is stuck, not working.
        """
        llm_error: list[str] = []

        async def llm_chat_fn(
            msgs: list[dict[str, Any]], tools: list[dict[str, Any]],
        ) -> tuple[str, list[dict[str, Any]]]:
            """Call the LLM with tools; parse a JSON tool_calls message back into
            the message list so ToolLoopRunner can execute them."""
            raw = await self.llm.chat(msgs, tools=tools, disable_thinking=True)
            if raw.strip() == "(no output)":
                # Providers use "(no output)" for an empty response — treat it
                # as empty so the loop's forced-synthesis retry / the concrete
                # fallback message kick in instead of showing cryptic text.
                raw = ""
            if raw.startswith("[Error") or raw.startswith("[LM Studio"):
                if not llm_error:
                    llm_error.append(raw)
                return raw, msgs
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("tool_calls"):
                parsed.pop("role", None)
                for tc in parsed["tool_calls"]:
                    func = tc.get("function") or {}
                    args_str = func.get("arguments", "{}")
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        # LM Studio sometimes emits literal newlines or unescaped
                        # chars inside JSON string values; repair by escaping
                        # control characters where they are actually illegal —
                        # inside string literals only.  A whole-text replace would
                        # also mangle the whitespace between tokens of pretty-
                        # printed output and wipe a valid multi-line payload to {}.
                        func["arguments"] = json.dumps(
                            _repair_tool_arguments(args_str), ensure_ascii=False)
                updated = list(msgs)
                updated.append(
                    {"role": "assistant",
                     "content": parsed.get("content") or "", **parsed})
                text = str(parsed.get("content") or "")
                return text, updated
            # Plain text answer — the loop terminates.
            updated = list(msgs)
            updated.append({"role": "assistant", "content": raw})
            return raw, updated

        final_text = ""
        final_messages = messages
        continuations = 0
        #: Last non-empty answer of a chained run — a byte-identical repeat
        #: means the model is stuck emitting the same incomplete answer.
        last_answer = ""
        #: One correlation id per TURN: every chained run of this chat_nlp
        #: call shares it, so a single task is linkable across its traces
        #: (decision #050).
        with CorrelationIdContext() as turn_cid:
            #: One run id per turn, reused by every experience row this turn
            #: writes (decision attribution): the llm_decisions provenance log
            #: and the chat_turn roll-up row share the llm_decision row's id,
            #: so a whole turn is linkable in one join.
            self._turn_run_id = turn_cid
            while True:
                #: Per-run trace writer (one JSONL file per run() invocation,
                #: decision #029).  AGENT_NO_TRACE=1 disables trace capture.
                #: Model/profile are stamped on every record so a trace is
                #: self-describing for review and cross-model comparison
                #: (decision #050).
                trace_writer = (
                    TraceWriter(
                        meta={
                            "model": self.model_name,
                            "profile": getattr(self.llm, "_profile_name", None) or "",
                        }
                    )
                    if (trace_enabled() and TraceWriter is not None)
                    else None
                )
                if trace_writer is not None:
                    trace_writer.emit_task_begin(user_input)
                #: Expose active trace writer to the shutdown handler so
                #: SIGBREAK can flush/close it before exit (decision #049).
                self._active_trace_writer = trace_writer
                #: File-effects recording (self-improvement): armed only while a
                #: trace sink exists — untraced runs stay byte-identical (decision
                #: #048).  The buffer is drained by _take_trace_effects after each
                #: tool call and discarded when the run ends.
                self._pending_effects = [] if trace_writer is not None else None
                loop = ToolLoopRunner(
                    max_iterations=150, display_mode=display_mode, trace=trace_writer
                )
                final_text, final_messages = await loop.run(
                    messages=final_messages,
                    llm_chat_fn=llm_chat_fn,
                    execute_tool_fn=self._execute_tool_call,
                    tools=filter_tool_schemas(NLP_TOOL_SCHEMAS, self.mode),
                    seen_calls=seen_calls,
                    effects_fn=self._take_trace_effects,
                    stream_fn=_resolve_stream_fn(self.llm),
                )
                self._pending_effects = None
                self._active_trace_writer = None
                if trace_writer is not None:
                    trace_writer.close()
                reason = loop.termination_reason
                #: Record this run's LLM outcome as an experience for the
                #: self-improvement harness (decision #049/#014): a direct,
                #: never-raising SQLite write that mirrors MCP record_experience.
                _run_outcome = 1.0 if not llm_error else 0.0
                _run_success = bool(not llm_error and reason in ("answer", "cap"))
                #: Provider telemetry for this run, when the transport reports
                #: it: latency + tokens land in the llm_decisions provenance
                #: row so cost/latency can be compared per model later.
                _metrics = get_last_metrics(getattr(self.llm, "_provider", None))
                self._record_llm_experience(
                    action="llm_decision",
                    outcome=_run_outcome,
                    context={
                        "model": self.model_name,
                        "profile": getattr(self.llm, "_profile_name", None) or "",
                        "verdict": reason,
                        "continuations": continuations,
                        "final_text_len": len(final_text),
                    },
                    success=_run_success,
                    decision_prompt=final_messages,
                    run_id=self._turn_run_id or CORRELATION_ID_CTX.get() or None,
                    latency_ms=getattr(_metrics, "latency_ms", None),
                    token_usage=getattr(_metrics, "total_tokens", None),
                )
                # A provider-level failure is not an answer: never auto-continue —
                # chaining would only re-burn the same broken LLM call.
                if llm_error:
                    break
                # A bare answer to an action request with ZERO tool calls is a
                # claim about repo state the model never observed (2026-09-28:
                # "commit changes" -> "everything is already committed and
                # pushed", with no git call ever made).  Re-enter the loop with
                # a note pointing at the git tool instead of accepting it.
                unverified = (
                    reason in ("answer", "no_progress")
                    and _unverified_action_claim(
                        user_input, getattr(loop, "tool_calls_made", 0) or 0
                    )
                )
                needs_more = (
                    reason == "cap"
                    or unverified
                    or (
                        reason in ("answer", "no_progress")
                        and _looks_incomplete(final_text)
                    )
                )
                if needs_more and continuations < _MAX_CHAINED_RUNS:
                    if final_text and final_text == last_answer:
                        # Same incomplete answer twice in a row — the model is
                        # stuck, not working.  Stop chaining instead of looping.
                        if display_mode != AgentDisplayMode.QUIET:
                            print(
                                magenta(
                                    "\n  [stopped] The model repeated the same answer "
                                )
                                + yellow("twice — ending the turn. ")
                                + gray(
                                    "Rephrase or ask something more specific to"
                                    " continue.\n")
                            )
                        break
                    last_answer = final_text
                    continuations += 1
                    # An unverified claim gets a targeted note (go look at the
                    # repo) instead of the generic "continue the task" note,
                    # which would just invite the same unverified answer again.
                    note = _VERIFY_NOTE if unverified else _CONTINUE_NOTE
                    if display_mode != AgentDisplayMode.QUIET:
                        why = {
                            "cap": "iteration budget exhausted",
                            "answer": "answer signals unfinished work",
                            "no_progress": "the final answer signals unfinished work",
                        }.get(reason, reason)
                        if unverified:
                            why = "answer claimed repo state without calling git"
                        print(
                            magenta(f"\n  [auto-continue] Run {continuations}: ")
                            + yellow(f"{why} — continuing automatically.\n")
                        )
                    final_messages = list(final_messages) + [
                        # User role: strict chat templates (qwen Jinja) reject
                        # system messages mid-conversation.  The tag marks this
                        # as loop-injected so the strip below cannot confuse it
                        # with a real user prompt.
                        {"role": "user", "content": note,
                         _CONTINUE_NOTE_TAG_KEY: _CONTINUE_NOTE_TAG},
                    ]
                    continue
                if (
                        reason in ("stuck", "no_progress")
                        and display_mode != AgentDisplayMode.QUIET
                ):
                    print(
                        magenta("\n  [stopped] The model stopped making progress ")
                        + yellow(
                            (
                                "stuck on repeated calls"
                                if reason == "stuck"
                                else "too many calls without making new progress"
                            )
                        )
                        + gray(
                            "The answer above is the best it produced — rephrase or"
                            " ask something more specific to continue.\n")
                    )
                break

        # The continuation note is only meant for the run it precedes — strip
        # every TAGGED loop note before the history is persisted, so a
        # finished task is not resumed by a future session.  (Tag-based, not
        # content-based: a user message that merely resembles the note text
        # must survive.)
        final_messages = [
            m for m in final_messages
            if m.get(_CONTINUE_NOTE_TAG_KEY) != _CONTINUE_NOTE_TAG
        ]
        return final_text, final_messages, (llm_error[0] if llm_error else None), loop

    def _finish_turn(
        self,
        final_text: str,
        llm_error: str | None,
        loop: ToolLoopRunner,
        display_mode: AgentDisplayMode,
    ) -> None:
        """Bound + persist the conversation and print the turn outcome.

        The final answer is ALWAYS printed — in every display mode, including
        QUIET (which only hides intermediate tool output, per the display-mode
        contract: "only the final answer is printed").
        """
        # Keep the conversation bounded and persist it so the next session
        # (or a follow-up prompt) can continue the dialogue.
        self._chat_history = _trim_chat_history(self._chat_history)
        mutated_files = self._mutating_files_this_turn()
        # Harness-layer observability ONLY (decision #014): bounded per-turn
        # outcome log + first production chat_turn experience.  Every hook
        # here is try/except-no-op — a broken state dir must never kill a
        # finished turn (same contract as _record_llm_experience).
        try:
            started = getattr(self, "_turn_started_at", None)
            duration = round((datetime.now() - started).total_seconds(), 3) \
                if started else None
            tail = str(getattr(self, "_last_user_input", "") or "")[-200:]
            record = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "user_input_tail": tail,
                "mutated_files": list(mutated_files or []),
                "llm_error": bool(llm_error),
                "duration": duration,
            }
            _append_turn_log(record)
            self._record_llm_experience(
                action="chat_turn",
                outcome=0.0 if llm_error else (1.0 if mutated_files else 0.7),
                context={
                    "plan_mode": bool(self.is_plan_mode()),
                    "files": list(mutated_files or []),
                },
                #: The turn roll-up is attributed to the same model that made
                #: the turn's decisions, and shares the turn's run id — so the
                #: llm_decision row and this quality row are one unit in the
                #: llm_decisions provenance log.
                decision_prompt=str(getattr(self, "_last_user_input", "") or "") or None,
                run_id=self._turn_run_id or CORRELATION_ID_CTX.get() or None,
            )
            # Meta-policy loop (B2): success/failure under the inferred
            # task type + active profile, with token/cost metrics from the
            # provider's last call; evolves+saves weights every 10 turns.
            # Never raises (same observability contract as above).
            record_turn_outcome(
                str(getattr(self, "_last_user_input", "") or ""),
                profile_name=getattr(self.llm, "_profile_name", None),
                success=not llm_error,
                latency_seconds=duration,
                provider=getattr(self.llm, "_provider", None),
                mutated_files=list(mutated_files or []),
            )
            # Per-turn quality (B3): one `turn`/`quality` event so the
            # harnessfix dashboard shows live quality.  Proxy path works
            # with tracing disabled; whole block is try/except no-op.
            _quality = _turn_quality_score(llm_error, mutated_files)
            if _quality is not None:
                from agent_core.monitoring.metrics_file import append_event
                append_event("turn", "quality", float(_quality))
        except Exception:  # noqa: BLE001 - observability must not break turns
            logger.debug("Turn-outcome hooks failed (no-op)", exc_info=True)
        self._save_chat_history()
        self._save_memory()

        clean = re.sub(r'</?tool_call>', '', final_text)
        clean = re.sub(r'</?function_call>', '', clean)

        if llm_error:
            # A provider-level failure was detected during the run: show the
            # actual error (not the generic fallback, not a green "answer").
            err = llm_error.strip()
            print(yellow(
                "\n  [llm-error] The model did not produce a usable response:"))
            print(red(f"  {err[:400]}"))
            if "reasoning" in err.lower():
                print(
                    yellow("  The reasoning model exhausted its thinking budget. ")
                    + gray(
                        "Switch to a non-reasoning model (model <name>) or retry"
                        " the request.\n")
                )
            self._nlp_workspace = None
            return

        if display_mode == AgentDisplayMode.CLEAN:
            # Fold any stray "I will ..." narration into a short preamble so the
            # report reads as one coherent answer (what changed / where / evidence).
            _NARRATION_PREFIX = re.compile(
                r"^\s*(i\s+(will|'ll|going\sto|am\ngoing)|let\s+me)\b", re.I,
            )
            if _NARRATION_PREFIX.match(clean):
                clean = "Plan: " + clean.strip()

        if clean.strip():
            # A streamed answer already appeared on the console token-by-token
            # (plan #9): printing it again would show the user the answer twice.
            # getattr, not attribute access: a loop-like object without the flag
            # (fakes, older runners) means "not streamed" -> print normally.
            if not getattr(loop, "final_answer_streamed", False):
                print(green(clean))
        else:
            # No usable answer: tell the user CONCRETELY what the loop did
            # instead of the cryptic "did not produce a response".
            print(yellow(_final_answer_fallback(loop)))
        if self.is_plan_mode() and clean.strip():
            self._persist_plan_answer(clean)
        if mutated_files and clean.strip():
            self._print_self_review_note(mutated_files)
        self._nlp_workspace = None

    async def chat_nlp(self, user_input: str,
                       images: list[str] | None = None) -> None:
        """Process natural language input through a structured tool-calling loop.

        The LLM receives native OpenAI-format tool schemas (``NLP_TOOL_SCHEMAS``)
        and must either emit a structured ``tool_calls`` or answer in text — there
        is no free-text tag format that lets it describe an action instead of
        taking it.  Every tool call is executed, its result is fed back, and the
        loop continues until the model answers in text or the iteration cap is
        reached.

        *images* is an optional list of base64 data URLs (``data:<mime>;base64,…``)
        sent as multimodal ``image_url`` content blocks alongside *user_input* —
        used by the ``paste_image`` command to let vision-capable models see an
        image (e.g. a screenshot, diagram, or photo).  Image blocks are never
        persisted to ``chat_history.json`` (they are stripped before saving, see
        :func:`_strip_image_blocks`), so a vision turn is resumable without
        bloating the on-disk history with multi-megabyte blobs.

        The loop auto-continues: if a run ends on the iteration cap, on repeated
        calls, or with an answer that signals unfinished work, a fresh run starts
        automatically (up to ``_MAX_CHAINED_RUNS``) so the model does not stop
        mid-task and wait for the end-user.

        The turn pipeline is split into four readable phases:
        1. ``_refresh_system_message`` — current ledger/plan-mode in prompt.
        2. ``_append_user_turn`` — multimodal user message + turn boundary.
        3. ``_run_chained_tool_loop`` — tool loop + auto-continue chaining.
        4. ``_finish_turn`` — trim/persist history, print the outcome.
        """
        self._refresh_system_message()
        self._read_streak = 0
        # Turn-outcome observability (A3): the input + start time feed the
        # bounded turn log and duration written by _finish_turn.
        self._last_user_input = user_input
        self._turn_started_at = datetime.now()
        # Meta-policy suggestion (B2): at most ONE line when the task's
        # mapped profile clearly outscores the running one, and only if the
        # user pinned no profile this session (restored-from-model.json
        # counts as pinned).  Auto-apply happens only behind the
        # `profile_auto` workspace pref (default off — no silent sampling
        # changes, decision #014).  Never raises: a broken pref file must
        # not kill the turn.
        try:
            if not getattr(self.llm, "_profile_pinned", False):
                _suggested = recommend_profile(
                    user_input, getattr(self.llm, "_profile_name", None),
                )
                if _suggested:
                    _auto = False
                    try:
                        import pathlib as _pathlib
                        from agent_core.llm.workspace_prefs import get_pref
                        _auto = get_pref(
                            _pathlib.Path(self._effective_ws_dir()),
                            "profile_auto",
                        ) is True
                    except Exception:  # noqa: BLE001 - pref file is optional
                        _auto = False
                    _quiet = _resolve_display_mode() == AgentDisplayMode.QUIET
                    if _auto:
                        try:
                            from agent_core.llm.model_profiles import get_profile
                            _prof = get_profile(_suggested)
                            self.llm._provider.apply_profile(
                                _suggested, _prof.temperature, _prof.max_tokens,
                            )
                            self.llm._profile_name = _suggested
                        except Exception:  # noqa: BLE001 - never break the turn
                            logger.debug(
                                "profile_auto apply failed (no-op)",
                                exc_info=True,
                            )
                        else:
                            if not _quiet:
                                print(gray(
                                    f"  [profile] auto-applied {_suggested}"
                                    " (profile_auto pref)"))
                    elif not _quiet:
                        print(gray(
                            f"  [suggest] profile {_suggested} for this task"
                            " — 'model profile use' or set pref"
                            " profile_auto"))
        except Exception:  # noqa: BLE001 - observability must not break turns
            logger.debug("Profile suggestion failed (no-op)", exc_info=True)
        self._append_user_turn(user_input, images)
        # Pre-trim: bound the history BEFORE the first LLM call so a large
        # restored session (e.g. 60 messages from chat_history.json) does not
        # blow the model's prefill budget and trigger an HTTP 500.  The
        # post-turn trim in _finish_turn still runs for incremental cleanup.
        self._chat_history = _trim_chat_history(self._chat_history)
        # Snapshot history right after the user turn so a transient LLM failure
        # can be retried from a clean slate — the failed run's tool messages are
        # discarded on retry (otherwise the retry would re-feed them and
        # duplicate work).  Without this the first timeout would end the task.
        history_snapshot = list(self._chat_history)

        final_text, final_messages, llm_error, loop = "", self._chat_history, None, None
        for attempt in range(1, _LLM_ERROR_MAX_RETRIES + 1):
            final_text, final_messages, llm_error, loop = (
                await self._run_chained_tool_loop(
                    user_input=user_input,
                    messages=self._chat_history,
                    display_mode=_resolve_display_mode(),
                    seen_calls={},
                )
            )
            # Only retry on a TRANSIENT provider failure (timeout/connection).
            # A permanent error (4xx/auth) must not be retried — it would loop
            # forever and never surface to the user.
            if not llm_error or not is_connection_failure(llm_error):
                break
            if attempt < _LLM_ERROR_MAX_RETRIES:
                self._chat_history = list(history_snapshot)
                if _resolve_display_mode() != AgentDisplayMode.QUIET:
                    print(yellow(
                        f"\n  [retry {attempt}/{_LLM_ERROR_MAX_RETRIES}] LLM call "
                        f"failed transiently ({llm_error[:120]}); retrying...\n"
                    ))
                await asyncio.sleep(_LLM_ERROR_BACKOFF_S * (2 ** (attempt - 1)))

        # Tagged continuation notes were stripped inside the chained-loop phase;
        # adopt its final message list as the session history.
        self._chat_history = final_messages
        self._finish_turn(
            final_text=final_text,
            llm_error=llm_error,
            loop=loop,
            display_mode=_resolve_display_mode(),
        )

    def _mutating_files_this_turn(self) -> list[str]:
        """Files written/edited by this turn (plan item B-#6).

        The NLP write/edit handlers append a ``[verify] py_compile ✓`` line to
        their result — that line is the per-file mutation marker.  Scanning it
        keeps this zero-cost while the loop runs; no extra bookkeeping.

        Only messages from the CURRENT turn (at or after ``_turn_start_index``,
        set at the top of :meth:`chat_nlp`) are scanned.  A session restored
        from chat_history.json contains tool results from previous sessions;
        without this boundary the self-review note would list stale files
        after every restart.
        """
        marker = "[verify] py_compile"
        files: list[str] = []
        history = self._chat_history
        start = max(0, min(self._turn_start_index, len(history)))
        for idx in range(start, len(history)):
            m = history[idx]
            if m.get("role") != "tool":
                continue
            content = str(m.get("content") or "")
            if marker not in content:
                continue
            first_line = content.splitlines()[0] if content else ""
            # Success lines start "Written <path> (N bytes)" / "Edited <path>";
            # take everything after the verb up to the trailing parenthetical.
            for prefix in ("Written ", "Edited ", "Skipped "):
                if first_line.startswith(prefix):
                    path = first_line[len(prefix):]
                    if path.endswith(")"):
                        path = path.rsplit("(", 1)[0].strip()
                    if path and path not in files:
                        files.append(path)
                    break
        return files

    def _print_self_review_note(self, mutated_files: list[str]) -> None:
        """Post-mutation self-review reminder (plan item B-#6).

        py_compile proves a file PARSES — not that it does what was asked.
        This prints one short nudge listing the changed files so the user
        knows exactly what to double-check (or ask the agent to re-verify).
        """
        def _short(f: str) -> str:
            try:
                return os.path.relpath(f, self.workspace).replace("\\", "/")
            except ValueError:
                return f

        shown = ", ".join(_short(f) for f in mutated_files[:6])
        more = f" (+{len(mutated_files) - 6} more)" if len(mutated_files) > 6 else ""
        print(magenta(
            f"\n  [self-review] This turn changed {len(mutated_files)} file(s): "
            f"{shown}{more}. py_compile verified syntax only — run 'tests' or "
            "'git diff' to confirm behaviour."
        ))

    def _persist_plan_answer(self, plan_text: str) -> None:
        """Save a plan-mode final answer to ``.docs/<ts>/plan_proposed.md``.

        Plan-mode answers used to evaporate as terminal text; persisting them
        gives the user a durable artifact to hand to ``implement``/``fix``
        later (plan item D-#14).  Best-effort: a write failure must never
        lose the already-printed answer or crash the turn.
        """
        with _suppress_and_log('Could not persist plan-mode answer:\n'):
            from agent_core.commands.doc_paths import new_run_dir

            out = new_run_dir(self.workspace) / "plan_proposed.md"
            out.write_text(
                "# Proposed plan\n\n"
                f"_Generated in plan mode on {datetime.now():%Y-%m-%d %H:%M}. "
                "Review before applying — switch to build mode to implement._\n\n"
                f"{plan_text}\n",
                encoding="utf-8",
            )
            print(yellow(f"\n  [plan] Saved to {out}"))

    def check_stale_files(self) -> list[str]:
        """Return files whose mtime has changed since last read."""
        stale = []
        for path in list(self._file_mtimes):
            try:
                if os.path.getmtime(path) != self._file_mtimes[path]:
                    stale.append(path)
            except FileNotFoundError:
                stale.append(path)
            except OSError as e:
                logger.warning("Could not stat %s: %s", path, e)
        return stale

    def invalidate_stale(self) -> int:
        """Remove stale entries from memory. Returns count of invalidated files."""
        stale = self.check_stale_files()
        for path in stale:
            self._file_mtimes.pop(path, None)
            self._files_read.discard(path)
        if stale:
            self._semantic_index.clear()
        return len(stale)

    def memory_stats(self) -> dict[str, Any]:
        """Return summary of current memory state."""
        stale = self.check_stale_files()
        return {
            "chat_history": len(self._chat_history),
            "files_read": len(self._files_read),
            "stale_files": len(stale),
            "history": len(self._history),
            "working_memory": len(self._working_memory),
            "semantic_index": len(self._semantic_index),
            "knowledge_graph": len(self._knowledge_graph),
        }

    def clear_history(self) -> None:
        """Clear all agent state."""
        self._history = []
        self._chat_history.clear()
        self._files_read.clear()
        self._file_mtimes.clear()
        self._knowledge_graph.clear()
        self._working_memory.clear()
        self._semantic_index.clear()
        try:
            os.remove(CHAT_HISTORY_JSON_PATH)
            os.remove(AGENT_MEMORY_JSON_PATH)
        except OSError as e:
            logger.warning("Could not remove persisted state files: %s", e)

    def _decision_constraints_block(self) -> str:
        """Design-decision constraints for the chat system prompt.

        Mirrors what ``implement``/``fix`` already inject (decision ledger
        ``.decisions.json`` via :func:`decisions_as_system_prompt`) so the
        conversational loop cannot contradict a recorded decision mid-chat.
        Matched on the files read this session; empty string when no decision
        touches them — the prompt stays byte-identical to before in that case.
        Never raises: a broken/absent ledger must not kill a chat turn.
        """
        try:
            if not self._files_read:
                return ""
            return decisions_as_system_prompt(
                self.workspace, sorted(self._files_read),
            )
        except Exception:
            logger.exception('Decision constraints unavailable:\n')
            return ""

    def _skill_index_block(self) -> str:
        """Compact index of workspace skills for the chat system prompt.

        Only ``- name — description`` lines are injected (see
        :func:`agent_core.skills.skill_index_block`); the runbook BODY is read
        on demand via the ``read_skill`` tool.  Rebuilt every turn so a skill
        added to ``skills/`` mid-session is advertised on the next turn.
        Empty string when the workspace has no skills — the prompt then stays
        byte-identical to before.  Never raises: a broken ``SKILL.md`` must not
        kill a chat turn (same rule as :meth:`_decision_constraints_block`).
        """
        try:
            return load_skill_index(self._effective_ws_dir())
        except Exception:
            logger.exception('Skill index unavailable:\n')
            return ""

    def _skill_hint_block(self, user_input: str) -> str:
        """Per-turn skill hint prepended to the user message when a skill matches.

        Only name + one-line description are injected (see
        :func:`agent_core.skills.skill_hint_block`); the runbook BODY still
        only enters context via an explicit ``read_skill`` call.  Never raises:
        a broken match must not kill a chat turn (same rule as
        :meth:`_skill_index_block`).

        Side-effects on a successful match: increments the shared
        ``skill.hint.matched`` metrics counter (mirrored to the event file so
        a standalone --serve dashboard sees it) and prints a short
        ``[skill-hints]`` status line suppressed in QUIET mode.
        """
        try:
            skills = discover_skills(self._effective_ws_dir())
            matched = match_skills_for_input(skills, user_input or "")
            if not matched:
                return ""
            # Observability: count the hit + mirror to shared event file.
            try:
                collector = self.get_metrics_collector()
                collector.increment_counter("skill.hint.matched")
                from agent_core.monitoring.metrics_file import append_event as _ae
                _ae("counter", "skill.hint.matched", 1.0)
            except Exception:
                pass  # metrics must never kill a turn
            # Status print (suppressed in QUIET mode, same contract as [plan mode]).
            if _resolve_display_mode() != AgentDisplayMode.QUIET:
                names = ", ".join(s.name for s in matched)
                print(yellow(f"  [skill-hints] matched: {names}"))
            return skill_hint_block(matched)
        except Exception:
            logger.exception("Skill hint unavailable:\n")
            return ""

    def _habits_block(self) -> str:
        """Mined user-habits block for the chat system prompt.

        Only the compact block (marker + at most 8 lines, see
        :func:`agent_core.habits.habits_block`) is injected; it is rebuilt
        every turn from the workspace ledger so a habit pinned or mined
        mid-session is advertised on the next turn.  Empty string when the
        workspace has no habits — the prompt then stays byte-identical to
        before.  Never raises: a broken ``.habits.json`` must not kill a chat
        turn (same rule as :meth:`_decision_constraints_block`).
        """
        try:
            return habits_block(load_habits(self._effective_ws_dir()))
        except Exception:
            logger.exception('Habits block unavailable:\n')
            return ""

    def _semantic_memory_block(self) -> str:
        """Embedding/KG memory block for the chat system prompt.

        Builds a compact block from indexed memories (see
        :func:`agent_core.memory.semantic_memory_block`).  It is
        rebuilt every turn from the workspace ledger so a memory indexed or
        updated mid-session is advertised on the next turn.  Empty string when
        the workspace has no indexed memories — the prompt then stays
        byte-identical to before.  Never raises: a broken ``.semantic_memory``
        must not kill a chat turn (same rule as
        :meth:`_decision_constraints_block`).
        """
        try:
            indexed_memories = load_semantic_memory(self._effective_ws_dir())
            return semantic_memory_block(indexed_memories, k=5)
        except Exception:
            logger.exception('Semantic memory block unavailable:\n')
            return ""

    # ------------------------------------------------------------------
    #  Persistent NLP chat history (chat_history.json)
    # ------------------------------------------------------------------

    def _load_chat_history(self) -> list[dict[str, Any]]:
        """Load the persisted NLP conversation from the previous session.

        The persisted history holds a bounded multi-exchange window (see
        ``_project_chat_history``), so a fresh session continues the dialogue
        instead of forgetting it.  A corrupt file is quarantined (see
        :func:`_read_json_quarantining`) instead of being silently dropped.
        """
        data = _read_json_quarantining(
            CHAT_HISTORY_JSON_PATH, "chat history",
        )
        if not isinstance(data, list):
            return []
        messages = [
            m for m in data
            if isinstance(m, dict) and m.get("role") in (
                    "system", "user", "assistant", "tool"
            )
            # Loop-internal tags never belong in a restored conversation.
            and _CONTINUE_NOTE_TAG_KEY not in m
        ]
        # Sanitize corrupt tool_call arguments from previous sessions
        # (truncated JSON, literal newlines, etc.) so the API never sees them.
        for m in messages:
            for tc in m.get("tool_calls", []):
                func = tc.get("function") or {}
                args_str = func.get("arguments", "{}")
                try:
                    parsed = json.loads(args_str)
                    if not isinstance(parsed, dict):
                        func["arguments"] = "{}"
                except (json.JSONDecodeError, TypeError):
                    func["arguments"] = "{}"
        return _project_chat_history(_strip_image_blocks(messages))

    def _save_chat_history(self) -> None:
        """Persist the NLP conversation so the next session can continue it.

        Image blocks are stripped first (see :func:`_strip_image_blocks`) so a
        vision turn does not bloat the on-disk history with base64 blobs.

        The write is atomic: content goes to ``chat_history.json.tmp`` first
        and is then ``os.replace()``d into place, so a crash mid-write can
        never leave a half-written JSON that would be silently dropped on the
        next load.
        """
        payload = json.dumps(
            _project_chat_history(_strip_image_blocks(self._chat_history)),
            ensure_ascii=False, indent=2,
        )
        # Derive the temp file from the CURRENT json path so tests that
        # monkeypatch CHAT_HISTORY_JSON_PATH keep the atomic-write pair
        # consistent (a stale module-level tmp path pointing into a
        # non-existent ~/.agent1/models dir made every save a silent no-op).
        tmp_path = CHAT_HISTORY_JSON_PATH + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, CHAT_HISTORY_JSON_PATH)
        except OSError as e:
            logger.warning("Failed to save chat history: %s", e)

    # ------------------------------------------------------------------
    #  Persistent agent memory (agent_memory.json)
    # ------------------------------------------------------------------

    def _load_memory(self) -> None:
        """Load cross-session memory from the previous session.

        Restores files read, the semantic index, the knowledge graph, and
        working memory so a fresh session resumes with accumulated context
        instead of starting from zero.  A corrupt file is quarantined as
        ``agent_memory.json.bad-<timestamp>`` before falling back to the
        empty defaults, so the bytes stay inspectable instead of being
        clobbered by the next save.
        """
        data = _read_json_quarantining(AGENT_MEMORY_JSON_PATH, "agent memory")
        if not isinstance(data, dict):
            return
        files = data.get("files_read")
        if isinstance(files, list):
            self._files_read = {str(p) for p in files}
        index = data.get("semantic_index")
        if isinstance(index, dict):
            self._semantic_index = defaultdict(
                set,
                {k: set(v) for k, v in index.items() if isinstance(v, list)},
            )
        kg = data.get("knowledge_graph")
        if isinstance(kg, dict):
            self._knowledge_graph = kg
        wm = data.get("working_memory")
        if isinstance(wm, list):
            self._working_memory = wm
        hist = data.get("history")
        if isinstance(hist, list):
            self._history = hist

    def _save_memory(self) -> None:
        """Persist cross-session memory so the next session resumes with it.

        Atomic like :meth:`_save_chat_history` (tmp file + ``os.replace``).
        """
        data = {
            "files_read": sorted(self._files_read),
            "semantic_index": {
                k: sorted(v) for k, v in self._semantic_index.items()
            },
            "knowledge_graph": self._knowledge_graph,
            "working_memory": self._working_memory,
            "history": self._history,
        }
        # Derive the temp file from the CURRENT json path (same rule as
        # _save_chat_history): a stale module-level AGENT_MEMORY_TMP_PATH can
        # point onto another volume (CI: repo on D:, temp on C:) and
        # os.replace then dies with WinError 17 / OSError EXDEV.
        tmp_path = AGENT_MEMORY_JSON_PATH + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, AGENT_MEMORY_JSON_PATH)
        except OSError:
            logger.warning("Failed to save agent memory:\n%s", traceback.format_exc())

    def _record_llm_experience(
        self, action: str, outcome: float, context: dict[str, Any] | None = None,
        success: bool | None = None, decision_llm: str | None = None,
        decision_prompt: Any = None, run_id: str | None = None,
        latency_ms: float | None = None, token_usage: int | None = None,
    ) -> int | None:
        """Record an LLM decision/outcome into agent_memory.db (experiences).

        Mirrors the MCP ``record_experience`` schema so a direct write is
        reconcilable with the memory server's own inserts.  outcome must be in
        [0.0, 1.0]; success defaults to ``outcome >= 0.5``.

        This is harness-layer only (decision #014): it never alters model or
        prompt behavior — it only persists observability of decisions for the
        self-improvement loop.  It must NEVER raise: if agent_memory.db is
        absent, unwritable, or the table missing, we silently no-op so untraced
        / non-memory runs stay byte-identical (decisions #048/#049).

        Decision attribution (plan: memory-collection LLM attribution)
        -------------------------------------------------------------
        Every row is stamped with the LLM that *decided* it, so a later model
        swap can be measured against ground-truth memory outcomes:

        ``decision_llm``
            The deciding model.  Defaults to this agent's active model
            (:func:`agent_core.memory.attribution.resolve_decision_llm`), so
            the stamp cannot be forgotten by a call site.
        ``decision_prompt_hash``
            SHA-256 of the prompt/conversation behind the decision (canonical,
            tag-insensitive — see ``prompt_sha256``), or ``NULL`` when the
            caller passes no prompt.  Never a hash of the empty string, which
            would falsely claim "the empty prompt was the decision input".
        ``decision_timestamp``
            When the decision was taken (same instant as ``timestamp`` for a
            direct write).

        An attributed write also appends one ``llm_decisions`` provenance row
        (run_id, model, prompt hash, the new experience rowid, latency, token
        usage).  The invariant is exact: one ``llm_decisions`` row per
        experience whose ``decision_llm`` is non-NULL.

        Schema compatibility: the first five columns keep the MCP server's
        exact names/order/types; the three attribution columns are APPENDED
        and nullable.  The server reads with ``SELECT *`` and formats rows by
        column name, and both writers name their columns explicitly — so old
        rows read back ``NULL`` and the server is unaffected.  Additive DDL
        only: no row is ever rewritten, so MCP-authored tables are extended in
        place, never clobbered.
        """
        try:
            outcome = float(outcome)
            if not 0.0 <= outcome <= 1.0:
                return None
            if success is None:
                success = 1 if outcome >= 0.5 else 0
            ctx_str = json.dumps(context or {}, ensure_ascii=False, sort_keys=True)
            ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            import sqlite3

            from agent_core.memory.attribution import (
                ensure_attribution_schema, prompt_sha256, record_llm_decision,
                resolve_decision_llm,
            )
            if decision_llm is None:
                decision_llm = resolve_decision_llm(self)
            prompt_hash = (
                prompt_sha256(decision_prompt) if decision_prompt is not None else None
            )
            db_path = AGENT_MEMORY_JSON_PATH.replace(".json", ".db")
            conn = sqlite3.connect(db_path, timeout=5.0)
            try:
                #: Ensure the schema exists.  No other in-repo code path
                #: creates these tables (the MCP memory server only
                #: reads/writes ``experiences``), so without this a fresh
                #: workspace would silently no-op forever.  Idempotent and
                #: additive: it creates ``experiences`` with the MCP columns
                #: first and ALTERs the attribution columns in when a
                #: pre-existing (MCP-written) table is found.
                ensure_attribution_schema(conn)
                if decision_llm is None:
                    # Unattributable (no model identity available): write the
                    # plain MCP-shaped row and leave the attribution columns
                    # NULL rather than inventing an author.
                    cur = conn.execute(
                        "INSERT INTO experiences "
                        "(timestamp, action, outcome, context, success) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (ts, action, outcome, ctx_str, 1 if success else 0),
                    )
                    conn.commit()
                    return int(cur.lastrowid)
                cur = conn.execute(
                    "INSERT INTO experiences (timestamp, action, outcome, "
                    "context, success, decision_llm, decision_prompt_hash, "
                    "decision_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts, action, outcome, ctx_str, 1 if success else 0,
                     decision_llm, prompt_hash, ts),
                )
                rowid = int(cur.lastrowid)
                record_llm_decision(
                    conn, run_id=run_id, model_name=decision_llm,
                    prompt_sha256=prompt_hash, outcome_experience_id=rowid,
                    latency_ms=latency_ms, token_usage=token_usage,
                )
                conn.commit()
                return rowid
            finally:
                conn.close()
        except Exception as exc:  # noqa: BLE001 - must never raise per contract
            logger.debug("Could not record LLM experience (no-op): %s", exc)
            return None



def _read_json_quarantining(path: str, label: str) -> Any:
    """Read JSON from *path*, quarantining corrupt bytes instead of dropping.

    Returns the parsed object, or ``None`` when the file is missing.  When
    the file exists but is unreadable/corrupt (truncated by an old non-atomic
    writer, encoding damage, ...), it is renamed to ``<path>.bad-<timestamp>``
    so the bytes stay inspectable, and ``None`` is returned.  Without this,
    the next save would silently clobber the only evidence of what was lost.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine = f"{path}.bad-{stamp}"
        try:
            os.replace(path, quarantine)
        except OSError:
            logger.warning("Failed to quarantine corrupt %s at %s", label, path)
        else:
            logger.warning(
                "Corrupt %s file moved to %s", label, quarantine,
            )
        return None


_MAX_CHAT_MESSAGES = 60

#: Rough character budget for the chat-history BODY (system prompt excluded;
#: ~4 chars per token, so ~15k chars ≈ 4k tokens — conservative enough for
#: small local models (bonsai-27b@4.4GB) that choke on large prefill, while
#: still leaving room for tool schemas and the answer).  When the body
#: exceeds it, oldest messages are dropped (see _trim_chat_history).
_HISTORY_CHAR_BUDGET = 15000

_HISTORY_TRIM_NOTE = (
    "[context compaction] {dropped} earlier message(s) were dropped to fit "
    "the context window. Earlier file contents are no longer visible — "
    "re-read anything you still need."
)

#: How many automatic continuation runs chat_nlp may chain before handing
#: control back to the user.  The model cannot predict its own tool budget,
#: so instead of stopping mid-task it continues with a fresh budget.  Each
#: chained run has its own guards (no-progress, stuck, deadline), so a high
#: cap cannot turn into an infinite loop — it only bounds total work.
_MAX_CHAINED_RUNS = 6

#: Transient LLM-failure retry for a whole turn.  A single timeout/connection
#: blip (e.g. the hosted gateway hiccups, or a failover to a slow local model
#: times out) must NOT destroy an in-flight task — the turn is retried a few
#: times with backoff.  Only *transient* failures (detected via
#: :func:`is_connection_failure`) are retried; a permanent 4xx/auth error is
#: surfaced immediately so it cannot loop forever.  See :meth:`chat_nlp`.
_LLM_ERROR_MAX_RETRIES = 2
_LLM_ERROR_BACKOFF_S = 5.0

#: Metadata key marking a loop-INJECTED user note (currently the continuation
#: note).  Tagged messages are removed from the history when the turn ends.
#: This replaces fragile content matching, which had a real failure mode: a
#: user whose prompt is byte-identical to the note text got their message
#: silently dropped from the conversation.
_CONTINUE_NOTE_TAG = "continue"

#: Hard ceiling for the NLP ``run`` tool's model-supplied timeout (seconds).
_MAX_RUN_TIMEOUT_S = 600

#: Hard ceiling for the NLP ``read`` tool's per-call line limit.
_MAX_READ_LINES = 500

#: Cap on distinct files a single ``read`` call may fetch via the ``paths``
#: array (multi-file read) — a runaway model must not pull the whole workspace
#: into context in one shot.
_MAX_READ_FILES = 10

#: File-size threshold (KB) above which .py files get an AST-based definition
#: summary instead of full content.  Set via USE_CONTEXT_AST_STRATEGY env var
#: (default 50).  0 disables the strategy.
_CONTEXT_AST_THRESHOLD_KB = int(os.environ.get("USE_CONTEXT_AST_STRATEGY", "50"))

#: Consecutive ``read`` calls allowed within one turn before the read-loop
#: guard starts appending a steering note.  Traces from the 2026-08-25
#: stalls (a01f1bde / 39a90f8f) show the failure spiral: dozens of sequential
#: reads balloon the prompt toward the char budget, every later request then
#: re-processes a huge context, prefill/decode crawl, and the turn ends as
#: ``stuck``/``no_progress`` after 35+ minutes.  Breaking the read loop at
#: the source keeps prompts small enough to stay fast.
_MAX_CONSECUTIVE_READS = 6

_READ_LOOP_NOTE = (
    "\n[read-loop guard] {n} consecutive reads this turn without acting. "
    "Stop paging files — each extra page inflates the prompt and slows every "
    "later call. Work with what you have: use definitions/references/search "
    "for targeted lookups, or give your answer / take the next action now."
)

#: Delegation limits for the NLP ``delegate`` tool.  Children exist to KEEP
#: contexts small — a pile of simultaneously running subagents re-creates
#: the very pressure they were meant to avoid, so the cap is deliberately
#: tight and the timeout mirrors the engine stall cap philosophy (a hung
#: child must surface as an error, not hold the parent turn hostage).
_MAX_ACTIVE_SUBAGENTS = 3
_DELEGATE_TIMEOUT_S = 600.0

#: Message key marking loop-injected notes (see LOOP_NOTE_TAG_KEY in
#: agent_core.constants — re-exported here for the chat_nlp loop).
_CONTINUE_NOTE_TAG_KEY = LOOP_NOTE_TAG_KEY

_CONTINUE_NOTE = (
    "The previous tool session ended before you finished. A fresh tool budget "
    "is available now. CONTINUE THE TASK where you left off — use the tools as "
    "needed, complete the remaining work, then give your final answer. Do not "
    "describe what remains; do it."
)

_INCOMPLETE_MARKERS = (
    "budget", "tool budget", "not finished", "could not", "couldn't",
    "would need", "needs one more", "needs 1 more", "needs 2 more",
    "remaining", "unfinished", "incomplete", "exhausted", "ran out",
    "out of tool", "more tool call", "cannot complete", "still needs",
    "not covered", "follow-up",
)

#: Strong completion signals — their presence means the answer is finished
#: even when a weak marker (budget/remaining/could not) also appears, which
#: reduces false auto-continues (plan FIX item 21).
_COMPLETE_MARKERS = (
    "all done", "is done", "done.", "completed", "finished",
    "is complete", "fully resolved", "all fixed", "task is complete",
)


def _looks_incomplete(text: str) -> bool:
    """Heuristic: did the model's final answer signal unfinished work?

    Strong completion signals (``all done``, ``is complete``, ...) override
    weak markers — "The remaining issue is fixed — all done." is finished even
    though it contains the word "remaining".
    """
    low = text.lower()
    if any(marker in low for marker in _COMPLETE_MARKERS):
        return False
    return any(marker in low for marker in _INCOMPLETE_MARKERS)


#: Imperative work-order verbs that mean "do this to the repository", as
#: opposed to asking *about* git.  Word-bounded so "committed"/"pushing" in a
#: question do not match.
_ACTION_VERB_RE = re.compile(r"\b(commit|push|stage)\b", re.I)

#: Leading tokens that mark a sentence as a question rather than an order.
#: "what does commit mean" is a question, "commit the changes" is an order.
_QUESTION_LEAD_RE = re.compile(
    r"^\s*(what|why|how|when|where|who|which|is|are|was|were|does|did|do|"
    r"can|could|should|would|will|has|have|am|explain|tell|show)\b",
    re.I,
)


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
    # A question is not a work order: do not force tools on "why did my commit
    # fail?" — that needs an explanation, and an answer is the right output.
    if "?" in text or _QUESTION_LEAD_RE.match(text):
        return False
    return True


#: Steering note injected when an action request was answered with no tool
#: calls.  Tagged like the continuation note so it is stripped from the
#: persisted history when the turn ends.
_VERIFY_NOTE = (
    "You answered without calling a single tool, so you have not verified "
    "anything about this repository — you cannot know whether changes are "
    "staged, committed or pushed. Do not assert that state. Call the git tool "
    "now (subcommand='status' first), then do the work and report what it "
    "actually returned."
)


def _resolve_stream_fn(
    llm: Any,
) -> Callable[[list[dict[str, Any]]], Awaitable[str]] | None:
    """Return the streaming final-answer callable for *llm*, or None (plan #9).

    Opt-in and capability-gated, both deliberately:

    * ``AGENT_STREAM_FINAL_ANSWER`` (default false) — streaming changes what the
      console shows, so an existing run must stay byte-identical unless the
      user asks for it (same contract as the other display settings, #048).
    * ``provider_supports_streaming`` — Opencode/OpenRouter/Lemonade implement
      ``chat_stream`` as a bare ``return await self.chat(...)`` that prints
      NOTHING.  On those, the returned text has not appeared on screen, so
      treating it as "already shown" would make ``_finish_turn`` skip its print
      and show the user no answer at all.

    Never raises: any failure here just means "no streaming", i.e. the
    pre-existing blocking behaviour.
    """
    try:
        settings = load_agent_settings()
        if not getattr(settings, "stream_final_answer", False):
            return None
    except Exception:  # noqa: BLE001 - settings failure must not break a turn
        return None
    provider = getattr(llm, "_provider", None) or llm
    if not provider_supports_streaming(provider):
        return None
    stream = getattr(llm, "chat_stream", None)
    if not callable(stream):
        return None
    return cast("Callable[[list[dict[str, Any]]], Awaitable[str]]", stream)


def _resolve_display_mode() -> AgentDisplayMode:
    """Resolve the agent's display mode from settings/env.

    Falls back to VERBOSE (current behaviour) when no value is configured, so
    existing REPL output and tests are unchanged by default."""
    try:
        return load_agent_settings().display_mode
    except Exception:
        raw = os.environ.get("AGENT_DISPLAY_MODE", "").strip().lower() or "verbose"
        try:
            return AgentDisplayMode(raw)
        except ValueError:
            return AgentDisplayMode.VERBOSE

#: Destructive shell patterns the ``run`` tool refuses (word-boundary,
#: case-insensitive) — the command injection surface of the NLP loop.
_DANGEROUS_SHELL_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"rm\s+-r[f]?", re.I), "recursive file removal (rm -r/-rf)"),
    (re.compile(r"\bdeltree\b", re.I), "deltree"),
    (re.compile(r"\brd\s+/s", re.I), "rd /s"),
    (re.compile(r"\brmdir\s+/s", re.I), "rmdir /s"),
    (re.compile(r"\bdel\s+/[sqf]", re.I), "del /s /q /f"),
    (re.compile(r"\bformat\s+[a-z]:", re.I), "format <drive>:"),
    (re.compile(r"\bshutdown\b", re.I), "shutdown"),
    (re.compile(r"\breboot\b", re.I), "reboot"),
    (re.compile(r"restart-computer", re.I), "restart-computer"),
    (re.compile(r"stop-computer", re.I), "stop-computer"),
    (re.compile(r"\bdiskpart\b", re.I), "diskpart"),
    (re.compile(r"\bmkfs\b", re.I), "mkfs"),
    (re.compile(r"wipefs", re.I), "wipefs"),
    (re.compile(r"\bdd\s+of=", re.I), "dd of="),
    (re.compile(r"taskkill\s+/f", re.I), "taskkill /f"),
    (re.compile(r"\breg\s+delete", re.I), "reg delete"),
    (re.compile(r"remove-item\s+-recurse", re.I), "Remove-Item -Recurse"),
    (re.compile(r"clear-recyclebin", re.I), "Clear-RecycleBin"),
    (re.compile(r"format-volume", re.I), "Format-Volume"),
    (re.compile(r"invoke-expression", re.I), "Invoke-Expression"),
]


def _unix_command_hint() -> str:
    """Return a hint appended to run results when a Unix-ism fails on
    Windows (tail/grep/ls/mypy-as-command are not available in cmd.exe)."""
    return (
        f"\nHint: this shell ({_detect_shell()}) has no Unix tools like tail/grep/ls, "
        "and installed Python tools must be called as 'python -m <tool>'. Use Python "
        "one-liners (python -c \"...\") or the built-in tools — run output is"
        " truncated "
        "automatically, so pipes like '2>&1 | tail -40' are neither needed nor"
        " supported."
    )


def _windows_command_hint() -> str:
    """Return a hint when a cmd.exe builtin / PowerShell cmdlet fails on a
    POSIX shell (Linux/macOS) — the mirror of :func:`_unix_command_hint`."""
    return (
        f"\nHint: this shell ({_detect_shell()}) is POSIX — there are no cmd.exe "
        "builtins (dir/type/findstr/copy/del/where) and no PowerShell cmdlets "
        "(Get-ChildItem/Select-String/Get-Content). Use ls/cat/grep/find, or a "
        "Python one-liner (python -c \"...\"); run output is auto-truncated."
    )


def _shell_command_hint() -> str:
    """The wrong-shell-command hint for the CURRENT platform (Unix-ism on
    Windows, Windows-ism on POSIX)."""
    return _unix_command_hint() if os.name == "nt" else _windows_command_hint()


def _final_answer_fallback(loop: "ToolLoopRunner") -> str:
    """Concrete stand-in when the model produced no usable answer.

    Instead of the cryptic "did not produce a response", report what the loop
    actually did (budget spent, tools used, last action, termination reason)
    so the user can react concretely — especially in QUIET mode, where tool
    noise is hidden and this is the only signal.
    """
    used = ", ".join(
        f"{tool}x{count}"
        for tool, count in sorted(loop.tools_used.items(), key=lambda kv: -kv[1])
    )
    return (
        f"The model made {loop.tool_calls_made} tool call(s) over "
        f"{loop.iterations_used} iteration(s) but produced no final answer "
        f"(terminated: {loop.termination_reason}). "
        f"Tools used: {used or 'none'}. Last action: {loop.last_tool_call or '-'}. "
        "Rephrase or ask something more specific to continue."
    )


def _kill_process_tree(proc: "subprocess.Popen[Any]") -> None:
    """Kill *proc* and its whole child tree (Windows: taskkill /T /F).

    A plain kill only terminates the shell; the orphaned children keep
    running and hold the captured pipes open.  Best effort — never raises.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
            )
        else:
            proc.kill()
    except Exception:
        logger.warning(
            "Failed to kill process tree for PID %s", getattr(proc, "pid", "?")
        )


def _git_branch() -> str:
    """Return the current Git branch name, or ``"(unknown)"`` if not available."""
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8", errors="replace",
            timeout=5,
            check=False,
        )
        branch = result.stdout.strip()
        if branch:
            return branch
    except (OSError, subprocess.SubprocessError):
        pass
    return "(unknown)"


def _turn_quality_score(llm_error: str | None, mutated_files: list[str]) -> float | None:
    """B3: per-turn quality score in [0,1] for the shared metrics log.

    With harnessfix tracing enabled *and* a turn-scoped events API present
    in ``harnessfix.tracing``, the trace events are scored with
    ``score_run``.  No such API exists today, so the traced path is
    skipped (never score the whole session as if it were this turn).
    With tracing off the cheap proxy is used: ``llm_error`` -> 0.0,
    mutated files -> 0.8, clean turn -> 0.7.  Never raises; returns
    ``None`` only if the check itself fails (nothing is recorded then).
    """
    try:
        from harnessfix.tracing import trace_enabled

        if trace_enabled():
            import harnessfix.tracing as _tr

            events: list[dict[str, Any]] | None = None
            for _name in ("current_turn_events", "take_turn_events",
                          "drain_turn_events", "turn_events"):
                _getter = getattr(_tr, _name, None)
                if callable(_getter):
                    events = _getter()
                    if events:
                        break
            if not events:
                for _attr in ("_writer", "current_writer", "get_writer"):
                    _obj = getattr(_tr, _attr, None)
                    if callable(_obj):
                        _obj = _obj()
                    if _obj is None:
                        continue
                    events = (getattr(_obj, "events", None)
                              or getattr(_obj, "_events", None))
                    if events:
                        break
            if events:
                from harnessfix.evolution_metrics import score_run

                return float(score_run(events))
            # Traced but no turn-scoped events: fall through to the proxy
            # rather than scoring the entire session.
        if llm_error:
            return 0.0
        if mutated_files:
            return 0.8
        return 0.7
    except Exception:  # noqa: BLE001 - observability must not break turns
        return None


def _append_turn_log(record: dict[str, Any]) -> None:
    """Append one turn-outcome line to ``turn_log.jsonl`` (bounded, atomic-ish).

    Harness-layer observability only (decision #014): the log feeds habit
    mining and is rewritten to its newest 500 lines whenever it grows past
    that bound, so it can never bloat the state dir.  Never raises — a broken
    or missing state dir must not break a finished turn.
    """
    try:
        path = TURN_LOG_PATH
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        lines: list[str] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
        lines.append(json.dumps(record, ensure_ascii=False))
        if len(lines) > 500:
            lines = lines[-500:]
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - observability must not break turns
        logger.debug("Could not append turn log (no-op)", exc_info=True)


def _strip_dynamic_system_blocks(text: str) -> str:
    """Remove previously injected dynamic blocks from a system prompt.

    ``chat_nlp`` rebuilds the constraints block (decision ledger) and the
    plan-mode suffix on every turn; without stripping, a long-lived session
    would accumulate one stale block per turn.  The BASE prompt — everything
    before the first dynamic marker — is what survives.
    """
    markers = (
        "\n\nCRITICAL DESIGN CONSTRAINTS",
        SKILL_INDEX_MARKER,
        HABITS_MARKER,
        SEMANTIC_MEMORY_MARKER,
        "\n\nSESSION MODE: PLAN",
    )
    cut = len(text)
    for marker in markers:
        pos = text.find(marker)
        if pos != -1:
            cut = min(cut, pos)
    return text[:cut]


def _detect_shell() -> str:
    """Return a human-readable name for the shell the run tool will use.

    Delegates to :func:`agent_core.subprocess_utils.shell_info` which walks
    the process tree to find the *actual* user shell, even when the run-tool
    injects ``cmd.exe`` as an intermediary.
    """
    from agent_core.subprocess_utils import shell_info
    info = shell_info()
    name = info["name"].lower()
    if name in ("powershell", "powershell.exe", "pwsh.exe"):
        return "PowerShell"
    if name in ("cmd", "cmd.exe"):
        return "cmd.exe (Windows Command Prompt)"
    if name in ("bash", "sh", "zsh", "fish", "dash", "ash", "ksh"):
        return f"{name} (or the default POSIX shell)"
    return info["name"]


def _shell_name_token() -> str:
    """Return a short machine-readable shell identifier.

    One of ``bash``, ``cmd``, or ``powershell`` — the canonical names used in
    system-prompt instructions so the LLM gets an unambiguous statement of its
    execution environment.  Derived from the real detected shell, not from
    environment variables.
    """
    from agent_core.subprocess_utils import shell_info
    info = shell_info()
    name = info["name"].lower()
    if name in ("powershell", "powershell.exe", "pwsh.exe"):
        return "powershell"
    if name in ("cmd", "cmd.exe"):
        return "cmd"
    return "bash"


#: Explicit shell identity propagated to the LLM system prompt.
#: Derived from the real detected shell via shell_info() — no env var needed.
_SHELL_NAME: str = _shell_name_token()


#: System prompt for the NLP tool loop, built once at import time (the only
#: dynamic piece is the detected shell name).  Kept as a module constant so
#: chat_nlp stays focused on the loop itself and the prompt lives in one
#: greppable place.
_SYSTEM_PROMPT = (
    "You are a senior coding assistant working inside this project workspace.\n"
    "The user speaks natural language; you have tools to search, read, write, "
    "edit, run, and test the code.\n"
    "You also talk normally: for greetings, small talk, or general questions "
    "that do not need the workspace, reply directly in text without calling "
    "any tool.\n\n"
    "WORKING METHOD:\n"
    "1. Understand the request.\n"
    "2. Take concrete action with the tools — never just describe what you "
    "would do. If you intend to read, search, edit or run something, call the tool.\n"
    "3. After a write/edit, a py_compile verification summary is returned "
    "automatically — report it.\n"
    "4. Finish with a short report: what you changed, where, and the "
    "verification/test evidence.\n\n"
    "RULES:\n"
    f"You are running inside SHELL_NAME={_SHELL_NAME} ({_detect_shell()}). "
    "This is a hard constraint: on Windows (cmd/PowerShell) there is NO "
    "tail/grep/ls/cat/sed/awk/wc/find; on POSIX there are NO cmd.exe builtins "
    "(dir/type/findstr/copy/del/where) and NO PowerShell cmdlets "
    "(Get-ChildItem/Select-String/Get-Content). Use Python one-liners "
    "(python -c \"...\") or the built-in tools instead of shell pipes — never "
    "construct commands that mix Unix and Windows syntax in a single invocation.\n"
    f"- The shell for the run tool is: {_detect_shell()}. On Windows there "
    "are NO Unix tools (tail/head/grep/ls/cat/sed/awk/wc) and no piping to "
    "them — use dir/type/findstr/more, PowerShell cmdlets, or Python "
    "one-liners (python -c \"...\"). Run output is truncated to 5000 chars "
    "automatically, so '2>&1 | tail -40' is neither needed nor supported. The "
    "run tool reports non-zero exit codes as [EXIT CODE: N] at the end of its "
    "output — read it from there, never chain '...; echo $?' (the model cannot "
    "see $? afterwards).\n"
    "- Python's sys.exitcode is only set at interpreter exit — use "
    "sys.exit(). Shell test tools ship with pytest (scripts exit non-zero "
    "on failure).\n"
    "- You have an effectively unlimited tool budget: do not rush, do not "
    "stop early, and do not plan around a budget. Take as many tool calls "
    "as the task needs — but make steady progress: when exploration is "
    "done, actually edit/write/fix instead of only reading.\n"
    "- read returns up to 5000 chars per call by default — never request "
    "tiny slices (limit < 1000). Read big chunks and continue with the "
    "offset hint; avoid many small read calls.\n"
    "- For 'where is X used/defined?' use references (one call) instead of "
    "search + reads. To orient inside a large file, call definitions first, "
    "then read only the line window you need — do not page blindly.\n"
    "- Prefer targeted edits over rewriting whole files.\n"
    "- If a tool fails, read the error and try a different approach.\n"
    "- When the request is ambiguous, make a reasonable assumption and state it, "
    "or ask one clarifying question before acting.\n"
    "- Never assert facts about this repo that you have not verified with a "
    "tool. project_plan.md / project_tasks.md are HISTORICAL phase docs and "
    "may be outdated — verify claims against the actual code.\n"
    f"- You are currently on Git branch: {_git_branch()}.\n"
    "- Git: call the git tool with subcommand='status' FIRST, before staging or "
    "committing. Stage everything with subcommand='add', args='-A' — a bare "
    "'-' is NOT a pathspec and will fail. Commit with subcommand='commit', "
    "args='-m \"message\"'. A request to \"commit changes\" means the FULL "
    "sequence status -> add -A -> commit -m -> (only if asked) push; never "
    "push alone, because `git push` only sends commits and would silently "
    "leave staged work behind. Never invent flags.\n"
    "- Verify numbers (e.g. how many tests exist) with the tests tool or git "
    "log before claiming them.\n"
    "- NEVER claim a repository state you have not observed with a tool this "
    "turn. Output the user saw in the terminal is NOT in your context — `git "
    "status` typed as a chat command is executed by the REPL and printed to "
    "stdout, so you never see it. If you called no tool, you do not know what "
    "is staged, committed or pushed. Call the git tool before stating it; an "
    "unverified \"already committed and pushed\" is a fabrication.\n"
    "- If a search finds nothing in source files, state that the symbol does "
    "not exist in the current code — never repeat the same search.\n"
    "- A failing test is a real signal: fix the implementation — never weaken "
    "or delete an assertion just to make it pass. Only change a test if the "
    "test itself is demonstrably wrong, and say why.\n"
    "- Every bug fix ships a permanent regression test in tests/ (pytest). "
    "Delete scratch scripts (_tmp_*.py) before finishing.\n"
    "- Verify fixes against the REAL code path (import the actual function), "
    "not a copied simulation of it.\n"
    "- Be concise. Answer in the user's language.\n"
    "\nSAFETY — AVOID FATAL CHANGES:\n"
    "- Never make unnecessary fatal changes: no `rm -rf`, no `git reset "
    "--hard`, no `git clean -f`, no force push, no deleting or overwriting "
    "files wholesale when a targeted edit will do.\n"
    "- Prefer the smallest targeted edit that solves the problem; keep diffs "
    "reviewable. When a change would be irreversible or destructive, STOP "
    "and ask the user first.\n"
    "- Destructive commands are blocked by the harness — never try to work "
    "around the block; report it instead.\n"
    "- IMPORTANT: If the user message is a greeting (hi, hello, hey, etc.) "
    "or small talk, reply with a short text greeting. Do NOT call any tools."
    "\n\nJEV DECISION ENGINE (fast, cheap, typed — use it for judgements):\n"
    "You have jev_decide: a dedicated non-thinking small model that answers "
    "TYPED questions with probabilities instead of prose. kind='yesno' (the "
    "question is a proposition -> P(yes)/P(no) plus TRUE/FALSE/UNKNOWN), "
    "kind='choice' (put the candidates in options -> a probability "
    "distribution), kind='score' (put the rubric in rubric -> 0-100 plus "
    "spread). Pass the evidence you already gathered in 'state' and set "
    "'threshold' (default 0.7). It is read-only and runs on a local small "
    "model, so it is fast and cheap — far cheaper than another reasoning pass.\n"
    "- Act on a yesno only when the decision is TRUE (P >= threshold); "
    "otherwise treat it as undecided and say so — never overclaim a guess.\n"
    "- Prefer it over guessing for bounded judgements: is this true / risky / "
    "needed? which of A/B/C is the best target? how well does this draft meet "
    "the rubric?\n"
    "- Do NOT use it for open-ended generation, code, or explanations — you "
    "write those yourself. It is a decision model, not a writer.\n"
    "\nSPECULATE — parallel grounded answers on demand:\n"
    "You have speculate: it runs N independent speculative branches in "
    "parallel (each reasons as you do and may call read-only tools to ground "
    "itself), judge-scores every branch, and COMMITs only the best when it "
    "meets the threshold — otherwise it REFUSES with the reasons. Use it when "
    "a fast corroborated answer beats your own long chain of thought: "
    "'where/what/how' questions about this repo (branches bring back file:line "
    "evidence), or any question where independent opinions are worth more than "
    "one. Speed knobs: judge='jev' scores on the dedicated small model; "
    "branch_model='jev' runs the WHOLE deliberation on it (fastest, but weak "
    "at grounding repo questions); fewer branches = faster. Read-only and "
    "allowed in plan mode."
    "\n\nMANAGER WORKFLOW (for complex tasks):\n"
    "1. Analyze if the task benefits from parallel specialists.\n"
    "2. Use create_subagent to spawn specialists (e.g. one for research, one for coding).\n"
    "3. Use run_subagent_task to assign work to each.\n"
    "4. Collect results and synthesize the final answer."
)


def _truncate_output(output: str | None, limit: int = 5000) -> str:
    """Cap shell output for the model's context: head + tail around a marker.

    Shared by the ``run``, ``git``, ``diff`` and ``tests`` NLP tools so every
    subprocess result is bounded identically (the system prompt promises
    "run output is truncated to 5000 chars automatically").

    ``output`` may be ``None`` when a subprocess produced no stdout; it is
    treated as an empty string in that case.
    """
    if output is None:
        output = ""
    if len(output) > limit:
        half = limit // 2
        return output[:half] + "\n... [truncated] ...\n" + output[-half:]
    return output if output else "(no output)"


def _run_subprocess_captured(
    cmd: list[str], cwd: str, timeout: float, label: str,
) -> tuple[str, str | None]:
    """Run *cmd* (arg-list, no shell), capture stdout+stderr, bound the result.

    Shared by the ``git``/``diff``/``tests`` NLP tools: identical timeout,
    stderr tagging ("[STDERR]\n…") and 5000-char truncation everywhere.
    Returns ``(output, error)`` — *error* is ``"<label> error: …"`` when the
    process could not be launched or timed out, else None.
    """
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=cwd, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"{label} timed out after {int(timeout)}s", None
    except Exception as e:
        return "", f"{label} error: {e}"
    output = r.stdout
    if r.stderr:
        output += f"\n[STDERR]\n{r.stderr}"
    return _truncate_output(output), None


def _shape_run_stderr(err: str | None, output: str | None, returncode: int | None) -> str:
    """Append stderr (plus Unix-ism hints) to ``run``-tool output.

    Shared by the ``run`` NLP tool so its stderr handling lives in one
    testable place instead of being inlined in the handler.

    When the command returns a non-zero exit code, the code is appended so
    the model can observe it directly — no need to capture it via shell
    pipelines (which often fail on Windows).

    ``output`` may be ``None`` when a subprocess produced no stdout; it is
    normalised to ``""`` up front.
    """
    if output is None:
        output = ""
    if not err:
        # cmd.exe silently fails whole pipelines (rc 255, no output)
        # when a pipe element or command does not exist.
        if os.name == "nt" and returncode == 255 and not output:
            return output + _shell_command_hint()
        if returncode is not None and returncode != 0:
            output += f"\n[EXIT CODE: {returncode}]"
        return output
    output += f"\n[STDERR]\n{err}"
    if re.search(r"is not recognized|not found|command not found", err, re.I):
        output += _shell_command_hint()
    if returncode is not None and returncode != 0:
        output += f"\n[EXIT CODE: {returncode}]"
    return output


def _blocked_shell_command(command: str) -> str | None:
    """Return a description of the first blocked destructive pattern in
    *command*, or None if the command passes the safety scan."""
    for pattern, desc in _DANGEROUS_SHELL_PATTERNS:
        if pattern.search(command):
            return desc
    return None


#: Repo-root .env consulted for the full-pytest budget (conftest reads it too).
_ENV_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
_PYTEST_FULL_SUITE_TIMEOUT_KEY = "PYTEST_FULL_SUITE_TIMEOUT"
_PYTEST_LAST_FULL_RUN_KEY = "PYTEST_LAST_FULL_RUN_SECONDS"
#: Must match conftest._FULL_SUITE_MARGIN so tool and watchdog agree.
_FULL_SUITE_MARGIN = 0.50


def _read_env_value(key: str) -> str:
    """Budget value for *key*: process env wins, else the repo .env."""
    if key in os.environ:
        return os.environ[key].strip()
    try:
        with open(_ENV_FILE_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                env_key, _, val = stripped.partition("=")
                if env_key.strip() == key:
                    return val.strip()
    except OSError:
        pass
    return ""


def _pytest_full_suite_timeout() -> float:
    """Timeout (seconds) to grant a full pytest run.

    The model regularly guesses a timeout (often 600s) for
    ``python -m pytest -q --no-cov``, which is below the suite's real budget on
    slower machines — the run tool then killed it mid-suite and the whole run
    was wasted.  Reuse the SAME budget conftest's watchdog enforces:
    ``max(PYTEST_FULL_SUITE_TIMEOUT, PYTEST_LAST_FULL_RUN_SECONDS * 1.5)``,
    read from the process env then the repo ``.env``.  Falling back to
    ``_MAX_RUN_TIMEOUT_S`` keeps behaviour identical on a bare checkout.
    """
    try:
        floor = float(
            _read_env_value(_PYTEST_FULL_SUITE_TIMEOUT_KEY) or _MAX_RUN_TIMEOUT_S
        )
    except ValueError:
        floor = float(_MAX_RUN_TIMEOUT_S)
    try:
        last = float(_read_env_value(_PYTEST_LAST_FULL_RUN_KEY) or 0.0)
    except ValueError:
        last = 0.0
    return max(floor, last * (1.0 + _FULL_SUITE_MARGIN))


#: Tokens that are part of the interpreter/pytest invocation itself, not a path.
_PYTEST_BARE_TOKENS = {
    "python", "python2", "python3", "py", "pytest", "py.test", "m", "set",
    "call", "&", "&&", "|", "||", ";",
}

#: pytest options that consume the FOLLOWING token as their value.  Without
#: this the value (``no:cacheprovider`` in ``-p no:cacheprovider``, ``not slow``
#: in ``-m "not slow"``, ``4`` in ``-n 4``) is mistaken for a test path, the
#: command is misclassified as TARGETED, and a full run is killed at the 600s
#: default instead of the suite budget.  Values attached with ``=`` are handled
#: by the generic flag check.
_PYTEST_VALUE_FLAGS = {
    "-p", "-k", "-m", "-o", "-c", "-n", "-W", "-r",
    "--deselect", "--ignore", "--ignore-glob", "--maxfail", "--tb",
    "--durations", "--rootdir", "--confcutdir", "--basetemp", "--junitxml",
    "--junit-prefix", "--log-level", "--log-file", "--log-format", "--dist",
    "--import-mode", "--cache-dir", "--override-ini", "--numprocesses",
    "--lfnf", "--last-failed-no-failures",
}


#: Flags that SELECT A SUBSET, so the invocation is not a full run: ``--lf``
#: / ``--nf`` (only failed / only new) and ``--testmon`` (only what changed).
#: Without this, ``pytest --lf`` looked like a full run and got both the suite
#: budget and the full-run gate - i.e. the cheap lanes were penalised exactly
#: like the expensive one.  Mirrors conftest._SUBSET_FLAGS.
_PYTEST_SUBSET_FLAGS = {
    "--lf", "--last-failed", "--lfnf", "--last-failed-no-failures",
    "--nf", "--new-first", "--failed-first", "--testmon",
}


def _is_full_pytest_command(command: str) -> bool:
    """True when *command* invokes pytest with no explicit test path.

    Mirrors conftest's definition of a full run: no positional path at all, or
    the whole test tree spelled out (``tests`` / ``tests/`` / ``.`` — the
    configured ``testpaths``).  Spelling the suite explicitly (``pytest
    tests/``) used to be misread as a targeted run, so it got the 120s default
    instead of the suite budget and was killed mid-run.  A subset selector
    (``--lf`` / ``--nf`` / ``--testmon``) is NOT a full run.  Segments without
    pytest, ``K=V`` assignments and redirections (``2>&1``, ``>nul``) are
    ignored; values consumed by :data:`_PYTEST_VALUE_FLAGS` are not paths;
    any other positional token marks a targeted run.
    """
    for segment in re.split(r"&&|\|\||[;|]", command):
        try:
            tokens = shlex.split(segment, posix=True)
        except ValueError:
            tokens = segment.split()
        if not any(t in ("pytest", "py.test") for t in tokens):
            continue
        skip_next = False
        for token in tokens:
            if skip_next:
                skip_next = False
                continue
            if token in _PYTEST_SUBSET_FLAGS:
                # Checked BEFORE the value-flag list: ``--lfnf`` is in both and
                # is boolean in pytest, so it must not swallow the next token.
                return False  # runs a subset, not the suite
            if token in _PYTEST_VALUE_FLAGS:
                skip_next = True
                continue
            if token.startswith("-"):
                continue
            if token in _PYTEST_BARE_TOKENS or "=" in token:
                continue
            if re.match(r"^\d*>&?\S*$", token):
                continue
            if token.replace("\\", "/").rstrip("/") in ("tests", "."):
                continue  # the whole test tree spelled out is still a full run
            return False
        return True
    return False


def _drop_orphan_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ``tool`` messages that are orphans or duplicate responses.

    - **Orphan**: a ``tool`` message whose ``tool_call_id`` has no matching
      assistant ``tool_calls`` message.
    - **Duplicate**: a second (or later) ``tool`` message for a
      ``tool_call_id`` already answered — strict gateways reject extras with
      HTTP 400 ("Messages with role 'tool' must be a response to a preceding
      message with 'tool_calls'").

    Trimming can cut between an assistant tool_calls message and its tool
    result, leaving an orphan; a backfill/sanitizer bug can leave duplicates.
    """
    valid_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    valid_ids.add(str(tc["id"]))
    # Dedupe WITHIN a contiguous run of tool results (reset on any non-tool
    # message): a later turn may legitimately reuse an id, so a global set
    # would wrongly drop a valid response.
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("role") == "tool":
            tid = str(m.get("tool_call_id") or "")
            if tid not in valid_ids or tid in seen:
                continue
            seen.add(tid)
            out.append(m)
        else:
            seen = set()
            out.append(m)
    return out


def _repair_unanswered_tool_calls(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Strip assistant ``tool_calls`` that have no matching tool response.

    The inverse of :func:`_drop_orphan_tool_messages`: an assistant message
    can announce a ``tool_calls`` batch whose results never arrive.  The
    projection removes loop-steering results ("NOTE: This ...") as cross-session
    noise, which orphans the assistant call that requested them; trimming a
    restored window can also cut the results.  Strict gateways reject the
    payload with HTTP 400 ("An assistant message with 'tool_calls' must be
    followed by tool messages responding to each 'tool_call_id'").

    The assistant's text content is preserved — only the unanswered calls are
    removed.  A message left with neither content nor calls is dropped.
    """
    answered = {
        str(m.get("tool_call_id") or "")
        for m in messages
        if m.get("role") == "tool"
    }
    out: list[dict[str, Any]] = []
    for m in messages:
        calls = m.get("tool_calls")
        if m.get("role") == "assistant" and calls:
            kept = [
                tc for tc in calls
                if isinstance(tc, dict) and str(tc.get("id") or "") in answered
            ]
            if len(kept) != len(calls):
                if kept:
                    m = {**m, "tool_calls": kept}
                else:
                    m = {k: v for k, v in m.items() if k != "tool_calls"}
                    if not str(m.get("content") or ""):
                        continue
        out.append(m)
    return out


def _strip_tool_args(m: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *m* with large tool-call arguments truncated.

    The LLM only needs to know WHICH tool was called (function name) and
    the tool result — not the full multi-KB argument payload (e.g. a
    ``write`` call with 10KB of markdown).  Stripping these before sending
    history to the LLM dramatically cuts prefill time and avoids HTTP 500
    on small local models.
    """
    tcs = m.get("tool_calls")
    if not tcs or not isinstance(tcs, list):
        return m
    stripped = []
    for tc in tcs:
        if not isinstance(tc, dict):
            stripped.append(tc)
            continue
        func = tc.get("function") or {}
        args = func.get("arguments") or ""
        if isinstance(args, str) and len(args) > 500:
            # Must remain VALID JSON: the gateway validates function.arguments
            # and rejects a "[truncated]" fragment with HTTP 400 ("Assistant
            # tool call function.arguments must be valid JSON").  A compact
            # placeholder keeps the call identifiable without a half-cut string.
            func = {
                **func,
                "arguments": json.dumps(
                    {"_truncated": True, "_original_len": len(args)}
                ),
            }
            stripped.append({**tc, "function": func})
        else:
            stripped.append(tc)
    return {**m, "tool_calls": stripped}


def _message_size(m: dict[str, Any]) -> int:
    """Approximate character size of one message (4 chars ≈ 1 token).

    Tool-call argument JSON counts too — a single big write call can dwarf
    its tiny ``content`` string.
    """
    total = len(str(m.get("content") or ""))
    for tc in m.get("tool_calls") or []:
        if isinstance(tc, dict):
            func = tc.get("function") or {}
            total += len(str(func.get("arguments") or ""))
            total += len(str(func.get("name") or ""))
    return total


def _trim_chat_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the conversation within a bounded context; returns a new list.

    Two caps apply (plan item B-#5 — the old code had only the count cap, so
    a handful of huge read/write messages could blow the context window while
    staying under 60 messages):

    - ``_MAX_CHAT_MESSAGES``: at most this many messages (system prompt +
      newest tail), as before.
    - ``_HISTORY_CHAR_BUDGET``: the body must also fit a rough char budget;
      when it does not, the OLDEST body messages are dropped until it fits
      (contiguous tail — no holes), and a compaction note tells the next turn
      that earlier context was dropped.

    Orphan tool messages are dropped both before and after the cut (see
    :func:`_drop_orphan_tool_messages`): a count-based slice can cut between
    an assistant tool_calls message and its tool result, and strict gateways
    reject those orphans with HTTP 400.  The reverse case — an assistant
    ``tool_calls`` message whose results were dropped — is repaired by
    :func:`_repair_unanswered_tool_calls`.
    """
    messages = _drop_orphan_tool_messages(messages)
    if len(messages) <= _MAX_CHAT_MESSAGES:
        head = messages[:1]
        body = list(messages[1:])
    else:
        head = messages[:1]
        body = list(messages[-(_MAX_CHAT_MESSAGES - 1):])

    # Strip large tool-call arguments from assistant messages in the body:
    # the LLM only needs to know WHICH tool was called, not the full payload
    # (e.g. a 10KB write() call).  This keeps the char-budget accurate for
    # prefill without losing conversation structure.
    body = [_strip_tool_args(m) for m in body]

    # Char-budget trim: walk from the NEWEST body message backwards until the
    # budget is exhausted.  A candidate is only accepted when it is not an
    # assistant-tool_calls/tool boundary split — i.e. we may drop a prefix
    # ending anywhere EXCEPT between an assistant(tool_calls) message and its
    # following tool result.
    total = sum(_message_size(m) for m in body)
    if total <= _HISTORY_CHAR_BUDGET:
        return _repair_unanswered_tool_calls(
            _drop_orphan_tool_messages(head + body)
        )
    keep_from = 0
    running = total
    for i, m in enumerate(body):
        if running <= _HISTORY_CHAR_BUDGET:
            keep_from = i
            break
        running -= _message_size(m)
        role = m.get("role")
        prev_role = body[i - 1].get("role") if i else None
        boundary_safe = not (
            role == "tool" or prev_role == "assistant" and "tool_calls" in body[i - 1]
        )
        if i == len(body) - 1 or boundary_safe:
            keep_from = i + 1
    trimmed = body[keep_from:]
    note = {
        "role": "user",
        "content": _HISTORY_TRIM_NOTE.format(dropped=keep_from),
    }
    return _repair_unanswered_tool_calls(
        _drop_orphan_tool_messages(head + [note] + trimmed)
    )


def _strip_image_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return *messages* with any multimodal image content stripped out.

    Vision turns carry base64 data URLs that can be several megabytes each;
    persisting them to ``chat_history.json`` would bloat the file and waste
    context on reload.  A user message that was purely an image (no text block)
    is dropped entirely; a mixed text+image message keeps its text and loses
    the image blocks so the conversation stays coherent without the payload.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            text_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            image_blocks = [
                b for b in content
                if isinstance(b, dict) and b.get("type") == "image_url"
            ]
            if not image_blocks:
                out.append(m)
                continue
            if not text_blocks:
                # Purely an image message — drop it (nothing to continue on).
                continue
            out.append({**m, "content": text_blocks})
            continue
        out.append(m)
    return out


def _project_chat_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project a conversation down to what the NEXT session should see.

    Keeps the system prompt plus a bounded multi-exchange window (see
    :func:`_trim_chat_history`) so a fresh session continues the dialogue
    instead of forgetting it.  Loop steering notes (``NOTE: This ...`` tool
    messages), empty assistant placeholders, and orphan tool messages are
    dropped as cross-session noise.
    """
    if not messages:
        return []
    head = messages[:1] if messages and messages[0].get("role") == "system" else []
    body = messages[1:] if head else messages
    cleaned = []
    for m in body:
        content = str(m.get("content") or "")
        if m.get("role") == "tool" and content.startswith("NOTE: This"):
            continue  # loop steering noise — meaningless in a fresh session
        if m.get("role") == "assistant" and not content and not m.get("tool_calls"):
            continue  # empty placeholder with no action
        cleaned.append(m)
    return _trim_chat_history(head + cleaned)


# ---------------------------------------------------------------------------
# Shared metrics plumbing / dashboard entry points.
#
# Moved to agent_dashboard.py; re-exported here so existing imports and test
# monkeypatches on ``agent`` (e.g. ``monkeypatch.setattr(agent, "_shared_metrics_collector", None)``) continue to work — the functions below reference this module's binding of that name, which is the same object imported from agent_dashboard.
from agent_dashboard import (  # noqa: F401
    _build_dashboard,
    _dashboard_port,
    _default_alert_rules,
    _emit_command_metrics,
    _emit_tool_metrics,
    _shared_metrics_collector,
    get_metrics_collector,
    record_command_metrics,
    run_dashboard_server,
    start_dashboard_thread,
)


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
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            cwd=agent.workspace, timeout=10,
        )
        if r.returncode != 0:
            return  # not a git repo (or git missing) — nothing to say
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if not lines:
            return
        print(yellow(
            f"\n  [git] {len(lines)} uncommitted change(s) in the workspace — "
            "commit them now (AGENTS.md invariant #4: uncommitted work is "
            "unrecoverable). Top offenders:"
        ))
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
    return (
        f"\n[{now:%Y-%m-%d %H:%M}] {branch} "
        f"{prompt_color(mode_tag)}"
        f"{prompt_color('> ')}"
    )


async def run_interactive() -> None:
    """Interactive mode - allows user to input commands."""

    # Create agent instance (resolves persisted model choice from model.json)
    agent = Agent(workspace=Agent.DEFAULT_WORKSPACE)

    banner = blue("=" * 50)
    print(banner)
    print(blue("Agent Interactive Mode with LM Studio"))
    print(f"Workspace: {cyan(Agent.DEFAULT_WORKSPACE)}")
    model_label = agent.llm.model_name
    profile_part = (
        f"  |  Profile: {agent.llm._profile_name}"
        if agent.llm._profile_name else "")
    print(f"Model: {cyan(model_label)}{gray(profile_part)}")
    try:
        from agent_core.llm.lmstudio import get_models_status
        models = get_models_status()
        loaded = [m for m in models if m.get("loaded")]
        status = (
            f"LM Studio: online ({len(loaded)}/{len(models)} models loaded)"
            if models else "LM Studio: online")
        print(green(status) if models else yellow(status))
    except Exception:
        print(yellow("LM Studio: offline"))
    print(blue("=" * 50))
    # Single source of truth: the banner derives its command list from the
    # registry itself (each command's help_text synopsis), so a new or
    # changed command can never drift from what the banner shows.
    registry = _build_registry()
    for name in sorted(registry.names()):
        cmd = registry.get(name)
        synopsis = cmd.help_text.splitlines()[0].strip() if cmd else name
        print(f"  {cyan(synopsis)}")
    # Surface the safe read-only generation path prominently: `propose` (and the
    # --propose flags on implement/fix) generate a reviewed diff bundle without
    # ever touching the working tree — the recommended way to try changes.
    print(gray(
        "  Tip: `propose <taskplan.md>` (or `implement --propose` / "
        "`fix --mypy --propose`) generates a reviewed diff bundle; "
        "it never writes the working tree."))
    print(f"  {cyan('quit')} - {gray('Exit')}")
    print(blue("=" * 50))
    _register_commands(registry)

    # Set up command registry with simple commands
    # Warn once per change wave when loaded code changed on disk.
    from agent_core.commands.freshness import (
        diff_snapshots,
        format_stale_warning,
        loaded_module_fingerprints,
    )
    _code_snapshot = loaded_module_fingerprints(__file__)
    _install_signal_handlers(agent)

    while True:
        try:
            # Get user input — _interruptible_input uses a background thread
            # so the process can be killed via SIGBREAK on Windows.
            branch = _current_git_branch() or "?"
            user_input = _interruptible_input(
                _build_chat_prompt(agent, branch, datetime.now())
            )
            if user_input is None:
                # Shutdown requested or stdin exhausted
                agent._save_memory()
                save_meta_policy_weights()
                _warn_uncommitted(agent)
                break
            if not user_input:
                continue

            # Check for quit command
            if user_input.lower() in ["quit", "exit", "q"]:
                agent._save_memory()
                save_meta_policy_weights()
                _warn_uncommitted(agent)
                print(green("Goodbye!"))
                break

            # Warn once per change wave when loaded code changed on disk.
            _stale_files = diff_snapshots(_code_snapshot)
            if _stale_files:
                print(yellow(format_stale_warning(_stale_files)))
                _code_snapshot = loaded_module_fingerprints(__file__)

            # Parse and execute commands
            try:
                parts = shlex.split(user_input, posix=False)
            except ValueError:
                parts = user_input.split(maxsplit=20)
            command = parts[0].lower()

            # Try commands from registry (driven by the registry itself so
            # newly registered commands are dispatched without touching the
            # whitelist here — "review" fell through to chat_nlp once).
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
            print(yellow(
                "\nInterrupted — the current run was stopped. Use 'quit' to"
                " exit."))
        except EOFError:
            if not sys.stdin.isatty():
                print(yellow("\n[stdin] Input stream ended — shutting down."))
            agent._save_memory()
            save_meta_policy_weights()
            _warn_uncommitted(agent)
            break


async def main() -> None:
    """Main entry point - runs interactive mode or the web dashboard."""
    if "--serve" in sys.argv:
        # run_dashboard_server() is synchronous (ThreadingHTTPServer.serve_forever);
        # awaiting it raised "object NoneType can't be used in 'await' expression".
        run_dashboard_server()
        return
    if "--dashboard" in sys.argv:
        # Interactive REPL + live web dashboard in the same process: the REPL
        # feeds get_metrics_collector(), the daemon thread serves it.
        start_dashboard_thread(_dashboard_port())
    # No --serve flag: fall back to the classic interactive CLI.
    await run_interactive()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
