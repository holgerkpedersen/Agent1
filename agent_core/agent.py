"""Agent core (refactor step r08): the AgentCore class.

The former body of ``agent.py``.  ``agent.py`` is now a thin shim that
re-exports this module (plus the CLI layer) so ``import agent``,
``python -m agent`` and every monkeypatch on the ``agent`` module keep
working unchanged.
"""
"""Agent implementation with workspace management and tool execution.."""
from agent_core.bootstrap import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.monitoring import MetricsCollector
    from agent_core.subagent import SubAgent

logger = logging.getLogger(__name__)


def _agent_mod() -> Any:
    """The ``agent`` shim module, resolved lazily.

    Names that tests monkeypatch on ``agent`` (``LLMClient``, ``FileSystem``,
    ``_resolve_display_mode``, ...) are read through it, so patching the shim
    keeps reaching this class exactly as it did when the class lived in
    ``agent.py``.
    """
    import agent as _agent_mod
    return _agent_mod

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
        if ch == '\\':
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            in_string = False
            out.append(ch)
            continue
        code = ord(ch)
        if code < 32:
            out.append({'\n': '\\n', '\r': '\\r', '\t': '\\t'}.get(ch, f'\\u{code:04x}'))
            continue
        out.append(ch)
    return ''.join(out)

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
from agent_core.suppress_log import _suppress_and_log
from agent_core.signal_handlers import _signal_break_with_cleanup, install as _install_signal_handlers, current_git_branch as _current_git_branch, interruptible_input as _interruptible_input
from agent_core.handlers.nlp_tools import _run_subprocess_captured
from agent_core.shell_utils import detect_shell as _detect_shell, shell_name_token as _shell_name_token, current_git_branch as _git_branch, unix_command_hint as _unix_command_hint, windows_command_hint as _windows_command_hint, shell_command_hint as _shell_command_hint, truncate_output as _truncate_output, run_subprocess_captured as _shell_run_subprocess_captured, shape_run_stderr as _shape_run_stderr, blocked_shell_command as _blocked_shell_command, read_env_value as _read_env_value, pytest_full_suite_timeout as _pytest_full_suite_timeout, is_full_pytest_command as _is_full_pytest_command, is_pytest_command as _is_pytest_command, is_subset_pytest_command as _is_subset_pytest_command, SYSTEM_PROMPT as _SYSTEM_PROMPT
from agent_core.json_utils import _read_json_quarantining
from agent_core.history_compaction import _recover_original_goal, _strip_image_blocks, _trim_chat_history, _project_chat_history, _HISTORY_CHAR_BUDGET, _MAX_CHAT_MESSAGES, _HISTORY_TRIM_NOTE, _is_compaction_note, _strip_turn_wrappers, _content_tokens
from agent_core.llm.tool_loop import _message_size
from agent_core.run_utils import _start_kanban_inbound, kill_process_tree as _kill_process_tree, unverified_action_claim as _unverified_action_claim, resolve_stream_fn as _resolve_stream_fn, resolve_display_mode as _resolve_display_mode, final_answer_fallback as _final_answer_fallback, turn_quality_score as _turn_quality_score, begin_trace_turn as _begin_trace_turn, append_turn_log as _append_turn_log, strip_dynamic_system_blocks as _strip_dynamic_system_blocks
NlpToolHandler = Callable[..., Awaitable[str]]
from agent_core.types import _GOAL_RELATED_THRESHOLD
from agent_core.llm.client import LLMClient
from agent_dashboard import _emit_command_metrics, _emit_tool_metrics, get_metrics_collector, record_command_metrics
from agent_core.handlers.nlp_tools import (
    _nlp_delegate,
    _nlp_delegate_batch,
    _run_command_quietly,
    _effective_ws_dir,
    _nlp_create_subagent,
    _nlp_run_subagent_task,
    _nlp_tool_handlers,
    _nlp_mcp_tools,
    _nlp_mcp_call,
    _nlp_definitions,
    _nlp_references,
    _nlp_jev_decide,
    _nlp_speculate,
    _nlp_search,
    _nlp_read,
    _nlp_read_many,
    _read_page,
    _nlp_read_skill,
    _nlp_list_files,
    _nlp_fix,
    _nlp_write,
    _nlp_run,
    _nlp_git,
    _nlp_merge,
    _nlp_diff,
    _nlp_edit,
    _nlp_tests,
    _nlp_analyze,
    _nlp_web_search,
    _nlp_get_current_datetime,
    _nlp_hue_control,
    _get_plan_state,
    _nlp_plan_status,
    _nlp_plan_start,
    _nlp_plan_step,
    _nlp_plan_finish,
)

from agent_core.agent_files import (
    _tool_read_file,
    _tool_write_file,
    _tool_apply_patch,
    _tool_edit_file,
    _tool_search,
    _tool_list_files,
    _tool_delete_file,
    _tool_analyze_file,
    _tool_llm_analyze,
    execute_tool,
    _normalize_path,
    _safe_path,
)

from agent_core.agent_effects import (
    _resolve_nlp_path,
    _save_verify_note,
    _verify_file,
    _note_effect,
    _take_trace_effects,
)

from agent_core.turn_loop import (
    _run_chained_tool_loop,
    _reset_failover_notice,
    _report_failover,
    _finish_turn,
    _mutating_files_this_turn,
    _print_self_review_note,
    check_stale_files,
    invalidate_stale,
)

from agent_core.prompts.context_blocks import (
    _decision_constraints_block,
    _skill_index_block,
    _skill_hint_block,
    _habits_block,
    _file_history_block,
    _semantic_memory_block,
    _ensure_original_goal,
    _recent_user_texts,
    _prompts_related,
    _suspend_history_for_new_task,
    _original_goal_block,
)

from agent_core.memory.persistence import (
    _persist_plan_answer,
    memory_stats,
    clear_history,
    _load_chat_history,
    _save_chat_history,
    _load_memory,
    _save_memory,
    _record_llm_experience,
)


class AgentCore:
    """Main agent class with workspace management and tool execution."""
    DEFAULT_WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, workspace: str | None=None, model_name: str | None=None):
        self.workspace = os.path.abspath(_agent_mod().to_windows_path(workspace or self.DEFAULT_WORKSPACE))
        self.model_name = _agent_mod().resolve_model(model_name)
        self._semantic_index: dict[str, set[int]] = defaultdict(set)
        self._files_read: set[str] = set()
        self._file_mtimes: dict[str, float] = {}
        self._full_run_gate = _agent_mod().FullRunGate(self.workspace)
        self._pending_effects: list[str] | None = None
        self._knowledge_graph: dict[str, Any] = {}
        self._working_memory: list[Any] = []
        self._original_goal: str = ''
        self._goal_from_restore: bool = False
        self._prior_chat_history: list[dict[str, Any]] = []
        self._history: list[Any] = []
        self._chat_history: list[dict[str, Any]] = self._load_chat_history()
        self._turn_start_index: int = len(self._chat_history)
        self._turn_run_id: str | None = None
        self._active_trace_writer: Any = None
        self._read_streak: int = 0
        self._delegating: bool = False
        self._active_subagents: set[str] = set()
        self._delegate_counter: int = 0
        self._nlp_workspace: str | None = None
        self.mode: str = MODE_BUILD
        self._load_memory()
        self._ensure_original_goal()
        self.llm = _agent_mod().LLMClient(model_name=self.model_name)
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
                    self.llm._provider.apply_profile(prof_name, prof.temperature, prof.max_tokens)
        except Exception as _seam_err:
            logger.exception('build_transport seam wiring failed:\n')
        self.fs = _agent_mod().FileSystem(self.workspace)
        self.searcher = _agent_mod().FileSearcher(self.workspace)
        self.dispatcher = _agent_mod().ToolDispatcher(on_tool=_emit_tool_metrics)
        self._register_tool_handlers()

    def _register_tool_handlers(self) -> None:
        """Register tool handlers with the dispatcher."""
        self.dispatcher.register('read_file', lambda args: self._tool_read_file(**args))
        self.dispatcher.register('write_file', lambda args: self._tool_write_file(**args))
        self.dispatcher.register('apply_patch', lambda args: self._tool_apply_patch(**args))
        self.dispatcher.register('edit_file', lambda args: self._tool_edit_file(**args))
        self.dispatcher.register('search', lambda args: self._tool_search(**args))
        self.dispatcher.register('search_file', lambda args: self._tool_search(**args))
        self.dispatcher.register('list_files', lambda args: self._tool_list_files(**args))
        self.dispatcher.register('delete_file', lambda args: self._tool_delete_file(**args))
        self.dispatcher.register('analyze_file', lambda args: self._tool_analyze_file(**args))
        self.dispatcher.register('llm_analyze', lambda args: self._tool_llm_analyze(**args))

    def build_transport(self, client: 'LLMClient') -> Any:
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

    def spawn_subagent(self, name: str, workspace: str | None=None, role: str | None=None) -> 'SubAgent':
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

    def run_parallel_tasks(self, tasks: list[Callable[[], str]], max_workers: int=10) -> list[str]:
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
                except Exception as exc:
                    results.append(f'[task-error] {exc}')
        return results

    def record_demo_activity(self, activity: str='analyze', latency_ms: float=120.0) -> dict[str, float]:
        """Record one synthetic activity into the shared metrics collector.

        Mirrors exactly what ``record_command_metrics()`` writes for a real
        REPL command, so the dashboard treats demo data identically:
        ``command.<name>.count`` counter + elapsed-seconds histogram sample
        + ``last.command.seconds`` gauge. Returns the written values.
        """
        elapsed_s = max(0.001, latency_ms / 1000.0)
        _emit_command_metrics(activity, elapsed_s)
        collector = self.get_metrics_collector()
        return {'events': 3, 'counter': collector.get_counter_value(f'command.{activity}.count'), 'elapsed_s': elapsed_s}

    def get_metrics_collector(self) -> 'MetricsCollector':
        """Instance delegate for the process-wide collector (test seam)."""
        return get_metrics_collector()

    def _resolve_nlp_path(self, path: str) -> str:
        return _resolve_nlp_path(self, path)

    async def _save_verify_note(self, path: str, content: str, ok_msg: str) -> str:
        return await _save_verify_note(self, path, content, ok_msg)

    async def _verify_file(self, path: str) -> str:
        return await _verify_file(self, path)

    def _note_effect(self, path: str) -> None:
        _note_effect(self, path)

    def _take_trace_effects(self, name: str, args: dict[str, Any]) -> list[str]:
        return _take_trace_effects(self, name, args)

    def set_mode(self, mode: str) -> None:
        """Switch the session mode (``build`` | ``plan``).

        Unknown tags raise ``ValueError`` — the ``mode`` command validates
        before calling, so this is a programming-error guard, not a silent
        fallback to a mode the user did not ask for.
        """
        if mode not in (MODE_BUILD, 'plan'):
            raise ValueError(f'unknown mode: {mode!r}')
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
        if name != 'read':
            self._read_streak = 0
        rejection = check_tool_allowed(name, self.mode)
        if rejection:
            return rejection
        handler = self._nlp_tool_handlers().get(name)
        if handler is None:
            return f"Unknown tool: {name}. Available: {', '.join(sorted(NLP_TOOL_NAMES))}"
        try:
            return await handler(args)
        except Exception as e:
            logger.exception('NLP tool %r failed', name)
            return f'{name} error: {e}'

    async def _nlp_delegate(self, args: dict[str, Any]) -> str:
        return await _nlp_delegate(self, args)

    async def _nlp_delegate_batch(self, args: dict[str, Any]) -> str:
        return await _nlp_delegate_batch(self, args)

    async def _run_command_quietly(self, command: Command, args: list[str], done_msg: str) -> str:
        return await _run_command_quietly(self, command, args, done_msg)

    def _effective_ws_dir(self) -> str:
        return _agent_mod()._effective_ws_dir(self)

    async def _nlp_create_subagent(self, args: dict[str, Any]) -> str:
        return await _nlp_create_subagent(self, args)

    async def _nlp_run_subagent_task(self, args: dict[str, Any]) -> str:
        return await _nlp_run_subagent_task(self, args)

    def _nlp_tool_handlers(self) -> dict[str, NlpToolHandler]:
        return _nlp_tool_handlers(self)

    async def _nlp_mcp_tools(self, args: dict[str, Any]) -> str:
        return await _nlp_mcp_tools(self, args)

    async def _nlp_mcp_call(self, args: dict[str, Any]) -> str:
        return await _nlp_mcp_call(self, args)

    async def _nlp_definitions(self, args: dict[str, Any]) -> str:
        return await _nlp_definitions(self, args)

    async def _nlp_references(self, args: dict[str, Any]) -> str:
        return await _nlp_references(self, args)

    async def _nlp_jev_decide(self, args: dict[str, Any]) -> str:
        return await _nlp_jev_decide(self, args)

    async def _nlp_speculate(self, args: dict[str, Any]) -> str:
        return await _nlp_speculate(self, args)

    async def _nlp_search(self, args: dict[str, Any]) -> str:
        return await _nlp_search(self, args)

    async def _nlp_read(self, args: dict[str, Any]) -> str:
        return await _nlp_read(self, args)

    async def _nlp_read_many(self, raw_paths: list[Any], args: dict[str, Any]) -> str:
        return await _nlp_read_many(self, raw_paths, args)

    async def _read_page(self, path: str, args: dict[str, Any], original: Any) -> str:
        return await _read_page(self, path, args, original)

    async def _nlp_read_skill(self, args: dict[str, Any]) -> str:
        return await _nlp_read_skill(self, args)

    async def _nlp_list_files(self, args: dict[str, Any]) -> str:
        return await _nlp_list_files(self, args)

    async def _nlp_fix(self, args: dict[str, Any]) -> str:
        return await _nlp_fix(self, args)

    async def _nlp_write(self, args: dict[str, Any]) -> str:
        return await _nlp_write(self, args)

    async def _nlp_run(self, args: dict[str, Any]) -> str:
        return await _nlp_run(self, args)

    async def _nlp_git(self, args: dict[str, Any]) -> str:
        return await _nlp_git(self, args)

    async def _nlp_merge(self, args: dict[str, Any]) -> str:
        return await _nlp_merge(self, args)

    async def _nlp_diff(self, args: dict[str, Any]) -> str:
        return await _nlp_diff(self, args)

    async def _nlp_edit(self, args: dict[str, Any]) -> str:
        return await _nlp_edit(self, args)

    async def _nlp_tests(self, args: dict[str, Any]) -> str:
        return await _nlp_tests(self, args)

    async def _nlp_analyze(self, args: dict[str, Any]) -> str:
        return await _nlp_analyze(self, args)

    async def _nlp_web_search(self, args: dict[str, Any]) -> str:
        return await _nlp_web_search(self, args)

    async def _nlp_get_current_datetime(self, args: dict[str, Any]) -> str:
        return await _nlp_get_current_datetime(self, args)

    async def _nlp_hue_control(self, args: dict[str, Any]) -> str:
        return await _nlp_hue_control(self, args)

    async def _get_plan_state(self) -> dict[str, Any]:
        return await _get_plan_state(self)

    async def _nlp_plan_status(self, args: dict[str, Any]) -> str:
        return await _nlp_plan_status(self, args)

    async def _nlp_plan_start(self, args: dict[str, Any]) -> str:
        return await _nlp_plan_start(self, args)

    async def _nlp_plan_step(self, args: dict[str, Any]) -> str:
        return await _nlp_plan_step(self, args)

    async def _nlp_plan_finish(self, args: dict[str, Any]) -> str:
        return await _nlp_plan_finish(self, args)

    async def _tool_read_file(self, path: str, **kwargs: Any) -> str:
        return await _tool_read_file(self, path, **kwargs)

    async def _tool_write_file(self, path: str, content: str, **kwargs: Any) -> str:
        return await _tool_write_file(self, path, content, **kwargs)

    async def _tool_apply_patch(self, path: str, find: str, replace: str, **kwargs: Any) -> str:
        return await _tool_apply_patch(self, path, find, replace, **kwargs)

    async def _tool_edit_file(self, path: str, content: str, **kwargs: Any) -> str:
        return await _tool_edit_file(self, path, content, **kwargs)

    async def _tool_search(self, query: str, path: str='.', **kwargs: Any) -> str:
        return await _tool_search(self, query, path, **kwargs)

    async def _tool_list_files(self, path: str='.', pattern: str='*', **kwargs: Any) -> str:
        return await _tool_list_files(self, path, pattern, **kwargs)

    async def _tool_delete_file(self, path: str, **kwargs: Any) -> str:
        return await _tool_delete_file(self, path, **kwargs)

    async def _tool_analyze_file(self, path: str, **kwargs: Any) -> str:
        return await _tool_analyze_file(self, path, **kwargs)

    async def _tool_llm_analyze(self, path: str, **kwargs: Any) -> str:
        return await _tool_llm_analyze(self, path, **kwargs)

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name using the dispatcher."""
        return await execute_tool(self, tool_name, arguments)

    def _normalize_path(self, path: str) -> str:
        """Normalize and validate paths with security checks."""
        return _normalize_path(self, path)

    def _safe_path(self, path: str) -> str:
        """Validate and normalize path in one step."""
        return _safe_path(self, path)

    async def read_file(self, path: str, track_read: bool=True) -> str:
        local_path = self._safe_path(path)
        if os.path.isdir(local_path):
            return f'Not a file (directory): {path} — read/analyze take a FILE path; omit the path to analyze the whole workspace, or use list_files/search to inspect a directory.'
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if track_read:
                self._files_read.add(local_path)
                try:
                    self._file_mtimes[local_path] = os.path.getmtime(local_path)
                except OSError as e:
                    logger.warning('Could not stat %s: %s', local_path, e)
            return content
        except IsADirectoryError:
            return f'Not a file (directory): {path}'
        except PermissionError as e:
            return f'Permission denied reading {path}: {e}'
        except FileNotFoundError:
            return f'File not found: {path}'
        except Exception as e:
            return f'Error reading file: {e}'

    async def write_file(self, path: str, content: str) -> str:
        local_path = self._safe_path(path)
        try:
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f'Successfully wrote to {path}'
        except PermissionError:
            return f'Permission denied: {path}'
        except Exception as e:
            return f'Error writing file: {e}'

    async def apply_patch(self, path: str, find: str, replace: str) -> str:
        local_path = self._safe_path(path)
        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if find not in content:
                return 'Pattern not found in file'
            count = content.count(find)
            if count > 1:
                return f'Error: find text matches {count} locations. Add more context to make it unique.'
            new_content = content.replace(find, replace, 1)
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return 'Patch applied successfully'
        except PermissionError:
            return f'Permission denied: {path}'
        except Exception as e:
            return f'Error applying patch: {e}'

    async def edit_file(self, path: str, content: str) -> str:
        local_path = self._safe_path(path)
        try:
            if save_file_py(local_path, content, auto_yes=True):
                return f'Successfully edited {path}'
            return f'Skipped {path} (no changes)'
        except PermissionError:
            return f'Permission denied: {path}'
        except Exception as e:
            return f'Error editing file: {e}'

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

    async def search_file(self, query: str, path: str | None=None) -> str:
        """Search workspace files for text.

        Delegates to the shared :class:`FileSearcher` (excludes git-ignored
        state/cache/binary files, returns ``path:lineno: content`` lines), so
        the REPL ``search`` command and the NLP ``search`` tool behave alike.
        """
        results = await self.searcher.search(query, path or self.workspace)
        if not results or results == 'No matches found':
            return 'No matches found'
        return results

    def _refresh_system_message(self, query: str | None=None) -> None:
        """Ensure history starts with a system message and rebuild its dynamic
        blocks (decision-constraints / skill index / plan-mode suffix).

        *query* is the current user message (when known) — forwarded to
        :meth:`_semantic_memory_block` so the SEMANTIC MEMORY block shows the
        ledger entries most relevant to THIS turn.

        The stored BASE prompt (position 0 minus previously injected dynamic
        blocks) is preserved — only the dynamic blocks are rebuilt, so nothing
        accumulates and a restored session's prompt is never clobbered.  A
        long-lived session must see the CURRENT decision ledger and the CURRENT
        skill index (a runbook added to ``skills/`` mid-session is advertised on
        the next turn), not the snapshot taken on the first turn.
        """
        if not self._chat_history:
            self._chat_history.append({'role': 'system', 'content': _SYSTEM_PROMPT + (plan_mode_system_suffix() if self.is_plan_mode() else '')})
        elif self._chat_history[0].get('role') != 'system':
            self._chat_history.insert(0, {'role': 'system', 'content': _SYSTEM_PROMPT})
        self._chat_history[0] = {'role': 'system', 'content': _strip_dynamic_system_blocks(str(self._chat_history[0].get('content') or _SYSTEM_PROMPT)) + self._original_goal_block() + self._decision_constraints_block() + self._skill_index_block() + self._habits_block() + self._semantic_memory_block(query) + self._file_history_block(query) + (plan_mode_system_suffix() if self.is_plan_mode() else '')}

    def _append_user_turn(self, user_input: str, images: list[str] | None) -> None:
        """Append this turn's user message (multimodal when *images* given).

        Plan mode (read-only) prepends a steering note telling the model to
        research and end with a plan instead of promising changes it is not
        allowed to make (mutating tools are additionally blocked in the
        executor — see ``_execute_tool_call``).

        Also sets ``_turn_start_index``: everything appended after this line
        belongs to THIS turn — the boundary per-turn scans such as
        ``_mutating_files_this_turn`` use, so tool results restored from a
        previous session's chat_history.json are never rescanned.

        The RAW ``user_input`` (before any plan-mode note or skill hint is
        prepended) is offered to :meth:`_ensure_original_goal`, so the pinned
        goal is what the user actually asked for rather than our steering
        wrapper around it.
        """
        self._ensure_original_goal(user_input)
        if self.is_plan_mode() and (not images):
            if _agent_mod()._resolve_display_mode() != AgentDisplayMode.QUIET:
                print(yellow("  [plan mode] Read-only research — mutating tools blocked. Switch back with 'mode build' to apply changes."))
            user_input = f'{plan_mode_turn_note()}\n\n{user_input}'
        hint = '' if images else self._skill_hint_block(user_input)
        if hint:
            user_input = f'{hint}\n\n{user_input}'
        self._turn_start_index = len(self._chat_history)
        if images:
            content: Any = [{'type': 'text', 'text': user_input or '(see the attached image)'}]
            for img in images:
                content.append({'type': 'image_url', 'image_url': {'url': img}})
            self._chat_history.append({'role': 'user', 'content': content})
        else:
            self._chat_history.append({'role': 'user', 'content': user_input})

    async def _run_chained_tool_loop(self, user_input: str, messages: list[dict[str, Any]], display_mode: AgentDisplayMode, seen_calls: dict[tuple[str, str], int]) -> tuple[str, list[dict[str, Any]], str | None, ToolLoopRunner]:
        return await _run_chained_tool_loop(self, user_input, messages, display_mode, seen_calls)

    def _reset_failover_notice(self) -> None:
        _reset_failover_notice(self)

    def _report_failover(self) -> None:
        _report_failover(self)

    def _finish_turn(self, final_text: str, llm_error: str | None, loop: ToolLoopRunner, display_mode: AgentDisplayMode) -> None:
        _finish_turn(self, final_text, llm_error, loop, display_mode)

    async def chat_nlp(self, user_input: str, images: list[str] | None=None) -> None:
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
        1. ``_refresh_system_message`` — current ledger/plan-mode in prompt
           (semantic memories query-ranked for this user message).
        2. ``_append_user_turn`` — multimodal user message + turn boundary.
        3. ``_run_chained_tool_loop`` — tool loop + auto-continue chaining.
        4. ``_finish_turn`` — trim/persist history, print the outcome.
        """
        self._ensure_original_goal(user_input)
        self._refresh_system_message(user_input)
        self._read_streak = 0
        self._reset_failover_notice()
        self._last_user_input = user_input
        self._turn_started_at = datetime.now()
        try:
            if not getattr(self.llm, '_profile_pinned', False):
                _suggested = _agent_mod().recommend_profile(user_input, getattr(self.llm, '_profile_name', None))
                if _suggested:
                    _auto = False
                    try:
                        import pathlib as _pathlib
                        from agent_core.llm.workspace_prefs import get_pref
                        _auto = get_pref(_pathlib.Path(self._effective_ws_dir()), 'profile_auto') is True
                    except Exception:
                        _auto = False
                    _quiet = _agent_mod()._resolve_display_mode() == AgentDisplayMode.QUIET
                    if _auto:
                        try:
                            from agent_core.llm.model_profiles import get_profile
                            _prof = get_profile(_suggested)
                            self.llm._provider.apply_profile(_suggested, _prof.temperature, _prof.max_tokens)
                            self.llm._profile_name = _suggested
                        except Exception:
                            logger.debug('profile_auto apply failed (no-op)', exc_info=True)
                        else:
                            if not _quiet:
                                print(gray(f'  [profile] auto-applied {_suggested} (profile_auto pref)'))
                    elif not _quiet:
                        print(gray(f"  [suggest] profile {_suggested} for this task — 'model profile use' or set pref profile_auto"))
        except Exception:
            logger.debug('Profile suggestion failed (no-op)', exc_info=True)
        self._append_user_turn(user_input, images)
        self._chat_history = _trim_chat_history(self._chat_history, self._original_goal)
        history_snapshot = list(self._chat_history)
        final_text, final_messages, llm_error, loop = ('', self._chat_history, None, None)
        for attempt in range(1, _LLM_ERROR_MAX_RETRIES + 1):
            final_text, final_messages, llm_error, loop = await self._run_chained_tool_loop(user_input=user_input, messages=self._chat_history, display_mode=_agent_mod()._resolve_display_mode(), seen_calls={})
            if not llm_error or not is_connection_failure(llm_error):
                break
            if attempt < _LLM_ERROR_MAX_RETRIES:
                self._chat_history = list(history_snapshot)
                if _agent_mod()._resolve_display_mode() != AgentDisplayMode.QUIET:
                    print(yellow(f'\n  [retry {attempt}/{_LLM_ERROR_MAX_RETRIES}] LLM call failed transiently ({llm_error[:120]}); retrying...\n'))
                await asyncio.sleep(_LLM_ERROR_BACKOFF_S * 2 ** (attempt - 1))
        self._chat_history = final_messages
        self._report_failover()
        self._finish_turn(final_text=final_text, llm_error=llm_error, loop=loop, display_mode=_agent_mod()._resolve_display_mode())

    def _mutating_files_this_turn(self) -> list[str]:
        return _mutating_files_this_turn(self)

    def _print_self_review_note(self, mutated_files: list[str]) -> None:
        _print_self_review_note(self, mutated_files)

    def _persist_plan_answer(self, plan_text: str) -> None:
        _persist_plan_answer(self, plan_text)

    def check_stale_files(self) -> list[str]:
        return check_stale_files(self)

    def invalidate_stale(self) -> int:
        return invalidate_stale(self)

    def memory_stats(self) -> dict[str, Any]:
        return memory_stats(self)

    def clear_history(self) -> None:
        clear_history(self)

    def _decision_constraints_block(self) -> str:
        return _decision_constraints_block(self)

    def _skill_index_block(self) -> str:
        return _skill_index_block(self)

    def _skill_hint_block(self, user_input: str) -> str:
        return _skill_hint_block(self, user_input)

    def _habits_block(self) -> str:
        return _habits_block(self)

    def _file_history_block(self, query: str | None=None) -> str:
        return _file_history_block(self, query)

    def _semantic_memory_block(self, query: str | None=None) -> str:
        return _semantic_memory_block(self, query)

    def _ensure_original_goal(self, pending_user_input: str | None=None) -> None:
        _ensure_original_goal(self, pending_user_input)

    def _recent_user_texts(self, limit: int=10) -> list[str]:
        return _recent_user_texts(self, limit)

    def _prompts_related(self, new_prompt: str, prior_texts: list[str], threshold: float=_GOAL_RELATED_THRESHOLD) -> bool:
        return _prompts_related(self, new_prompt, prior_texts, threshold)

    def _suspend_history_for_new_task(self) -> None:
        _suspend_history_for_new_task(self)

    def _original_goal_block(self) -> str:
        return _original_goal_block(self)

    def _load_chat_history(self) -> list[dict[str, Any]]:
        return _load_chat_history(self)

    def _save_chat_history(self) -> None:
        _save_chat_history(self)

    def _load_memory(self) -> None:
        _load_memory(self)

    def _save_memory(self) -> None:
        _save_memory(self)

    def _record_llm_experience(self, action: str, outcome: float, context: dict[str, Any] | None=None, success: bool | None=None, decision_llm: str | None=None, decision_prompt: Any=None, run_id: str | None=None, latency_ms: float | None=None, token_usage: int | None=None) -> int | None:
        return _record_llm_experience(self, action, outcome, context, success, decision_llm, decision_prompt, run_id, latency_ms, token_usage)
_MAX_CHAINED_RUNS = 6
_LLM_ERROR_MAX_RETRIES = 2
_LLM_ERROR_BACKOFF_S = 5.0
_CONTINUE_NOTE_TAG = 'continue'
_MAX_RUN_TIMEOUT_S = 600
_MAX_READ_LINES = 500
_MAX_READ_FILES = 10
_CONTEXT_AST_THRESHOLD_KB = int(os.environ.get('USE_CONTEXT_AST_STRATEGY', '50'))
_MAX_CONSECUTIVE_READS = 6
_READ_LOOP_NOTE = '\n[read-loop guard] {n} consecutive reads this turn without acting. Stop paging files — each extra page inflates the prompt and slows every later call. Work with what you have: use definitions/references/search for targeted lookups, or give your answer / take the next action now.'
_MAX_ACTIVE_SUBAGENTS = 3
_DELEGATE_TIMEOUT_S = 600.0
_CONTINUE_NOTE_TAG_KEY = LOOP_NOTE_TAG_KEY
_CONTINUE_NOTE = 'The previous tool session ended before you finished. A fresh tool budget is available now. CONTINUE THE TASK where you left off — use the tools as needed, complete the remaining work, then give your final answer. Do not describe what remains; do it.'
_VERIFY_NOTE = "You answered without calling a single tool, so you have not verified anything about this repository — you cannot know whether changes are staged, committed or pushed. Do not assert that state. Call the git tool now (subcommand='status' first), then do the work and report what it actually returned."


# Public alias: the class was renamed AgentCore; agent.py exposes it as Agent.
Agent = AgentCore


# Full public+private surface re-exported by the agent.py shim (frozen by
# tests/test_agent_module_surface.py via star-import expansion).
__all__ = (
    'AGENT_MEMORY_JSON_PATH',
    'AGENT_MEMORY_TMP_PATH',
    'Agent',
    'AgentCore',
    'AgentDisplayMode',
    'AnalyzeCommand',
    'Any',
    'Awaitable',
    'CHAT_HISTORY_JSON_PATH',
    'CHAT_HISTORY_TMP_PATH',
    'CORRELATION_ID_CTX',
    'Callable',
    'CleanupCommand',
    'ClearCommand',
    'Command',
    'CommandRegistry',
    'CorrelationIdContext',
    'DEFAULT_FULL_SUITE_TIMEOUT',
    'DEFAULT_PAGE_LINES',
    'DESTRUCTIVE_SHELL_PATTERNS',
    'DecideCommand',
    'DemoDataCommand',
    'DisplayCommand',
    'EntitiesCommand',
    'FILE_HISTORY_MARKER',
    'FileSearcher',
    'FileSystem',
    'FixCommand',
    'FlowStopped',
    'FullRunGate',
    'GitCommand',
    'HABITS_MARKER',
    'HueBridge',
    'HueBridgeError',
    'ImplementCommand',
    'Iterator',
    'LLMClient',
    'LOOP_NOTE_TAG_KEY',
    'MAX_GOAL_CHARS',
    'MODE_BUILD',
    'ModelCommand',
    'MultiLlmCommand',
    'NLP_TOOL_NAMES',
    'NLP_TOOL_SCHEMAS',
    'NlpToolHandler',
    'NonUtf8TextError',
    'ORIGINAL_GOAL_MARKER',
    'OptimizeCommand',
    'Optional',
    'PasteCommand',
    'PasteImageCommand',
    'PerfCommand',
    'PerfTracker',
    'PlanCommand',
    'ReadCommand',
    'ReconstructCommand',
    'ReviewCommand',
    'RunCommand',
    'SEMANTIC_MEMORY_MARKER',
    'SKILL_INDEX_MARKER',
    'SearchCommand',
    'SelfHealCommand',
    'SkillError',
    'TURN_LOG_PATH',
    'TYPE_CHECKING',
    'TaskplanCommand',
    'ToolDispatcher',
    'ToolLoopRunner',
    'TraceWriter',
    'UnrepresentableTextError',
    'WorkflowCommand',
    'WriteCommand',
    '_CONTEXT_AST_THRESHOLD_KB',
    '_CONTINUE_NOTE',
    '_CONTINUE_NOTE_TAG',
    '_CONTINUE_NOTE_TAG_KEY',
    '_DELEGATE_TIMEOUT_S',
    '_GOAL_RELATED_THRESHOLD',
    '_HISTORY_CHAR_BUDGET',
    '_HISTORY_TRIM_NOTE',
    '_LLM_ERROR_BACKOFF_S',
    '_LLM_ERROR_MAX_RETRIES',
    '_MAX_ACTIVE_SUBAGENTS',
    '_MAX_CHAINED_RUNS',
    '_MAX_CHAT_MESSAGES',
    '_MAX_CONSECUTIVE_READS',
    '_MAX_READ_FILES',
    '_MAX_READ_LINES',
    '_MAX_RUN_TIMEOUT_S',
    '_READ_LOOP_NOTE',
    '_SYSTEM_PROMPT',
    '_VERIFY_NOTE',
    '_append_turn_log',
    '_begin_trace_turn',
    '_blocked_shell_command',
    '_content_tokens',
    '_current_git_branch',
    '_decision_constraints_block',
    '_detect_shell',
    '_effective_ws_dir',
    '_ensure_original_goal',
    '_escape_json_string_controls',
    '_file_history_block',
    '_final_answer_fallback',
    '_finish_turn',
    '_get_plan_state',
    '_git_branch',
    '_habits_block',
    '_install_signal_handlers',
    '_interruptible_input',
    '_is_compaction_note',
    '_is_full_pytest_command',
    '_is_pytest_command',
    '_is_subset_pytest_command',
    '_kill_process_tree',
    '_load_chat_history',
    '_load_memory',
    '_message_size',
    '_mutating_files_this_turn',
    '_nlp_analyze',
    '_nlp_create_subagent',
    '_nlp_definitions',
    '_nlp_delegate',
    '_nlp_delegate_batch',
    '_nlp_diff',
    '_nlp_edit',
    '_nlp_fix',
    '_nlp_get_current_datetime',
    '_nlp_git',
    '_nlp_hue_control',
    '_nlp_jev_decide',
    '_nlp_list_files',
    '_nlp_mcp_call',
    '_nlp_mcp_tools',
    '_nlp_merge',
    '_nlp_plan_finish',
    '_nlp_plan_start',
    '_nlp_plan_status',
    '_nlp_plan_step',
    '_nlp_read',
    '_nlp_read_many',
    '_nlp_read_skill',
    '_nlp_references',
    '_nlp_run',
    '_nlp_run_subagent_task',
    '_nlp_search',
    '_nlp_speculate',
    '_nlp_tests',
    '_nlp_tool_handlers',
    '_nlp_web_search',
    '_nlp_write',
    '_normalize_path',
    '_note_effect',
    '_original_goal_block',
    '_persist_plan_answer',
    '_print_self_review_note',
    '_project_chat_history',
    '_prompts_related',
    '_pytest_full_suite_timeout',
    '_read_env_value',
    '_read_json_quarantining',
    '_read_page',
    '_recent_user_texts',
    '_record_llm_experience',
    '_recover_original_goal',
    '_repair_tool_arguments',
    '_report_failover',
    '_reset_failover_notice',
    '_resolve_display_mode',
    '_resolve_nlp_path',
    '_resolve_stream_fn',
    '_run_chained_tool_loop',
    '_run_command_quietly',
    '_run_subprocess_captured',
    '_safe_path',
    '_save_chat_history',
    '_save_memory',
    '_save_verify_note',
    '_semantic_memory_block',
    '_shape_run_stderr',
    '_shell_command_hint',
    '_shell_name_token',
    '_signal_break_with_cleanup',
    '_skill_hint_block',
    '_skill_index_block',
    '_start_kanban_inbound',
    '_strip_dynamic_system_blocks',
    '_strip_image_blocks',
    '_strip_turn_wrappers',
    '_suppress_and_log',
    '_suspend_history_for_new_task',
    '_take_trace_effects',
    '_tool_analyze_file',
    '_tool_apply_patch',
    '_tool_delete_file',
    '_tool_edit_file',
    '_tool_list_files',
    '_tool_llm_analyze',
    '_tool_read_file',
    '_tool_search',
    '_tool_write_file',
    '_trim_chat_history',
    '_truncate_output',
    '_turn_quality_score',
    '_unix_command_hint',
    '_verify_file',
    '_windows_command_hint',
    'asyncio',
    'blue',
    'cast',
    'chat_stoppable',
    'check_stale_files',
    'check_tool_allowed',
    'clear_history',
    'clear_stop',
    'collect_definitions',
    'collect_references',
    'concurrent',
    'contextlib',
    'cyan',
    'datetime',
    'decisions_as_system_prompt',
    'defaultdict',
    'discover_skills',
    'encode_in_encoding',
    'execute_tool',
    'file_notes_block',
    'filter_tool_schemas',
    'find_destructive_shell_pattern',
    'format_skill_page',
    'get_last_metrics',
    'get_role',
    'gray',
    'green',
    'habits_block',
    'invalidate_stale',
    'io',
    'is_connection_failure',
    'is_plan_mode',
    'json',
    'load_agent_settings',
    'load_habits',
    'load_semantic_memory',
    'load_skill_index',
    'logger',
    'logging',
    'magenta',
    'match_skills_for_input',
    'memory_stats',
    'original_goal_block',
    'original_goal_from_block',
    'os',
    'plan_mode_system_suffix',
    'plan_mode_turn_note',
    'provider_supports_streaming',
    're',
    'read_skill',
    'read_text_preserving',
    'recommend_profile',
    'record_full_run_seconds',
    'record_turn_outcome',
    'red',
    'resolve_model',
    'resolve_path',
    'role_names',
    'safe_path',
    'save_file_py',
    'save_meta_policy_weights',
    'semantic_memory_block',
    'semantic_search_memories',
    'shlex',
    'signal',
    'skill_hint_block',
    'subprocess',
    'sys',
    'time',
    'to_windows_path',
    'trace_enabled',
    'traceback',
    'yellow',
)
