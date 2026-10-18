"""Agent implementation with workspace management and tool execution.."""
from agent_core.bootstrap import *
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_core.monitoring import MetricsCollector
    from agent_core.subagent import SubAgent

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
from agent_core.shell_utils import detect_shell as _detect_shell, shell_name_token as _shell_name_token, current_git_branch as _git_branch, unix_command_hint as _unix_command_hint, windows_command_hint as _windows_command_hint, shell_command_hint as _shell_command_hint, truncate_output as _truncate_output, run_subprocess_captured as _run_subprocess_captured, shape_run_stderr as _shape_run_stderr, blocked_shell_command as _blocked_shell_command, read_env_value as _read_env_value, pytest_full_suite_timeout as _pytest_full_suite_timeout, is_full_pytest_command as _is_full_pytest_command, is_pytest_command as _is_pytest_command, is_subset_pytest_command as _is_subset_pytest_command, SYSTEM_PROMPT as _SYSTEM_PROMPT
from agent_core.json_utils import _read_json_quarantining
from agent_core.history_compaction import _recover_original_goal, _strip_image_blocks, _trim_chat_history, _project_chat_history, _HISTORY_CHAR_BUDGET, _MAX_CHAT_MESSAGES, _HISTORY_TRIM_NOTE, _is_compaction_note, _strip_turn_wrappers, _content_tokens
from agent_core.llm.tool_loop import _message_size
from agent_core.run_utils import _start_kanban_inbound, kill_process_tree as _kill_process_tree, unverified_action_claim as _unverified_action_claim, resolve_stream_fn as _resolve_stream_fn, resolve_display_mode as _resolve_display_mode, final_answer_fallback as _final_answer_fallback, turn_quality_score as _turn_quality_score, begin_trace_turn as _begin_trace_turn, append_turn_log as _append_turn_log, strip_dynamic_system_blocks as _strip_dynamic_system_blocks
NlpToolHandler = Callable[..., Awaitable[str]]
from agent_core.types import _GOAL_RELATED_THRESHOLD
from agent_core.llm.client import LLMClient
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


class Agent:
    """Main agent class with workspace management and tool execution."""
    DEFAULT_WORKSPACE = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, workspace: str | None=None, model_name: str | None=None):
        self.workspace = os.path.abspath(to_windows_path(workspace or self.DEFAULT_WORKSPACE))
        self.model_name = resolve_model(model_name)
        self._semantic_index: dict[str, set[int]] = defaultdict(set)
        self._files_read: set[str] = set()
        self._file_mtimes: dict[str, float] = {}
        self._full_run_gate = FullRunGate(self.workspace)
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
        self._read_streak: int = 0
        self._delegating: bool = False
        self._active_subagents: set[str] = set()
        self._delegate_counter: int = 0
        self._nlp_workspace: str | None = None
        self.mode: str = MODE_BUILD
        self._load_memory()
        self._ensure_original_goal()
        self.llm = LLMClient(model_name=self.model_name)
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
        self.fs = FileSystem(self.workspace)
        self.searcher = FileSearcher(self.workspace)
        self.dispatcher = ToolDispatcher(on_tool=_emit_tool_metrics)
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
            return f'Skipped {path} (no changes)'
        verify = await self._verify_file(path) if path.endswith('.py') else ''
        self._note_effect(path)
        return f'{ok_msg}\n{verify}'.strip()

    async def _verify_file(self, path: str) -> str:
        """Run py_compile on *path* and return a short verification summary."""
        try:
            cwd = os.path.abspath(to_windows_path(self._nlp_workspace or self.workspace))
            if not os.path.isdir(cwd):
                cwd = self.workspace
            r = subprocess.run([sys.executable, '-m', 'py_compile', path], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=cwd)
        except OSError as e:
            return f'[verify] py_compile could not run: {e}'
        if r.returncode == 0:
            return '[verify] py_compile ✓'
        return f'[verify] py_compile ✗: {r.stderr.strip()[:300]}'

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
        return _effective_ws_dir(self)

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
            if _resolve_display_mode() != AgentDisplayMode.QUIET:
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

        async def llm_chat_fn(msgs: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
            """Call the LLM with tools; parse a JSON tool_calls message back into
            the message list so ToolLoopRunner can execute them."""
            raw = await self.llm.chat(msgs, tools=tools, disable_thinking=True)
            if raw.strip() == '(no output)':
                raw = ''
            if raw.startswith('[Error') or raw.startswith('[LM Studio'):
                if not llm_error:
                    llm_error.append(raw)
                return (raw, msgs)
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get('tool_calls'):
                parsed.pop('role', None)
                for tc in parsed['tool_calls']:
                    func = tc.get('function') or {}
                    args_str = func.get('arguments', '{}')
                    try:
                        json.loads(args_str)
                    except (json.JSONDecodeError, TypeError):
                        func['arguments'] = json.dumps(_repair_tool_arguments(args_str), ensure_ascii=False)
                updated = list(msgs)
                updated.append({'role': 'assistant', 'content': parsed.get('content') or '', **parsed})
                text = str(parsed.get('content') or '')
                return (text, updated)
            updated = list(msgs)
            updated.append({'role': 'assistant', 'content': raw})
            return (raw, updated)
        final_text = ''
        final_messages = messages
        continuations = 0
        last_answer = ''
        with CorrelationIdContext() as turn_cid:
            self._turn_run_id = turn_cid
            _begin_trace_turn()
            while True:
                trace_writer = TraceWriter(meta={'model': self.model_name, 'profile': getattr(self.llm, '_profile_name', None) or ''}) if trace_enabled() and TraceWriter is not None else None
                if trace_writer is not None:
                    trace_writer.emit_task_begin(user_input)
                self._active_trace_writer = trace_writer
                self._pending_effects = [] if trace_writer is not None else None
                loop = ToolLoopRunner(max_iterations=150, display_mode=display_mode, trace=trace_writer)
                final_text, final_messages = await loop.run(messages=final_messages, llm_chat_fn=llm_chat_fn, execute_tool_fn=self._execute_tool_call, tools=filter_tool_schemas(NLP_TOOL_SCHEMAS, self.mode), seen_calls=seen_calls, effects_fn=self._take_trace_effects, stream_fn=_resolve_stream_fn(self.llm))
                self._pending_effects = None
                self._active_trace_writer = None
                if trace_writer is not None:
                    trace_writer.close()
                reason = loop.termination_reason
                _run_outcome = 1.0 if not llm_error else 0.0
                _run_success = bool(not llm_error and reason in ('answer', 'cap'))
                _metrics = get_last_metrics(getattr(self.llm, '_provider', None))
                self._record_llm_experience(action='llm_decision', outcome=_run_outcome, context={'model': self.model_name, 'profile': getattr(self.llm, '_profile_name', None) or '', 'verdict': reason, 'continuations': continuations, 'final_text_len': len(final_text)}, success=_run_success, decision_prompt=final_messages, run_id=self._turn_run_id or CORRELATION_ID_CTX.get() or None, latency_ms=getattr(_metrics, 'latency_ms', None), token_usage=getattr(_metrics, 'total_tokens', None))
                if llm_error:
                    break
                unverified = reason in ('answer', 'no_progress') and _unverified_action_claim(user_input, getattr(loop, 'tool_calls_made', 0) or 0)
                needs_more = reason == 'cap' or unverified or (reason in ('answer', 'no_progress') and _looks_incomplete(final_text))
                if needs_more and continuations < _MAX_CHAINED_RUNS:
                    if final_text and final_text == last_answer:
                        if display_mode != AgentDisplayMode.QUIET:
                            print(magenta('\n  [stopped] The model repeated the same answer ') + yellow('twice — ending the turn. ') + gray('Rephrase or ask something more specific to continue.\n'))
                        break
                    last_answer = final_text
                    continuations += 1
                    note = _VERIFY_NOTE if unverified else _CONTINUE_NOTE
                    if display_mode != AgentDisplayMode.QUIET:
                        why = {'cap': 'iteration budget exhausted', 'answer': 'answer signals unfinished work', 'no_progress': 'the final answer signals unfinished work'}.get(reason, reason)
                        if unverified:
                            why = 'answer claimed repo state without calling git'
                        print(magenta(f'\n  [auto-continue] Run {continuations}: ') + yellow(f'{why} — continuing automatically.\n'))
                    final_messages = list(final_messages) + [{'role': 'user', 'content': note, _CONTINUE_NOTE_TAG_KEY: _CONTINUE_NOTE_TAG}]
                    continue
                if reason in ('stuck', 'no_progress') and display_mode != AgentDisplayMode.QUIET:
                    print(magenta('\n  [stopped] The model stopped making progress ') + yellow('stuck on repeated calls' if reason == 'stuck' else 'too many calls without making new progress') + gray('The answer above is the best it produced — rephrase or ask something more specific to continue.\n'))
                break
        final_messages = [m for m in final_messages if m.get(_CONTINUE_NOTE_TAG_KEY) != _CONTINUE_NOTE_TAG]
        return (final_text, final_messages, llm_error[0] if llm_error else None, loop)

    def _reset_failover_notice(self) -> None:
        """Drop any pending failover notice on the wrapped provider.

        Called once at turn start so a failover that happened in an EARLIER
        turn is never reported again (the record would otherwise be stale
        forever).  No-op when the provider is not a failover chain.
        """
        reset = getattr(getattr(self.llm, '_provider', None), 'reset_failover_notice', None)
        if callable(reset):
            try:
                reset()
            except Exception:
                logger.debug('failover notice reset failed (no-op)', exc_info=True)

    def _report_failover(self) -> None:
        """Print which LLM answered when a failover happened this turn.

        Failover is currently invisible to the user: the switch is only a
        ``logger.warning`` in :class:`FailoverProvider`, so the REPL silently
        answers from a different model than the one the user selected — which
        looks like a hallucination when the answer style changes.  This prints
        one line naming the model that answered and what was skipped.

        Honours the display-mode contract: suppressed in QUIET, which promises
        only the final answer.  Never raises — reporting must not kill a turn.
        """
        try:
            notice = getattr(getattr(self.llm, '_provider', None), 'last_failover', None)
            if not notice:
                return
            if _resolve_display_mode() == AgentDisplayMode.QUIET:
                return
            answered = notice.get('answered_model') or notice.get('answered_provider') or 'next provider'
            attempted = notice.get('attempted') or []
            skipped = ', '.join((str(a.get('model') or a.get('provider') or '?') for a in attempted))
            print(magenta(f'\n  [failover] answered by {answered}' + (f' after {skipped} unreachable' if skipped else '')))
            for entry in attempted:
                err = str(entry.get('error') or '').strip()
                if err:
                    print(gray(f'    skipped {err[:160]}'))
        except Exception:
            logger.debug('failover report failed (no-op)', exc_info=True)

    def _finish_turn(self, final_text: str, llm_error: str | None, loop: ToolLoopRunner, display_mode: AgentDisplayMode) -> None:
        """Bound + persist the conversation and print the turn outcome.

        The final answer is ALWAYS printed — in every display mode, including
        QUIET (which only hides intermediate tool output, per the display-mode
        contract: "only the final answer is printed").
        """
        self._chat_history = _trim_chat_history(self._chat_history, self._original_goal)
        mutated_files = self._mutating_files_this_turn()
        try:
            started = getattr(self, '_turn_started_at', None)
            duration = round((datetime.now() - started).total_seconds(), 3) if started else None
            tail = str(getattr(self, '_last_user_input', '') or '')[-200:]
            record = {'ts': datetime.now().isoformat(timespec='seconds'), 'user_input_tail': tail, 'mutated_files': list(mutated_files or []), 'llm_error': bool(llm_error), 'duration': duration}
            _append_turn_log(record)
            self._record_llm_experience(action='chat_turn', outcome=0.0 if llm_error else 1.0 if mutated_files else 0.7, context={'plan_mode': bool(self.is_plan_mode()), 'files': list(mutated_files or [])}, decision_prompt=str(getattr(self, '_last_user_input', '') or '') or None, run_id=self._turn_run_id or CORRELATION_ID_CTX.get() or None)
            _quality = _turn_quality_score(llm_error, mutated_files)
            record_turn_outcome(str(getattr(self, '_last_user_input', '') or ''), profile_name=getattr(self.llm, '_profile_name', None), success=not llm_error, latency_seconds=duration, provider=getattr(self.llm, '_provider', None), mutated_files=list(mutated_files or []), quality=_quality)
            if _quality is not None:
                from agent_core.monitoring.metrics_file import append_event
                append_event('turn', 'quality', float(_quality))
        except Exception:
            logger.debug('Turn-outcome hooks failed (no-op)', exc_info=True)
        self._save_chat_history()
        self._save_memory()
        clean = re.sub('</?tool_call>', '', final_text)
        clean = re.sub('</?function_call>', '', clean)
        if llm_error:
            err = llm_error.strip()
            print(yellow('\n  [llm-error] The model did not produce a usable response:'))
            print(red(f'  {err[:400]}'))
            if 'reasoning' in err.lower():
                print(yellow('  The reasoning model exhausted its thinking budget. ') + gray('Switch to a non-reasoning model (model <name>) or retry the request.\n'))
            self._nlp_workspace = None
            return
        if display_mode == AgentDisplayMode.CLEAN:
            _NARRATION_PREFIX = re.compile("^\\s*(i\\s+(will|'ll|going\\sto|am\\ngoing)|let\\s+me)\\b", re.I)
            if _NARRATION_PREFIX.match(clean):
                clean = 'Plan: ' + clean.strip()
        if clean.strip():
            if not getattr(loop, 'final_answer_streamed', False):
                print(green(clean))
        else:
            print(yellow(_final_answer_fallback(loop)))
        if self.is_plan_mode() and clean.strip():
            self._persist_plan_answer(clean)
        if mutated_files and clean.strip():
            self._print_self_review_note(mutated_files)
        self._nlp_workspace = None

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
                _suggested = recommend_profile(user_input, getattr(self.llm, '_profile_name', None))
                if _suggested:
                    _auto = False
                    try:
                        import pathlib as _pathlib
                        from agent_core.llm.workspace_prefs import get_pref
                        _auto = get_pref(_pathlib.Path(self._effective_ws_dir()), 'profile_auto') is True
                    except Exception:
                        _auto = False
                    _quiet = _resolve_display_mode() == AgentDisplayMode.QUIET
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
            final_text, final_messages, llm_error, loop = await self._run_chained_tool_loop(user_input=user_input, messages=self._chat_history, display_mode=_resolve_display_mode(), seen_calls={})
            if not llm_error or not is_connection_failure(llm_error):
                break
            if attempt < _LLM_ERROR_MAX_RETRIES:
                self._chat_history = list(history_snapshot)
                if _resolve_display_mode() != AgentDisplayMode.QUIET:
                    print(yellow(f'\n  [retry {attempt}/{_LLM_ERROR_MAX_RETRIES}] LLM call failed transiently ({llm_error[:120]}); retrying...\n'))
                await asyncio.sleep(_LLM_ERROR_BACKOFF_S * 2 ** (attempt - 1))
        self._chat_history = final_messages
        self._report_failover()
        self._finish_turn(final_text=final_text, llm_error=llm_error, loop=loop, display_mode=_resolve_display_mode())

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
        marker = '[verify] py_compile'
        files: list[str] = []
        history = self._chat_history
        start = max(0, min(self._turn_start_index, len(history)))
        for idx in range(start, len(history)):
            m = history[idx]
            if m.get('role') != 'tool':
                continue
            content = str(m.get('content') or '')
            if marker not in content:
                continue
            first_line = content.splitlines()[0] if content else ''
            for prefix in ('Written ', 'Edited ', 'Skipped '):
                if first_line.startswith(prefix):
                    path = first_line[len(prefix):]
                    if path.endswith(')'):
                        path = path.rsplit('(', 1)[0].strip()
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
                return os.path.relpath(f, self.workspace).replace('\\', '/')
            except ValueError:
                return f
        shown = ', '.join((_short(f) for f in mutated_files[:6]))
        more = f' (+{len(mutated_files) - 6} more)' if len(mutated_files) > 6 else ''
        print(magenta(f"\n  [self-review] This turn changed {len(mutated_files)} file(s): {shown}{more}. py_compile verified syntax only — run 'tests' or 'git diff' to confirm behaviour."))

    def _persist_plan_answer(self, plan_text: str) -> None:
        """Save a plan-mode final answer to ``.docs/<ts>/plan_proposed.md``.

        Plan-mode answers used to evaporate as terminal text; persisting them
        gives the user a durable artifact to hand to ``implement``/``fix``
        later (plan item D-#14).  Best-effort: a write failure must never
        lose the already-printed answer or crash the turn.
        """
        with _suppress_and_log('Could not persist plan-mode answer:\n'):
            from agent_core.commands.doc_paths import new_run_dir
            out = new_run_dir(self.workspace) / 'plan_proposed.md'
            out.write_text(f'# Proposed plan\n\n_Generated in plan mode on {datetime.now():%Y-%m-%d %H:%M}. Review before applying — switch to build mode to implement._\n\n{plan_text}\n', encoding='utf-8')
            print(yellow(f'\n  [plan] Saved to {out}'))

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
                logger.warning('Could not stat %s: %s', path, e)
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
        return {'chat_history': len(self._chat_history), 'files_read': len(self._files_read), 'stale_files': len(stale), 'history': len(self._history), 'working_memory': len(self._working_memory), 'semantic_index': len(self._semantic_index), 'knowledge_graph': len(self._knowledge_graph)}

    def clear_history(self) -> None:
        """Clear all agent state."""
        self._history = []
        self._chat_history.clear()
        if hasattr(self, '_conversation'):
            self._conversation.clear()
        self._turn_start_index = 0
        self._turn_run_id = None
        self._read_streak = 0
        self._delegating = False
        self._active_subagents.clear()
        self._delegate_counter = 0
        self._pending_effects = None
        self._original_goal = ''
        self._goal_from_restore = False
        self._prior_chat_history = []
        self._files_read.clear()
        self._file_mtimes.clear()
        self._knowledge_graph.clear()
        self._working_memory.clear()
        self._semantic_index.clear()
        try:
            os.remove(CHAT_HISTORY_JSON_PATH)
            os.remove(AGENT_MEMORY_JSON_PATH)
        except OSError as e:
            logger.warning('Could not remove persisted state files: %s', e)
        try:
            self._full_run_gate.reset()
        except Exception:
            logger.debug('full-run gate reset failed', exc_info=True)

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
                return ''
            return decisions_as_system_prompt(self.workspace, sorted(self._files_read))
        except Exception:
            logger.exception('Decision constraints unavailable:\n')
            return ''

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
            return ''

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
            matched = match_skills_for_input(skills, user_input or '')
            if not matched:
                return ''
            try:
                collector = self.get_metrics_collector()
                collector.increment_counter('skill.hint.matched')
                from agent_core.monitoring.metrics_file import append_event as _ae
                _ae('counter', 'skill.hint.matched', 1.0)
            except Exception:
                pass
            if _resolve_display_mode() != AgentDisplayMode.QUIET:
                names = ', '.join((s.name for s in matched))
                print(yellow(f'  [skill-hints] matched: {names}'))
            return skill_hint_block(matched)
        except Exception:
            logger.exception('Skill hint unavailable:\n')
            return ''

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
            return ''

    def _file_history_block(self, query: str | None=None) -> str:
        """Per-file execution-history block for the chat system prompt.

        The workspace already keeps a trace corpus and an execution ledger
        (``reports/traces``, ``reports/history/executions.jsonl``); only
        implement/fix ever read them.  This method makes the same history
        visible to the conversational loop: the files *query* mentions are
        looked up with :func:`harnessfix.history.file_history` (the real
        matcher — a directory arg counts for its direct children) and the
        most important past events render as one compact block.

        Rebuilt every turn from the current workspace, same contract as
        :meth:`_habits_block`: empty string when nothing is relevant, so the
        prompt stays byte-identical to before; never raises — a broken
        ledger must not kill a chat turn.
        """
        try:
            return file_notes_block(query, self._effective_ws_dir())
        except Exception:
            logger.exception('File-history block unavailable:\n')
            return ''

    def _semantic_memory_block(self, query: str | None=None) -> str:
        """Embedding/KG memory block for the chat system prompt.

        Builds a compact block from indexed memories (see
        :func:`agent_core.memory.semantic_memory_block`).  It is
        rebuilt every turn from the workspace ledger so a memory indexed or
        updated mid-session is advertised on the next turn.  Empty string when
        the workspace has no indexed memories — the prompt then stays
        byte-identical to before.  Never raises: a broken ``.semantic_memory``
        must not kill a chat turn (same rule as
        :meth:`_decision_constraints_block`).

        When *query* (the current user message) is known, the ledger entries
        are ranked with :func:`semantic_search_memories`
        (:meth:`MemoryStore.semantic_search`) and the top-5 most relevant ones
        are injected, prefixed with a ``QUERY:`` line.  When embeddings are
        unavailable the ranking returns ``[]`` and the block falls back to the
        unranked ledger order WITHOUT the query line (byte-identical to the
        old behaviour).
        """
        try:
            indexed_memories = load_semantic_memory(self._effective_ws_dir())
            if query and indexed_memories:
                try:
                    ranked = semantic_search_memories(indexed_memories, query, k=5, embedding_service=getattr(self, '_semantic_embedding_service', None))
                except Exception:
                    ranked = []
                if ranked:
                    return semantic_memory_block(ranked, query=query, k=5)
            return semantic_memory_block(indexed_memories, k=5)
        except Exception:
            logger.exception('Semantic memory block unavailable:\n')
            return ''

    def _ensure_original_goal(self, pending_user_input: str | None=None) -> None:
        """Record the session's original request once, and only once.

        The goal is whatever the user asked FIRST: the first user message
        already in a restored history, or the input of the first turn of a
        fresh session (``pending_user_input`` — captured before the plan-mode
        note and skill hints are prepended to it).  Later prompts never replace
        it: the point is to pin what the session was *originally* about, which
        is exactly what chat-history compaction drops first.

        Cross-session exception: when the goal was RESTORED (agent_memory.json
        or the previous session's history, tracked by ``_goal_from_restore``)
        and the first live prompt of this session is topically unrelated to it
        (:meth:`_prompts_related` over the goal plus the recent user texts),
        the restored goal is REPLACED by the live prompt and the old
        conversation is suspended (:meth:`_suspend_history_for_new_task`).  A
        stale goal block otherwise steers every answer of the new session away
        from what the user is actually asking now.

        Capped at ``MAX_GOAL_CHARS`` so a first prompt that is itself a file
        dump cannot bloat the system prompt or ``agent_memory.json``.
        """
        if self._original_goal:
            if self._goal_from_restore and pending_user_input:
                self._goal_from_restore = False
                if not self._prompts_related(pending_user_input, [self._original_goal, *self._recent_user_texts()]):
                    self._original_goal = pending_user_input[:MAX_GOAL_CHARS]
                    self._suspend_history_for_new_task()
            return
        recovered = _recover_original_goal(self._chat_history)
        if recovered is not None:
            source = recovered
            if recovered:
                self._goal_from_restore = True
        elif pending_user_input:
            source = pending_user_input
        else:
            source = ''
        if source:
            self._original_goal = source[:MAX_GOAL_CHARS]

    def _recent_user_texts(self, limit: int=10) -> list[str]:
        """The newest *limit* raw user texts from the current chat history.

        Steering wrappers (plan-mode note, skill hints) are stripped and
        compaction notes skipped, so the texts are comparable with a live
        prompt for the relevance check in :meth:`_ensure_original_goal`.
        """
        texts: list[str] = []
        for m in reversed(self._chat_history):
            if m.get('role') != 'user':
                continue
            content = m.get('content')
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text = ' '.join((str(b.get('text') or '') for b in content if isinstance(b, dict) and b.get('type') == 'text')).strip()
            else:
                text = ''
            if text and (not _is_compaction_note(text)):
                texts.append(_strip_turn_wrappers(text))
            if len(texts) >= limit:
                break
        texts.reverse()
        return texts

    def _prompts_related(self, new_prompt: str, prior_texts: list[str], threshold: float=_GOAL_RELATED_THRESHOLD) -> bool:
        """Whether *new_prompt* looks topically related to any prior text.

        Token-overlap heuristic over content words (:func:`_content_tokens`):
        a short prompt (<2 content tokens, e.g. "continue") is conservatively
        treated as related so a bare nudge can never evict the goal.
        """
        new_tokens = _content_tokens(new_prompt)
        if len(new_tokens) < 2:
            return True
        prior: set[str] = set()
        for text in prior_texts:
            prior |= _content_tokens(text or '')
        if not prior:
            return False
        overlap = new_tokens & prior
        return len(overlap) / len(new_tokens) >= threshold

    def _suspend_history_for_new_task(self) -> None:
        """Archive a restored conversation whose goal just got replaced.

        The previous session's history belonged to a different task; keeping
        it inline would let old context steer (and contaminate) the new task.
        The body is archived in ``_prior_chat_history`` and — best-effort — on
        disk as ``chat_history_prior.json`` (path derived from the CURRENT
        ``CHAT_HISTORY_JSON_PATH`` so test monkeypatches are honoured), so
        nothing is lost.  Only the system message stays inline (rebuilt fresh
        by ``_refresh_system_message`` on the next turn).  Task-agnostic
        memory stores (files_read, semantic index, knowledge graph) are kept.
        """
        body = [dict(m) for m in self._chat_history if m.get('role') != 'system']
        if not body:
            return
        system = [dict(m) for m in self._chat_history if m.get('role') == 'system'][:1]
        self._prior_chat_history = body
        self._chat_history = system
        self._turn_start_index = len(self._chat_history)
        prior_path = os.path.join(os.path.dirname(CHAT_HISTORY_JSON_PATH) or '.', 'chat_history_prior.json')
        try:
            with open(prior_path, 'w', encoding='utf-8') as f:
                json.dump(body, f, ensure_ascii=False, indent=2)
        except OSError:
            logger.debug('Could not archive prior chat history to %s', prior_path, exc_info=True)

    def _original_goal_block(self) -> str:
        """The pinned ORIGINAL TASK block for the chat system prompt.

        Rebuilt every turn from ``_original_goal`` (same contract as
        :meth:`_decision_constraints_block`); empty string when no goal is
        recorded, so the prompt stays byte-identical to before in that case.
        Never raises: a missing goal must not kill a chat turn.
        """
        try:
            return original_goal_block(self._original_goal)
        except Exception:
            logger.exception('Original goal block unavailable:\n')
            return ''

    def _load_chat_history(self) -> list[dict[str, Any]]:
        """Load the persisted NLP conversation from the previous session.

        The persisted history holds a bounded multi-exchange window (see
        ``_project_chat_history``), so a fresh session continues the dialogue
        instead of forgetting it.  A corrupt file is quarantined (see
        :func:`_read_json_quarantining`) instead of being silently dropped.
        """
        data = _read_json_quarantining(CHAT_HISTORY_JSON_PATH, 'chat history')
        if not isinstance(data, list):
            return []
        messages = [m for m in data if isinstance(m, dict) and m.get('role') in ('system', 'user', 'assistant', 'tool') and (_CONTINUE_NOTE_TAG_KEY not in m)]
        for m in messages:
            for tc in m.get('tool_calls', []):
                func = tc.get('function') or {}
                args_str = func.get('arguments', '{}')
                try:
                    parsed = json.loads(args_str)
                    if not isinstance(parsed, dict):
                        func['arguments'] = '{}'
                except (json.JSONDecodeError, TypeError):
                    func['arguments'] = '{}'
        return _project_chat_history(_strip_image_blocks(messages), getattr(self, '_original_goal', ''))

    def _save_chat_history(self) -> None:
        """Persist the NLP conversation so the next session can continue it.

        Image blocks are stripped first (see :func:`_strip_image_blocks`) so a
        vision turn does not bloat the on-disk history with base64 blobs.

        The write is atomic: content goes to ``chat_history.json.tmp`` first
        and is then ``os.replace()``d into place, so a crash mid-write can
        never leave a half-written JSON that would be silently dropped on the
        next load.
        """
        payload = json.dumps(_project_chat_history(_strip_image_blocks(self._chat_history), self._original_goal), ensure_ascii=False, indent=2)
        tmp_path = CHAT_HISTORY_JSON_PATH + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(payload)
            os.replace(tmp_path, CHAT_HISTORY_JSON_PATH)
        except OSError as e:
            logger.warning('Failed to save chat history: %s', e)

    def _load_memory(self) -> None:
        """Load cross-session memory from the previous session.

        Restores files read, the semantic index, the knowledge graph, and
        working memory so a fresh session resumes with accumulated context
        instead of starting from zero.  A corrupt file is quarantined as
        ``agent_memory.json.bad-<timestamp>`` before falling back to the
        empty defaults, so the bytes stay inspectable instead of being
        clobbered by the next save.
        """
        data = _read_json_quarantining(AGENT_MEMORY_JSON_PATH, 'agent memory')
        if not isinstance(data, dict):
            return
        files = data.get('files_read')
        if isinstance(files, list):
            self._files_read = {str(p) for p in files}
        index = data.get('semantic_index')
        if isinstance(index, dict):
            self._semantic_index = defaultdict(set, {k: set(v) for k, v in index.items() if isinstance(v, list)})
        kg = data.get('knowledge_graph')
        if isinstance(kg, dict):
            self._knowledge_graph = kg
        wm = data.get('working_memory')
        if isinstance(wm, list):
            self._working_memory = wm
        hist = data.get('history')
        if isinstance(hist, list):
            self._history = hist
        goal = data.get('original_goal')
        if isinstance(goal, str):
            self._original_goal = goal
            if goal:
                self._goal_from_restore = True

    def _save_memory(self) -> None:
        """Persist cross-session memory so the next session resumes with it.

        Atomic like :meth:`_save_chat_history` (tmp file + ``os.replace``).
        """
        data = {'files_read': sorted(self._files_read), 'semantic_index': {k: sorted(v) for k, v in self._semantic_index.items()}, 'knowledge_graph': self._knowledge_graph, 'working_memory': self._working_memory, 'history': self._history, 'original_goal': self._original_goal}
        tmp_path = AGENT_MEMORY_JSON_PATH + '.tmp'
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, AGENT_MEMORY_JSON_PATH)
        except OSError:
            logger.warning('Failed to save agent memory:\n%s', traceback.format_exc())

    def _record_llm_experience(self, action: str, outcome: float, context: dict[str, Any] | None=None, success: bool | None=None, decision_llm: str | None=None, decision_prompt: Any=None, run_id: str | None=None, latency_ms: float | None=None, token_usage: int | None=None) -> int | None:
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
            ts = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            import sqlite3
            from agent_core.memory.attribution import ensure_attribution_schema, prompt_sha256, record_llm_decision, resolve_decision_llm
            if decision_llm is None:
                decision_llm = resolve_decision_llm(self)
            prompt_hash = prompt_sha256(decision_prompt) if decision_prompt is not None else None
            db_path = AGENT_MEMORY_JSON_PATH.replace('.json', '.db')
            conn = sqlite3.connect(db_path, timeout=5.0)
            try:
                ensure_attribution_schema(conn)
                if decision_llm is None:
                    cur = conn.execute('INSERT INTO experiences (timestamp, action, outcome, context, success) VALUES (?, ?, ?, ?, ?)', (ts, action, outcome, ctx_str, 1 if success else 0))
                    conn.commit()
                    return int(cur.lastrowid)
                cur = conn.execute('INSERT INTO experiences (timestamp, action, outcome, context, success, decision_llm, decision_prompt_hash, decision_timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (ts, action, outcome, ctx_str, 1 if success else 0, decision_llm, prompt_hash, ts))
                rowid = int(cur.lastrowid)
                record_llm_decision(conn, run_id=run_id, model_name=decision_llm, prompt_sha256=prompt_hash, outcome_experience_id=rowid, latency_ms=latency_ms, token_usage=token_usage)
                conn.commit()
                return rowid
            finally:
                conn.close()
        except Exception as exc:
            logger.debug('Could not record LLM experience (no-op): %s', exc)
            return None
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
_INCOMPLETE_MARKERS = ('budget', 'tool budget', 'not finished', 'could not', "couldn't", 'would need', 'needs one more', 'needs 1 more', 'needs 2 more', 'remaining', 'unfinished', 'incomplete', 'exhausted', 'ran out', 'out of tool', 'more tool call', 'cannot complete', 'still needs', 'not covered', 'follow-up')
_COMPLETE_MARKERS = ('all done', 'is done', 'done.', 'completed', 'finished', 'is complete', 'fully resolved', 'all fixed', 'task is complete')

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
_ACTION_VERB_RE = re.compile('\\b(commit|push|stage)\\b', re.I)
_QUESTION_LEAD_RE = re.compile('^\\s*(what|why|how|when|where|who|which|is|are|was|were|does|did|do|can|could|should|would|will|has|have|am|explain|tell|show)\\b', re.I)

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
    text = (user_input or '').strip()
    if not text or tool_calls_made > 0:
        return False
    if not _ACTION_VERB_RE.search(text):
        return False
    if '?' in text or _QUESTION_LEAD_RE.match(text):
        return False
    return True
_VERIFY_NOTE = "You answered without calling a single tool, so you have not verified anything about this repository — you cannot know whether changes are staged, committed or pushed. Do not assert that state. Call the git tool now (subcommand='status' first), then do the work and report what it actually returned."
from agent_dashboard import _build_dashboard, _dashboard_port, _default_alert_rules, _emit_command_metrics, _emit_tool_metrics, _shared_metrics_collector, get_metrics_collector, record_command_metrics, run_dashboard_server, start_dashboard_thread

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

def _warn_uncommitted(agent: 'Agent') -> None:
    """Print a reminder when the session leaves uncommitted changes.

    AGENTS.md invariant #4: "Commit after every session. Uncommitted work is
    unrecoverable" — the deleted-agent.py incident (decision #058) lost
    written files because they were never committed.  This runs on every
    REPL shutdown path (quit / stdin end / EOF) as a last-line reminder.
    Best-effort and silent outside a git repo; never blocks exit.
    """
    try:
        r = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=agent.workspace, timeout=10)
        if r.returncode != 0:
            return
        lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
        if not lines:
            return
        print(yellow(f'\n  [git] {len(lines)} uncommitted change(s) in the workspace — commit them now (AGENTS.md invariant #4: uncommitted work is unrecoverable). Top offenders:'))
        for ln in lines[:5]:
            print(f'    {ln}')
        if len(lines) > 5:
            print(gray(f'    ... and {len(lines) - 5} more'))
    except Exception as e:
        logger.debug('Uncommitted-changes check skipped: %s', e)

def _build_chat_prompt(agent: 'Agent', branch: str, now: datetime) -> str:
    """Build the interactive chat prompt, colored by session mode.

    The prompt reads ``[build]`` in green (full mutating toolset) or
    ``[plan]`` in blue (read-only research mode), so the active mode is
    unmistakable at a glance.  Falls back to green/build styling when the
    agent has no ``is_plan_mode`` (defensive — the prompt must always render).
    """
    plan = getattr(agent, 'is_plan_mode', None)
    plan = bool(plan() if callable(plan) else False)
    mode_tag = '[plan] ' if plan else '[build] '
    prompt_color = blue if plan else green
    return f'\n[{now:%Y-%m-%d %H:%M}] {branch} {prompt_color(mode_tag.rstrip())}'
CHAT_INPUT_PROMPT = '> '

async def run_interactive() -> None:
    """Interactive mode - allows user to input commands."""
    agent = Agent(workspace=Agent.DEFAULT_WORKSPACE)
    banner = blue('=' * 50)
    print(banner)
    print(blue('Agent Interactive Mode with LM Studio'))
    print(f'Workspace: {cyan(Agent.DEFAULT_WORKSPACE)}')
    model_label = agent.llm.model_name
    profile_part = f'  |  Profile: {agent.llm._profile_name}' if agent.llm._profile_name else ''
    print(f'Model: {cyan(model_label)}{gray(profile_part)}')
    try:
        from agent_core.llm.lmstudio import get_models_status
        models = get_models_status()
        loaded = [m for m in models if m.get('loaded')]
        status = f'LM Studio: online ({len(loaded)}/{len(models)} models loaded)' if models else 'LM Studio: online'
        print(green(status) if models else yellow(status))
    except Exception:
        print(yellow('LM Studio: offline'))
    print(blue('=' * 50))
    registry = _build_registry()
    for name in sorted(registry.names()):
        cmd = registry.get(name)
        synopsis = cmd.help_text.splitlines()[0].strip() if cmd else name
        print(f'  {cyan(synopsis)}')
    print(gray('  Tip: `propose <taskplan.md>` (or `implement --propose` / `fix --mypy --propose`) generates a reviewed diff bundle; it never writes the working tree.'))
    print(f"  {cyan('quit')} - {gray('Exit')}")
    print(blue('=' * 50))
    _register_commands(registry)
    from agent_core.commands.freshness import diff_snapshots, format_stale_warning, loaded_module_fingerprints
    _code_snapshot = loaded_module_fingerprints(__file__)
    _install_signal_handlers(agent)
    while True:
        try:
            branch = _current_git_branch() or '?'
            print(_build_chat_prompt(agent, branch, datetime.now()))
            user_input = _interruptible_input(CHAT_INPUT_PROMPT)
            if user_input is None:
                agent._save_memory()
                save_meta_policy_weights()
                _warn_uncommitted(agent)
                break
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                agent._save_memory()
                save_meta_policy_weights()
                _warn_uncommitted(agent)
                print(green('Goodbye!'))
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
                print(f'  [cmd] {cyan(command)} running...')
                clear_stop()
                _llm = getattr(agent, 'llm', None)
                _chat = getattr(_llm, 'chat', None)
                if _llm is not None and _chat is not None:
                    _llm.chat = chat_stoppable(_chat)
                try:
                    try:
                        await registry.execute(command, parts[1:], agent)
                    except FlowStopped:
                        print(yellow('  Flow stopped by user.'))
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
            print(yellow("\nInterrupted — the current run was stopped. Use 'quit' to exit."))
        except EOFError:
            if not sys.stdin.isatty():
                print(yellow('\n[stdin] Input stream ended — shutting down.'))
            agent._save_memory()
            save_meta_policy_weights()
            _warn_uncommitted(agent)
            break

async def main() -> None:
    """Main entry point - runs interactive mode or the web dashboard."""
    _start_kanban_inbound()
    if '--serve' in sys.argv:
        run_dashboard_server()
        return
    if '--dashboard' in sys.argv:
        start_dashboard_thread(_dashboard_port())
    await run_interactive()
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nInterrupted.')