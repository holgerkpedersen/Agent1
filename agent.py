from typing import Any, cast
#!/usr/bin/env python3
"""Agent implementation with workspace management and tool execution."""

import asyncio
import contextlib
import io
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from agent_core import to_windows_path
from agent_core.colors import cyan, green, yellow, blue, magenta, gray, red
from agent_core.constants import (
    resolve_model,
    CHAT_HISTORY_JSON_PATH,
    AGENT_MEMORY_JSON_PATH,
)
from agent_core.config import load_agent_settings, AgentDisplayMode
from agent_core.file_system import FileSystem
from agent_core.file_searcher import FileSearcher
from agent_core.tool_dispatcher import ToolDispatcher
from agent_core.tool_schemas import NLP_TOOL_SCHEMAS, NLP_TOOL_NAMES
from agent_core.llm.tool_loop import ToolLoopRunner
from agent_core.context_management import CorrelationIdContext
try:
    from harnessfix.tracing import TraceWriter, trace_enabled
except Exception:  # pragma: no cover - tracing degrades gracefully if unavailable
    TraceWriter = None  # type: ignore[assignment, misc]

    def trace_enabled() -> bool:
        return False
from agent_core.commands.base import save_file_py, chat_stoppable, clear_stop, FlowStopped
from agent_core.commands.registry import CommandRegistry
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
from agent_core.commands.implement_cmd import ImplementCommand
from agent_core.commands.fix_cmd import FixCommand
from agent_core.commands.workflow_cmd import WorkflowCommand
from agent_core.commands.optimize_cmd import OptimizeCommand
from agent_core.commands.paste_cmd import PasteCommand
from agent_core.commands.perf_cmd import PerfCommand, PerfTracker
from agent_core.commands.display_cmd import DisplayCommand
from agent_core.commands.decide_cmd import DecideCommand
from agent_core.commands.review_cmd import ReviewCommand
from agent_core.commands.run_cmd import RunCommand
from agent_core.commands.self_heal_cmd import SelfHealCommand
from pathlib import Path
import subprocess
import shlex
from typing import Any


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
        # Restore active profile from model.json on startup
        try:
            from agent_core.constants import load_model_json
            data = load_model_json()
            prof_name = data.get("profile")
            if prof_name:
                from agent_core.llm.model_profiles import get_profile
                profile = get_profile(prof_name)
                self._profile_name = prof_name
                self._provider._profile_name = prof_name
                self._provider.temperature = profile.temperature
                self._provider.max_tokens = profile.max_tokens
        except Exception:
            print("Warning: silenced exception in agent.py:86")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value
        self._provider.model_name = value

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **kwargs: Any) -> str:
        """Send chat request to LLM via LM Studio (pass-through wrapper)."""
        return await self._provider.chat(messages, tools, **kwargs)
    
    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """Chat with real-time token streaming to console."""
        return await self._provider.chat_stream(messages)
    
    async def chat_with_continuation(self, messages: list[dict[str, Any]], max_continues: int = 3, max_tokens: int | None = None) -> str:
        """Chat with auto-resume if response gets truncated at token limit."""
        full_response = ""
        current_messages = [dict(m) for m in messages]

        for i in range(max_continues):
            result = await self.chat(current_messages, max_tokens=max_tokens)

            if result.startswith("[Error") or result.startswith("[LM Studio"):
                return full_response or result

            full_response += result

            stripped = full_response.rstrip()
            if stripped and not stripped.endswith(('```', '}', ')', ']', '"', "'", '.', '\n')):
                print(magenta(f"\n[auto-resume] Truncated ({len(result)} chars), continuing ({i+1}/{max_continues})..."))
                current_messages.append({"role": "assistant", "content": result})
                current_messages.append({"role": "user", "content": "Continue exactly where you stopped. Output the remaining code without repeating anything."})
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
        self.workspace = os.path.abspath(to_windows_path(workspace or self.DEFAULT_WORKSPACE))
        self.model_name = resolve_model(model_name)

        self._semantic_index: dict[str, set[int]] = defaultdict(set)
        self._files_read: set[str] = set()
        self._file_mtimes: dict[str, float] = {}
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
        self._nlp_workspace: str | None = None  # workspace override for NLP tools (set by paste --workspace)

        #: Cross-session memory (files read, semantic index, knowledge graph,
        #: working memory) — restored from agent_memory.json so work done in a
        #: previous session is not forgotten.
        self._load_memory()

        # Initialize LLM client for AI analysis (LM Studio)
        self.llm = LLMClient(model_name=self.model_name)

        # Initialize extracted components
        self.fs = FileSystem(self.workspace)
        self.searcher = FileSearcher(self.workspace)
        self.dispatcher = ToolDispatcher()
        self._register_tool_handlers()

    def _register_tool_handlers(self) -> None:
        """Register tool handlers with the dispatcher."""
        self.dispatcher.register("read_file", lambda args: self._tool_read_file(**args))
        self.dispatcher.register("write_file", lambda args: self._tool_write_file(**args))
        self.dispatcher.register("apply_patch", lambda args: self._tool_apply_patch(**args))
        self.dispatcher.register("edit_file", lambda args: self._tool_edit_file(**args))
        self.dispatcher.register("search", lambda args: self._tool_search(**args))
        self.dispatcher.register("search_file", lambda args: self._tool_search(**args))
        self.dispatcher.register("list_files", lambda args: self._tool_list_files(**args))
        self.dispatcher.register("delete_file", lambda args: self._tool_delete_file(**args))
        self.dispatcher.register("analyze_file", lambda args: self._tool_analyze_file(**args))
        self.dispatcher.register("llm_analyze", lambda args: self._tool_llm_analyze(**args))

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

    async def _verify_file(self, path: str) -> str:
        """Run py_compile on *path* and return a short verification summary."""
        try:
            cwd = os.path.abspath(to_windows_path(self._nlp_workspace or self.workspace))
            if not os.path.isdir(cwd):
                cwd = self.workspace
            r = subprocess.run(
                [sys.executable, "-m", "py_compile", path],
                capture_output=True, text=True, cwd=cwd,
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

    async def _execute_tool_call(self, name: str, args: dict[str, Any]) -> str:
        """Execute a native tool call from the NLP conversation.

        *name* must be one of :data:`NLP_TOOL_NAMES`; *args* is the parsed JSON
        arguments dict.  Writing tools (write/edit) append a py_compile
        verification summary so the model can report verified results.
        """
        ws_dir = self._nlp_workspace or self.workspace
        # subprocess cwd must be an existing directory; a bad override (e.g.
        # a Git-Bash /c/... path from paste --workspace) must not break tools.
        ws_dir = os.path.abspath(to_windows_path(ws_dir))
        if not os.path.isdir(ws_dir):
            ws_dir = self.workspace
        name = name.lower()

        if name == "search":
            query = str(args.get("query", ""))
            search_path = self._resolve_nlp_path(str(args.get("path") or "."))
            try:
                results = await self.searcher.search(query, search_path)
                if not results or results == "No matches found":
                    return "No files found matching that query."
                lines = results.splitlines()
                return "\n".join(f"  {line}" for line in lines[:30])
            except Exception as e:
                return f"Search error: {e}"

        if name == "read":
            path = self._resolve_nlp_path(str(args.get("path", "")).strip('"').strip("'"))
            try:
                # Paging is LINE-BASED (1-indexed): models and callers pass the
                # starting line number and a line count, not character offsets —
                # char offsets kept landing every read in the same short
                # docstring, causing repeated-read loops.
                offset = max(1, int(args.get("offset") or 1))
                limit = max(1, int(args.get("limit") or 100))
                content = await self.read_file(path, track_read=False)
                if content.startswith("File not found") or content.startswith("Error"):
                    return content
                self._note_effect(path)
                lines = content.splitlines()
                if offset > len(lines):
                    return f"Offset {offset} is beyond the end of {path} ({len(lines)} lines)."
                chunk_lines = lines[offset - 1: offset - 1 + limit]
                chunk = "\n".join(chunk_lines)
                if offset - 1 + limit < len(lines):
                    return f"{chunk}\n[truncated — use read with offset={offset + limit} to continue]"
                return chunk
            except Exception as e:
                return f"Read error: {e}"

        if name == "list_files":
            raw_path = str(args.get("path") or ".").strip('"').strip("'")
            path = self._resolve_nlp_path(raw_path)
            try:
                import os as _os
                abs_path = path if _os.path.isabs(path) else _os.path.abspath(path)
                if _os.path.isdir(abs_path):
                    entries = _os.listdir(abs_path)[:50]
                    lines = []
                    for entry in sorted(entries):
                        full = _os.path.join(abs_path, entry)
                        suffix = "/" if _os.path.isdir(full) else ""
                        lines.append(f"  {entry}{suffix}")
                    return "\n".join(lines)
                return f"Not a directory: {path}"
            except Exception as e:
                return f"List error: {e}"

        if name == "fix":
            resolved_args = [str(a) for a in args.get("args", [])]
            if resolved_args and not resolved_args[0].startswith("--"):
                resolved_args[0] = self._resolve_nlp_path(resolved_args[0])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                await FixCommand().execute(resolved_args, self)
            output = buf.getvalue()
            if resolved_args and not resolved_args[0].startswith("--"):
                self._note_effect(resolved_args[0])
            return output or "Fix executed. Check the file for changes."

        if name == "write":
            path = self._resolve_nlp_path(str(args.get("path", "")))
            content = str(args.get("content", ""))
            try:
                if save_file_py(path, content, auto_yes=True):
                    verify = await self._verify_file(path) if path.endswith(".py") else ""
                    self._note_effect(path)
                    return f"Written {path} ({len(content)} bytes)\n{verify}".strip()
                return f"Skipped {path} (no changes)"
            except Exception as e:
                return f"Write error: {e}"

        if name == "run":
            cmd_to_run = str(args.get("command", "")).strip()
            timeout = int(args.get("timeout") or 120)
            if not cmd_to_run:
                return "Error: run requires a command."
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
                    cwd=ws_dir,
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
            if err:
                output += f"\n[STDERR]\n{err}"
                if re.search(r"is not recognized|not found", err, re.I):
                    output += _unix_command_hint()
            elif os.name == "nt" and proc.returncode == 255 and not output:
                # cmd.exe silently fails whole pipelines (rc 255, no output)
                # when a pipe element or command does not exist.
                output += _unix_command_hint()
            if len(output) > 5000:
                output = output[:2500] + "\n... [truncated] ...\n" + output[-2500:]
            return output if output else "(no output)"

        if name == "git":
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
                return f"Unknown git command: {subcmd}. Available: {', '.join(git_cmds.keys())}"
            try:
                extra = shlex.split(git_args)
            except ValueError as e:
                return f"Git error: invalid arguments: {e}"
            try:
                # Arg-list execution (no shell): shell metacharacters in
                # model-supplied args become literal git arguments.
                r = subprocess.run(
                    ["git"] + base + extra,
                    capture_output=True,
                    text=True,
                    cwd=ws_dir,
                    timeout=30,
                )
                output = r.stdout
                if r.stderr:
                    output += f"\n[STDERR]\n{r.stderr}"
                if len(output) > 5000:
                    output = output[:2500] + "\n... [truncated] ...\n" + output[-2500:]
                return output if output else "(no output)"
            except Exception as e:
                return f"Git error: {e}"

        if name == "diff":
            file1 = self._resolve_nlp_path(str(args.get("file1", "")))
            file2 = args.get("file2")
            if not file1:
                return "Error: diff requires at least one file."
            cmd = ["git", "diff", "--no-color", "--"]
            if file2:
                cmd += [file1, self._resolve_nlp_path(str(file2))]
            else:
                cmd += [file1]
            try:
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=ws_dir,
                    timeout=30,
                )
                output = r.stdout
                if len(output) > 5000:
                    output = output[:2500] + "\n... [truncated] ...\n" + output[-2500:]
                return output if output else "(no differences found)"
            except Exception as e:
                return f"Diff error: {e}"

        if name == "edit":
            path = self._resolve_nlp_path(str(args.get("path", "")))
            old_text = str(args.get("old_text", ""))
            new_text = str(args.get("new_text", ""))
            try:
                content = open(path, "r", encoding="utf-8", errors="replace").read()
                if old_text not in content:
                    return f"Text not found in {path}. Make sure old text matches exactly (including whitespace)."
                new_content = content.replace(old_text, new_text, 1)
                ctx = getattr(self, "context", None)
                if save_file_py(path, new_content, auto_yes=True):
                    if ctx is not None:
                        ctx.mark_modified(path)
                    verify = await self._verify_file(path) if path.endswith(".py") else ""
                    self._note_effect(path)
                    return f"Edited {path}\n{verify}".strip()
                return f"Skipped {path} (no changes)"
            except Exception as e:
                return f"Edit error: {e}"

        if name == "tests":
            test_path = self._resolve_nlp_path(str(args.get("path") or "."))
            framework = str(args.get("framework") or "pytest")
            #: The full suite takes ~2.5 minutes, so 120s is too short and made
            #: the agent split runs into subsets. 300s covers whole-suite runs.
            timeout = 300
            try:
                if framework == "pytest":
                    cmd = [sys.executable, "-m", "pytest", test_path, "-v"]
                else:
                    cmd = [sys.executable, "-m", "unittest", test_path, "-v"]
                r = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=ws_dir,
                    timeout=timeout,
                )
                output = r.stdout
                if r.stderr:
                    output += f"\n[STDERR]\n{r.stderr}"
                if len(output) > 5000:
                    output = output[:2500] + "\n... [truncated] ...\n" + output[-2500:]
                return output if output else "(no output)"
            except subprocess.TimeoutExpired:
                return f"Tests timed out after {timeout}s"
            except Exception as e:
                return f"Error: {e}"

        if name == "analyze":
            analyze_args: list[str] = []
            if args.get("path"):
                analyze_args = [self._resolve_nlp_path(str(args["path"]))]
            try:
                from agent_core.commands.analyze_cmd import AnalyzeCommand
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    await AnalyzeCommand().execute(analyze_args, self)
                output = buf.getvalue()
                return output or "Analysis complete."
            except Exception as e:
                return f"Analyze error: {e}"

        if name == "web_search":
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
                max_results = max(1, min(int(args.get("max_results") or 5), MAX_RESULTS_LIMIT))
                clean_query = sanitize_query(query)
                results = search_ddg(clean_query, max_results=max_results)
                output = format_results(clean_query, results)
                return output if len(output) <= 5000 else output[:5000]
            except Exception as e:
                return f"Web search error: {e}"

        return f"Unknown tool: {name}. Available: {', '.join(sorted(NLP_TOOL_NAMES))}"


    async def _tool_read_file(self, path: str, **kwargs: Any) -> str:
        result = await self.fs.read(path)
        if not result.startswith("File not found") and not result.startswith("Error"):
            safe = self.fs.safe_path(path)
            self._files_read.add(safe)
            try:
                self._file_mtimes[safe] = os.path.getmtime(safe)
            except OSError:
                print("Warning: silenced exception in agent.py:506")
        return result

    async def _tool_write_file(self, path: str, content: str, **kwargs: Any) -> str:
        return await self.fs.write(path, content)

    async def _tool_apply_patch(self, path: str, find: str, replace: str, **kwargs: Any) -> str:
        return await self.fs.apply_patch(path, find, replace)

    async def _tool_edit_file(self, path: str, content: str, **kwargs: Any) -> str:
        return await self.fs.edit(path, content)

    async def _tool_search(self, query: str, path: str = ".", **kwargs: Any) -> str:
        return await self.searcher.search(query, path)

    async def _tool_list_files(self, path: str = ".", pattern: str = "*", **kwargs: Any) -> str:
        return await self.fs.list_files(path, pattern)

    async def _tool_delete_file(self, path: str, **kwargs: Any) -> str:
        return await self.fs.delete(path)

    async def _tool_analyze_file(self, path: str, **kwargs: Any) -> str:
        return cast(str, await self.llm.analyze_code(await self.read_file(path)))  # type: ignore[redundant-cast]

    async def _tool_llm_analyze(self, path: str, **kwargs: Any) -> str:
        file_content = await self.read_file(path, track_read=False)
        if file_content.startswith("File not found:") or file_content.startswith("Error reading file:"):
            return f"Could not analyze: {file_content}"
        return cast(str, await self.llm.analyze_code(file_content))  # type: ignore[redundant-cast]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name using the dispatcher."""
        return await self.dispatcher.execute(tool_name, arguments)

    def _normalize_path(self, path: str) -> str:
        """Normalize and validate paths with security checks."""
        normalized = to_windows_path(path)

        try:
            abs_path = Path(normalized).resolve()
            return str(abs_path)
        except (OSError, RuntimeError):
            if normalized.startswith(("C:\\", "D:\\", "/")):
                return str(normalized)
            raise ValueError(f"Invalid path: {path}")

    def _safe_path(self, path: str) -> str:
        """Validate and normalize path in one step."""
        if path.startswith(("./", ".\\")):
            path = path[2:]
        return self._normalize_path(path)

    async def read_file(self, path: str, track_read: bool = True) -> str:
        local_path = self._safe_path(path)

        try:
            with open(local_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if track_read:
                self._files_read.add(local_path)
                try:
                    self._file_mtimes[local_path] = os.path.getmtime(local_path)
                except OSError:
                    print("Warning: silenced exception in agent.py:570")

            return content

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
                return f"Error: find text matches {count} locations. Add more context to make it unique."

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

    async def chat_nlp(self, user_input: str) -> None:
        """Process natural language input through a structured tool-calling loop.

        The LLM receives native OpenAI-format tool schemas (``NLP_TOOL_SCHEMAS``)
        and must either emit a structured ``tool_calls`` or answer in text — there
        is no free-text tag format that lets it describe an action instead of
        taking it.  Every tool call is executed, its result is fed back, and the
        loop continues until the model answers in text or the iteration cap is
        reached.

        The loop auto-continues: if a run ends on the iteration cap, on repeated
        calls, or with an answer that signals unfinished work, a fresh run starts
        automatically (up to ``_MAX_CHAINED_RUNS``) so the model does not stop
        mid-task and wait for the end-user.
        """
        if not self._chat_history:
            self._chat_history.append({
                "role": "system",
                "content": (
                    "You are a senior coding assistant working inside this project workspace.\n"
                    "The user speaks natural language; you have tools to search, read, write, "
                    "edit, run, and test the code.\n\n"
                    "WORKING METHOD:\n"
                    "1. Understand the request.\n"
                    "2. Take concrete action with the tools — never just describe what you "
                    "would do. If you intend to read, search, edit or run something, call the tool.\n"
                    "3. After a write/edit, a py_compile verification summary is returned "
                    "automatically — report it.\n"
                    "4. Finish with a short report: what you changed, where, and the "
                    "verification/test evidence.\n\n"
                    "RULES:\n"
                    f"- The shell for the run tool is: {_detect_shell()}. On Windows there "
                    "is NO tail/grep/ls/find. Never pipe with '2>&1 | tail -40' — use Python "
                    "one-liners (python -c \"...\") or the built-in tools; run output is "
                    "truncated to 5000 chars automatically.\n"
                    "- You have an effectively unlimited tool budget: do not rush, do not "
                    "stop early, and do not plan around a budget. Take as many tool calls "
                    "as the task needs — but make steady progress: when exploration is "
                    "done, actually edit/write/fix instead of only reading.\n"
                    "- read returns up to 5000 chars per call by default — never request "
                    "tiny slices (limit < 1000). Read big chunks and continue with the "
                    "offset hint; avoid many small read calls.\n"
                    "- Prefer targeted edits over rewriting whole files.\n"
                    "- If a tool fails, read the error and try a different approach.\n"
                    "- When the request is ambiguous, make a reasonable assumption and state it, "
                    "or ask one clarifying question before acting.\n"
                    "- Never assert facts about this repo that you have not verified with a "
                    "tool. project_plan.md / project_tasks.md are HISTORICAL phase docs and "
                    "may be outdated — verify claims against the actual code.\n"
                    "- Verify numbers (e.g. how many tests exist) with the tests tool or git "
                    "log before claiming them.\n"
                    "- If a search finds nothing in source files, state that the symbol does "
                    "not exist in the current code — never repeat the same search.\n"
                    "- Be concise. Answer in the user's language."
                ),
            })

        self._chat_history.append({"role": "user", "content": user_input})

        #: Provider-level failures (reasoning-budget exhaustion, HTTP errors,
        #: unreachable server, ...) are DETECTED here.  They must never be
        #: mistaken for a model answer, fed into auto-continue chaining, or
        #: printed as the green final answer — bad LLM behavior is surfaced
        #: explicitly instead.
        llm_error: list[str] = []
        #: Last non-empty answer of a chained run — a byte-identical repeat
        #: means the model is stuck emitting the same incomplete answer.
        last_answer = ""

        async def llm_chat_fn(
            messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        ) -> tuple[str, list[dict[str, Any]]]:
            """Call the LLM with tools; parse a JSON tool_calls message back into
            the message list so ToolLoopRunner can execute them."""
            raw = await self.llm.chat(messages, tools=tools, disable_thinking=True)
            if raw.strip() == "(no output)":
                # Providers use "(no output)" for an empty response — treat it
                # as empty so the loop's forced-synthesis retry / the concrete
                # fallback message kick in instead of showing cryptic text.
                raw = ""
            if raw.startswith("[Error") or raw.startswith("[LM Studio"):
                if not llm_error:
                    llm_error.append(raw)
                return raw, messages
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("tool_calls"):
                parsed.pop("role", None)
                updated = list(messages)
                updated.append({"role": "assistant", "content": parsed.get("content") or "", **parsed})
                text = str(parsed.get("content") or "")
                return text, updated
            # Plain text answer — the loop terminates.
            updated = list(messages)
            updated.append({"role": "assistant", "content": raw})
            return raw, updated

        final_text = ""
        final_messages = self._chat_history
        continuations = 0
        display_mode = _resolve_display_mode()
        #: Cross-run call registry: how many times each exact call has been
        #: executed across ALL chained runs of this turn, so repeated probes
        #: cannot hide behind a fresh run's duplicate counter.
        seen_calls: dict[tuple[str, str], int] = {}
        #: One correlation id per TURN: every chained run of this chat_nlp
        #: call shares it, so a single task is linkable across its traces
        #: (decision #050).
        with CorrelationIdContext():
            while True:
                #: Per-run trace writer (one JSONL file per run() invocation,
                #: decision #029).  AGENT_NO_TRACE=1 disables trace capture.
                #: Model/profile are stamped on every record so a trace is
                #: self-describing for review and cross-model comparison (decision #050).
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
                    tools=list(NLP_TOOL_SCHEMAS),
                    seen_calls=seen_calls,
                    effects_fn=self._take_trace_effects,
                )
                self._pending_effects = None
                if trace_writer is not None:
                    trace_writer.close()
                reason = loop.termination_reason
                # A provider-level failure is not an answer: never auto-continue —
                # chaining would only re-burn the same broken LLM call.
                if llm_error:
                    break
                # "cap" (budget run-out while progressing), an "answer" that
                # signals unfinished work, and — as a safety net — a "no_progress"
                # verdict whose forced answer STILL signals unfinished work justify
                # chaining.  Plain "stuck"/"no_progress" verdicts are explicit
                # "model is not making progress" signals — continuing would only
                # re-enter the same loop.
                needs_more = (
                    reason == "cap"
                    or (reason in ("answer", "no_progress") and _looks_incomplete(final_text))
                )
                if needs_more and continuations < _MAX_CHAINED_RUNS:
                    if final_text and final_text == last_answer:
                        # Same incomplete answer twice in a row — the model is
                        # stuck, not working.  Stop chaining instead of looping.
                        if display_mode != AgentDisplayMode.QUIET:
                            print(
                                magenta("\n  [stopped] The model repeated the same answer ")
                                + yellow("twice — ending the turn. ")
                                + gray("Rephrase or ask something more specific to continue.\n")
                            )
                        break
                    last_answer = final_text
                    continuations += 1
                    if display_mode != AgentDisplayMode.QUIET:
                        why = {
                            "cap": "iteration budget exhausted",
                            "answer": "answer signals unfinished work",
                            "no_progress": "the final answer signals unfinished work",
                        }.get(reason, reason)
                        print(
                            magenta(f"\n  [auto-continue] Run {continuations}: ")
                            + yellow(f"{why} — continuing automatically.\n")
                        )
                    final_messages = list(final_messages) + [
                        # User role: strict chat templates (qwen Jinja) reject
                        # system messages mid-conversation.
                        {"role": "user", "content": _CONTINUE_NOTE},
                    ]
                    continue
                if reason in ("stuck", "no_progress") and display_mode != AgentDisplayMode.QUIET:
                    print(
                        magenta("\n  [stopped] The model stopped making progress ")
                        + yellow(f"{'stuck on repeated calls' if reason == 'stuck' else 'too many calls without making new progress'}). ")
                        + gray("The answer above is the best it produced — rephrase or ask something more specific to continue.\n")
                    )
                break

        # The continuation note is only meant for the run it precedes — strip
        # it (and any earlier ones) before the history is persisted, so a
        # finished task is not resumed by a future session.
        final_messages = [
            m for m in final_messages
            if not (m.get("content") == _CONTINUE_NOTE)
        ]
        self._chat_history = final_messages
        # Keep the conversation bounded and persist it so the next session
        # (or a follow-up prompt) can continue the dialogue.
        self._chat_history = _trim_chat_history(self._chat_history)
        self._save_chat_history()
        self._save_memory()

        clean = re.sub(r'</?tool_call>', '', final_text)
        clean = re.sub(r'</?function_call>', '', clean)

        if llm_error:
            # A provider-level failure was detected during the run: show the
            # actual error (not the generic fallback, not a green "answer").
            err = llm_error[0].strip()
            print(yellow("\n  [llm-error] The model did not produce a usable response:"))
            print(red(f"  {err[:400]}"))
            if "reasoning" in err.lower():
                print(
                    yellow("  The reasoning model exhausted its thinking budget. ")
                    + gray("Switch to a non-reasoning model (model <name>) or retry the request.\n")
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

        # The final answer is ALWAYS printed — in every display mode, including
        # QUIET (which only hides intermediate tool output, per the display-mode
        # contract: "only the final answer is printed").
        if clean.strip():
            print(green(clean))
        else:
            # No usable answer: tell the user CONCRETELY what the loop did
            # instead of the cryptic "did not produce a response".
            print(yellow(_final_answer_fallback(loop)))
        self._nlp_workspace = None

    def check_stale_files(self) -> list[str]:
        """Return files whose mtime has changed since last read."""
        stale = []
        for path in list(self._file_mtimes):
            try:
                if os.path.getmtime(path) != self._file_mtimes[path]:
                    stale.append(path)
            except FileNotFoundError:
                stale.append(path)
            except OSError:
                print("Warning: silenced exception in agent.py:863")
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
        except OSError:
            print("Warning: silenced exception in agent.py:901")

    # ------------------------------------------------------------------
    #  Persistent NLP chat history (chat_history.json)
    # ------------------------------------------------------------------

    def _load_chat_history(self) -> list[dict[str, Any]]:
        """Load the persisted NLP conversation from the previous session.

        The persisted history holds a bounded multi-exchange window (see
        ``_project_chat_history``), so a fresh session continues the dialogue
        instead of forgetting it.
        """
        try:
            with open(CHAT_HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            messages = [
                m for m in data
                if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant", "tool")
            ]
            return _project_chat_history(messages)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def _save_chat_history(self) -> None:
        """Persist the NLP conversation so the next session can continue it."""
        try:
            with open(CHAT_HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(_project_chat_history(self._chat_history), f, ensure_ascii=False, indent=2)
        except OSError:
            print("Warning: silenced exception in agent.py:933")

    # ------------------------------------------------------------------
    #  Persistent agent memory (agent_memory.json)
    # ------------------------------------------------------------------

    def _load_memory(self) -> None:
        """Load cross-session memory from the previous session.

        Restores files read, the semantic index, the knowledge graph, and
        working memory so a fresh session resumes with accumulated context
        instead of starting from zero.  Missing or corrupt files fall back to
        the empty defaults.  File mtimes are intentionally NOT persisted —
        staleness detection is scoped to the current session.
        """
        try:
            with open(AGENT_MEMORY_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
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
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass

    def _save_memory(self) -> None:
        """Persist cross-session memory so the next session resumes with it."""
        try:
            data = {
                "files_read": sorted(self._files_read),
                "semantic_index": {
                    k: sorted(v) for k, v in self._semantic_index.items()
                },
                "knowledge_graph": self._knowledge_graph,
                "working_memory": self._working_memory,
                "history": self._history,
            }
            with open(AGENT_MEMORY_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            print("Warning: silenced exception in agent.py:1009")


_MAX_CHAT_MESSAGES = 60

#: How many automatic continuation runs chat_nlp may chain before handing
#: control back to the user.  The model cannot predict its own tool budget,
#: so instead of stopping mid-task it continues with a fresh budget.  Each
#: chained run has its own guards (no-progress, stuck, deadline), so a high
#: cap cannot turn into an infinite loop — it only bounds total work.
_MAX_CHAINED_RUNS = 6

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
        "one-liners (python -c \"...\") or the built-in tools — run output is truncated "
        "automatically, so pipes like '2>&1 | tail -40' are neither needed nor supported."
    )


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
        print("Warning: silenced exception in agent.py:1055")


def _detect_shell() -> str:
    """Return a human-readable name for the shell the run tool will use.

    The run tool executes via ``subprocess.run(..., shell=True)``, which on
    Windows means %COMSPEC% (cmd.exe by default, PowerShell if configured).
    """
    if os.name != "nt":
        return "bash (or the default POSIX shell)"
    comspec = os.environ.get("COMSPEC", "").lower()
    if comspec.endswith(("powershell.exe", "pwsh.exe")):
        return "PowerShell"
    return "cmd.exe (Windows Command Prompt)"


def _blocked_shell_command(command: str) -> str | None:
    """Return a description of the first blocked destructive pattern in
    *command*, or None if the command passes the safety scan."""
    for pattern, desc in _DANGEROUS_SHELL_PATTERNS:
        if pattern.search(command):
            return desc
    return None


def _drop_orphan_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop ``tool`` messages whose ``tool_call_id`` has no matching assistant
    ``tool_calls`` message.

    Trimming can cut between an assistant tool_calls message and its tool
    result, leaving an orphan — strict gateways (opencode Console Go) reject
    those with HTTP 400 ("Messages with role 'tool' must be a response to a
    preceding message with 'tool_calls'").
    """
    valid_ids: set[str] = set()
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("id"):
                    valid_ids.add(str(tc["id"]))
    return [
        m for m in messages
        if not (m.get("role") == "tool" and m.get("tool_call_id") not in valid_ids)
    ]


def _trim_chat_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the system prompt plus the last N-1 messages so the conversation
    stays within a bounded context; returns a new list.

    Also drops orphan tool messages that a cut could leave behind (see
    :func:`_drop_orphan_tool_messages`).
    """
    if len(messages) <= _MAX_CHAT_MESSAGES:
        return _drop_orphan_tool_messages(messages)
    head = messages[:1]
    return _drop_orphan_tool_messages(head + list(messages[-(_MAX_CHAT_MESSAGES - 1):]))


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


def _is_similar(content1: str, content2: str, threshold: float = 0.8) -> bool:
    """Check if two pieces of content are similar enough (based on line structure)."""
    if not content1 or not content2:
        return False
    lines1 = set(content1.splitlines())
    lines2 = set(content2.splitlines())
    if not lines1 or not lines2:
        return False
    intersection = len(lines1 & lines2)
    union = len(lines1 | lines2)
    similarity = intersection / union if union > 0 else 0
    return similarity >= threshold


async def run_interactive() -> None:
    """Interactive mode - allows user to input commands."""
    
    # Create agent instance (resolves persisted model choice from model.json)
    agent = Agent(workspace=Agent.DEFAULT_WORKSPACE)

    banner = blue("=" * 50)
    print(banner)
    print(blue("Agent Interactive Mode with LM Studio"))
    print(f"Workspace: {cyan(Agent.DEFAULT_WORKSPACE)}")
    model_label = agent.llm.model_name
    profile_part = (f"  |  Profile: {agent.llm._profile_name}" if agent.llm._profile_name else "")
    print(f"Model: {cyan(model_label)}{gray(profile_part)}")
    try:
        from agent_core.llm.lmstudio import get_models_status
        models = get_models_status()
        loaded = [m for m in models if m.get("loaded")]
        status = f"LM Studio: online ({len(loaded)}/{len(models)} models loaded)" if models else "LM Studio: online"
        print(green(status) if models else yellow(status))
    except Exception:
        print(yellow("LM Studio: offline"))
    print(cyan("Commands:"))
    _CMD_LIST = [
        ("read <path>", "Read a file"),
        ("write <path> <content>", "Write content to file"),
        ("search <query>", "Search for string in files"),
        ("analyze <file> [--desc \"q\"] [--stdin] [--deep]", "AI analysis via LM Studio"),
        ("plan <analysis.md> <plan.md>", "Generate coding plan from analysis"),
        ("entities <analysis.md> <plan.md> [entities.md]", "Generate shared entities"),
        ("taskplan <analysis.md> <plan.md> [tasks.md]", "Generate implementation tasks"),
        ("implement <taskplan.md> ... [--keep] [--force] [--fix] [--retry] [--review] [--workspace <path>]", "Implement files"),
        ('fix "<traceback>"', "Paste a traceback to auto-fix the error"),
        ('fix <file> --desc "text" [--full]', "Describe an issue, LLM analyzes full codebase and fixes it"),
        ("fix --mypy [path...] [--limit N] [--rounds N] [--yes]", "Batch-fix mypy errors via LLM"),
        ("cleanup", "Show unreferenced files and reference graph"),
        ("workflow <target> ... [--from spec.md] [--stdin] [--brainstorm] [--desc \"text\"] [--features spec.md] [--force] [--workspace <path>]", "Full pipeline"),
        ("model [list|load|unload|reload|name|profile]", "Manage models via LM Studio API"),
        ("optimize <file|dir> [--apply] [--yes] [--stdin]", "Find and apply optimizations"),
        ("perf [--detail|--reset|--html]", "Command performance dashboard"),
        ("clear", "Clear agent memory"),
        ("display [verbose|clean|quiet]", "Show/set NLP output verbosity"),
        ("paste [--workspace <path>]", "Paste multiline text for AI analysis (Ctrl+Z / Ctrl+D to finish)"),
        ("run <command>", "Execute a shell command directly, no LLM"),
        ("self_heal [path] [--rounds N] [--yes]", "Patch failing tests and re-run until green"),
        ("decide ...", "Track design decisions (add/list/show/check/resolve/link/extract/review)"),
        ("review <refresh|list|show|label|auto|export>", "Human gate over failed task traces (label <task> auto = agent reviews)"),
        ("quit", "Exit"),
    ]
    for name, desc in _CMD_LIST:
        print(f"  {cyan(name)} - {gray(desc)}")
    print(blue("=" * 50))
    
    # Set up command registry with simple commands
    registry = CommandRegistry()
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
    registry.register(ImplementCommand())
    registry.register(FixCommand())
    registry.register(WorkflowCommand())
    registry.register(OptimizeCommand())
    registry.register(PerfCommand())
    registry.register(PasteCommand())
    registry.register(DisplayCommand())
    registry.register(DecideCommand())
    registry.register(ReviewCommand())
    registry.register(RunCommand())
    registry.register(SelfHealCommand())

    while True:
        try:
            # Get user input
            user_input = input(f"\n[{datetime.now():%Y-%m-%d %H:%M}] > ").strip()
            
            if not user_input:
                continue
            
            # Check for quit command
            if user_input.lower() in ["quit", "exit", "q"]:
                agent._save_memory()
                print(green("Goodbye!"))
                break
            
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
                continue

            else:
                await agent.chat_nlp(user_input)
        except KeyboardInterrupt:
            print(yellow("\nInterrupted — the current run was stopped. Use 'quit' to exit."))
        except EOFError:
            agent._save_memory()
            break


async def main() -> None:
    """Main entry point - runs interactive mode."""
    await run_interactive()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
