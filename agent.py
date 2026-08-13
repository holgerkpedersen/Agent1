from typing import Any, cast
#!/usr/bin/env python3
"""Agent implementation with workspace management and tool execution."""

import asyncio
import contextlib
import io
import json
import os
import platform
import re
import sys
from collections import defaultdict
from agent_core import to_windows_path
from agent_core.constants import resolve_model, CHAT_HISTORY_JSON_PATH
from agent_core.llm.lmstudio import LMStudioProvider
from agent_core.file_system import FileSystem
from agent_core.file_searcher import FileSearcher
from agent_core.tool_dispatcher import ToolDispatcher
from agent_core.tool_schemas import NLP_TOOL_SCHEMAS, NLP_TOOL_NAMES
from agent_core.llm.tool_loop import ToolLoopRunner
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
from agent_core.commands.decide_cmd import DecideCommand
from pathlib import Path
import subprocess
import shlex
from typing import Any


class LLMClient:
    """Thin wrapper around LMStudioProvider for backward compatibility.
    
    Delegates to LMStudioProvider for all LLM operations.
    """
    
    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self._model_name: str = resolve_model(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._provider = LMStudioProvider(model_name=self._model_name, api_key=self.api_key)
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
            pass
    
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
                print(f"\n[auto-resume] Truncated ({len(result)} chars), continuing ({i+1}/{max_continues})...")
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
        self._knowledge_graph: dict[str, Any] = {}
        self._working_memory: list[Any] = []
        self._history: list[Any] = []
        #: NLP conversation context — persisted to chat_history.json so a new
        #: session continues where the previous one left off.
        self._chat_history: list[dict[str, Any]] = self._load_chat_history()
        self._nlp_workspace: str | None = None  # workspace override for NLP tools (set by paste --workspace)

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
        """Resolve a path for NLP tool use, honouring _nlp_workspace if set."""
        import os as _os
        if self._nlp_workspace and not _os.path.isabs(path):
            return _os.path.normpath(_os.path.join(self._nlp_workspace, path))
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
                if not results:
                    return "No files found matching that query."
                return "\n".join(f"  {r}" for r in results[:30])
            except Exception as e:
                return f"Search error: {e}"

        if name == "read":
            path = self._resolve_nlp_path(str(args.get("path", "")).strip('"').strip("'"))
            try:
                content = await self.read_file(path, track_read=False)
                if content.startswith("File not found") or content.startswith("Error"):
                    return content
                return content[:5000]
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
            return output or "Fix executed. Check the file for changes."

        if name == "write":
            path = self._resolve_nlp_path(str(args.get("path", "")))
            content = str(args.get("content", ""))
            try:
                if save_file_py(path, content, auto_yes=True):
                    verify = await self._verify_file(path) if path.endswith(".py") else ""
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
            try:
                r = subprocess.run(
                    cmd_to_run,
                    shell=True,
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
                return f"Command timed out after {timeout}s"
            except Exception as e:
                return f"Error: {e}"

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
                    return f"Edited {path}\n{verify}".strip()
                return f"Skipped {path} (no changes)"
            except Exception as e:
                return f"Edit error: {e}"

        if name == "tests":
            test_path = self._resolve_nlp_path(str(args.get("path") or "."))
            framework = str(args.get("framework") or "pytest")
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
                    timeout=120,
                )
                output = r.stdout
                if r.stderr:
                    output += f"\n[STDERR]\n{r.stderr}"
                if len(output) > 5000:
                    output = output[:2500] + "\n... [truncated] ...\n" + output[-2500:]
                return output if output else "(no output)"
            except subprocess.TimeoutExpired:
                return "Tests timed out after 120s"
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

        return f"Unknown tool: {name}. Available: {', '.join(sorted(NLP_TOOL_NAMES))}"


    async def _tool_read_file(self, path: str, **kwargs: Any) -> str:
        result = await self.fs.read(path)
        if not result.startswith("File not found") and not result.startswith("Error"):
            safe = self.fs.safe_path(path)
            self._files_read.add(safe)
            try:
                self._file_mtimes[safe] = os.path.getmtime(safe)
            except OSError:
                pass
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
                    pass

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

    async def _search_files(self, query: str, local_path: str) -> list[str]:
        """Search files with platform-appropriate command and fallback."""
        results = []

        if platform.system() == "Windows":
            try:
                proc = await asyncio.create_subprocess_exec(
                    "findstr", "/S", "/N", "/C:" + query, local_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await proc.communicate()

                if proc.returncode == 0:
                    results = stdout.decode().splitlines()
                else:
                    results = await self._fallback_search(query, local_path)

            except FileNotFoundError:
                results = await self._fallback_search(query, local_path)
        else:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rn", query, local_path,
                stdout=asyncio.subprocess.PIPE
            )
            stdout, _ = await proc.communicate()
            results = stdout.decode().splitlines()

        return results

    async def _fallback_search(self, query: str, path: str) -> list[str]:
        """Fallback search using Python's os.walk with chunked reading."""
        results = []
        chunk_size = 8192

        for root, dirs, files in os.walk(path):
            for file in files:
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        while True:
                            chunk = f.read(chunk_size)
                            if not chunk:
                                break
                            if query in chunk:
                                results.append(filepath)
                                break
                except Exception:
                    pass

        return results

    async def search_file(self, query: str, path: str | None = None) -> str:
        local_path = self._safe_path(path or self.workspace)
        
        results = await self._search_files(query, local_path)
        
        if not results:
            return "No matches found"
        
        return "\n".join(results)

    async def chat_nlp(self, user_input: str) -> None:
        """Process natural language input through a structured tool-calling loop.

        The LLM receives native OpenAI-format tool schemas (``NLP_TOOL_SCHEMAS``)
        and must either emit a structured ``tool_calls`` or answer in text — there
        is no free-text tag format that lets it describe an action instead of
        taking it.  Every tool call is executed, its result is fed back, and the
        loop continues until the model answers in text or the iteration cap is
        reached.
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
                    "- Prefer targeted edits over rewriting whole files.\n"
                    "- If a tool fails, read the error and try a different approach.\n"
                    "- When the request is ambiguous, make a reasonable assumption and state it, "
                    "or ask one clarifying question before acting.\n"
                    "- Be concise. Answer in the user's language."
                ),
            })

        self._chat_history.append({"role": "user", "content": user_input})

        async def llm_chat_fn(
            messages: list[dict[str, Any]], tools: list[dict[str, Any]],
        ) -> tuple[str, list[dict[str, Any]]]:
            """Call the LLM with tools; parse a JSON tool_calls message back into
            the message list so ToolLoopRunner can execute them."""
            raw = await self.llm.chat(messages, tools=tools, disable_thinking=True)
            if raw.startswith("[Error") or raw.startswith("[LM Studio"):
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

        loop = ToolLoopRunner(max_iterations=20)
        final_text, final_messages = await loop.run(
            messages=self._chat_history,
            llm_chat_fn=llm_chat_fn,
            execute_tool_fn=self._execute_tool_call,
            tools=list(NLP_TOOL_SCHEMAS),
        )
        self._chat_history = final_messages
        # Keep the conversation bounded and persist it so the next session
        # (or a follow-up prompt) can continue the dialogue.
        self._chat_history = _trim_chat_history(self._chat_history)
        self._save_chat_history()

        clean = re.sub(r'</?tool_call>', '', final_text)
        clean = re.sub(r'</?function_call>', '', clean)
        if clean.strip():
            print(clean)
        else:
            print("  (The assistant did not produce a response. Try rephrasing.)")
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
                pass
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
        except OSError:
            pass

    # ------------------------------------------------------------------
    #  Persistent NLP chat history (chat_history.json)
    # ------------------------------------------------------------------

    def _load_chat_history(self) -> list[dict[str, Any]]:
        """Load the persisted NLP conversation from the previous session."""
        try:
            with open(CHAT_HISTORY_JSON_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                return []
            messages = [
                m for m in data
                if isinstance(m, dict) and m.get("role") in ("system", "user", "assistant", "tool")
            ]
            return _trim_chat_history(messages)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def _save_chat_history(self) -> None:
        """Persist the NLP conversation so the next session can continue it."""
        try:
            with open(CHAT_HISTORY_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(self._chat_history, f, ensure_ascii=False, indent=2)
        except OSError:
            pass


_MAX_CHAT_MESSAGES = 60

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


def _blocked_shell_command(command: str) -> str | None:
    """Return a description of the first blocked destructive pattern in
    *command*, or None if the command passes the safety scan."""
    for pattern, desc in _DANGEROUS_SHELL_PATTERNS:
        if pattern.search(command):
            return desc
    return None


def _trim_chat_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the system prompt plus the last N-1 messages so the conversation
    stays within a bounded context; returns a new list."""
    if len(messages) <= _MAX_CHAT_MESSAGES:
        return list(messages)
    head = messages[:1]
    return head + list(messages[-(_MAX_CHAT_MESSAGES - 1):])


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

    print("=" * 50)
    print("Agent Interactive Mode with LM Studio")
    print(f"Workspace: {Agent.DEFAULT_WORKSPACE}")
    model_label = agent.llm.model_name
    print(f"Model: {model_label}" + (f"  |  Profile: {agent.llm._profile_name}" if agent.llm._profile_name else ""))
    try:
        from agent_core.llm.lmstudio import get_models_status
        models = get_models_status()
        loaded = [m for m in models if m.get("loaded")]
        print(f"LM Studio: online ({len(loaded)}/{len(models)} models loaded)" if models else "LM Studio: online")
    except Exception:
        print("LM Studio: offline")
    print("Commands:")
    print("  read <path>        - Read a file")
    print("  write <path> <content> - Write content to file")
    print("  search <query>     - Search for string in files")
    print("  analyze <file> [--desc \"q\"] [--stdin] [--deep] — AI analysis via LM Studio")
    print("  plan <analysis.md> <plan.md> - Generate coding plan from analysis")
    print("  entities <analysis.md> <plan.md> [entities.md] - Generate shared entities")
    print("  taskplan <analysis.md> <plan.md> [tasks.md] - Generate implementation tasks")
    print("  implement <taskplan.md> [analysis.md] [plan.md] [entities.md] [--keep] [--force] [--fix] [--retry] [--review] [--workspace <path>] — Implement files")
    print("  fix \"<traceback>\"   - Paste a traceback to auto-fix the error")
    print("  fix <file> --desc \"text\" [--full] - Describe an issue, LLM analyzes full codebase and fixes it")
    print("  fix --mypy [path...] [--limit N] [--rounds N] [--yes] - Batch-fix mypy errors via LLM")
    print("  cleanup             - Show unreferenced files and reference graph")
    print("  workflow <target> [--from spec.md] [--stdin] [--brainstorm] [--desc \"text\"] [--features spec.md] [--force] [--workspace <path>] — Full pipeline")
    print("  model [list|load|unload|reload|name|profile] — Manage models via LM Studio API")
    print("  optimize <file|dir> [--apply] [--yes] [--stdin] — Find and apply optimizations")
    print("  perf [--detail|--reset|--html] — Command performance dashboard")
    print("  clear              - Clear agent memory")
    print("  paste [--workspace <path>] - Paste multiline text for AI analysis (Ctrl+Z / Ctrl+D to finish)")
    print("  quit               - Exit")
    print("=" * 50)
    
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
    registry.register(DecideCommand())

    while True:
        try:
            # Get user input
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Check for quit command
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            # Parse and execute commands
            try:
                parts = shlex.split(user_input, posix=False)
            except ValueError:
                parts = user_input.split(maxsplit=20)
            command = parts[0].lower()

            # Try commands from registry
            if command in ["read", "write", "search", "clear", "model", "analyze", "plan", "entities", "taskplan", "cleanup", "implement", "fix", "workflow", "optimize", "perf", "paste", "decide"]:
                import time as _time
                _start = _time.perf_counter()
                clear_stop()
                _llm = getattr(agent, "llm", None)
                _chat = getattr(_llm, "chat", None)
                if _llm is not None and _chat is not None:
                    _llm.chat = chat_stoppable(_chat)
                try:
                    try:
                        await registry.execute(command, parts[1:], agent)
                    except FlowStopped:
                        print("  Flow stopped by user.")
                finally:
                    if _llm is not None and _chat is not None:
                        _llm.chat = _chat
                PerfTracker.record(command, _time.perf_counter() - _start, user_input)
                continue

            else:
                await agent.chat_nlp(user_input)
        except KeyboardInterrupt:
            print("\nInterrupted — the current run was stopped. Use 'quit' to exit.")
        except EOFError:
            break


async def main() -> None:
    """Main entry point - runs interactive mode."""
    await run_interactive()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
