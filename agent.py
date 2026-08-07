#!/usr/bin/env python3
"""Agent implementation with workspace management and tool execution."""

import asyncio
import contextlib
import io
import os
import platform
import re
from collections import defaultdict
from agent_core import to_windows_path
from agent_core.constants import KNOWN_MODELS, DEFAULT_MODEL, resolve_model, persist_model_choice
from agent_core.llm.lmstudio import LMStudioProvider
from agent_core.file_system import FileSystem
from agent_core.file_searcher import FileSearcher
from agent_core.tool_dispatcher import ToolDispatcher
from agent_core.commands.base import Command
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
from datetime import datetime
from pathlib import Path
import json
import subprocess
import shlex
import difflib


class LLMClient:
    """Thin wrapper around LMStudioProvider for backward compatibility.
    
    Delegates to LMStudioProvider for all LLM operations.
    """
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self._model_name = resolve_model(model_name)
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
    
    async def chat(self, messages: list[dict], tools: list[dict] | None = None, **kwargs) -> str:
        """Send chat request to LLM via LM Studio (pass-through wrapper)."""
        return await self._provider.chat(messages, tools, **kwargs)
    
    async def chat_stream(self, messages: list[dict]) -> str:
        """Chat with real-time token streaming to console."""
        return await self._provider.chat_stream(messages)
    
    async def chat_with_continuation(self, messages: list[dict], max_continues: int = 3, max_tokens: int | None = None) -> str:
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

    DEFAULT_WORKSPACE = "/c/Dev/Agent1"

    def __init__(self, workspace: str = None, model_name: str = None):
        self.workspace = workspace or self.DEFAULT_WORKSPACE
        self.model_name = resolve_model(model_name)

        self._semantic_index: dict[str, set[int]] = defaultdict(set)
        self._files_read: set[str] = set()
        self._file_mtimes: dict[str, float] = {}
        self._knowledge_graph: dict = {}
        self._working_memory: list = []
        self._history: list = []
        self._chat_history: list[dict] = []  # NLP conversation context
        self._nlp_workspace: str | None = None  # workspace override for NLP tools (set by paste --workspace)

        # Initialize LLM client for AI analysis (LM Studio)
        self.llm = LLMClient(model_name=self.model_name)

        # Initialize extracted components
        self.fs = FileSystem(self.workspace)
        self.searcher = FileSearcher(self.workspace)
        self.dispatcher = ToolDispatcher()
        self._register_tool_handlers()

    def _register_tool_handlers(self):
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

    async def _execute_nlp_tool(self, tool_text: str) -> str:
        """Execute a tool call from the NLP conversation and return the result."""
        # Strip nested XML tags that some models add (e.g. <arg_key>path</arg_key>)
        tool_text = re.sub(r'<arg_key>[^<]*</arg_key>', ' ', tool_text)
        tool_text = re.sub(r'</?arg_value>', '', tool_text)
        tool_text = re.sub(r' +', ' ', tool_text).strip()
        parts = shlex.split(tool_text, posix=False) if tool_text else []
        if not parts:
            return "Error: empty tool call"

        cmd = parts[0].lower()
        if cmd == "search":
            query = " ".join(parts[1:])
            search_path = self._nlp_workspace or self.workspace
            try:
                results = await self.searcher.search(query, search_path)
                if not results:
                    return "No files found matching that query."
                return "\n".join(f"  {r}" for r in results[:30])
            except Exception as e:
                return f"Search error: {e}"

        if cmd == "read":
            path = self._resolve_nlp_path(" ".join(parts[1:]).strip('"').strip("'"))
            try:
                content = await self.read_file(path, track_read=False)
                if content.startswith("File not found") or content.startswith("Error"):
                    return content
                return content[:5000]
            except Exception as e:
                return f"Read error: {e}"

        if cmd == "list_files" or cmd == "list":
            raw_path = " ".join(parts[1:]).strip('"').strip("'") or "."
            path = self._resolve_nlp_path(raw_path)
            try:
                import os as _os
                abs_path = path if _os.path.isabs(path) else _os.path.abspath(path)
                if _os.path.isdir(abs_path):
                    entries = _os.listdir(abs_path)[:50]
                    lines = []
                    for e in sorted(entries):
                        full = _os.path.join(abs_path, e)
                        suffix = "/" if _os.path.isdir(full) else ""
                        lines.append(f"  {e}{suffix}")
                    return "\n".join(lines)
                return f"Not a directory: {path}"
            except Exception as e:
                return f"List error: {e}"

        if cmd == "fix":
            resolved_args = list(parts[1:])
            if resolved_args and not resolved_args[0].startswith("--"):
                resolved_args[0] = self._resolve_nlp_path(resolved_args[0])
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                await FixCommand().execute(resolved_args, self)
            output = buf.getvalue()
            return output or "Fix executed. Check the file for changes."

        if cmd == "write":
            # Content is everything after the first newline in tool_text
            nl = tool_text.find('\n')
            if nl == -1:
                return "Error: write requires path on first line, content after"
            path = self._resolve_nlp_path(tool_text[len(cmd):nl].strip())
            content = tool_text[nl + 1:]
            try:
                filepath = Path(path)
                filepath.parent.mkdir(parents=True, exist_ok=True)
                filepath.write_text(content, encoding="utf-8")
                return f"Written {filepath} ({len(content)} bytes)"
            except Exception as e:
                return f"Write error: {e}"

        if cmd == "run":
            # run <command> [timeout]
            run_args = " ".join(parts[1:])
            # Check for timeout flag
            timeout = 120
            timeout_match = re.search(r'--timeout[=\s]+(\d+)', run_args)
            if timeout_match:
                timeout = int(timeout_match.group(1))
                run_args = run_args[:timeout_match.start()] + run_args[timeout_match.end():]
            cmd_to_run = run_args.strip()
            if not cmd_to_run:
                return "Error: run requires a command. Usage: run python main.py"
            # Safety check
            dangerous = ["rm -rf /", "del /s /q", "format c:", "shutdown", "reboot"]
            for d in dangerous:
                if d in cmd_to_run.lower():
                    return f"Error: Dangerous command blocked: {cmd_to_run}"
            try:
                r = subprocess.run(
                    cmd_to_run,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.ws_dir,
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

        if cmd == "git":
            # git <subcommand> [args]
            subcmd = parts[1] if len(parts) > 1 else "status"
            git_args = " ".join(parts[2:]) if len(parts) > 2 else ""
            git_cmds = {
                "status": "git status",
                "diff": "git diff",
                "log": "git log --oneline -15",
                "add": "git add",
                "commit": "git commit",
                "push": "git push",
                "pull": "git pull",
                "branch": "git branch",
                "checkout": "git checkout",
                "stash": "git stash",
                "show": "git show",
                "blame": "git blame",
                "remote": "git remote -v",
                "branches": "git branch -a",
            }
            base = git_cmds.get(subcmd)
            if not base:
                return f"Unknown git command: {subcmd}. Available: {', '.join(git_cmds.keys())}"
            full_cmd = f"{base} {git_args}".strip()
            try:
                r = subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.ws_dir,
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

        if cmd == "diff":
            # diff <file1> [file2]
            if len(parts) < 2:
                return "Error: diff requires at least one file. Usage: diff file1.py [file2.py]"
            file1 = self._resolve_nlp_path(parts[1])
            if len(parts) > 2:
                file2 = self._resolve_nlp_path(parts[2])
                full_cmd = f'diff "{file1}" "{file2}"'
            else:
                full_cmd = f'git diff "{file1}"'
            try:
                r = subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.ws_dir,
                    timeout=30,
                )
                output = r.stdout
                if len(output) > 5000:
                    output = output[:2500] + "\n... [truncated] ...\n" + output[-2500:]
                return output if output else "(no differences found)"
            except Exception as e:
                return f"Diff error: {e}"

        if cmd == "edit":
            # edit <path> | <old text> | <new text>
            # Uses pipe delimiter to separate parts
            edit_args = " ".join(parts[1:])
            pipe_parts = edit_args.split("|")
            if len(pipe_parts) < 3:
                return "Error: edit requires path, old text, and new text separated by |. Usage: edit file.py | old text | new text"
            path = self._resolve_nlp_path(pipe_parts[0].strip())
            old_text = pipe_parts[1].strip()
            new_text = pipe_parts[2].strip()
            try:
                content = open(path, "r", encoding="utf-8", errors="replace").read()
                if old_text not in content:
                    return f"Text not found in {path}. Make sure old text matches exactly (including whitespace)."
                new_content = content.replace(old_text, new_text, 1)
                open(path, "w", encoding="utf-8").write(new_content)
                self.context.mark_modified(path)
                return f"Edited {path}"
            except Exception as e:
                return f"Edit error: {e}"

        if cmd == "tests":
            # tests [path] [--framework pytest|unittest]
            test_path = "."
            framework = "pytest"
            for i, p in enumerate(parts[1:], 1):
                if p == "--framework" and i + 1 < len(parts):
                    framework = parts[i + 1]
                elif not p.startswith("--"):
                    test_path = self._resolve_nlp_path(p)
            if framework == "pytest":
                full_cmd = f"python -m pytest {test_path} -v"
            else:
                full_cmd = f"python -m unittest {test_path} -v"
            try:
                r = subprocess.run(
                    full_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=self.ws_dir,
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

        return f"Unknown tool: {cmd}. Available: search, read, list_files, fix, write, run, git, diff, edit, tests"

    async def _tool_read_file(self, path: str, **kwargs) -> str:
        result = await self.fs.read(path)
        if not result.startswith("File not found") and not result.startswith("Error"):
            safe = self.fs.safe_path(path)
            self._files_read.add(safe)
            try:
                self._file_mtimes[safe] = os.path.getmtime(safe)
            except OSError:
                pass
        return result

    async def _tool_write_file(self, path: str, content: str, **kwargs) -> str:
        return await self.fs.write(path, content)

    async def _tool_apply_patch(self, path: str, find: str, replace: str, **kwargs) -> str:
        return await self.fs.apply_patch(path, find, replace)

    async def _tool_edit_file(self, path: str, content: str, **kwargs) -> str:
        return await self.fs.edit(path, content)

    async def _tool_search(self, query: str, path: str = ".", **kwargs) -> str:
        return await self.searcher.search(query, path)

    async def _tool_list_files(self, path: str = ".", pattern: str = "*", **kwargs) -> str:
        return await self._list_files(path, pattern)

    async def _tool_delete_file(self, path: str, **kwargs) -> str:
        return await self._delete_file(path)

    async def _tool_analyze_file(self, path: str, **kwargs) -> str:
        return await self._analyze_file(path)

    async def _tool_llm_analyze(self, path: str, **kwargs) -> str:
        file_content = await self.read_file(path, track_read=False)
        if file_content.startswith("File not found:") or file_content.startswith("Error reading file:"):
            return f"Could not analyze: {file_content}"
        return await self.llm.analyze_code(file_content)

    async def execute_tool(self, tool_name: str, arguments: dict) -> str:
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
                return normalized
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
            with open(local_path, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"Successfully edited {path}"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            return f"Error editing file: {e}"

    def _build_semantic_index(self, words: list[str], idx: int):
        """Build semantic index with memory management."""
        MAX_INDEX_SIZE = 10000
        
        if len(self._semantic_index) > MAX_INDEX_SIZE:
            self._cleanup_semantic_index()
        
        for word in words:
            normalized_word = word.lower()
            self._semantic_index[normalized_word].add(idx)

    def _cleanup_semantic_index(self):
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

    async def search_file(self, query: str, path: str = None) -> str:
        local_path = self._safe_path(path or self.workspace)
        
        results = await self._search_files(query, local_path)
        
        if not results:
            return "No matches found"
        
        return "\n".join(results)

    def _parse_natural_language(self, query: str) -> tuple:
        """Parse natural language into tool actions."""
        workspace = self.workspace
        query_lower = query.lower()

        if "search" in query_lower and "file" in query_lower:
            search_term = query.replace("search", "").replace("file", "").strip()
            for prep in ["for", "in", "inside"]:
                if prep in search_term:
                    search_term = search_term.split(prep, 1)[1].strip()
                    break
            return ("search_file", {"query": search_term, "path": workspace})

        if "read" in query_lower and ".py" in query:
            filename = query.split()[-1] if query.split() else ""
            return ("read_file", {"path": f"{workspace}/{filename}"})

        if "write" in query_lower:
            parts = query.replace("write", "").strip().split("to")
            if len(parts) == 2:
                path = parts[1].strip()
                content = parts[0].strip()
                return ("write_file", {"path": f"{workspace}/{path}", "content": content})

        if "analyze" in query_lower or "explain" in query_lower:
            parts = query.split()
            file_path = None
            for i, part in enumerate(parts):
                if part.endswith((".py", ".txt", ".md", ".json", ".js", ".ts", ".html", ".css")) or part.startswith(("/", "./", ".\\")):
                    file_path = part
                    if file_path.startswith(("./", ".\\")):
                        file_path = file_path[2:]
                    file_path = to_windows_path(file_path)
                    if not file_path.startswith(("C:", "D:")):
                        file_path = f"{workspace}/{file_path}"
                    break
            if file_path:
                return ("llm_analyze", {"path": file_path})
            return ("llm_analyze", {"path": workspace})

        return ("unknown", {})

    async def chat_nlp(self, user_input: str) -> None:
        """Process natural language input through the ReAct tool-calling loop."""
        # Skip heuristic parsing for long/pasted text — keywords like
        # "analyze" in command output cause false matches.
        if len(user_input) > 500 or self._nlp_workspace:
            tool_action, args = "unknown", {}
        else:
            tool_action, args = self._parse_natural_language(user_input)

        if tool_action == "unknown":
            if not self._chat_history:
                self._chat_history.append({
                    "role": "system",
                    "content": (
                        "You are an AI coding assistant embedded in Agent1.\n"
                        "The user interacts with you through a REPL with these commands:\n"
                        "- workflow, implement, fix, analyze, optimize — LLM-assisted code generation/repair\n"
                        "- model [list|load|unload|reload|name] — manage LLM models via LM Studio API\n"
                        "- model profile [list|save|delete|use|name] — manage model profiles\n"
                        "- clear, cleanup, perf, read, write, search, plan, entities, taskplan, paste — utilities\n"
                        "- Any text not matching a command is sent to you as natural language.\n\n"
                        "TOOLS — write these on their OWN SEPARATE LINE wrapped in <tool_call></tool_call> tags:\n\n"
                        "SEARCH & READ:\n"
                        "<tool_call>search <text></tool_call>     — searches workspace files for <text>\n"
                        "<tool_call>read <path></tool_call>       — reads a file (use absolute paths)\n"
                        "<tool_call>list_files [path]</tool_call> — lists directory (dirs marked with /)\n\n"
                        "WRITE & EDIT:\n"
                        "<tool_call>write <path></tool_call>      — creates/overwrites a file (code on next lines)\n"
                        "<tool_call>edit <path> | <old text> | <new text></tool_call> — edits a file by replacing text\n\n"
                        "RUN & TEST:\n"
                        "<tool_call>run <command></tool_call>     — runs a shell command (e.g., run python main.py)\n"
                        "<tool_call>tests [path] [pytest|unittest]</tool_call> — runs tests on a file/directory\n\n"
                        "VERSION CONTROL:\n"
                        "<tool_call>git <subcommand> [args]    — runs git commands (status, diff, log, add, commit, push, pull, branch, stash, blame)\n"
                        "<tool_call>diff <file1> [file2]      — shows file differences (git diff if one file, diff if two)\n\n"
                        "FIX & ANALYZE:\n"
                        "<tool_call>fix <path> --desc \"error\" — fixes a file based on the description\n\n"
                        "RULES:\n"
                        "- PLAIN TEXT ONLY between tags — no <arg_key>, <arg_value>, or nested XML\n"
                        " The <tool_call> line must contain NOTHING else. Not part of a sentence.\n"
                        " The result will be fed back to you automatically.\n"
                        "- When you see errors, FIX them immediately using the appropriate tool\n"
                        "- Do NOT give generic advice. Take concrete action with tools.\n"
                        "- Be concise. Prefer targeted edits over rewriting entire files.\n"
                        "- For edit, use pipe | to separate path, old text, and new text"
                    ),
                })

            self._chat_history.append({"role": "user", "content": user_input})

            for _ in range(6):
                result = await self.llm.chat(self._chat_history[-20:])

                tool_text: str | None = None
                tool_match = re.search(r'<tool_call>(.+?)</tool_call>', result, re.DOTALL)
                if tool_match:
                    tool_text = tool_match.group(1).strip()

                if not tool_text:
                    self._chat_history.append({"role": "assistant", "content": result})
                    clean = re.sub(r'</?tool_call>', '', result)
                    clean = re.sub(r'</?function_call>', '', clean)
                    print(clean)
                    break

                tool_text = re.sub(r'</?tool_call>', '', tool_text).strip()
                tool_result = await self._execute_nlp_tool(tool_text)
                print(f"  [tool] {tool_text[:80]} -> {len(tool_result)} bytes")

                self._chat_history.append({
                    "role": "assistant",
                    "content": f"<tool_call>{tool_text}</tool_call>",
                })
                self._chat_history.append({
                    "role": "user",
                    "content": f"Tool result:\n{tool_result[:3000]}\n\nContinue your answer based on this.",
                })
            else:
                self._chat_history.append({
                    "role": "user",
                    "content": "You have enough information. Provide your final answer now — text only, no more tool calls.",
                })
                final = await self.llm.chat(self._chat_history[-20:])

                if re.search(r'<tool_call>', final):
                    self._chat_history.append({"role": "assistant", "content": final})
                    self._chat_history.append({
                        "role": "user",
                        "content": "STOP requesting tools. You MUST synthesize your findings into a plain-text answer RIGHT NOW. No tags, no commands — just your analysis.",
                    })
                    final = await self.llm.chat(self._chat_history[-20:])

                self._chat_history.append({"role": "assistant", "content": final})
                clean = re.sub(r'</?tool_call>', '', final)
                clean = re.sub(r'</?function_call>', '', clean)
                if clean.strip():
                    print(clean)
                else:
                    print("  (The LLM did not produce a synthesis. Try repeating the question.)")
        else:
            result = await self.execute_tool(tool_action, args)
            print(result)
        self._nlp_workspace = None

    async def process_query(self, query: str) -> str:
        tool_action, args = self._parse_natural_language(query)

        if tool_action == "unknown":
            return await self.llm.chat([{"role": "user", "content": query}])

        if tool_action == "llm_analyze":
            path = args.get("path", "")
            
            if os.path.isdir(path):
                py_files = []
                for root, _, files in os.walk(path):
                    if ".git" in root or "__pycache__" in root:
                        continue
                    for f in files:
                        if f.endswith(".py"):
                            py_files.append(os.path.join(root, f))
                
                combined = ""
                for pf in py_files:
                    content = await self.read_file(pf, track_read=False)
                    if not content.startswith("File not found:") and not content.startswith("Error"):
                        combined += f"\n\n# ---- {pf} ----\n{content}"
                
                if not combined:
                    return "No Python files found to analyze"
                return await self.llm.analyze_code(combined)
            
            file_content = await self.read_file(path, track_read=False)

            if file_content.startswith("File not found:") or file_content.startswith("Error reading file:"):
                return f"Could not analyze: {file_content}"

            return await self.llm.analyze_code(file_content)

        return await self.execute_tool(tool_action, args)

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

    def memory_stats(self) -> dict:
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

    def clear_history(self):
        """Clear all agent state."""
        self._history = []
        self._chat_history.clear()
        self._files_read.clear()
        self._file_mtimes.clear()
        self._knowledge_graph.clear()
        self._working_memory.clear()
        self._semantic_index.clear()


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


async def run_interactive():
    """Interactive mode - allows user to input commands."""
    
    print("=" * 50)
    print("Agent Interactive Mode with LM Studio")
    print(f"Workspace: {Agent.DEFAULT_WORKSPACE}")
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
    print("  cleanup             - Show unreferenced files and reference graph")
    print("  workflow <target> [--from spec.md] [--stdin] [--brainstorm] [--desc \"text\"] [--features spec.md] [--force] [--workspace <path>] — Full pipeline")
    print("  model [list|load|unload|reload|name|profile] — Manage models via LM Studio API")
    print("  optimize <file|dir> [--apply] [--yes] [--stdin] — Find and apply optimizations")
    print("  perf [--detail|--reset|--html] — Command performance dashboard")
    print("  clear              - Clear agent memory")
    print("  paste [--workspace <path>] - Paste multiline text for AI analysis (Ctrl+Z / Ctrl+D to finish)")
    print("  quit               - Exit")
    print("=" * 50)
    
    # Create agent instance
    agent = Agent(workspace=Agent.DEFAULT_WORKSPACE)

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
            if command in ["read", "write", "search", "clear", "model", "analyze", "plan", "entities", "taskplan", "cleanup", "implement", "fix", "workflow", "optimize", "perf", "paste"]:
                import time as _time
                _start = _time.perf_counter()
                result = await registry.execute(command, parts[1:], agent)
                PerfTracker.record(command, _time.perf_counter() - _start, user_input)
                continue

            else:
                await agent.chat_nlp(user_input)
        except KeyboardInterrupt:
            print("\nInterrupted. Use 'quit' to exit.")
        except EOFError:
            break


async def main():
    """Main entry point - runs interactive mode."""
    await run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
