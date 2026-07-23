#!/usr/bin/env python3
"""Agent implementation with workspace management and tool execution."""

import asyncio
import os
import platform
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import subprocess


DEFAULT_MODEL = "qwen3.6-27b-mtp"


class LLMClient:
    """LLM client for AI-powered analysis and conversation."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        self.model_name = model_name or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

        # LM Studio configuration - typically runs on localhost:1234
        self.lmstudio_url = os.environ.get("LMSTUDIO_URL", "http://localhost:1234/v1")
    
    async def chat(self, messages: list[dict]) -> str:
        """Send chat request to LLM via LM Studio."""

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 50000
        }

        try:
            import urllib.request

            data = json.dumps(payload).encode('utf-8')

            req = urllib.request.Request(
                f"{self.lmstudio_url}/chat/completions",
                data=data,
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}'
                },
                method='POST'
            )

            with urllib.request.urlopen(req, timeout=1800) as response:
                result = json.loads(response.read().decode())

                if 'choices' in result and len(result['choices']) > 0:
                    message = result['choices'][0]['message']
                    content = message.get('content') or message.get('reasoning_content') or ""
                    return content

        except (asyncio.TimeoutError, TimeoutError):
            return "[Error: Request timed out - model is taking too long]"
        except Exception as e:
            return f"[LM Studio error: {e}]"

        return "No response from LLM"
    
    async def analyze_code(self, code: str) -> str:
        """Analyze code using LLM."""

        prompt = f"""Analyze this Python code and identify:
1. Bugs or issues
2. Code quality concerns
3. Potential improvements
4. Circular imports - which modules import each other, creating cycles
5. Missing or broken cross-module references

Code:
{code}"""

        messages = [
            {"role": "system", "content": "You are an expert code reviewer. Analyze the provided code and give detailed feedback."},
            {"role": "user", "content": prompt}
        ]

        return await self.chat(messages)


class Agent:
    """Main agent class with workspace management and tool execution."""

    DEFAULT_WORKSPACE = "/c/Dev/Agent1"

    def __init__(self, workspace: str = None, model_name: str = None):
        self.workspace = workspace or self.DEFAULT_WORKSPACE
        self.model_name = model_name or DEFAULT_MODEL

        self._semantic_index: dict[str, set[int]] = defaultdict(set)
        self._files_read: set[str] = set()
        self._knowledge_graph: dict = {}
        self._working_memory: list = []
        self._history: list = []

        # Initialize LLM client for AI analysis (LM Studio)
        self.llm = LLMClient(model_name=self.model_name)

        # Chat history for context
        self.chat_history = []

    def _normalize_path_strict(self, path: str) -> str:
        """Normalize path and ensure consistent casing for comparison."""
        if path.startswith("/c/"):
            normalized = "C:\\" + path[3:].replace("/", "\\")
        elif path.startswith("/d/"):
            normalized = "D:\\" + path[3:].replace("/", "\\")
        elif path.startswith("/"):
            normalized = "C:\\" + path[1:].replace("/", "\\")
        else:
            normalized = path

        # Resolve to absolute path for consistent comparison
        try:
            abs_path = Path(normalized).resolve()
            return str(abs_path)
        except (OSError, RuntimeError):
            # Fallback: just return the normalized path
            return normalized

    def _normalize_path(self, path: str) -> str:
        """Normalize and validate paths with security checks."""
        if path.startswith("/c/"):
            normalized = "C:\\" + path[3:].replace("/", "\\")
        elif path.startswith("/d/"):
            normalized = "D:\\" + path[3:].replace("/", "\\")
        elif path.startswith("/"):
            normalized = "C:\\" + path[1:].replace("/", "\\")
        else:
            normalized = path

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

    async def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool with proper duplicate read prevention."""

        if tool_name == "read_file":
            path = args.get("path", "")

            try:
                normalized_path = self._normalize_path(path)
            except ValueError as e:
                return f"Invalid path: {e}"

            if normalized_path in self._files_read:
                return f"File already read: {path}"

            result = await self.read_file(path, track_read=False)
            if not result.startswith("Error") and not result.startswith("File not found"):
                self._files_read.add(normalized_path)
            return result

        elif tool_name == "write_file":
            return await self.write_file(
                args.get("path", ""),
                args.get("content", "")
            )

        elif tool_name == "apply_patch":
            return await self.apply_patch(
                args.get("path", ""),
                args.get("find", ""),
                args.get("replace", "")
            )

        elif tool_name == "edit_file":
            return await self.edit_file(
                args.get("path", ""),
                args.get("content", "")
            )

        elif tool_name == "search_file":
            return await self.search_file(
                args.get("query", ""),
                args.get("path", None)
            )

        elif tool_name == "llm_analyze":
            path = args.get("path", "")
            file_content = await self.read_file(path, track_read=False)
            if file_content.startswith("File not found:") or file_content.startswith("Error reading file:"):
                return f"Could not analyze: {file_content}"
            return await self.llm.analyze_code(file_content)

        return f"Unknown tool: {tool_name}"

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
                    if not file_path.startswith("/c/") and not file_path.startswith("C:") and not file_path.startswith("D:"):
                        file_path = f"{workspace}/{file_path}"
                    break
            if file_path:
                return ("llm_analyze", {"path": file_path})
            return ("llm_analyze", {"path": workspace})

        return ("unknown", {})

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

    def clear_history(self):
        """Clear all agent state."""
        self._history = []
        self._files_read.clear()
        self._knowledge_graph.clear()
        self._working_memory.clear()
        self._semantic_index.clear()


class ToolRegistry:
    """Registry for managing tool definitions and execution."""

    def __init__(self, workspace: str = None, model_name: str = None):
        self.workspace = workspace or Agent.DEFAULT_WORKSPACE
        self._agent = Agent(workspace=self.workspace, model_name=model_name or DEFAULT_MODEL)

    async def read_file(self, path: str) -> str:
        return await self._agent.read_file(path)

    async def write_file(self, path: str, content: str) -> str:
        return await self._agent.write_file(path, content)

    async def apply_patch(self, path: str, find: str, replace: str) -> str:
        return await self._agent.apply_patch(path, find, replace)

    async def edit_file(self, path: str, content: str) -> str:
        return await self._agent.edit_file(path, content)

    async def search_file(self, query: str, path: str = None) -> str:
        return await self._agent.search_file(query, path)

    async def execute_tool(self, tool_name: str, args: dict) -> str:
        return await self._agent.execute_tool(tool_name, args)


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
    print("  analyze <file> [analysis.md] - AI analysis via LM Studio")
    print("  plan <analysis.md> <plan.md> - Generate coding plan from analysis")
    print("  entities <analysis.md> <plan.md> [entities.md] - Generate shared entities")
    print("  taskplan <analysis.md> <plan.md> [tasks.md] - Generate implementation tasks")
    print("  implement <taskplan.md> [--keep] [--force] [--workspace <path>] - Implement files")
    print("  workflow <target> [--from spec.md] [--desc \"text\"] [--features spec.md] [--force] [--workspace <path>] - Full pipeline")
    print("  clear              - Clear agent memory")
    print("  quit               - Exit")
    print("=" * 50)
    
    # Create agent instance
    agent = Agent(workspace=Agent.DEFAULT_WORKSPACE)
    
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
            parts = user_input.split(maxsplit=20)
            command = parts[0].lower()
            
            if command == "read":
                if len(parts) < 2:
                    print("Usage: read <path>")
                    continue
                    
                path = parts[1]
                result = await agent.read_file(path)
                print(result)
                
            elif command == "write":
                if len(parts) < 3:
                    print("Usage: write <path> <content>")
                    continue
                    
                path = parts[1]
                content = parts[2]
                result = await agent.write_file(path, content)
                print(result)
                
            elif command == "search":
                if len(parts) < 2:
                    print("Usage: search <query>")
                    continue
                    
                query = parts[1]
                result = await agent.search_file(query)
                print(result)
                
            elif command in ["clear", "reset"]:
                agent.clear_history()
                print("Agent memory cleared.")
                
            # Analyze command - uses LM Studio LLM
            elif command == "analyze":
                if len(parts) < 2:
                    print("Usage: analyze <path> [analysis.md]")
                    continue
                    
                path = parts[1]
                output_file = parts[2] if len(parts) > 2 else None
                result = await agent.process_query(f"analyze {path}")
                
                if output_file:
                    with open(output_file, "w", encoding="utf-8") as f:
                        f.write(f"# Analysis of {path}\n\n")
                        f.write(result)
                    print(f"Analysis written to {output_file}")
                else:
                    print(result)
                
            # Plan command - generates coding plan from analysis file
            elif command == "plan":
                if len(parts) < 3:
                    print("Usage: plan <analysis.md> <plan.md>")
                    continue
                    
                analysis_file = parts[1]
                plan_file = parts[2]
                
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        analysis_content = f.read()
                except FileNotFoundError:
                    print(f"Error: File not found: {analysis_file}")
                    continue
                
                messages = [
                    {"role": "system", "content": "You are an expert software architect. Based on the code analysis provided, create a detailed coding plan with specific implementation steps, prioritized by impact and dependencies."},
                    {"role": "user", "content": f"Create a coding plan based on this analysis:\n\n{analysis_content}"}
                ]
                plan = await agent.llm.chat(messages)
                
                with open(plan_file, "w", encoding="utf-8") as f:
                    f.write(f"# Coding Plan\n\n")
                    f.write(plan)
                print(f"Coding plan written to {plan_file}")
                
            # Entities command - generates entities.md from analysis and plan
            elif command == "entities":
                if len(parts) < 3:
                    print("Usage: entities <analysis.md> <plan.md> [entities.md]")
                    continue
                    
                analysis_file = parts[1]
                plan_file = parts[2]
                entities_file = parts[3] if len(parts) > 3 else "entities.md"
                
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        analysis_content = f.read()
                    with open(plan_file, "r", encoding="utf-8") as f:
                        plan_content = f.read()
                except FileNotFoundError as e:
                    print(f"Error: File not found: {e}")
                    continue
                
                messages = [
                    {"role": "system", "content": "You are an expert Python developer. Create a comprehensive entities.md file that defines shared data structures, classes, and types that should be imported across multiple Python files. Include class definitions with attributes, types, and clear docstrings."},
                    {"role": "user", "content": f"Extract and define all shared entities from this analysis and plan:\n\n## Analysis:\n{analysis_content}\n\n## Plan:\n{plan_content}\n\nCreate an entities.md file with Python-ready entity definitions that can be centralized in an entities.py file for import across the project."}
                ]
                entities = await agent.llm.chat(messages)
                
                with open(entities_file, "w", encoding="utf-8") as f:
                    f.write(f"# Shared Entities\n\n")
                    f.write(entities)
                print(f"Entities written to {entities_file}")
                
            # Taskplan command - generates implementation task plan
            elif command == "taskplan":
                if len(parts) < 3:
                    print("Usage: taskplan <analysis.md> <plan.md> [tasks.md]")
                    continue
                    
                analysis_file = parts[1]
                plan_file = parts[2]
                tasks_file = parts[3] if len(parts) > 3 else "tasks.md"
                
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        analysis_content = f.read()
                    with open(plan_file, "r", encoding="utf-8") as f:
                        plan_content = f.read()
                except FileNotFoundError as e:
                    print(f"Error: File not found: {e}")
                    continue
                
                entities_content = ""
                entities_py = os.path.join(os.path.dirname(analysis_file), "entities.py")
                if os.path.exists(entities_py):
                    with open(entities_py, "r", encoding="utf-8") as f:
                        entities_content = f.read()
                
                messages = [
                    {"role": "system", "content": "You are an expert project manager. Create a detailed task plan for implementing code changes. Break down work into concrete, actionable tasks with clear descriptions. Include task dependencies and priority."},
                    {"role": "user", "content": f"Create a task implementation plan from this analysis and plan:\n\n## Analysis:\n{analysis_content}\n\n## Plan:\n{plan_content}\n\n## Existing entities.py:\n{entities_content if entities_content else 'No entities.py found'}\n\nGenerate a tasks.md file with specific implementation tasks, organized by file, with clear steps for new and existing files. Ensure tasks respect the entity definitions in entities.py."}
                ]
                tasks = await agent.llm.chat(messages)
                
                with open(tasks_file, "w", encoding="utf-8") as f:
                    f.write(f"# Implementation Tasks\n\n")
                    f.write(tasks)
                print(f"Tasks written to {tasks_file}")
                
            # Implement command - implements all files from taskplan
            elif command == "implement":
                if len(parts) < 2:
                    print("Usage: implement <taskplan.md> [analysis.md] [plan.md] [entities.md] [--keep] [--refresh] [--force] [--fix] [--workspace <path>]")
                    continue

                keep_mode = "--keep" in parts
                refresh_cache = "--refresh" in parts
                force_mode = "--force" in parts
                fix_mode = "--fix" in parts
                
                target_workspace = agent.workspace
                if "--workspace" in parts:
                    ws_idx = parts.index("--workspace")
                    if ws_idx + 1 < len(parts):
                        target_workspace = parts[ws_idx + 1]
                
                skip_tokens = ["--keep", "--refresh", "--force", "--fix", "--workspace", target_workspace]
                filtered_parts = [p for p in parts if p not in skip_tokens]
                
                taskplan_file = filtered_parts[1]
                analysis_file = filtered_parts[2] if len(filtered_parts) > 2 else "analysis.md"
                plan_file = filtered_parts[3] if len(filtered_parts) > 3 else "plan.md"
                entities_file = filtered_parts[4] if len(filtered_parts) > 4 else "entities.md"
                
                cache_file = os.path.join(os.path.dirname(os.path.realpath(taskplan_file)) if os.path.isabs(taskplan_file) else ".", ".implement_cache.json")
                if not os.path.isabs(taskplan_file):
                    cache_file = os.path.join(".", ".implement_cache.json")
                
                analysis_content = ""
                plan_content = ""
                entities_content = ""
                taskplan_content = ""
                
                try:
                    with open(taskplan_file, "r", encoding="utf-8") as f:
                        taskplan_content = f.read()
                except FileNotFoundError:
                    print(f"Error: File not found: {taskplan_file}")
                    continue
                
                try:
                    with open(analysis_file, "r", encoding="utf-8") as f:
                        analysis_content = f.read()
                except FileNotFoundError:
                    print(f"Warning: {analysis_file} not found")
                    
                try:
                    with open(plan_file, "r", encoding="utf-8") as f:
                        plan_content = f.read()
                except FileNotFoundError:
                    print(f"Warning: {plan_file} not found")
                    
                try:
                    with open(entities_file, "r", encoding="utf-8") as f:
                        entities_content = f.read()
                except FileNotFoundError:
                    print(f"Warning: {entities_file} not found")
                
                all_files = None
                
                if keep_mode and not refresh_cache and os.path.exists(cache_file):
                    try:
                        with open(cache_file, "r", encoding="utf-8") as f:
                            cache_data = json.load(f)
                        if cache_data.get("taskplan") == taskplan_file:
                            all_files = cache_data.get("files", [])
                            print(f"Using cached file list ({len(all_files)} files): {', '.join(all_files)}")
                    except Exception:
                        pass
                
                if all_files is None:
                    
                    print("Analyzing task plan to identify all files...")
                    
                    list_messages = [
                        {"role": "system", "content": "You are an expert Python developer. List ALL files that need to be implemented from the task plan. Reply with ONLY a list of filenames, one per line, with the file path. Include .py, .json, .yaml, .yml, .env, .md files. No explanations, just the list."},
                        {"role": "user", "content": f"List every file that needs to be created or modified from this task plan:\n\n## Task Plan:\n{taskplan_content}\n\n## Analysis:\n{analysis_content if analysis_content else 'N/A'}\n\n## Plan:\n{plan_content if plan_content else 'N/A'}\n\n## Entities:\n{entities_content if entities_content else 'N/A'}"}
                    ]
                    
                    file_list_response = await agent.llm.chat(list_messages)
                    
                    if not file_list_response or file_list_response.startswith("[") or "error" in file_list_response.lower():
                        print(f"Error: LM Studio API not responding or returned an error: {file_list_response}")
                        print("Please ensure LM Studio is running and try again.")
                        continue
                        
                    file_lines = [line.strip() for line in file_list_response.strip().split('\n') if line.strip() and not line.startswith('#')]
                    all_files = [f for f in file_lines if f.endswith(('.py', '.json', '.yaml', '.yml', '.env', '.md', '.txt', '.cfg', '.ini', '.toml'))]
                    
                    if not all_files:
                        all_files = re.findall(r'`([^`]+\.(?:py|json|yaml|yml|env|txt|cfg|ini|toml))`', file_list_response)
                    
                    cache_data = {"taskplan": taskplan_file, "files": all_files}
                    try:
                        with open(cache_file, "w", encoding="utf-8") as f:
                            json.dump(cache_data, f)
                        print(f"Cached file list to {cache_file}")
                    except Exception:
                        pass
                
                print(f"Found {len(all_files)} files to implement: {', '.join(all_files)}")
                
                protected_files = set()
                if os.path.exists(".protected"):
                    with open(".protected", "r", encoding="utf-8") as pf:
                        for line in pf:
                            line = line.strip()
                            if line and not line.startswith("#"):
                                protected_files.add(line)
                    print(f"Protected files: {protected_files}")
                
                analyzed_file = ""
                match = re.search(r'# Analysis of (\S+)', analysis_content)
                if match:
                    analyzed_file = match.group(1)
                    print(f"Analyzed file from analysis.md: {analyzed_file}")
                
                def file_needs_generation(fname):
                    from pathlib import Path
                    raw_ws = target_workspace
                    if raw_ws.startswith('/c/') or raw_ws.startswith('/C/'):
                        raw_ws = 'C:' + raw_ws[2:]
                    fpath = Path(raw_ws) / fname
                    if not fpath.exists():
                        return True, "not found"
                    if fpath.stat().st_size == 0:
                        return True, "empty"
                    if fname.endswith(".py"):
                        result = subprocess.run(
                            ["python", "-m", "py_compile", os.path.realpath(fpath)],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode != 0:
                            return True, f"compile failed: {result.stderr.strip()}"
                        return False, "OK"
                    return False, "OK"
                
                implemented = []
                if keep_mode and not force_mode:
                    files_to_skip = []
                    for fname in all_files:
                        is_analyzed = analyzed_file and fname == analyzed_file
                        if is_analyzed:
                            files_to_skip.append(f"{fname}: force-regenerate (was analyzed)")
                            continue
                        needs_gen, reason = file_needs_generation(fname)
                        if needs_gen:
                            files_to_skip.append(f"{fname}: {reason}")
                        else:
                            files_to_skip.append(f"{fname}: already exists, compile OK")
                            if fname not in implemented:
                                implemented.append(fname)
                    
                    print(f"\nFiles to skip (already exist and compile): {len(files_to_skip)}")
                    for f in files_to_skip:
                        print(f"  - {f}")
                    
                    files_to_generate = [fname for fname in all_files if fname not in implemented]
                    print(f"\nFiles to generate: {len(files_to_generate)}: {', '.join(files_to_generate)}")
                    
                    if not files_to_generate:
                        print("All files already exist and compile. Nothing to do.")
                        if not fix_mode:
                            continue
                        print("\n[fix] Running validation on existing files...")
                        implemented = [f for f in all_files if f.endswith(".py")]
                        all_files = implemented
                        # Continue to fix logic below
                    
                    all_files = files_to_generate
                
                errors = []
                batch_size = 1
                
                for i in range(0, len(all_files), batch_size):
                    batch = all_files[i:i+batch_size]
                    print(f"\nImplementing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}: {batch}")
                    
                    batch_files_md = "\n".join([f"- {f}" for f in batch])
                    
                    impl_messages = [
                        {"role": "system", "content": "You are an expert Python developer. Implement or update the specified files.\n\nFor NEW files: create complete code.\nFor EXISTING files (given below): ONLY add necessary imports at the top and replace old logic with calls to the new modules. Keep all existing code that doesn't need changes. Do NOT rewrite from scratch.\n\nFormat each file as:\n[FILE: filename.py]\n```python\n# code\n```"},
                        {"role": "user", "content": f"Files to implement:\n{batch_files_md}\n\n## Task Plan:\n{taskplan_content}\n\n## Analysis:\n{analysis_content if analysis_content else 'N/A'}\n\nImplement or update these files. For existing files, ONLY add imports and replace old implementations with new module calls."}
                    ]
                    
                    impl_response = None
                    for attempt in range(3):
                        try:
                            impl_response = await agent.llm.chat(impl_messages)
                            if impl_response and not impl_response.startswith("[Error:"):
                                break
                            print(f"  Attempt {attempt + 1} failed, retrying...")
                        except Exception as e:
                            print(f"  Attempt {attempt + 1} error: {e}, retrying...")
                            if attempt == 2:
                                impl_response = None
                    
                    if not impl_response or impl_response.startswith("[Error:"):
                        print(f"  Failed after 3 attempts, skipping batch")
                        continue
                    
                    patterns = [
                        r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```\s*$',
                        r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```',
                        r'\[FILE:\s*([^\]]+)\]\s*\n+(.*?)(?=\[FILE:|$)',
                    ]
                    
                    matches = []
                    for pattern in patterns:
                        matches = list(re.findall(pattern, impl_response, re.DOTALL))
                        if matches:
                            print(f"  Parsed {len(matches)} files using pattern")
                            break
                    
                    if not matches:
                        print(f"  Warning: Could not parse files from batch response")
                        print(f"  Raw response: {impl_response}")
                        continue
                    
                    for filename, content in matches:
                        content = content.strip()
                        raw_workspace = target_workspace
                        if raw_workspace.startswith('/c/') or raw_workspace.startswith('/C/'):
                            raw_workspace = 'C:' + raw_workspace[2:]
                        workspace = Path(raw_workspace)
                        filepath = workspace / filename
                        
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        
                        skip_reason = None
                        
                        is_analyzed_file = analyzed_file and filename == analyzed_file
                        
                        if not force_mode and not is_analyzed_file and filepath.exists() and filepath.stat().st_size > 0:
                            if filename.endswith(".py"):
                                filepath_str = os.path.realpath(filepath)
                                result = subprocess.run(
                                    ["python", "-m", "py_compile", filepath_str],
                                    capture_output=True,
                                    text=True
                                )
                                if result.returncode == 0:
                                    skip_reason = "Already exists and compiles OK"
                        
                        if skip_reason:
                            print(f"  Skipping {filename}: {skip_reason}")
                            if filename not in implemented:
                                implemented.append(filename)
                            continue
                        
                        if filename in protected_files:
                            print(f"  Protected: {filename} (skipped)")
                            if filename not in implemented:
                                implemented.append(filename)
                            continue
                        
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)
                        implemented.append(filename)
                        print(f"  Written: {filename}")
                        
                        if filename.endswith(".py"):
                            filepath_str = os.path.realpath(filepath)
                            
                            is_truncated = False
                            stripped = content.rstrip()
                            if stripped:
                                last_line = stripped.splitlines()[-1].strip()
                                # Heuristics for truncation: ends mid-expression
                                if last_line.endswith(('(', '[', '{', ':', ',', '+', '-', '*', '/', '=', '\\')):
                                    is_truncated = True
                                elif not last_line.endswith((')', '}', ']', "'", '"', 'pass', '...', 'True', 'False', 'None')):
                                    # Might be truncated mid-identifier or mid-string
                                    if not last_line.endswith(('.py', '.md', '.json', '.yaml')) and not last_line.startswith(('#', '//', '"""')):
                                        if last_line and last_line[-1].isalnum():
                                            is_truncated = True
                            
                            if is_truncated:
                                print(f"  WARNING: {filename} appears truncated, re-requesting...")
                                retry_msgs = [
                                    {"role": "system", "content": "Generate the COMPLETE and FULL code for this single file. Output as:\n[FILE: filename.py]\n```python\n# complete code here\n```\n\nIMPORTANT: Generate the ENTIRE file. Do not truncate. Every function, class, and method must be complete."},
                                    {"role": "user", "content": f"Generate the complete code for {filename}. The previous generation was truncated."}
                                ]
                                retry_content = await agent.llm.chat(retry_msgs)
                                if not retry_content.startswith("[Error"):
                                    match = re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', retry_content, re.DOTALL)
                                    if match:
                                        new_content = match.group(2).strip()
                                        if len(new_content) > len(content) * 0.8:
                                            with open(filepath, "w", encoding="utf-8") as f:
                                                f.write(new_content)
                                            print(f"  Re-written: {filename} ({len(new_content)} bytes)")
                                            content = new_content
                            
                            result = subprocess.run(
                                ["python", "-m", "py_compile", filepath_str],
                                capture_output=True,
                                text=True
                            )
                            if result.returncode != 0:
                                errors.append(f"{filename}: {result.stderr}")
                                print(f"  Compile error in {filename}")
                            else:
                                print(f"  Compiled OK: {filename}")
                
                print(f"\n{'='*50}")
                print(f"Implementation complete: {len(implemented)}/{len(all_files)} files")
                
                if implemented:
                    print(f"\nImplemented files:")
                    for f in implemented:
                        print(f"  - {f}")
                
                if errors:
                    print(f"\nCompilation errors ({len(errors)}):")
                    for err in errors:
                        print(f"  - {err}")
                else:
                    print("\nAll Python files compiled successfully!")
                
                missing_files = set(all_files) - set(implemented)
                if missing_files:
                    print(f"\nMissing files ({len(missing_files)}):")
                    for f in missing_files:
                        print(f"  - {f}")
                
                if fix_mode and implemented:
                    print(f"\n{'='*50}")
                    print(f"[fix] Deep validation: checking imports + class instantiation...")
                    print(f"{'='*50}")
                    
                    for fix_attempt in range(3):
                        errors_found = []
                        
                        for fname in implemented:
                            if not fname.endswith(".py"):
                                continue
                            ws = target_workspace
                            if ws.startswith("/c/") or ws.startswith("/C/"):
                                ws = "C:" + ws[2:]
                            fp = Path(ws) / fname
                            fpath_str = os.path.realpath(fp)
                            
                            # Step 1: py_compile
                            r = subprocess.run(["python", "-m", "py_compile", fpath_str], capture_output=True, text=True)
                            if r.returncode != 0:
                                errors_found.append((fname, fpath_str, f"COMPILE: {r.stderr.strip()}"))
                                continue
                            
                            # Step 2: import test
                            import tempfile
                            mod_name = fname[:-3].replace('\\', '.').replace('/', '.')
                            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
                                tf.write(f"import sys; sys.path.insert(0, r'{ws}')\n")
                                tf.write(f"import {mod_name}\n")
                                tf.write("print('OK')\n")
                                tfpath = tf.name
                            r = subprocess.run(["python", tfpath], capture_output=True, text=True, cwd=str(Path(ws)))
                            os.unlink(tfpath)
                            if r.returncode != 0:
                                errors_found.append((fname, fpath_str, f"IMPORT: {r.stderr.strip()}"))
                                continue
                            
                            # Step 3: try to instantiate every class defined in the file
                            with open(fpath_str, "r", encoding="utf-8") as f:
                                source = f.read()
                            class_names = re.findall(r'^class\s+(\w+)', source, re.MULTILINE)
                            for cn in class_names:
                                import tempfile
                                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tf:
                                    tf.write(f"import sys; sys.path.insert(0, r'{ws}')\n")
                                    tf.write(f"import {mod_name}\n")
                                    tf.write(f"c={mod_name}.{cn}\n")
                                    tf.write(f"import inspect\n")
                                    tf.write(f"try:\n    sig=inspect.signature(c)\n    print(f'OK: {cn}'+str(list(sig.parameters.keys())))\n")
                                    tf.write(f"except (ValueError, TypeError):\n    print(f'OK: {cn} (builtin/Protocol/TypedDict)')\n")
                                    tfpath = tf.name
                                r = subprocess.run(["python", tfpath], capture_output=True, text=True, cwd=str(Path(ws)))
                                os.unlink(tfpath)
                                if r.returncode != 0:
                                    errors_found.append((fname, fpath_str, f"CLASS {cn}: {r.stderr.strip()}"))
                                else:
                                    print(f"  {fname}: {r.stdout.strip()}")
                        
                        if not errors_found:
                            print("\n[fix] All files pass deep validation (compile + import + class instantiation)!")
                            break
                        
                        # Find root cause: file with COMPILE error (IMPORT errors are cascade)
                        err_root = errors_found[0][0]
                        for fname, _, err in errors_found:
                            if "COMPILE:" in err:
                                err_root = fname
                                break
                        print(f"\n[fix] Root error file: {err_root} (fixing this first)")
                        
                        print(f"\n[fix] Attempt {fix_attempt + 1}: {len(errors_found)} errors")
                        for fname, fpath, err in errors_found:
                            print(f"  - {fname}:")
                            print(f"    {err}")
                        
                        if fix_attempt >= 2:
                            print("[fix] Max attempts reached.")
                            break
                        
                        for fname, fpath, err in errors_found:
                            print(f"\n[fix] Fixing {fname}...")
                            with open(fpath, "r", encoding="utf-8") as f:
                                current_code = f.read()
                            
                            fix_msgs = [
                                {"role": "system", "content": "Fix the error. Output ONLY the corrected file. Start with [FILE: filename.py] immediately. No explanations."},
                                {"role": "user", "content": f"Error in {fname}:\n{err}\n\nCurrent code:\n```python\n{current_code}\n```\n\nOutput the fixed file."}
                            ]
                            fixed = await agent.llm.chat(fix_msgs)
                            if fixed.startswith("[Error") or fixed.startswith("[LM Studio"):
                                print(f"  LLM error: {fixed}")
                                continue
                            
                            match = re.search(r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)\n```', fixed, re.DOTALL)
                            if match:
                                new_code = match.group(2).strip()
                                # Guard: must be actual Python code
                                if not re.search(r'\b(import|def |class )\b', new_code):
                                    print(f"  WARNING: Fix for {fname} is not valid Python code, skipping")
                                    continue
                                # Guard: must not be cascade error from another file
                                if fname != err_root and len(new_code) < 100:
                                    print(f"  Skipping cascade fix for {fname} (root issue is in {err_root})")
                                    continue
                                # Guard: new content must be at least 10% of original (don't replace with stub)
                                if len(new_code) < len(current_code) * 0.1:
                                    print(f"  WARNING: Fix is {len(new_code)} bytes vs original {len(current_code)} bytes, skipping")
                                    continue
                                with open(fpath, "w", encoding="utf-8") as f:
                                    f.write(new_code)
                                print(f"  Fixed: {fname} ({len(new_code)} bytes)")
                    
                    print(f"\n[fix] Complete")
                        
            elif command == "workflow":
                if len(parts) < 2:
                    print("Usage: workflow <target> [--from <spec.md>] [--force] [--workspace <path>]")
                    print("  target: .  |  --desc/-from spec | --features spec (file or inline)")
                    continue
                
                force = "--force" in parts
                
                spec_file = None
                greenfield = False
                features_file = None
                
                import tempfile
                
                if "--desc" in parts:
                    di = parts.index("--desc")
                    end = di + 1
                    while end < len(parts) and not parts[end].startswith("--"):
                        end += 1
                    desc_text = " ".join(parts[di + 1:end])
                    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
                    tmp.write(f"# Project Specification\n\n{desc_text}")
                    tmp.close()
                    spec_file = tmp.name
                    greenfield = True
                    print(f"\n[desc] {desc_text}")
                elif "--from" in parts:
                    fi = parts.index("--from")
                    end = fi + 1
                    while end < len(parts) and not parts[end].startswith("--"):
                        end += 1
                    spec_file = " ".join(parts[fi + 1:end])
                    greenfield = True
                
                if "--features" in parts:
                    fi = parts.index("--features")
                    end = fi + 1
                    while end < len(parts) and not parts[end].startswith("--"):
                        end += 1
                    feat_val = " ".join(parts[fi + 1:end])
                    if os.path.isfile(feat_val):
                        features_file = feat_val
                    else:
                        import tempfile
                        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8")
                        tmp.write(f"# Feature Requirements\n\n{feat_val}")
                        tmp.close()
                        features_file = tmp.name
                        print(f"\n[features] {feat_val}")
                
                target = [p for p in parts if not p.startswith("--") and p not in [spec_file, features_file]][1]
                
                target_workspace = agent.workspace
                if "--workspace" in parts:
                    ws_idx = parts.index("--workspace")
                    if ws_idx + 1 < len(parts):
                        target_workspace = parts[ws_idx + 1]
                
                if spec_file and not os.path.isabs(spec_file):
                    spec_file = os.path.join(target_workspace, spec_file)
                if spec_file and (spec_file.startswith("/c/") or spec_file.startswith("/C/")):
                    spec_file = "C:" + spec_file[2:]
                
                if features_file and not os.path.isabs(features_file):
                    features_file = os.path.join(target_workspace, features_file)
                if features_file and (features_file.startswith("/c/") or features_file.startswith("/C/")):
                    features_file = "C:" + features_file[2:]
                
                ws = target_workspace
                if ws.startswith("/c/") or ws.startswith("/C/"):
                    ws = "C:" + ws[2:]
                ws_path = Path(ws)
                ws_path.mkdir(parents=True, exist_ok=True)
                print(f"Workspace: {ws_path}")
                
                analysis_md = str(ws_path / "project_analysis.md")
                plan_md = str(ws_path / "project_plan.md")
                entities_md = str(ws_path / "project_entities.md")
                tasks_md = str(ws_path / "project_tasks.md")
                
                def step_ok(result):
                    return not (result.startswith("[Error") or result.startswith("[LM Studio"))
                
                if spec_file:
                    # --- GREENFIELD from specification ---
                    with open(spec_file, "r", encoding="utf-8") as f:
                        spec_content = f.read()
                    print(f"\n[spec] Loaded from {spec_file}")
                    
                    if not force and os.path.exists(plan_md):
                        print(f"\n[Skipping plan] exists")
                    else:
                        print(f"\n[plan] Creating plan...")
                        r = await agent.llm.chat([
                            {"role": "system", "content": "You are an expert software architect. Create a detailed coding plan with ALL files needed."},
                            {"role": "user", "content": f"Create coding plan:\n\n{spec_content}"}
                        ])
                        if not step_ok(r): print(f"[plan] FAILED: {r[:200]}"); continue
                        with open(plan_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[plan] Written")
                    
                    if not force and os.path.exists(entities_md):
                        print(f"\n[Skipping entities] exists")
                    else:
                        with open(plan_md, "r", encoding="utf-8") as f: plan = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "Extract shared classes/types. Avoid circular imports."},
                            {"role": "user", "content": f"Extract entities:\n\n## Spec:\n{spec_content}\n\n## Plan:\n{plan}"}
                        ])
                        if not step_ok(r): print(f"[entities] FAILED: {r[:200]}"); continue
                        with open(entities_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[entities] Written")
                    
                    if not force and os.path.exists(tasks_md):
                        print(f"\n[Skipping taskplan] exists")
                    else:
                        with open(plan_md, "r", encoding="utf-8") as f: plan = f.read()
                        with open(entities_md, "r", encoding="utf-8") as f: entities = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "Create task plan with files in dependency order."},
                            {"role": "user", "content": f"Create task plan:\n\n## Spec:\n{spec_content}\n\n## Plan:\n{plan}\n\n## Entities:\n{entities}"}
                        ])
                        if not step_ok(r): print(f"[taskplan] FAILED: {r[:200]}"); continue
                        with open(tasks_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[taskplan] Written")
                    
                    print(f"\nNext: implement {tasks_md} {plan_md} {entities_md} --workspace {target_workspace} --force")
                    
                elif features_file:
                    # --- BROWNFIELD EXTENSION: analyze existing + add features ---
                    with open(features_file, "r", encoding="utf-8") as f:
                        features = f.read()
                    print(f"\n[features] Loaded from {features_file}")
                    
                    # Step 1: Analyze existing code
                    if not force and os.path.exists(analysis_md):
                        print(f"\n[Skipping analyze] exists")
                    else:
                        print(f"\n[analyze] Scanning existing py files...")
                        py_files = []
                        for dp, dn, filenames in os.walk(ws_path):
                            if ".git" in dp or "__pycache__" in dp: continue
                            for fn in filenames:
                                if fn.endswith(".py"): py_files.append(os.path.join(dp, fn))
                        if not py_files:
                            print("[analyze] No py files. Use --from for greenfield.")
                            continue
                        combined = ""
                        for pf in py_files:
                            try:
                                with open(pf, "r", encoding="utf-8") as f:
                                    combined += f"\n\n# ---- {pf} ----\n{f.read()}"
                            except: pass
                        r = await agent.llm.analyze_code(combined)
                        if not step_ok(r): print(f"[analyze] FAILED: {r[:200]}"); continue
                        with open(analysis_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[analyze] Written")
                    
                    # Step 2: Plan that extends existing code with new features
                    if not force and os.path.exists(plan_md):
                        print(f"\n[Skipping plan] exists")
                    else:
                        with open(analysis_md, "r", encoding="utf-8") as f: analysis = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "You are an expert software architect. Create a plan that extends the EXISTING codebase with these new features. Preserve existing architecture. Show what to ADD and what MINIMAL changes are needed in existing files."},
                            {"role": "user", "content": f"## Existing code analysis:\n{analysis}\n\n## New features to add:\n{features}\n\nCreate a plan that integrates these features into the existing codebase."}
                        ])
                        if not step_ok(r): print(f"[plan] FAILED: {r[:200]}"); continue
                        with open(plan_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[plan] Written")
                    
                    # Step 3: Extract new entities (preserve existing)
                    if not force and os.path.exists(entities_md):
                        print(f"\n[Skipping entities] exists")
                    else:
                        with open(plan_md, "r", encoding="utf-8") as f: plan = f.read()
                        with open(entities_file, "r") as f: entities_existing = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "Extract ONLY NEW shared entities needed for these features. Preserve existing entities. Avoid circular imports with existing code."},
                            {"role": "user", "content": f"## Plan:\n{plan}\n\n## Existing entities:\n{entities_existing}\n\nExtract only new entities needed."}
                        ])
                        if not step_ok(r): print(f"[entities] FAILED: {r[:200]}"); continue
                        with open(entities_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[entities] Written")
                    
                    # Step 4: Taskplan for adding features
                    if not force and os.path.exists(tasks_md):
                        print(f"\n[Skipping taskplan] exists")
                    else:
                        with open(analysis_md, "r", encoding="utf-8") as f: analysis = f.read()
                        with open(plan_md, "r", encoding="utf-8") as f: plan = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "Create task plan for ADDING these features to existing code. Mark existing files as 'modify' with only the necessary changes. New files as 'create'."},
                            {"role": "user", "content": f"## Analysis:\n{analysis}\n\n## Plan:\n{plan}\n\nCreate implementation tasks."}
                        ])
                        if not step_ok(r): print(f"[taskplan] FAILED: {r[:200]}"); continue
                        with open(tasks_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[taskplan] Written")
                    
                    print(f"\nNext: implement {tasks_md} {analysis_md} {plan_md} {entities_md} --workspace {target_workspace} --keep")
                    
                else:
                    # --- BROWNFIELD from existing code ---
                    if not force and os.path.exists(analysis_md):
                        print(f"\n[Skipping analyze] exists")
                    else:
                        print(f"\n[analyze] Scanning py files...")
                        py_files = []
                        for dp, dn, filenames in os.walk(ws_path):
                            if ".git" in dp or "__pycache__" in dp: continue
                            for fn in filenames:
                                if fn.endswith(".py"): py_files.append(os.path.join(dp, fn))
                        if not py_files:
                            print("[analyze] No py files. Use --from spec.md for greenfield.")
                            continue
                        combined = ""
                        for pf in py_files:
                            try:
                                with open(pf, "r", encoding="utf-8") as f:
                                    combined += f"\n\n# ---- {pf} ----\n{f.read()}"
                            except: pass
                        r = await agent.llm.analyze_code(combined)
                        if not step_ok(r): print(f"[analyze] FAILED: {r[:200]}"); continue
                        with open(analysis_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[analyze] Written")
                    
                    if not force and os.path.exists(plan_md):
                        print(f"\n[Skipping plan] exists")
                    else:
                        with open(analysis_md, "r", encoding="utf-8") as f: analysis = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "Create a detailed coding plan."},
                            {"role": "user", "content": f"Create plan:\n\n{analysis}"}
                        ])
                        if not step_ok(r): print(f"[plan] FAILED: {r[:200]}"); continue
                        with open(plan_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[plan] Written")
                    
                    if not force and os.path.exists(entities_md):
                        print(f"\n[Skipping entities] exists")
                    else:
                        with open(analysis_md, "r", encoding="utf-8") as f: analysis = f.read()
                        with open(plan_md, "r", encoding="utf-8") as f: plan = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "Extract shared entities. Avoid circular imports."},
                            {"role": "user", "content": f"Extract entities:\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}"}
                        ])
                        if not step_ok(r): print(f"[entities] FAILED: {r[:200]}"); continue
                        with open(entities_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[entities] Written")
                    
                    if not force and os.path.exists(tasks_md):
                        print(f"\n[Skipping taskplan] exists")
                    else:
                        with open(analysis_md, "r", encoding="utf-8") as f: analysis = f.read()
                        with open(plan_md, "r", encoding="utf-8") as f: plan = f.read()
                        r = await agent.llm.chat([
                            {"role": "system", "content": "Create task plan with files in dependency order."},
                            {"role": "user", "content": f"Create task plan:\n\n## Analysis:\n{analysis}\n\n## Plan:\n{plan}"}
                        ])
                        if not step_ok(r): print(f"[taskplan] FAILED: {r[:200]}"); continue
                        with open(tasks_md, "w", encoding="utf-8") as f: f.write(r)
                        print(f"[taskplan] Written")
                    
                    print(f"\nNext: implement {tasks_md} {analysis_md} {plan_md} {entities_md} --workspace {target_workspace} --keep")
                        
            else:
                # Try natural language processing with LLM
                tool_action, args = agent._parse_natural_language(user_input)
                
                if tool_action == "unknown":
                    # Use LM Studio for general conversation
                    result = await agent.llm.chat([{"role": "user", "content": user_input}])
                    print(result)
                else:
                    result = await agent.execute_tool(tool_action, args)
                    print(result)
                    
        except KeyboardInterrupt:
            print("\nInterrupted. Use 'quit' to exit.")
        except EOFError:
            break


async def main():
    """Main entry point - runs interactive mode."""
    await run_interactive()


if __name__ == "__main__":
    asyncio.run(main())
