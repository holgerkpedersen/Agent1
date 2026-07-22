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
            "max_tokens": 8192
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

            with urllib.request.urlopen(req, timeout=600) as response:
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


async def run_interactive():
    """Interactive mode - allows user to input commands."""
    
    print("=" * 50)
    print("Agent Interactive Mode with LM Studio")
    print(f"Workspace: {Agent.DEFAULT_WORKSPACE}")
    print("Commands:")
    print("  read <path>        - Read a file")
    print("  write <path> <content> - Write content to file")
    print("  search <query>     - Search for string in files")
    print("  analyze <file> [output.md] - AI analysis via LM Studio")
    print("  plan <output.md> <plan.md> - Generate coding plan from analysis")
    print("  entities <analysis.md> <plan.md> [entities.md] - Generate shared entities")
    print("  taskplan <analysis.md> <plan.md> [tasks.md] - Generate implementation tasks")
    print("  implement <taskplan.md> [analysis.md] [plan.md] [entities.md] - Implement all files")
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
            parts = user_input.split(maxsplit=5)
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
                    print("Usage: analyze <path> [output.md]")
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
                    print("Usage: plan <output.md> <plan.md>")
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
                    print("Usage: implement <taskplan.md> [analysis.md] [plan.md] [entities.md]")
                    continue
                    
                taskplan_file = parts[1]
                analysis_file = parts[2] if len(parts) > 2 else "analysis.md"
                plan_file = parts[3] if len(parts) > 3 else "plan.md"
                entities_file = parts[4] if len(parts) > 4 else "entities.md"
                
                try:
                    with open(taskplan_file, "r", encoding="utf-8") as f:
                        taskplan_content = f.read()
                except FileNotFoundError:
                    print(f"Error: File not found: {taskplan_file}")
                    continue
                
                analysis_content = ""
                plan_content = ""
                entities_content = ""
                
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
                    
                import re
                file_lines = [line.strip() for line in file_list_response.strip().split('\n') if line.strip() and not line.startswith('#')]
                all_files = [f for f in file_lines if f.endswith(('.py', '.json', '.yaml', '.yml', '.env', '.md', '.txt', '.cfg', '.ini', '.toml'))]
                
                if not all_files:
                    all_files = re.findall(r'`([^`]+\.(?:py|json|yaml|yml|env|txt|cfg|ini|toml))`', file_list_response)
                
                print(f"Found {len(all_files)} files to implement: {', '.join(all_files)}")
                
                implemented = []
                errors = []
                batch_size = 2
                
                for i in range(0, len(all_files), batch_size):
                    batch = all_files[i:i+batch_size]
                    print(f"\nImplementing batch {i//batch_size + 1}/{(len(all_files) + batch_size - 1)//batch_size}: {batch}")
                    
                    batch_files_md = "\n".join([f"- {f}" for f in batch])
                    
                    impl_messages = [
                        {"role": "system", "content": "You are an expert Python developer. Implement the specified files completely. For each Python file, output the complete, working code. Format each file as:\n\n[FILE: filename.py]\n```python\n# complete code\n```\n\nFor config files use:\n[FILE: config.json]\n```json\n# config here\n```\n\nFor yaml files:\n[FILE: config.yaml]\n```yaml\n# yaml content\n```\n\nEnsure all imports are correct, code is complete, and follows best practices."},
                        {"role": "user", "content": f"Implement these files:\n{batch_files_md}\n\n## Task Plan:\n{taskplan_content}\n\n## Analysis:\n{analysis_content if analysis_content else 'N/A'}\n\n## Plan:\n{plan_content if plan_content else 'N/A'}\n\n## Entities:\n{entities_content if entities_content else 'N/A'}\n\nImplement ALL files in this batch with complete, working code."}
                    ]
                    
                    impl_response = await agent.llm.chat(impl_messages)
                    
                    patterns = [
                        r'\[FILE:\s*([^\]]+)\]\s*\n*(?:```\w*\n)?(.*?)```',
                        r'\[FILE:\s*([^\]]+)\]\s*\n+(.*?)(?=\[FILE:|$)',
                        r'File:\s*([^\.]+\.\w+)\s*\n+```\w*\n?(.*?)```',
                    ]
                    
                    matches = []
                    for pattern in patterns:
                        matches = list(re.findall(pattern, impl_response, re.DOTALL))
                        if matches:
                            print(f"  Parsed {len(matches)} files using pattern")
                            break
                    
                    if not matches:
                        print(f"  Warning: Could not parse files from batch response")
                        print(f"  Raw response (first 500 chars): {impl_response[:500]}")
                        continue
                    
                    for filename, content in matches:
                        content = content.strip()
                        from pathlib import Path
                        raw_workspace = agent.workspace
                        if raw_workspace.startswith('/c/') or raw_workspace.startswith('/C/'):
                            raw_workspace = 'C:' + raw_workspace[2:]
                        workspace = Path(raw_workspace)
                        filepath = workspace / filename
                        
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(filepath, "w", encoding="utf-8") as f:
                            f.write(content)
                        implemented.append(filename)
                        print(f"  Written: {filename}")
                        
                        if filename.endswith(".py"):
                            filepath_str = os.path.realpath(filepath)
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
