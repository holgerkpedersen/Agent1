import asyncio
import json
import logging
import os
import urllib.error
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")

TOOL_DEFINITIONS = {
    "read_file": {
        "description": "Read the contents of a file at the given path",
        "parameters": {
            "path": {"type": "string", "description": "File path, e.g. /c/Dev/Agent1/agent.py", "required": True}
        },
        "example_args": {"path": "/c/Dev/Agent1/agent.py"}
    },
    "apply_patch": {
        "description": "Apply a find-and-replace patch to a file. Find text must match exactly once.",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file to patch", "required": True},
            "find": {"type": "string", "description": "Exact text to find", "required": True},
            "replace": {"type": "string", "description": "Replacement text", "required": True}
        },
        "example_args": {"path": "/c/Dev/Agent1/agent.py", "find": "old text", "replace": "new text"}
    },
    "edit_file": {
        "description": "Create or overwrite a file with new content",
        "parameters": {
            "path": {"type": "string", "description": "Path to the file to write", "required": True},
            "content": {"type": "string", "description": "Full file content", "required": True}
        },
        "example_args": {"path": "/c/Dev/Agent1/agent.py", "content": "full file"}
    },
    "search_file": {
        "description": "Search for text in files within a directory",
        "parameters": {
            "query": {"type": "string", "description": "Text or pattern to search for", "required": True},
            "path": {"type": "string", "description": "Directory to search, default is current directory", "required": False}
        },
        "example_args": {"query": "search term", "path": "/c/Dev/Agent1"}
    }
}

def _build_system_prompt() -> str:
    json_examples = "\n".join(
        json.dumps({"tool": name, "args": d["example_args"]})
        for name, d in TOOL_DEFINITIONS.items()
    )
    return f"""You are a coding agent. Think, then output a JSON tool call.

{json_examples}

Always read a file before editing it.
Paths use /c/Dev/file.txt format"""

def _build_tool_schemas() -> list[dict]:
    schemas = []
    for name, d in TOOL_DEFINITIONS.items():
        properties = {}
        required = []
        for pname, pinfo in d["parameters"].items():
            properties[pname] = {"type": pinfo["type"], "description": pinfo["description"]}
            if pinfo.get("required", False):
                required.append(pname)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": d["description"],
                "parameters": {"type": "object", "properties": properties, "required": required}
            }
        })
    return schemas

SYSTEM_PROMPT = _build_system_prompt()
TOOL_SCHEMAS = _build_tool_schemas()


def _normalize_path(path: str) -> str:
    if path.startswith("/c/"):
        return "C:\\" + path[3:].replace("/", "\\")
    elif path.startswith("/d/"):
        return "D:\\" + path[3:].replace("/", "\\")
    elif path.startswith("/"):
        return "C:\\" + path[1:].replace("/", "\\")
    return path


def _safe_path(path: str) -> str:
    """Validate and normalize a file path, blocking path traversal attempts.

    Converts Unix-style paths (e.g., /c/Users/...) to Windows format,
    resolves the path to an absolute path, and checks for directory
    traversal attempts using '..' components.

    Args:
        path: The file path to validate. Can be in Unix /c/... format
              or Windows C:\\... format.

    Returns:
        The resolved absolute path in Windows format.

    Raises:
        ValueError: If the path is empty, invalid, or contains traversal
                    attempts (e.g., '..' components).
    """
    if not path or not isinstance(path, str):
        raise ValueError("Path must be a non-empty string")

    # Normalize to Windows format first for consistent checking
    normalized = _normalize_path(path)

    # Check for path traversal components before resolution
    if ".." in Path(normalized).parts:
        raise ValueError(f"Path traversal blocked: {path}")

    # Resolve to absolute path (handles relative paths and symlinks)
    try:
        resolved = Path(normalized).resolve()
    except (OSError, RuntimeError) as e:
        raise ValueError(f"Invalid path: {path}") from e

    return str(resolved)


class ToolRegistry:
    @staticmethod
    async def read_file(path: str) -> str:
        try:
            if not isinstance(path, str) or not path:
                return "Error: Invalid path"
            local_path = _safe_path(path)
            if not os.path.exists(local_path):
                return f"File not found: {path}"
            if not os.path.isfile(local_path):
                return f"Not a file: {path}"
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > 16000:
                return content[:16000] + "\n\n... [truncated]"
            return content
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            logger.error("read_file failed for %s: %s", path, e)
            return f"Error reading file: {e}"

    @staticmethod
    async def apply_patch(path: str, find: str, replace: str) -> str:
        try:
            if not all(isinstance(x, str) for x in [path, find, replace]):
                return "Error: All arguments must be strings"
            if not find:
                return "Error: find text cannot be empty"
            local_path = _safe_path(path)
            if not os.path.exists(local_path):
                return f"File not found: {path}"
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if find not in content:
                return f"ERROR: find text not found in {path}. Check whitespace and indentation."
            count = content.count(find)
            if count > 1:
                return f"ERROR: find text matches {count} locations. Add more context to make it unique."
            new_content = content.replace(find, replace, 1)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(new_content)
            logger.info("Patch applied to %s (%d -> %d bytes)", path, len(content), len(new_content))
            return f"Patch applied: {path} ({len(content)} -> {len(new_content)} bytes)"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            logger.error("apply_patch failed for %s: %s", path, e)
            return f"Error applying patch: {e}"

    @staticmethod
    async def edit_file(path: str, content: str) -> str:
        try:
            if not isinstance(path, str) or not path:
                return "Error: Invalid path"
            local_path = _safe_path(path)
            dir_name = os.path.dirname(local_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("File written: %s (%d bytes)", path, len(content))
            return f"File written: {path} ({len(content)} bytes)"
        except PermissionError:
            return f"Permission denied: {path}"
        except Exception as e:
            logger.error("edit_file failed for %s: %s", path, e)
            return f"Error writing file: {e}"

    @staticmethod
    async def search_file(query: str, path: str = ".") -> str:
        try:
            if not isinstance(query, str) or not query:
                return "Error: Invalid query"
            local_path = _safe_path(path) if path != "." else "."
            import platform
            if platform.system() == "Windows":
                search_pattern = local_path + "\\*.*"
                proc = await asyncio.create_subprocess_exec(
                    "findstr", "/S", "/N", "/C:" + query, search_pattern,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    "grep", "-rn", query, local_path,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode(errors="replace")
            if output.strip():
                return output.strip()[:8000]
            err = stderr.decode(errors="replace").strip()
            if err:
                return f"Search error: {err}"
            return f"No matches for '{query}' in {path}"
        except FileNotFoundError:
            return "Error: search command not available"
        except asyncio.TimeoutError:
            return "Error: search timed out"
        except Exception as e:
            logger.error("search_file failed for %s in %s: %s", query, path, e)
            return f"Error searching: {e}"

    TOOLS = list(TOOL_DEFINITIONS.keys())


class Agent:
    def __init__(self, name: str, max_history: int = 20):
        self.name = name
        self._history: list[dict] = []
        self._max_history = max_history


class LMStudioAgent(Agent):
    BASE_URL = "http://localhost:1234/v1/chat/completions"

    def __init__(self, name: str, model: str = "gpt-3.5-turbo", max_history: int = 20, checkpoint_dir: str | None = None):
        super().__init__(name, max_history)
        self.model = model
        self._cancelled = False
        self._last_tool_call = None
        self._last_reasoning = None
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def _send(self, messages: list[dict]) -> str:
        import urllib.request
        self._last_tool_call = None
        self._last_reasoning = None
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
            "stop": ["\n\n"],
        }).encode()
        headers = {"Content-Type": "application/json"}
        for attempt in range(3):
            req = urllib.request.Request(self.BASE_URL, data=payload, headers=headers, method="POST")
            try:
                resp = await asyncio.get_running_loop().run_in_executor(None, lambda r=req: urllib.request.urlopen(r, timeout=300))
                data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    self._last_tool_call = tool_calls[0]
                    return content or ""
                if content:
                    content = self._strip_thinking(content)
                    if not content and not reasoning:
                        return ""
                    xml_tool = self._parse_minimax_tool_call(content) if content else None
                    if xml_tool:
                        return json.dumps({"tool": xml_tool[0], "args": xml_tool[1]})
                if not content and reasoning:
                    self._last_reasoning = reasoning
                    extracted = self._parse_tool_call(reasoning)
                    if extracted:
                        tool_name, args = extracted
                        return json.dumps({"tool": tool_name, "args": args})
                    return ""
                return content
            except urllib.error.HTTPError as e:
                logger.warning("HTTP %d on attempt %d: %s", e.code, attempt + 1, e.reason)
                if attempt == 2:
                    return f"HTTP Error {e.code}: {e.reason}"
            except urllib.error.URLError as e:
                logger.warning("Connection error on attempt %d: %s", attempt + 1, e.reason)
                if attempt == 2:
                    return f"Connection error: {e.reason}"
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON response: %s", e)
                return "Error: Invalid response from model"
            except Exception as e:
                logger.error("Request failed on attempt %d: %s", attempt + 1, e)
                if attempt == 2:
                    return f"Error: {e}"
            await asyncio.sleep(2 ** attempt)

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        if tool_name not in ToolRegistry.TOOLS:
            return f"Unknown tool: {tool_name}. Available: {', '.join(ToolRegistry.TOOLS)}"
        handler = getattr(ToolRegistry, tool_name)
        try:
            return await handler(**args)
        except TypeError as e:
            return f"Invalid arguments for {tool_name}: {e}"
        except Exception as e:
            logger.error("Tool %s failed: %s", tool_name, e)
            return f"Error: {e}"

    def _parse_tool_call(self, text: str) -> tuple[str, dict] | None:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                return (obj["tool"], obj["args"])
        except json.JSONDecodeError:
            pass
        start = text.find('{')
        while start != -1:
            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == '\\' and in_string:
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_string = not in_string
                    continue
                if not in_string:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                obj = json.loads(text[start:i+1])
                                if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                                    return (obj["tool"], obj["args"])
                            except json.JSONDecodeError:
                                pass
                            start = text.find('{', i+1)
                            break
            else:
                break
        return None

    def _parse_natural_language(self, text: str) -> tuple[str, dict] | None:
        import re
        text_lower = text.lower()
        backtick_pattern = re.compile(r'`([^`]+\.(?:py|txt|json|md|js|ts|html|css))`')
        files_mentioned = backtick_pattern.findall(text)
        if not files_mentioned:
            file_pattern = re.compile(r'(?:^|\s)([a-zA-Z_][\w-]*\.(?:py|txt|json|md|js|ts|html|css))\b')
            files_mentioned = file_pattern.findall(text)
        if not files_mentioned:
            return None
        filename = files_mentioned[0]
        is_read = any(w in text_lower for w in ['read', 'see', 'show', 'view', 'check', 'open', 'cat', 'analyze', 'explain', 'describe', 'review'])
        is_search = any(w in text_lower for w in ['search', 'find', 'grep', 'look for'])
        is_edit = any(w in text_lower for w in ['edit', 'change', 'modify', 'update', 'fix', 'improve', 'add', 'remove', 'write', 'patch'])
        if is_search:
            query = filename.replace('.py', '').replace('_', ' ')
            for word in ['def ', 'class ', 'import ', 'async ']:
                if word in text_lower:
                    idx = text_lower.index(word)
                    end = text.find(' ', idx + len(word) + 5)
                    if end == -1:
                        end = len(text)
                    query = text[idx:end].strip().strip('`')
                    break
            return ("search_file", {"query": query, "path": "/c/Dev/Agent1"})
        if is_read or is_edit:
            return ("read_file", {"path": f"/c/Dev/Agent1/{filename}"})
        return ("read_file", {"path": f"/c/Dev/Agent1/{filename}"})

    @staticmethod
    def _strip_thinking(text: str) -> str:
        import re
        text = re.sub(r'<think>.*?</think>\s*', '', text, count=1, flags=re.DOTALL)
        return text.rstrip()

    @staticmethod
    def _parse_minimax_tool_call(text: str) -> tuple[str, dict] | None:
        if "<minimax:tool_call>" not in text:
            return None
        import re
        match = re.search(r"<minimax:tool_call>(.*?)</minimax:tool_call>", text, re.DOTALL)
        if not match:
            return None
        block = match.group(1)
        invoke_match = re.search(r'<invoke name="([^"]+)"\s*>(.*?)</invoke>', block, re.DOTALL)
        if not invoke_match:
            return None
        tool_name = invoke_match.group(1)
        args: dict = {}
        for m in re.finditer(r'<parameter name="([^"]+)"\s*>(.*?)</parameter>', invoke_match.group(2), re.DOTALL):
            args[m.group(1)] = m.group(2).strip()
        return (tool_name, args)

    def save_checkpoint(self):
        if not self._checkpoint_dir:
            return
        try:
            import re
            safe_name = re.sub(r'[^\w\-]', '_', self.name)
            path = self._checkpoint_dir / f"{safe_name}_checkpoint.json"
            path.write_text(json.dumps({"name": self.name, "model": self.model, "history": self._history}, indent=2))
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)

    def load_checkpoint(self) -> bool:
        if not self._checkpoint_dir:
            return False
        import re
        safe_name = re.sub(r'[^\w\-]', '_', self.name)
        path = self._checkpoint_dir / f"{safe_name}_checkpoint.json"
        if not path.exists():
            return False
        try:
            self._history = json.loads(path.read_text()).get("history", [])
            return True
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return False

    def cancel(self):
        self._cancelled = True

    def _summarize_response(self, text: str) -> str:
        text = text.strip()
        if len(text) <= 100:
            return text
        sentences = text.replace("\n", " ").split(". ")
        summary = sentences[0].strip()
        if len(sentences) > 1 and len(summary) < 80:
            summary += ". " + sentences[1].strip()
        if len(summary) > 120:
            summary = summary[:117] + "..."
        return summary if summary else text[:100]

    async def converse(self, prompt: str, max_rounds: int = 15) -> str:
        self._cancelled = False
        if not self._history:
            self._history.append({"role": "system", "content": SYSTEM_PROMPT})
        self._history.append({"role": "user", "content": prompt})

        last_response = None
        no_tool_count = 0
        reasoning_count = 0
        for _ in range(max_rounds):
            if self._cancelled:
                return "[Cancelled]"

            if len(self._history) > self._max_history:
                self._history = [self._history[0]] + self._history[-(self._max_history - 1):]

            response = await self._send(self._history)

            if self._last_tool_call:
                tc = self._last_tool_call
                self._last_tool_call = None
                tool_name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}
                if tool_name not in ToolRegistry.TOOLS:
                    self._history.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    self._history.append({"role": "tool", "tool_call_id": tc["id"], "content": f"Unknown tool: {tool_name}"})
                    continue
                tool_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                if tool_key == last_response:
                    self._history.append({"role": "assistant", "content": None, "tool_calls": [tc]})
                    self._history.append({"role": "tool", "tool_call_id": tc["id"], "content": "[STOP] Same tool call repeated. Analyze and respond to user."})
                    last_response = None
                    continue
                last_response = tool_key
                no_tool_count = 0
                reasoning_count = 0
                result = await self._execute_tool(tool_name, args)
                self._history.append({"role": "assistant", "content": response if response.strip() else None, "tool_calls": [tc]})
                self._history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                continue

            if not response.strip():
                reasoning_count += 1
                if reasoning_count >= 5:
                    return "[No action after 5 reasoning rounds]"
                content = f"<think>\n{self._last_reasoning}\n</think>" if self._last_reasoning else "[thinking]"
                self._last_reasoning = None
                self._history.append({"role": "assistant", "content": content})
                self._history.append({"role": "user", "content": "Now output the tool call."})
                continue

            reasoning_count = 0

            tool_call = self._parse_tool_call(response)
            if not tool_call:
                tool_call = self._parse_minimax_tool_call(response)
            if not tool_call:
                tool_call = self._parse_natural_language(response)
            if not tool_call:
                no_tool_count += 1
                if no_tool_count >= 3:
                    self._history.append({"role": "assistant", "content": self._summarize_response(response)})
                    self.save_checkpoint()
                    return self._summarize_response(response)
                self._history.append({"role": "assistant", "content": self._summarize_response(response)})
                self._history.append({"role": "user", "content": "Only respond with a JSON tool call."})
                continue
            no_tool_count = 0

            tool_name, args = tool_call
            tool_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

            if tool_key == last_response:
                self._history.append({"role": "assistant", "content": response})
                self._history.append({"role": "user", "content": "[STOP] Same tool call repeated. Analyze results and respond to user."})
                last_response = None
                continue

            last_response = tool_key
            result = await self._execute_tool(tool_name, args)

            tc = {"id": f"call_{tool_name}", "type": "function", "function": {"name": tool_name, "arguments": json.dumps(args)}}
            self._history.append({"role": "assistant", "content": None, "tool_calls": [tc]})
            self._history.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

        return "[Max rounds reached]"


class SingletonMeta(type):
    _instances: dict[type, Any] = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LLMStudio(metaclass=SingletonMeta):
    def __init__(self):
        self.agents: dict[str, LMStudioAgent] = {}

    async def add_agent(self, name: str, model: str = "gpt-3.5-turbo", **kwargs):
        if name not in self.agents:
            agent = LMStudioAgent(name, model, **kwargs)
            agent.load_checkpoint()
            self.agents[name] = agent

    async def converse(self, prompt: str):
        for name, agent in list(self.agents.items()):
            yield f"{name}: {await agent.converse(prompt)}"

    def cancel_all(self):
        for agent in self.agents.values():
            agent.cancel()

    def shutdown(self):
        for agent in self.agents.values():
            agent.save_checkpoint()


async def main():
    llmstudio = LLMStudio()
    await llmstudio.add_agent("Holger", "minimax-m2.5@q2_k", checkpoint_dir="checkpoints")
    print("Agent ready. Type 'quit' to exit, 'cancel' to interrupt.\n")
    try:
        while True:
            prompt = await asyncio.get_running_loop().run_in_executor(None, input, "You: ")
            if prompt.strip().lower() in ("quit", "exit"):
                break
            if prompt.strip().lower() == "cancel":
                llmstudio.cancel_all()
                print("Cancelled.\n")
                continue
            async for msg in llmstudio.converse(prompt):
                print(msg)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
    finally:
        llmstudio.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
