import asyncio
import json
import logging
import os
import time
import urllib.error
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")

SYSTEM_PROMPT = """Respond with ONLY a JSON tool call. No explanation, no analysis, no thinking. Just the JSON.

{"tool": "read_file", "args": {"path": "/c/Dev/Agent1/agent.py"}}
{"tool": "apply_patch", "args": {"path": "/c/Dev/Agent1/agent.py", "find": "old text", "replace": "new text"}}
{"tool": "search_file", "args": {"query": "search term", "path": "/c/Dev/Agent1"}}
{"tool": "edit_file", "args": {"path": "/c/Dev/Agent1/agent.py", "content": "full file"}}

Rules:
- For ANY file request, output ONLY the JSON tool call immediately
- No text before or after the JSON
- Use read_file first to see content
- Use apply_patch for changes
- Paths: /c/Dev/file.txt"""


def _normalize_path(path: str) -> str:
    if path.startswith("/c/"):
        return "C:\\" + path[3:].replace("/", "\\")
    elif path.startswith("/d/"):
        return "D:\\" + path[3:].replace("/", "\\")
    elif path.startswith("/"):
        return "C:\\" + path[1:].replace("/", "\\")
    return path


class ToolRegistry:
    @staticmethod
    async def read_file(path: str) -> str:
        try:
            if not isinstance(path, str) or not path:
                return "Error: Invalid path"
            local_path = _normalize_path(path)
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
            local_path = _normalize_path(path)
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
            local_path = _normalize_path(path)
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
            local_path = _normalize_path(path) if path != "." else "."
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

    TOOLS = ["read_file", "apply_patch", "edit_file", "search_file"]


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
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def _send(self, messages: list[dict]) -> str:
        import urllib.request
        payload = json.dumps({"model": self.model, "messages": messages, "temperature": 0.1, "max_tokens": 2048, "stop": ["\n\n", "I need to", "Let me", "First,"]}).encode()
        req = urllib.request.Request(self.BASE_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        for attempt in range(3):
            try:
                resp = await asyncio.get_event_loop().run_in_executor(None, lambda: urllib.request.urlopen(req, timeout=300))
                data = json.loads(resp.read())
                msg = data["choices"][0]["message"]
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                if not content and reasoning:
                    return reasoning
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
        is_read = any(w in text_lower for w in ['read', 'see', 'show', 'view', 'check', 'open', 'cat'])
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
        if is_read:
            return ("read_file", {"path": f"/c/Dev/Agent1/{filename}"})
        if is_edit:
            if 'improve' in text_lower or 'add' in text_lower or 'fix' in text_lower:
                return ("read_file", {"path": f"/c/Dev/Agent1/{filename}"})
            return ("read_file", {"path": f"/c/Dev/Agent1/{filename}"})
        return ("read_file", {"path": f"/c/Dev/Agent1/{filename}"})

    def save_checkpoint(self):
        if not self._checkpoint_dir:
            return
        try:
            path = self._checkpoint_dir / f"{self.name}_checkpoint.json"
            path.write_text(json.dumps({"name": self.name, "model": self.model, "history": self._history}, indent=2))
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)

    def load_checkpoint(self) -> bool:
        if not self._checkpoint_dir:
            return False
        path = self._checkpoint_dir / f"{self.name}_checkpoint.json"
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

        seen_tools = set()
        last_response = None
        no_tool_count = 0
        for _ in range(max_rounds):
            if self._cancelled:
                return "[Cancelled]"

            if len(self._history) > self._max_history:
                self._history = [self._history[0]] + self._history[-(self._max_history - 1):]

            response = await self._send(self._history)

            tool_call = self._parse_tool_call(response)
            if not tool_call:
                tool_call = self._parse_natural_language(response)
            if not tool_call:
                no_tool_count += 1
                if no_tool_count >= 3:
                    self._history.append({"role": "assistant", "content": self._summarize_response(response)})
                    self.save_checkpoint()
                    return self._summarize_response(response)
                self._history.append({"role": "assistant", "content": self._summarize_response(response)})
                self._history.append({"role": "user", "content": "STOP. Output ONLY JSON tool call. No explanation. Example: {\"tool\": \"read_file\", \"args\": {\"path\": \"/c/Dev/Agent1/agent.py\"}}"})
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

            self._history.append({"role": "assistant", "content": response})
            self._history.append({"role": "user", "content": f"Tool {tool_name} result:\n{result}"})

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
    await llmstudio.add_agent("Holger", "qwen3.6-27b-mtp", checkpoint_dir="checkpoints")
    print("Agent ready. Type 'quit' to exit, 'cancel' to interrupt.\n")
    try:
        while True:
            prompt = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
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
