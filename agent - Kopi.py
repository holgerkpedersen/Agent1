import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("agent")

SYSTEM_PROMPT = """You are an AI assistant operating in a Git Bash (MINGW64) environment on Windows.
Your working directory uses Unix-style paths (e.g., /c/Dev/Agent1).
Shell commands should be executed via bash.exe, typical path: C:\\Program Files\\Git\\bin\\bash.exe

CRITICAL: When you need to read, edit, or search files, you MUST respond with ONLY a JSON object (no other text before or after it). Do NOT tell the user to run commands - use the tools directly by outputting the JSON.

Available tools:

1. read_file(path) - Read file contents
   Response: {"tool": "read_file", "args": {"path": "/c/Dev/Agent1/agent.py"}}

2. edit_file(path, content) - Write/create a file with the given content (replaces entire file)
   Response: {"tool": "edit_file", "args": {"path": "/c/Dev/Agent1/main.py", "content": "print('hello')"}}

3. search_file(query, path) - Search for content in files
   Response: {"tool": "search_file", "args": {"query": "def main", "path": "/c/Dev/Agent1"}}

WORKFLOW for requests like "improve", "fix", "add feature", "refactor":
1. First call read_file to get the current content
2. Analyze the content and plan improvements
3. Call edit_file with the improved content
4. Confirm what you changed

Examples:
- "read agent.py" → {"tool": "read_file", "args": {"path": "/c/Dev/Agent1/agent.py"}}
- "improve agent.py" → First: {"tool": "read_file", "args": {"path": "/c/Dev/Agent1/agent.py"}}
  Then after reading: {"tool": "edit_file", "args": {"path": "/c/Dev/Agent1/agent.py", "content": "... improved code ..."}}
- "fix main.py" → First: {"tool": "read_file", "args": {"path": "/c/Dev/Agent1/main.py"}}
  Then after reading: {"tool": "edit_file", "args": {"path": "/c/Dev/Agent1/main.py", "content": "... fixed code ..."}}
- "add error handling to server.py" → First: {"tool": "read_file", "args": {"path": "/c/Dev/Agent1/server.py"}}
  Then after reading: {"tool": "edit_file", "args": {"path": "/c/Dev/Agent1/server.py", "content": "... code with error handling ..."}}

When you receive a tool result, analyze it and proceed with the next step of the workflow.

IMPORTANT: Your ENTIRE response must be just the JSON object. No explanations, no markdown, no other text.

If the user asks a question that doesn't require tools, respond in plain text normally."""


class ToolRegistry:
    """Registry of available tools the agent can call."""

    @staticmethod
    async def read_file(path: str) -> str:
        try:
            import ntpath
            local_path = path
            if path.startswith("/c/"):
                local_path = "C:\\" + path[3:].replace("/", "\\")
            elif path.startswith("/d/"):
                local_path = "D:\\" + path[3:].replace("/", "\\")
            elif path.startswith("/"):
                local_path = "C:\\" + path[1:].replace("/", "\\")

            if os.path.exists(local_path):
                with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if len(content) > 8000:
                    return content[:8000] + "\n\n... [truncated, file is long]"
                return content

            proc = await asyncio.create_subprocess_exec(
                "cat", path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode != 0:
                return f"Error reading file: {stderr.decode().strip()}"
            content = stdout.decode()
            if len(content) > 8000:
                return content[:8000] + "\n\n... [truncated, file is long]"
            return content
        except FileNotFoundError:
            return f"File not found: {path}"
        except asyncio.TimeoutError:
            return "Timeout reading file"
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    async def edit_file(path: str, content: str) -> str:
        try:
            local_path = path
            if path.startswith("/c/"):
                local_path = "C:\\" + path[3:].replace("/", "\\")
            elif path.startswith("/d/"):
                local_path = "D:\\" + path[3:].replace("/", "\\")
            elif path.startswith("/"):
                local_path = "C:\\" + path[1:].replace("/", "\\")

            os.makedirs(os.path.dirname(local_path) if os.path.dirname(local_path) else ".", exist_ok=True)
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"File written successfully: {path} ({len(content)} bytes)"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    async def search_file(query: str, path: str = ".") -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "grep", "-rn", query, path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            output = stdout.decode()
            if proc.returncode != 0 and not output:
                return f"No matches found for '{query}' in {path}"
            if len(output) > 8000:
                return output[:8000] + "\n... [truncated]"
            return output.strip() if output else f"No matches found for '{query}' in {path}"
        except asyncio.TimeoutError:
            return "Timeout searching files"

    TOOLS = ["read_file", "edit_file", "search_file"]


class Agent:
    def __init__(self, name: str, max_history_messages: int = 50):
        self.name = name
        self._history: list[dict] = []
        self._max_history_messages = max_history_messages


class LMStudioAgent(Agent):
    BASE_URL = "http://localhost:1234/v1/chat/completions"

    def __init__(
        self,
        name: str,
        model: str = "gpt-3.5-turbo",
        max_history_messages: int = 50,
        max_retries: int = 3,
        base_retry_delay: float = 1.0,
        tool_call_delay: float = 0.5,
        checkpoint_dir: str | None = None,
    ):
        super().__init__(name, max_history_messages)
        self.model = model
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay
        self._tool_call_delay = tool_call_delay
        self._last_tool_call_time = 0.0
        self._cancelled = False
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self._checkpoint_dir:
            self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _trim_history(self):
        if len(self._history) <= self._max_history_messages:
            return
        system = self._history[0]
        excess = len(self._history) - self._max_history_messages
        trimmed = self._history[1:]
        summary_entry = {
            "role": "user",
            "content": f"[System: {excess} earlier messages were trimmed to fit context window.]",
        }
        self._history = [system] + trimmed[excess:] + [summary_entry]
        logger.info(
            "[%s] Trimmed %d messages from history (max=%d)",
            self.name, excess, self._max_history_messages,
        )

    async def _send(self, messages: list[dict]) -> str:
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }).encode("utf-8")

        req = urllib.request.Request(
            self.BASE_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error = None
        for attempt in range(self._max_retries):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, lambda: urllib.request.urlopen(req, timeout=60)
                )
                body = json.loads(response.read().decode("utf-8"))
                logger.debug("[%s] LLM response received (%d tokens)", self.name, body.get("usage", {}).get("total_tokens", 0))
                return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
                if e.code >= 500:
                    delay = self._base_retry_delay * (2 ** attempt)
                    logger.warning(
                        "[%s] Server error on attempt %d/%d: %s. Retrying in %.1fs",
                        self.name, attempt + 1, self._max_retries, last_error, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                return f"Error connecting to LM Studio: {last_error}"
            except urllib.error.URLError as e:
                last_error = str(e.reason)
                delay = self._base_retry_delay * (2 ** attempt)
                logger.warning(
                    "[%s] Connection error on attempt %d/%d: %s. Retrying in %.1fs",
                    self.name, attempt + 1, self._max_retries, last_error, delay,
                )
                await asyncio.sleep(delay)
            except (json.JSONDecodeError, KeyError) as e:
                return f"Error parsing response: {e}"

        return f"Error connecting to LM Studio after {self._max_retries} attempts: {last_error}"

    async def _execute_tool(self, tool_name: str, args: dict) -> str:
        elapsed = time.monotonic() - self._last_tool_call_time
        if elapsed < self._tool_call_delay:
            await asyncio.sleep(self._tool_call_delay - elapsed)

        handler = getattr(ToolRegistry, tool_name, None)
        if not handler:
            return f"Unknown tool: {tool_name}. Available: {ToolRegistry.TOOLS}"
        try:
            logger.info("[%s] Tool call: %s(%s)", self.name, tool_name, args)
            result = await handler(**args)
            self._last_tool_call_time = time.monotonic()
            return result
        except TypeError as e:
            return f"Error calling {tool_name}: missing or wrong arguments. {e}"
        except Exception as e:
            logger.error("[%s] Tool %s failed: %s", self.name, tool_name, e)
            return f"Error calling {tool_name}: {e}"

    def _parse_tool_call(self, text: str) -> tuple[str | None, dict] | None:
        text = text.strip()

        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                return (obj["tool"], obj["args"])
        except json.JSONDecodeError:
            pass

        import re
        json_pattern = r'\{[^{}]*"tool"\s*:\s*"[^"]+"\s*,\s*"args"\s*:\s*\{[^{}]*\}[^{}]*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                    return (obj["tool"], obj["args"])
            except json.JSONDecodeError:
                continue

        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return (obj["tool"], obj["args"])
                except json.JSONDecodeError:
                    continue

        for line in lines:
            line = line.strip().removeprefix("```json").removesuffix("```").strip()
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "tool" in obj and "args" in obj:
                        return (obj["tool"], obj["args"])
                except json.JSONDecodeError:
                    continue

        return None

    def cancel(self):
        self._cancelled = True
        logger.info("[%s] Cancellation requested", self.name)

    def _reset_cancel(self):
        self._cancelled = False

    def save_checkpoint(self):
        if not self._checkpoint_dir:
            return
        path = self._checkpoint_dir / f"{self.name}_checkpoint.json"
        data = {
            "name": self.name,
            "model": self.model,
            "history": self._history,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("[%s] Checkpoint saved to %s", self.name, path)

    def load_checkpoint(self) -> bool:
        if not self._checkpoint_dir:
            return False
        path = self._checkpoint_dir / f"{self.name}_checkpoint.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._history = data.get("history", [])
            logger.info("[%s] Checkpoint loaded from %s", self.name, path)
            return True
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[%s] Failed to load checkpoint: %s", self.name, e)
            return False

    async def converse(
        self,
        prompt: str,
        max_tool_rounds: int = 25,
        overall_timeout: float | None = 300.0,
    ) -> str:
        self._reset_cancel()
        start_time = time.monotonic()
        recent_tool_calls: list[str] = []

        if not self._history:
            self._history.append({"role": "system", "content": SYSTEM_PROMPT})

        self._history.append({"role": "user", "content": prompt})

        for round_num in range(1, max_tool_rounds + 1):
            if self._cancelled:
                logger.info("[%s] Conversation cancelled by user", self.name)
                return "[Cancelled by user]"

            if overall_timeout and (time.monotonic() - start_time) > overall_timeout:
                logger.warning("[%s] Conversation timed out after %.0fs", self.name, overall_timeout)
                return f"[Timed out after {overall_timeout}s — {round_num-1} tool rounds completed]"

            self._trim_history()
            response = await self._send(self._history)

            if self._cancelled:
                return "[Cancelled by user]"

            tool_call = self._parse_tool_call(response)
            if not tool_call:
                self._history.append({"role": "assistant", "content": response})
                self.save_checkpoint()
                return response

            tool_name, args = tool_call
            tool_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"

            is_duplicate = tool_key in recent_tool_calls

            logger.info(
                "[%s] Round %d/%d: %s(%s)%s",
                self.name, round_num, max_tool_rounds, tool_name,
                ", ".join(f"{k}={v!r}" for k, v in args.items()),
                " [DUPLICATE]" if is_duplicate else "",
            )
            result = await self._execute_tool(tool_name, args)

            if tool_name == "read_file" and not is_duplicate:
                result += "\n\n[FILE READ SUCCESSFULLY — Now use edit_file to make changes, or respond to the user with your analysis.]"

            recent_tool_calls.append(tool_key)
            if len(recent_tool_calls) > 5:
                recent_tool_calls.pop(0)

            if is_duplicate:
                result += "\n\n[ALREADY READ — Do NOT read this file again. Use edit_file to make changes, or respond to the user.]"

            self._history.append({"role": "assistant", "content": response})
            self._history.append({
                "role": "user",
                "content": f"[Tool Result — {tool_name}]:\n{result}"
            })

        logger.warning("[%s] Max tool rounds (%d) reached", self.name, max_tool_rounds)
        final_response = await self._send(self._history)
        self._history.append({"role": "assistant", "content": final_response})
        self.save_checkpoint()
        return final_response


class SingletonMeta(type):
    _instances: dict[type, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
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
            response = await agent.converse(prompt)
            yield f"{name}: {response}"

    def cancel_all(self):
        for agent in self.agents.values():
            agent.cancel()

    def shutdown(self):
        logger.info("Shutting down — saving all agent checkpoints...")
        for agent in self.agents.values():
            agent.save_checkpoint()


async def main():
    llmstudio = LLMStudio()
    await llmstudio.add_agent(
        "Alice",
        "qwen2.5-coder-7b-instruct",
        checkpoint_dir="checkpoints",
    )

    print("Agent ready. Type 'quit' to exit, 'cancel' to interrupt running task.\n")
    try:
        while True:
            prompt = await asyncio.get_event_loop().run_in_executor(None, input, "You: ")
            if prompt.strip().lower() in ("quit", "exit"):
                break
            if prompt.strip().lower() == "cancel":
                llmstudio.cancel_all()
                print("Cancellation requested for all agents.\n")
                continue
            async for message in llmstudio.converse(prompt):
                print(message)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted.")
    finally:
        llmstudio.shutdown()
        print("Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
