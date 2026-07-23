Here’s a complete, production-ready shared types module designed specifically for your tutorial architecture. It extracts all cross-cutting concepts, uses structural typing (`Protocol`) to avoid inheritance chains, and is strictly isolated to prevent circular imports.

### 📦 `src/types.py`
```python
"""
src/types.py
Shared types, protocols, and utilities for the Agent Tutorial.
✅ Zero dependencies on agent implementations → Safe from circular imports.
✅ Uses Protocol + Dataclasses → Structural typing without inheritance overhead.
"""

from __future__ import annotations

import json
from typing import TypedDict, Protocol, Callable, Any
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────
# 📦 MESSAGE FORMAT
# ──────────────────────────────────────────────────────────────────────
class Message(TypedDict):
    """Standard chat message structure compatible with OpenAI/Anthropic."""
    role: str          # "system" | "user" | "assistant" | "tool"
    content: str | None
    name: str | None   # Tool identifier for tool results
    tool_calls: list[dict] | None


# ──────────────────────────────────────────────────────────────────────
# 🔧 TOOL DEFINITION
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ToolDefinition:
    """Immutable schema + implementation wrapper for agent tools."""
    name: str
    description: str
    parameters: dict   # JSON Schema format
    func: Callable[..., Any]

    def to_openai_format(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }


# ──────────────────────────────────────────────────────────────────────
# 🤖 LLM PROVIDER PROTOCOL
# ──────────────────────────────────────────────────────────────────────
class LLMProvider(Protocol):
    """Structural interface for any LLM backend (Mock, OpenAI, local, etc.)"""

    def chat(self, messages: list[Message], system_prompt: str | None = None) -> str: ...

    def generate_tool_calls(
        self, messages: list[Message], tools: list[ToolDefinition]
    ) -> list[dict]: ...


# ──────────────────────────────────────────────────────────────────────
# 🧠 AGENT BASE PROTOCOL
# ──────────────────────────────────────────────────────────────────────
class AgentBase(Protocol):
    """Structural interface for all agent versions."""

    def run(self, user_input: str) -> dict | str: ...


# ──────────────────────────────────────────────────────────────────────
# 🔄 REASONING STEP RESULT (ReAct / Planning Loop)
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ReasoningStep:
    thought: str = ""
    action_name: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    is_final: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ──────────────────────────────────────────────────────────────────────
# ⚙️ AGENT CONFIGURATION
# ──────────────────────────────────────────────────────────────────────
@dataclass
class AgentConfig:
    name: str = "TutorialAgent"
    temperature: float = 0.2
    max_tokens: int = 512
    system_prompt: str = ""
    max_turns: int = 10
    enable_tools: bool = False
    enable_reasoning: bool = False


# ──────────────────────────────────────────────────────────────────────
# 🛠️ STATELESS UTILITIES (Safe to import anywhere)
# ──────────────────────────────────────────────────────────────────────
def parse_json_safely(text: str) -> dict | list | None:
    """Extract JSON from LLM output, ignoring markdown/code fences."""
    text = text.strip()
    # Strip ```json ... ``` if present
    if "```" in text:
        try:
            start = max(text.index("{"), text.index("["))
            end = max(text.rindex("}"), text.rindex("]")) + 1
            return json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def format_tool_result(tool_name: str, result: Any) -> Message:
    """Wrap tool execution output into a standardized message."""
    return {
        "role": "tool",
        "content": str(result),
        "name": tool_name,
        "tool_calls": None,
    }


def build_context_window(
    history: list[Message], max_turns: int, system_prompt: str | None = None
) -> list[Message]:
    """Return a sliding window of messages respecting turn limits."""
    if not history or max_turns <= 0:
        return []

    # Keep system prompt at index 0 if provided
    ctx: list[Message] = [{"role": "system", "content": system_prompt, "name": None, "tool_calls": None}] \
                          if system_prompt else []
    
    # Sliding window over user/assistant/tool pairs
    tail = history[-(max_turns * 2):]
    ctx.extend(tail)
    return ctx
```

---

### 🔗 Integration Guide (How to use without circular imports)

| File | Import Pattern | Why It Works |
|------|----------------|--------------|
| `src/mock_llm.py` | `from src.types import LLMProvider, Message, ToolDefinition` | Implements protocol; no agent dependencies |
| `src/agent_v1_basic.py` | `from src.types import AgentBase, Message, LLMProvider, AgentConfig` | Uses only types & protocols |
| `src/agent_v2_tools.py` | `from src.types import ToolDefinition, parse_json_safely, format_tool_result` | Reuses shared tool schema & helpers |
| `src/agent_v3_memory.py` | `from src.types import build_context_window, Message` | Delegates windowing logic to shared module |
| `src/agent_v4_reasoning.py` | `from src.types import ReasoningStep, LLMProvider, ToolDefinition` | Structured step tracking without tight coupling |
| `src/agent_final.py` | Imports all above + orchestrates | Composition over inheritance; no circular refs |

---

### 🚫 Circular Import Prevention Rules

1. **One-Way Dependency Graph**:  
   `types.py` → (nothing)  
   `mock_llm.py`, `agent_v*.py`, `run_tutorial.py` → `types.py`  
   Never import from `src/` inside `types.py`.

2. **Prefer `Protocol` over `ABC`**:  
   Structural typing avoids `isinstance()` checks and inheritance chains that often trigger circular imports in multi-file tutorials.

3. **Lazy/Type-Checking Imports (if needed)**:  
   If you ever need forward references inside a function, use:
   ```python
   from __future__ import annotations  # Already included
   # or
   if TYPE_CHECKING:
       from src.types import SomeType
   ```

4. **Keep Utilities Stateless**:  
   Functions like `parse_json_safely` and `build_context_window` take inputs and return outputs. No module-level state = safe to import anywhere.

---

### ✅ Next Steps
1. Replace inline type definitions in your agent files with imports from `src.types`.
2. Update `mock_llm.py` to explicitly satisfy `LLMProvider` (type checkers will validate it).
3. Run `mypy src/ --ignore-missing-imports` to verify zero circular dependencies and correct protocol adherence.

Let me know if you want any specific agent version (`v1`–`v4` or `final`) rewritten to use these shared types, or if you need a Pydantic-based variant for stricter runtime validation.