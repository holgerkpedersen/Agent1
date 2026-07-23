"""Core entities and type contracts for the Agent Tutorial System.

Provides dataclasses, TypedDicts, and Protocols used across all agent
implementations (v1–v4) and the final orchestrator.  Designed to avoid
circular imports by using structural typing (`Protocol`) instead of
deep inheritance chains.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypedDict


# ──────────────────────────────────────────────────────
# 1. Message Types
# ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class Message:
    """Immutable message exchanged between user, assistant, and tools."""

    role: str = "user"             # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_call_id: str | None = None   # links a tool result back to the call that produced it
    name: str | None = None           # optional agent/tool identifier

    def __str__(self) -> str:
        tag = self.name or self.role
        return f"[{tag}] {self.content}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize for LLM API payloads."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            d["name"] = self.name
        return d


# ──────────────────────────────────────────────────────
# 2. Tool Definitions & Execution Results
# ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolParameterSchema:
    """JSON-Schema-like description for a single tool parameter."""

    name: str
    param_type: str = "string"      # "string" | "number" | "integer" | "boolean" | "array" | "object"
    description: str = ""
    required: bool = True


@dataclass(frozen=True)
class ToolDefinition:
    """Metadata describing a callable tool available to the agent."""

    name: str
    description: str
    parameters: list[ToolParameterSchema] = field(default_factory=list)
    func: Callable[..., Any] | None = None   # actual implementation (may be injected later)

    def to_dict(self) -> dict[str, Any]:
        """Format for LLM tool-calling payloads."""
        props: dict[str, Any] = {}
        required: list[str] = []
        for p in self.parameters:
            props[p.name] = {"type": p.param_type, "description": p.description}
            if p.required:
                required.append(p.name)

        payload: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": props,
            },
        }
        if required:
            payload["parameters"]["required"] = required
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class ToolCallRequest:
    """Parsed intent from the LLM indicating a tool should be invoked."""

    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "arguments": self.arguments}


@dataclass(frozen=True)
class ToolExecutionResult:
    """Outcome of executing a tool call."""

    success: bool = True
    output: str = ""
    error_message: str | None = None
    tool_name: str = ""
    tool_call_id: str | ""

    def to_message(self) -> Message:
        content = self.output if self.success else f"Error ({self.tool_name}): {self.error_message}"
        return Message(role="tool", content=content, tool_call_id=self.tool_call_id, name=self.tool_name)


# ──────────────────────────────────────────────────────
# 3. Reasoning Steps (ReAct loop)
# ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReasoningStep:
    """One iteration inside a ReAct-style reasoning loop."""

    thought: str = ""
    action: str | None = None         # e.g. "search", "calculator"
    action_input: str | None = None   # arguments for the action
    observation: str | None = None    # result after executing the action
    is_final: bool = False            # True when the agent has a final answer

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"thought": self.thought}
        if self.action is not None:
            d["action"] = self.action
        if self.action_input is not None:
            d["action_input"] = self.action_input
        if self.observation is not None:
            d["observation"] = self.observation
        d["is_final"] = self.is_final
        return d


# ──────────────────────────────────────────────────────
# 4. Agent Configuration
# ──────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Runtime knobs shared across all agent versions."""

    system_prompt: str = "You are a helpful assistant."
    max_turns: int = 20              # memory buffer limit (pairs of user/assistant)
    max_reasoning_steps: int = 15    # ReAct loop ceiling
    temperature: float = 0.0         # passed to LLM provider when supported
    seed: int | None = None          # deterministic behaviour for mocking/testing
    use_tool_calling: bool = False   # whether the agent supports function calling
    log_observations: bool = True    # include tool results in trace output


# ──────────────────────────────────────────────────────
# 5. Protocols (Structural Typing)
# ──────────────────────────────────────────────────────

class LLMProvider(Protocol):
    """Minimal interface any language-model backend must satisfy."""

    def chat(self, messages: list[Message], system_prompt: str | None = None) -> Message: ...  # noqa: E704

    def generate_tool_calls(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        system_prompt: str | None = None,
    ) -> tuple[list[ToolCallRequest] | None, str]: ...  # noqa: E704


class AgentBase(Protocol):
    """Core contract for every agent version in the tutorial."""

    def run(self, user_input: str) -> Message: ...  # noqa: E704

    def reset_memory(self) -> None: ...  # noqa: E704


# ──────────────────────────────────────────────────────
# 6. TypedDict helpers (for external config / JSON payloads)
# ──────────────────────────────────────────────────────

class ToolSchemaDict(TypedDict, total=False):
    """Shape of tool definitions as they appear in `config.yaml`."""
    name: str
    description: str
    parameters: list[dict[str, Any]]


class AgentConfigDict(TypedDict, total=False):
    """Flat dictionary shape loaded from YAML / JSON config files."""
    system_prompt: str
    max_turns: int
    max_reasoning_steps: int
    temperature: float
    seed: int | None
    use_tool_calling: bool
    log_observations: bool


# ──────────────────────────────────────────────────────
# 7. Stateless Utilities
# ──────────────────────────────────────────────────────

def parse_json_safely(text: str) -> dict[str, Any] | list[Any] | None:
    """Attempt to parse *text* as JSON; return `None` on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        # Fallback: look for a JSON block surrounded by markdown code fences
        match = re.search(r"