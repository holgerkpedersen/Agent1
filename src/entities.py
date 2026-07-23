"""
Shared types, contracts, and stateless utilities for the agent framework.
Implements structural typing (Protocol + TypedDict) to avoid inheritance chains & circular imports.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict, Union


# =============================================================================
# Data Structures (TypedDicts)
# =============================================================================

class Message(TypedDict):
    """Standard message format for LLM context windows."""
    role: str  # "system", "user", or "assistant"
    content: str


class ToolCall(TypedDict):
    """Represents a single tool invocation requested by an LLM."""
    name: str
    arguments: Dict[str, Any]


class ToolDefinition(TypedDict):
    """Schema definition for a callable tool exposed to the LLM."""
    name: str
    description: str
    parameters: Optional[Dict[str, Any]]  # JSON Schema format


class ReasoningStep(TypedDict):
    """Tracks one iteration of a ReAct-style reasoning loop."""
    thought: str
    action: str
    action_input: str
    observation: Optional[str]
    is_final: bool


class AgentConfig(TypedDict, total=False):
    """Configuration payload for agent initialization and runtime behavior."""
    system_prompt: str
    max_turns: int
    max_reasoning_steps: int
    temperature: float
    use_mock_llm: bool


# =============================================================================
# Structural Protocols (Interfaces)
# =============================================================================

class LLMProvider(Protocol):
    """Contract for any synchronous LLM backend (mock, OpenAI, Anthropic, etc.)."""

    def chat(self, messages: List[Message], tools: Optional[List[ToolDefinition]] = None) -> str: ...

    def generate_tool_calls(
        self, messages: List[Message], tools: List[ToolDefinition]
    ) -> List[ToolCall]: ...


class AgentBase(Protocol):
    """Contract for all progressive agent implementations."""

    def run(self, user_input: str) -> Union[str, List[ReasoningStep]]: ...


# =============================================================================
# Stateless Utilities
# =============================================================================

def parse_json_safely(json_str: str) -> Any:
    """
    Safely parse a JSON string. 
    Includes fallback logic to extract JSON from markdown code blocks or messy LLM outputs.
    Returns `None` on complete failure.
    """
    if not isinstance(json_str, str):
        return None

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: attempt to isolate the first valid JSON object
    start = json_str.find('{')
    end = json_str.rfind('}') + 1
    if start != -1 and end > start:
        try:
            return json.loads(json_str[start:end])
        except (json.JSONDecodeError, TypeError):
            pass

    # Fallback for JSON arrays
    start_arr = json_str.find('[')
    end_arr = json_str.rfind(']') + 1
    if start_arr != -1 and end_arr > start_arr:
        try:
            return json.loads(json_str[start_arr:end_arr])
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def format_tool_result(tool_name: str, result: Any) -> str:
    """
    Formats the output of a tool execution for safe injection back into the LLM context.
    Ensures dict/list outputs are serialized and wrapped in a predictable prefix.
    """
    if isinstance(result, (dict, list)):
        payload = json.dumps(result, ensure_ascii=False)
    else:
        payload = str(result)

    return f"[Tool '{tool_name}' Result]: {payload}"


def build_context_window(messages: List[Message], max_turns: int) -> List[Message]:
    """
    Prunes message history to fit within `max_turns`.
    
    Assumes turns are pairs of user/assistant messages. Preserves a leading 
    system prompt if it exists at index 0. Returns an empty list on invalid input.
    """
    if not messages or max_turns <= 0:
        return []

    # Detect and preserve system prompt
    has_system = len(messages) > 0 and messages[0].get("role") == "system"
    context_start_index = 1 if has_system else 0
    
    history = messages[context_start_index:]
    
    # Each turn consists of a user message followed by an assistant message.
    max_history_messages = max_turns * 2
    pruned_history = history[-max_history_messages:]
    
    return (messages[:1] if has_system else []) + pruned_history


# =============================================================================
# Helper Classes
# =============================================================================

class ToolRegistry:
    """
    Simple registry to map tool names to executable functions and their definitions.
    Used by v2+ agents to dynamically resolve LLM requests.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, Callable[..., Any]] = {}
        self._definitions: List[ToolDefinition] = []

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a callable function as an LLM-accessible tool."""
        if name in self._tools:
            raise ValueError(f"Tool '{name}' is already registered.")
            
        self._tools[name] = func
        self._definitions.append({
            "name": name,
            "description": description,
            "parameters": parameters or {},
        })

    def get_definitions(self) -> List[ToolDefinition]:
        """Return a copy of all registered tool definitions."""
        return self._definitions.copy()

    def execute(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool by name with provided arguments."""
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool requested: '{tool_name}'")
        return self._tools[tool_name](**kwargs)