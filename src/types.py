"""
src/types.py
Shared type contracts, structural typing protocols, and stateless utilities 
for the progressive agent tutorial. Acts as the root dependency for all agent modules.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

logger = logging.getLogger(__name__)


# ==============================================================================
# DATA STRUCTURES (TypedDict / Dataclasses)
# ==============================================================================

class Message(TypedDict):
    """Standard message format for LLM context windows."""
    role: str
    content: str


@dataclass
class ToolDefinition:
    """Schema definition for a callable tool available to the agent."""
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    func: Any = None


@dataclass
class ReasoningStep:
    """Single iteration in a ReAct-style reasoning loop."""
    thought: str
    action: str | None = None
    input_data: str | None = None
    observation: str | None = None
    is_final: bool = False


@dataclass
class AgentConfig:
    """Configuration parameters for agent initialization."""
    system_prompt: str = "You are a helpful AI assistant."
    max_turns: int = 10
    temperature: float = 0.0
    use_mock_llm: bool = True
    tools: list[ToolDefinition] = field(default_factory=list)


# ==============================================================================
# STRUCTURAL TYPING (Protocols)
# ==============================================================================

class LLMProvider(Protocol):
    """Interface for any language model backend (mock or live)."""
    
    def chat(self, messages: list[Message]) -> str: ...
    
    def generate_tool_calls(
        self, 
        messages: list[Message], 
        tools: list[ToolDefinition]
    ) -> list[dict[str, Any]]: ...


class AgentBase(Protocol):
    """Interface that all agent implementations must satisfy."""
    
    def run(self, user_input: str) -> str: ...


# ==============================================================================
# STATELESS UTILITIES
# ==============================================================================

def parse_json_safely(text: str) -> dict | list[dict] | None:
    """
    Attempts to extract and parse a JSON object or array from raw text.
    Handles markdown code blocks, partial outputs, and malformed strings.
    Returns parsed data or None on failure.
    """
    if not isinstance(text, str):
        return None
    
    cleaned = text.strip()
    
    # Strip markdown code block formatting if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r'^```[a-z]*\n?', '', cleaned)
        cleaned = re.sub(r'\n?```$', '', cleaned)
    
    # Try to find JSON in the text
    match = re.search(r'\{.*\}|\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Try the entire text as JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None
