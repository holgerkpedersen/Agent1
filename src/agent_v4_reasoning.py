"""
Type definitions and utility functions for the Agent framework.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Protocol, TypedDict, Union


class Message(TypedDict):
    role: str
    content: str


class ToolDefinition(TypedDict):
    name: str
    description: str
    func: Callable[[str], Any]


class ReasoningStep(TypedDict):
    iteration: int
    thought: str
    action: Optional[str]
    input: str
    observation: Optional[str]
    is_final: bool


class LLMProvider(Protocol):
    def chat(self, messages: List[Message]) -> str: ...


class AgentBase:
    """Abstract base class for all agent implementations."""
    pass


def format_tool_result(result: Any) -> str:
    """Converts tool output into a string-safe observation."""
    if isinstance(result, (dict, list)):
        return json.dumps(result, ensure_ascii=False)
    return str(result)


def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
    """
    Attempts to extract and parse a JSON object from potentially noisy LLM output.
    Handles markdown code blocks, trailing text, and malformed JSON gracefully.
    """
    cleaned = text.strip()

    # Strip markdown formatting if present
    if cleaned.startswith("