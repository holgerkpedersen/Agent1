"""
src/types.py - Shared Type Contracts & Stateless Utilities for Agent Tutorial

This module defines the core data structures, structural interfaces (Protocols),
and helper functions used across all agent implementations. It enforces type safety
without requiring inheritance chains or external framework dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, TypedDict


# ==============================================================================
# 📦 CORE DATA STRUCTURES
# ==============================================================================

class Message(TypedDict):
    """Represents a single message in an LLM conversation.
    
    Attributes:
        role: The sender ('system', 'user', 'assistant', or 'tool').
        content: The text payload of the message.
        name: Optional identifier, typically used to link tool responses back 
              to their originating function call.
    """
    role: str
    content: str
    name: Optional[str]


@dataclass
class ToolDefinition:
    """Schema and metadata for a callable agent tool.
    
    Attributes:
        name: Unique identifier for the tool (e.g., 'calculator').
        description: Human-readable explanation of what the tool does & when to use it.
        parameters: JSON Schema-style dict describing expected arguments.
        func: The actual Python function to execute. Can be None if registered later.
    """
    name: str
    description: str
    parameters: dict[str, Any]
    func: Optional[Callable[..., Any]] = None


@dataclass
class ReasoningStep:
    """Tracks a single iteration in a ReAct-style reasoning loop.
    
    Attributes:
        thought: The agent's internal planning/logic for this step.
        action: Name of the tool/function to call, or 'final_answer'.
        action_input: Arguments passed to the action.
        observation: Result returned from executing the action (filled after execution).
        is_final: Flag indicating if this step concludes the reasoning process.
    """
    thought: str
    action: str
    action_input: str
    observation: Optional[str] = None
    is_final: bool = False


class AgentConfig(TypedDict, total=False):
    """Configuration settings for initializing and tuning an agent.
    
    All fields are optional to allow partial overrides during composition/testing.
    """
    name: str
    temperature: float
    max_tokens: int
    system_prompt: str
    max_turns: int
    max_steps: int
    tools: list[ToolDefinition]


# ==============================================================================
# 🧱 STRUCTURAL INTERFACES (PROTOCOLS)
# ==============================================================================

class LLMProvider(Protocol):
    """Structural interface for any Large Language Model wrapper.
    
    Any class implementing these methods can be passed to agents, 
    enabling easy swapping between MockLLM, OpenAI, Anthropic, etc.
    """
    def chat(self, messages: List[Message]) -> str: ...
    def generate_tool_calls(self, messages: List[Message], tools: list[ToolDefinition]) -> list[dict]: ...


class AgentBase(Protocol):
    """Structural interface for all agent implementations.
    
    Ensures every agent version (v1-v4) exposes a consistent `run()` method 
    regardless of internal complexity.
    """
    def run(self, user_input: str) -> Any: ...


# ==============================================================================
# 🛠️ STATELESS UTILITIES
# ==============================================================================

def parse_json_safely(text: str) -> dict | list | None:
    """Attempts to extract and parse a JSON object or array from an LLM response.
    
    Handles common formatting issues like markdown code fences, surrounding prose, 
    or trailing commas that often break strict parsers.
    
    Args:
        text: Raw string output from the LLM.
        
    Returns:
        Parsed dict/list if valid JSON is found, otherwise None.
    """
    if not isinstance(text, str):
        return None
    
    cleaned = text.strip()
    
    # Strip markdown code fences (