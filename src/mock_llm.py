"""
src/mock_llm.py
Zero-API-key LLM simulator for safe practice and deterministic testing.
Satisfies the LLMProvider contract with predictable outputs based on message content.
"""
import json
import re
from typing import Any, Dict, List, Optional

from src.types import ToolDefinition


class MockLLM:
    """
    A deterministic, offline LLM simulator that routes responses based on input keywords.
    Designed to mimic OpenAI-style chat and function-calling formats without external API calls.
    
    Attributes:
        config (dict): Configuration dictionary supporting 'temperature' and 'seed'.
        temperature (float): Simulated randomness factor (0.0 = fully deterministic).
        seed (int): Base value for reproducible non-deterministic variations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.temperature = float(self.config.get("temperature", 0.2))
        self.seed = int(self.config.get("seed", 42))
        
        # Pre-defined response templates for predictable simulation
        self._templates: Dict[str, str] = {
            "math": '{{"thought": "I will evaluate this math expression.", "action": "calculator", "input": "{expr}"}}',
            "weather": '{{"thought": "Fetching weather data.", "action": "get_weather", "input": "{loc}"}}',
        }

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Simulates a standard LLM chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            
        Returns:
            Deterministic text response based on the last user message content.
        """
        if not messages:
            return "No input provided."

        last_msg = messages[-1]
        content = last_msg.get("content", "").lower()

        # Keyword-based routing for predictable outputs
        if re.search(r'\b(math|calculate|\d+\s*[\+\-\*/]\s*\d+)\b', content):
            return self._templates["math"].format(expr="2 + 2")
        elif re.search(r'\b(weather|forecast|temperature)\b', content):
            return self._templates["weather"].format(loc="San Francisco")
            
        # Apply optional temperature noise to simulate non-determinism
        base_response = "Here is a helpful response based on your input."
        return self._apply_temperature_noise(base_response)

    def generate_tool_calls(
        self, messages: List[Dict[str, str]], tools: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Simulates OpenAI-style function calling.
        
        Parses message content to return predictable JSON tool calls matching 
        registered schemas from the provided `tools` list.
        
        Args:
            messages: Conversation history.
            tools: List of ToolDefinition objects or dicts describing available functions.
            
        Returns:
            List of simulated tool call dictionaries in standard function-calling format.
        """
        if not tools or not messages:
            return []

        last_msg = messages[-1]
        content = last_msg.get("content", "").lower()

        matched_tools = []
        for tool in tools:
            # Support both object-like and dict-like ToolDefinitions
            name = getattr(tool, "name", tool.get("name", "")) if tool else ""
            if isinstance(name, str) and name.lower() in content:
                matched_tools.append(tool)

        calls = []
        for i, tool in enumerate(matched_tools):
            name = getattr(tool, "name", tool.get("name", ""))
            args_def = getattr(tool, "args", tool.get("args", [])) or []
            
            # Generate deterministic mock arguments matching the schema
            mock_args = {arg: f"mock_{arg}_{i}" for arg in args_def}
            
            calls.append({
                "id": f"call_mock_{self.seed}_{i}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(mock_args)
                }
            })

        return calls

    def _apply_temperature_noise(self, text: str) -> str:
        """
        Optional config-driven randomness to simulate non-deterministic LLM behavior.
        Uses seed + temperature to deterministically alter output casing/punctuation.
        
        Args:
            text: Base response string.
            
        Returns:
            Slightly altered string if temperature > 0, otherwise original text.
        """
        if self.temperature == 0 or not text:
            return text

        import hashlib
        noise = int(hashlib.md5(f"{self.seed}_{text}".encode()).hexdigest(), 16) % 3
        
        if noise == 1:
            return f"Sure! {text}"
        elif noise == 2:
            return text.capitalize() + "?"
            
        return text