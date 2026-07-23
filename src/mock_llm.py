import json
import logging
from typing import Any, Dict, List, Optional

# Graceful imports to support both standalone execution and package-based runs
try:
    from src.types import LLMProvider
except ImportError:
    # Fallback base class for environments where types haven't been built yet
    class LLMProvider: pass

logger = logging.getLogger(__name__)


class MockLLM(LLMProvider):
    """
    Deterministic mock implementation of an LLM provider.
    
    Used for testing agent pipelines without external API dependencies or rate limits.
    Routes prompts to predictable tool calls or text responses based on keyword matching,
    ensuring consistent unit test outcomes across v1-v4 agent implementations.
    """

    def __init__(self, seed: int = 42, temperature: float = 0.0) -> None:
        """
        Initialize the mock provider with optional configuration knobs.
        
        Args:
            seed: Reserved for future deterministic randomness support.
            temperature: Mock parameter; kept for API parity with real providers.
        """
        self.seed = seed
        self.temperature = temperature
        logger.debug(f"MockLLM initialized (seed={seed}, temp={temperature})")

    def chat(self, messages: List[Dict[str, Any]]) -> str:
        """Simulates a standard text-only LLM completion."""
        if not messages:
            return ""

        # Extract last user message content
        user_content = self._extract_last_user_text(messages)
        
        text = str(user_content).lower()

        # Deterministic fallback responses based on intent keywords
        if any(kw in text for kw in ("hello", "hi", "hey")):
            return "Hello! How can I assist you today?"
        elif "time" in text:
            return "I don't have access to real-time clocks, but I can help with other things!"
        elif "weather" in text:
            return "I'm a mock LLM and don't know the weather. Try asking me to use tools!"
        else:
            return f"[Mock Response] You said: {user_content}"

    def generate_tool_calls(self, messages: List[Dict[str, Any]], tools: Optional[List[Any]] = None) -> Dict[str, Any]:
        """
        Simulates LLM deciding whether to call a tool or respond with text.
        
        Returns a dict containing either 'tool_calls' (list of structured calls) 
        or 'response' (plain string fallback).
        
        Args:
            messages: Conversation history in standard role/content format.
            tools: Optional list of registered tool definitions for routing context.
            
        Returns:
            Dict with either "tool_calls" key or "response" key.
        """
        if not messages:
            return {"response": "No input provided."}

        user_content = self._extract_last_user_text(messages)
        text = str(user_content).lower()

        # Simulate tool routing based on keywords
        if "time" in text:
            return {
                "tool_calls": [{
                    "name": "get_current_time",
                    "arguments": json.dumps({"timezone": "UTC"})
                }]
            }
        elif "weather" in text:
            return {
                "tool_calls": [{
                    "name": "get_weather",
                    "arguments": json.dumps({"location": "New York", "unit": "celsius"})
                }]
            }
        elif any(kw in text for kw in ("search", "look up", "find")):
            return {
                "tool_calls": [{
                    "name": "web_search",
                    "arguments": json.dumps({"query": user_content})
                }]
            }

        # Default to text response if no tool matches or tools aren't registered for this intent
        return {"response": self.chat(messages)}

    @staticmethod
    def _extract_last_user_text(messages: List[Dict[str, Any]]) -> str:
        """Helper to safely pull the most recent user message content."""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""