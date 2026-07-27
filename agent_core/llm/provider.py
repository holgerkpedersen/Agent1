"""LLM Provider Protocol - abstract interface for LLM backends."""
from typing import Protocol, Any


class LLMProvider(Protocol):
    """Abstract interface for LLM providers.
    
    This protocol defines the contract that all LLM providers must implement.
    It enables dependency inversion - Agent depends on this abstraction,
    not on concrete LMStudio implementation.
    """
    
    model_name: str
    
    async def chat(
        self, 
        messages: list[dict], 
        tools: list[dict] | None = None
    ) -> str:
        """Send chat request to LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of OpenAI-format tool schemas
            
        Returns:
            LLM response text, or JSON string if tools present and tool_calls returned
        """
        ...
    
    async def chat_stream(self, messages: list[dict]) -> str:
        """Chat with real-time token streaming to console.
        
        Args:
            messages: List of message dicts
            
        Returns:
            Complete response text
        """
        ...
    
    async def analyze_code(self, code: str) -> str:
        """Analyze code and return feedback.
        
        Args:
            code: Source code to analyze
            
        Returns:
            Analysis text with bugs, improvements, suggestions
        """
        ...
