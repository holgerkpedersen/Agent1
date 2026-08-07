"""LLM Provider Protocol - abstract interface for LLM backends."""
from typing import Protocol


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
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        disable_thinking: bool = False,
    ) -> str:
        """Send chat request to LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of OpenAI-format tool schemas
            max_tokens: Optional output token cap for this call
            disable_thinking: If True, send thinking: disabled (reasoning
                models otherwise burn the output budget on reasoning and
                return empty content)
            
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
