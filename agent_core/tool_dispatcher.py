"""Tool dispatcher with registry pattern for agent."""
from typing import Callable, Awaitable


class ToolDispatcher:
    """Dispatches tool calls to registered handlers.
    
    Replaces the if/elif chain in Agent.execute_tool with a registry pattern.
    Follows Open/Closed Principle - new tools can be added without modifying
    existing code.
    """
    
    def __init__(self):
        self._handlers: dict[str, Callable[..., Awaitable[str]]] = {}
    
    def register(self, name: str, handler: Callable[..., Awaitable[str]]):
        """Register a tool handler.
        
        Args:
            name: Tool name (e.g., 'read_file', 'write_file')
            handler: Async function that takes (args: dict) and returns result string
        """
        self._handlers[name] = handler
    
    async def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments dictionary for the tool
            
        Returns:
            Tool execution result as string
        """
        handler = self._handlers.get(tool_name)
        if handler:
            return await handler(args)
        return f"Unknown tool: {tool_name}"
    
    @property
    def available_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._handlers.keys())
