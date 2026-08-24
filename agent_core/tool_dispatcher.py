"""Tool dispatcher with registry pattern for agent."""
import time
from typing import Callable, Awaitable, Any


class ToolDispatcher:
    """Dispatches tool calls to registered handlers.
    
    Replaces the if/elif chain in Agent.execute_tool with a registry pattern.
    Follows Open/Closed Principle - new tools can be added without modifying
    existing code.

    An optional ``on_tool`` callback is invoked after every execution of a
    *registered* handler (name, elapsed seconds, success flag) so callers can
    feed observability metrics without the dispatcher knowing about them.
    """
    
    def __init__(self, on_tool: "Callable[[str, float, bool], None] | None" = None) -> None:
        self._handlers: dict[str, Callable[..., Awaitable[str]]] = {}
        self._on_tool = on_tool
    
    def register(self, name: str, handler: Callable[..., Awaitable[str]]) -> None:
        """Register a tool handler.
        
        Args:
            name: Tool name (e.g., 'read_file', 'write_file')
            handler: Async function that takes (args: dict) and returns result string
        """
        self._handlers[name] = handler
    
    async def execute(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            args: Arguments dictionary for the tool
            
        Returns:
            Tool execution result as string
        """
        handler = self._handlers.get(tool_name)
        if not handler:
            return f"Unknown tool: {tool_name}"
        start = time.perf_counter()
        ok = True
        try:
            return await handler(args)
        except Exception:
            ok = False
            raise
        finally:
            if self._on_tool is not None:
                try:
                    self._on_tool(tool_name, time.perf_counter() - start, ok)
                except Exception:
                    pass  # metrics must never break tool execution
    
    @property
    def available_tools(self) -> list[str]:
        """Return list of registered tool names."""
        return list(self._handlers.keys())
