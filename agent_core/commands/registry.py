"""Command registry for dispatching commands."""
from typing import TYPE_CHECKING

from .base import Command

if TYPE_CHECKING:
    from agent import Agent


class CommandRegistry:
    """Registry for managing and dispatching commands.
    
    Follows Open/Closed Principle - new commands can be registered
    without modifying existing code.
    """
    
    def __init__(self):
        self._commands: dict[str, Command] = {}
    
    def register(self, command: Command):
        """Register a command."""
        self._commands[command.name] = command
    
    def get(self, name: str) -> Command | None:
        """Get command by name."""
        return self._commands.get(name)
    
    async def execute(self, name: str, args: list[str], agent: 'Agent') -> bool:
        """Execute a command by name.
        
        Returns:
            True to continue REPL, False to exit
        """
        command = self._commands.get(name)
        if command:
            return await command.execute(args, agent)
        print(f"Unknown command: {name}. Type 'help' for commands.")
        return True
    
    def print_help(self):
        """Print all registered commands."""
        print("Commands:")
        for cmd in sorted(self._commands.values(), key=lambda c: c.name):
            print(f"  {cmd.help_text}")
