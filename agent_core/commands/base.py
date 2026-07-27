"""Command base class and registry for agent interactive mode."""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


class Command(ABC):
    """Abstract base class for interactive commands.
    
    All commands must implement execute() and provide a help text.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name used in REPL."""
        ...
    
    @property
    @abstractmethod
    def help_text(self) -> str:
        """Help text shown in commands list."""
        ...
    
    @abstractmethod
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        """Execute the command.
        
        Args:
            args: Arguments after the command name
            agent: Agent instance for LLM/file operations
            
        Returns:
            True to continue REPL, False to exit
        """
        ...
    
    def error(self, msg: str):
        """Print error message."""
        print(f"Error: {msg}")
    
    def success(self, msg: str):
        """Print success message."""
        print(msg)
