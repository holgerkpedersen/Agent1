"""Read command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class ReadCommand(Command):
    """Read a file and print its contents."""
    
    @property
    def name(self) -> str:
        return "read"
    
    @property
    def help_text(self) -> str:
        return "read <path>        - Read a file"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 1:
            self.error("Usage: read <path>")
            return True
        
        path = args[0]
        result = await agent.read_file(path)
        print(result)
        return True
