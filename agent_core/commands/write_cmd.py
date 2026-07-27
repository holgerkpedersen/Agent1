"""Write command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class WriteCommand(Command):
    """Write content to a file."""
    
    @property
    def name(self) -> str:
        return "write"
    
    @property
    def help_text(self) -> str:
        return "write <path> <content> - Write content to file"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 2:
            self.error("Usage: write <path> <content>")
            return True
        
        path = args[0]
        content = args[1]
        result = await agent.write_file(path, content)
        print(result)
        return True
