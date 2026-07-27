"""Search command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class SearchCommand(Command):
    """Search for text pattern in files."""
    
    @property
    def name(self) -> str:
        return "search"
    
    @property
    def help_text(self) -> str:
        return "search <query>     - Search for string in files"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        if len(args) < 1:
            self.error("Usage: search <query>")
            return True
        
        query = args[0]
        result = await agent.search_file(query)
        print(result)
        return True
