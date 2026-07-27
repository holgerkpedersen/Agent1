"""Clear command for agent interactive mode."""
from .base import Command

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agent import Agent


class ClearCommand(Command):
    """Clear agent memory."""
    
    @property
    def name(self) -> str:
        return "clear"
    
    @property
    def help_text(self) -> str:
        return "clear              - Clear agent memory"
    
    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        agent.clear_history()
        print("Agent memory cleared.")
        return True
