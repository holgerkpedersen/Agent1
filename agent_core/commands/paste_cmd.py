"""Paste command — read multiline text from stdin and feed into the NLP ReAct loop."""
import os as _os
import sys as _sys
from typing import TYPE_CHECKING
from .base import Command

if TYPE_CHECKING:
    from agent import Agent


class PasteCommand(Command):
    @property
    def name(self) -> str:
        return "paste"

    @staticmethod
    def end_key_hint() -> str:
        """Key that ends stdin on this platform (EOF). Ctrl+Z only on Windows."""
        if _sys.platform == "win32":
            return "Ctrl+Z and Enter"
        return "Ctrl+D"

    @property
    def help_text(self) -> str:
        return (f"paste [--workspace <path>] - Paste multiline text for AI analysis "
                f"({self.end_key_hint()} to finish)")

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        workspace = None
        parsed_args = list(args)
        if "--workspace" in parsed_args:
            idx = parsed_args.index("--workspace")
            if idx + 1 < len(parsed_args):
                workspace = _os.path.abspath(parsed_args[idx + 1])

        print(f"Paste text, then press {self.end_key_hint()} to finish:")
        text = _sys.stdin.read()
        if not text.strip():
            self.error("No text provided")
            return True

        if workspace:
            agent._nlp_workspace = workspace
        await agent.chat_nlp(text)
        return True
