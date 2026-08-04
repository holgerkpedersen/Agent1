"""Paste command — read multiline text from stdin and feed into the NLP ReAct loop."""
import os as _os
import sys as _sys
from .base import Command

if False:
    from agent import Agent


class PasteCommand(Command):
    @property
    def name(self) -> str:
        return "paste"

    @property
    def help_text(self) -> str:
        return "paste [--workspace <path>] - Paste multiline text for AI analysis (Ctrl+Z / Ctrl+D to finish)"

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        workspace = None
        parsed_args = list(args)
        if "--workspace" in parsed_args:
            idx = parsed_args.index("--workspace")
            if idx + 1 < len(parsed_args):
                workspace = _os.path.abspath(parsed_args[idx + 1])

        print("Paste text, then press Ctrl+Z and Enter to finish:")
        text = _sys.stdin.read()
        if not text.strip():
            self.error("No text provided")
            return True

        if workspace:
            agent._nlp_workspace = workspace
        await agent.chat_nlp(text)
        return True
