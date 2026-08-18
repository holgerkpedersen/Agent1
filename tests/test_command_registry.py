"""Registry-driven REPL dispatch (regression: "review" fell through to
chat_nlp because the dispatch whitelist was hardcoded in agent.py)."""

from pathlib import Path

from agent_core.commands.base import Command
from agent_core.commands.registry import CommandRegistry


class StubCommand(Command):
    @property
    def name(self) -> str:
        return "stub"

    @property
    def help_text(self) -> str:
        return "stub help"

    async def execute(self, args, agent) -> bool:
        return True


def test_registry_names_expose_registered_commands():
    registry = CommandRegistry()
    registry.register(StubCommand())
    assert registry.names() == {"stub"}


def test_repl_dispatch_is_registry_driven_not_hardcoded():
    """Regression: agent.py:1436 hardcoded the command whitelist, so the
    registered `review` command was never dispatched and its input fell
    through to chat_nlp.  Dispatch must read from the registry."""
    src = Path("agent.py").read_text(encoding="utf-8")
    assert "if command in registry.names()" in src
    assert 'if command in ["' not in src


def test_review_is_registered_for_dispatch():
    from agent_core.commands.review_cmd import ReviewCommand

    registry = CommandRegistry()
    registry.register(ReviewCommand())
    assert "review" in registry.names()
