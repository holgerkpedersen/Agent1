"""Regression: pasted text must reach agent.chat_nlp (the LLM path)."""
import asyncio
import io
from unittest import mock

from agent_core.commands.paste_cmd import PasteCommand


class FakeAgent:
    def __init__(self):
        self.received = []
        self._nlp_workspace = None

    async def chat_nlp(self, text, images=None):
        self.received.append(text)


def test_paste_sends_content_to_llm():
    agent = FakeAgent()
    cmd = PasteCommand()
    with mock.patch("sys.stdin", io.StringIO("line one\nline two\n")):
        result = asyncio.run(cmd.execute([], agent))
    assert result is True
    assert agent.received == ["line one\nline two\n"]


def test_paste_empty_input_does_not_call_llm():
    agent = FakeAgent()
    with mock.patch("sys.stdin", io.StringIO("   \n")):
        asyncio.run(PasteCommand().execute([], agent))
    assert agent.received == []


def test_paste_end_key_hint_matches_platform():
    import sys
    from unittest import mock
    cmd = PasteCommand()
    with mock.patch("agent_core.commands.paste_cmd._sys.platform", "linux"):
        assert "Ctrl+D" in cmd.help_text
        assert "Ctrl+Z" not in cmd.help_text
        assert cmd.end_key_hint() == "Ctrl+D"
    with mock.patch("agent_core.commands.paste_cmd._sys.platform", "win32"):
        assert "Ctrl+Z" in cmd.help_text
        assert cmd.end_key_hint() == "Ctrl+Z and Enter"
