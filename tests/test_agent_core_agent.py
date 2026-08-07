"""Tests for agent_core.agent — LLMAgent and its file-context integration."""

from __future__ import annotations

from typing import Final
from unittest.mock import MagicMock

import pytest

from agent_core.agent import LLMAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(**kwargs: object) -> LLMAgent:
    """Return an LLMAgent with an optional stub LLM client."""
    llm = kwargs.get("llm")
    if llm is None:
        return LLMAgent()
    return LLMAgent(llm_client=llm)  # type: ignore[arg-type]


def _stub_llm(response: str = "ok") -> MagicMock:
    """Return a mock whose .chat() returns *response*."""
    llm = MagicMock()
    llm.chat.return_value = response
    return llm


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    """LLMAgent instantiates cleanly with or without a client."""

    def test_no_client(self) -> None:
        agent = _make_agent()
        assert agent._llm_client is None

    def test_with_client(self) -> None:
        llm = _stub_llm()
        agent = _make_agent(llm=llm)
        assert agent._llm_client is llm

    def test_internal_state_empty(self) -> None:
        agent = _make_agent()
        assert agent._conversation == []
        assert agent._pending_temp_systems == []


# ---------------------------------------------------------------------------
# _detect_file_keywords
# ---------------------------------------------------------------------------

class TestDetectFileKeywords:
    """Keyword detection delegates to the file-context interface."""

    def test_no_keywords_returns_empty(self) -> None:
        agent = _make_agent()
        assert agent._detect_file_keywords("hello world") == []

    def test_delegates_to_interface(self) -> None:
        agent = _make_agent()
        agent._file_context_interface = MagicMock()
        agent._file_context_interface.process_request.return_value = ["ctx1"]
        result = agent._detect_file_keywords("see /foo.py")
        assert result == ["ctx1"]
        agent._file_context_interface.process_request.assert_called_once_with("see /foo.py")


# ---------------------------------------------------------------------------
# _insert_temp_system_messages  (the optimize fix target)
# ---------------------------------------------------------------------------

class TestInsertTempSystemMessages:
    """Temporary system messages are inserted with correct ordering."""

    def test_single_context(self) -> None:
        agent = _make_agent()
        agent._insert_temp_system_messages(["ctx1"])
        assert agent._pending_temp_systems == [
            {"role": "system", "content": "ctx1"},
        ]

    def test_multiple_contexts_preserve_order(self) -> None:
        """Contexts end up in original order after reversed+insert(0,...) cycle."""
        agent = _make_agent()
        agent._insert_temp_system_messages(["ctx1", "ctx2", "ctx3"])
        contents = [m["content"] for m in agent._pending_temp_systems]
        assert contents == ["ctx1", "ctx2", "ctx3"]

    def test_cumulative_insertions(self) -> None:
        agent = _make_agent()
        agent._insert_temp_system_messages(["a"])
        agent._insert_temp_system_messages(["b", "c"])
        contents = [m["content"] for m in agent._pending_temp_systems]
        assert contents == ["b", "c", "a"]

    def test_all_messages_are_system_role(self) -> None:
        agent = _make_agent()
        agent._insert_temp_system_messages(["x", "y"])
        assert all(m["role"] == "system" for m in agent._pending_temp_systems)


# ---------------------------------------------------------------------------
# _process_message
# ---------------------------------------------------------------------------

class TestProcessMessage:
    """_process_message records the user message and assembles the payload."""

    def test_user_message_recorded(self) -> None:
        agent = _make_agent()
        agent._process_message("hi")
        assert agent._conversation == [{"role": "user", "content": "hi"}]

    def test_payload_starts_with_temp_systems(self) -> None:
        agent = _make_agent()
        agent._file_context_interface = MagicMock()
        agent._file_context_interface.process_request.return_value = ["file-ctx"]
        payload = agent._process_message("load /foo.py")
        assert payload[0] == {"role": "system", "content": "file-ctx"}
        assert payload[-1] == {"role": "user", "content": "load /foo.py"}

    def test_no_contexts_no_system_messages(self) -> None:
        agent = _make_agent()
        payload = agent._process_message("plain message")
        assert len(payload) == 1
        assert payload[0]["role"] == "user"

    def test_returns_independent_copy(self) -> None:
        """Mutating the returned payload must not alter internal state."""
        agent = _make_agent()
        payload = agent._process_message("msg")
        payload.clear()
        assert agent._conversation == [{"role": "user", "content": "msg"}]


# ---------------------------------------------------------------------------
# respond
# ---------------------------------------------------------------------------

class TestRespond:
    """respond() generates a reply and clears temporary system messages."""

    def test_fallback_when_no_client(self) -> None:
        agent = _make_agent()
        reply = agent.respond("hello")
        assert "no LLM client" in reply.lower() or reply  # non-empty

    def test_uses_llm_client(self) -> None:
        llm = _stub_llm("LLM says hi")
        agent = _make_agent(llm=llm)
        reply = agent.respond("hello")
        assert reply == "LLM says hi"
        llm.chat.assert_called_once()

    def test_clears_temp_systems_after_response(self) -> None:
        agent = _make_agent()
        agent._file_context_interface = MagicMock()
        agent._file_context_interface.process_request.return_value = ["ctx"]
        agent.respond("load /x.py")
        assert agent._pending_temp_systems == []

    def test_conversation_grows(self) -> None:
        agent = _make_agent()
        agent.respond("a")
        agent.respond("b")
        roles = [m["role"] for m in agent._conversation]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_assistant_response_recorded(self) -> None:
        llm = _stub_llm("reply")
        agent = _make_agent(llm=llm)
        agent.respond("q")
        last = agent._conversation[-1]
        assert last == {"role": "assistant", "content": "reply"}


# ---------------------------------------------------------------------------
# _fallback_response
# ---------------------------------------------------------------------------

class TestFallbackResponse:
    def test_returns_nonempty_string(self) -> None:
        agent = _make_agent()
        text = agent._fallback_response()
        assert isinstance(text, str)
        assert len(text) > 0
