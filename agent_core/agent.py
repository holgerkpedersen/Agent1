"""Agent core module providing an LLM agent with file-context integration."""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal protocol describing the LLM client used by :class:`LLMAgent`."""

    def chat(self, messages: list[dict[str, str]]) -> str: ...


class LLMAgent:
    """LLM agent that integrates file-context retrieval for ``/file`` and ``@file``.

    The agent instantiates a :class:`FileContextRetriever` together with an
    :class:`AgentFileContextInterface`. When processing incoming messages the
    interface is asked to detect file keywords (``/file`` / ``@file``) and, if any
    are present, temporary system messages carrying retrieved file context are
    inserted before the LLM call. Those temporary messages are cleared once a
    response has been generated so they never linger in conversation history.
    """

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm_client = llm_client
        self._file_context_retriever = FileContextRetriever()
        self._context_optimizer = ContextOptimizer(self._file_context_retriever)
        self._file_context_interface = AgentFileContextInterface(
            self._file_context_retriever,
            optimizer=self._context_optimizer
        )
        self._conversation: list[dict[str, str]] = []
        self._pending_temp_systems: list[dict[str, str]] = []

    def _detect_file_keywords(self, message: str) -> list[str]:
        """Return the file keywords (``/file`` / ``@file``) found in *message*.

        Keyword detection is delegated to :class:`AgentFileContextInterface` via
        its ``process_request`` method which returns retrieved context snippets.
        """
        return self._file_context_interface.process_request(message)

    def _insert_temp_system_messages(self, contexts: list[str]) -> None:
        """Insert temporary system messages carrying file *contexts*.

        The first retrieved context becomes the topmost (most recent) system
        message so it takes precedence when handed to the LLM.
        """
        self._pending_temp_systems = [
            {"role": "system", "content": ctx} for ctx in contexts
        ] + self._pending_temp_systems

    def _process_message(self, message: str) -> list[dict[str, str]]:
        """Process *message*: record it and attach any file context as temps.

        Detects ``/file`` / ``@file`` keywords through the file-context interface,
        inserts temporary system messages before the (upcoming) LLM call, and
        returns the assembled payload to hand to the model.
        """
        self._conversation.append({"role": "user", "content": message})
        contexts = self._detect_file_keywords(message)
        if contexts:
            self._insert_temp_system_messages(contexts)
        return [*self._pending_temp_systems, *self._conversation]

    def respond(self, message: str) -> str:
        """Generate a response for *message* and clear temp system messages.

        Temporary file-context system messages are inserted before the LLM call
        (see :meth:`_process_message`) and cleared after the response is produced.
        """
        payload = self._process_message(message)
        if self._llm_client is not None:
            response_text = self._llm_client.chat(payload)
        else:
            response_text = self._fallback_response()
        # Clear temporary system messages after response generation.
        self._pending_temp_systems.clear()
        self._conversation.append({"role": "assistant", "content": response_text})
        return response_text

    def _fallback_response(self) -> str:
        """Return a basic acknowledgement when no LLM client is configured."""
        return "I received your message but have no LLM client available."


from agent_core.file_context_retriever import FileContextRetriever  # noqa: E402
from agent_core.agent_file_context_interface import AgentFileContextInterface  # noqa: E402
from agent_core.context_optimizer import ContextOptimizer  # noqa: E402