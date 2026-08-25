"""Sub-agent module providing child agents with isolated conversation history."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent import Agent


class SubAgent:
    """Child agent with isolated conversation history but shared filesystem.

    Subagents share the parent's workspace and model configuration, but each
    maintains its own conversation history so work done in one subagent does
    not pollute another's context or the parent's chat history.
    """

    def __init__(self, parent: "Agent", name: str, workspace: str | None = None) -> None:
        self.parent = parent
        self.name = name
        #: Real filesystem path — same as parent unless overridden.
        self.workspace: str = os.path.abspath(workspace or parent.workspace)
        self.model_name: str = parent.model_name
        # Isolated conversation history (not persisted to chat_history.json).
        self._conversation: list[dict[str, str]] = []
        #: LLM client instance — shares the same model as the parent.
        from agent import LLMClient  # local import to avoid circular deps
        self.llm = LLMClient(model_name=self.model_name)

    async def respond(self, message: str) -> str:
        """Process *message* with this subagent's own conversation history.

        The message is prefixed with the agent name so the model knows which
        subagent issued it.  Results are stored in ``_conversation`` for later
        retrieval via :meth:`get_context_summary`.
        """
        prefixed = f"[SubAgent:{self.name}] {message}"
        self._conversation.append({"role": "user", "content": prefixed})
        response_text = await self.llm.chat(list(self._conversation))
        self._conversation.append({"role": "assistant", "content": response_text})
        return response_text

    def get_context_summary(self, max_messages: int = 10) -> str:
        """Return a condensed summary of recent conversation turns.

        Used by the parent to review what a subagent has done without loading
        the full history into its own context window.
        """
        if not self._conversation:
            return f"<SubAgent:{self.name}> has no conversation yet."

        # Show last N messages (pairs of user/assistant).
        recent = self._conversation[-(max_messages * 2):]
        lines: list[str] = []
        for msg in recent:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                # Strip the prefix for readability.
                if content.startswith("[SubAgent:"):
                    content = content.split("] ", 1)[-1]
                lines.append(f"  > {content}")
            else:
                lines.append(f"  < {content[:500]}")  # cap length
        return f"[SubAgent:{self.name}] context summary:\n" + "\n".join(lines)

    def get_conversation(self) -> list[dict[str, str]]:
        """Return the full conversation history (for debugging/testing)."""
        return list(self._conversation)

    def reset(self) -> None:
        """Clear this subagent's conversation history."""
        self._conversation.clear()

    def __repr__(self) -> str:
        return f"SubAgent(name={self.name!r}, workspace={self.workspace!r})"


__all__ = ["SubAgent"]
