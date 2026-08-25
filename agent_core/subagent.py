"""Sub-agent module providing child agents with isolated conversation history.

Roles (plan phase 1-2): a SubAgent may be spawned with a *role* from
:mod:`agent_core.subagent_roles`.  A role gives the child

* a persona system prompt,
* its own session ``mode`` (capped by the parent's mode — a plan-mode
  parent keeps every child read-only, so the plan-mode guarantee "no file
  in the workspace may be changed during a plan turn" holds through
  delegation),
* an explicit tool whitelist enforced BEFORE the parent's
  ``_execute_tool_call`` choke point (which still applies plan-mode
  rejection), and
* a turn cap feeding the stuck-synthesis guard.

Without a role, behaviour is unchanged from the original design: plain
LLM chat with an isolated history and no tool access.
"""

from __future__ import annotations

import json
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

    def __init__(
        self, parent: "Agent", name: str,
        workspace: str | None = None, role: str | None = None,
    ) -> None:
        self.parent = parent
        self.name = name
        #: Real filesystem path — same as parent unless overridden.
        self.workspace: str = os.path.abspath(workspace or parent.workspace)
        self.model_name: str = parent.model_name

        # ---- role wiring (phase 1-2) -------------------------------------
        from agent_core.modes import MODE_PLAN, is_plan_mode
        from agent_core.subagent_roles import get_role

        self.role_name: str | None = None
        self.role_spec = None
        self._tools_allowed: frozenset[str] = frozenset()
        if role:
            spec = get_role(role)
            if spec is None:
                from agent_core.subagent_roles import role_names
                raise ValueError(
                    f"Unknown subagent role {role!r}. "
                    f"Available: {', '.join(role_names())}"
                )
            self.role_name = spec.name
            self.role_spec = spec
            self._tools_allowed = spec.tools_allowed
            # Parent mode caps child mode: delegation must never become a
            # plan-mode escape hatch.
            self.mode: str = (
                MODE_PLAN if is_plan_mode(parent.mode) else spec.mode
            )
        else:
            self.mode = parent.mode

        # Isolated conversation history (not persisted to chat_history.json).
        self._conversation: list[dict[str, str]] = []
        #: LLM client instance — shares the same model as the parent.
        from agent import LLMClient  # local import to avoid circular deps
        self.llm = LLMClient(model_name=self.model_name)

    # ------------------------------------------------------------------
    # Tool execution (role-gated, parent-enforced)
    # ------------------------------------------------------------------
    async def _execute_tool_call(self, name: str, args: dict[str, Any]) -> str:
        """Execute one tool call inside this subagent's whitelist.

        Order of enforcement:
        1. role whitelist (this method) — the role simply has no other tools;
        2. ``check_tool_allowed`` via the parent's ``_execute_tool_call``
           choke point — plan mode rejection and handler dispatch.
        """
        lowered = (name or "").lower()
        if lowered not in self._tools_allowed:
            return (
                f"[subagent:{self.name}] tool '{lowered}' is not allowed for "
                f"role '{self.role_name}'. Allowed: "
                f"{', '.join(sorted(self._tools_allowed))}"
            )
        return await self.parent._execute_tool_call(lowered, args)

    async def _chat_with_tools(self, msgs: list[dict[str, Any]]) -> str:
        """Run a bounded tool-calling loop for this subagent.

        Mirrors the parent's ``llm_chat_fn`` contract: providers return a
        plain string, or a JSON object with ``tool_calls`` when the model
        wants to act — parsed back into an assistant message so
        :class:`ToolLoopRunner` can execute them.  The runner supplies the
        stuck/duplicate/no-progress guards, so a subagent cannot spin
        forever even though its history is isolated.
        """
        from agent_core.llm.tool_loop import ToolLoopRunner
        from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

        schemas = [
            s for s in NLP_TOOL_SCHEMAS
            if s["function"]["name"] in self._tools_allowed
        ]

        async def llm_chat_fn(
            m: list[dict[str, Any]], tools: list[dict[str, Any]],
        ) -> tuple[str, list[dict[str, Any]]]:
            raw = await self.llm.chat(m, tools=tools, disable_thinking=True)
            if raw.strip() == "(no output)":
                raw = ""
            try:
                parsed = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, dict) and parsed.get("tool_calls"):
                parsed.pop("role", None)
                updated = list(m)
                updated.append(
                    {"role": "assistant", "content": parsed.get("content") or "", **parsed}
                )
                return str(parsed.get("content") or ""), updated
            updated = list(m)
            updated.append({"role": "assistant", "content": raw})
            return raw, updated

        assert self.role_spec is not None
        runner = ToolLoopRunner(max_iterations=self.role_spec.max_turns)
        final_text, _final_messages = await runner.run(
            msgs, llm_chat_fn, self._execute_tool_call, tools=schemas,
        )
        return final_text

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------
    async def respond(self, message: str) -> str:
        """Process *message* with this subagent's own conversation history.

        The message is prefixed with the agent name so the model knows which
        subagent issued it.  Results are stored in ``_conversation`` for later
        retrieval via :meth:`get_context_summary`.

        With a role, the persona system prompt is prepended per call and the
        answer comes from a bounded tool loop restricted to the role's
        whitelist; without a role this is a plain chat call (original
        behaviour).
        """
        prefixed = f"[SubAgent:{self.name}] {message}"

        # Turn cap: refuse once the role's budget is spent (the parent sees
        # the note and can collect/reset instead of silently burning tokens).
        if self.role_spec is not None:
            turns = sum(
                1 for m in self._conversation if m.get("role") == "assistant"
            )
            if turns >= self.role_spec.max_turns:
                return (
                    f"[subagent:{self.name}] turn cap reached "
                    f"({self.role_spec.max_turns} turns). Collect my summary "
                    f"via get_context_summary() or reset me for a new task."
                )

        self._conversation.append({"role": "user", "content": prefixed})

        if self.role_spec is not None:
            msgs: list[dict[str, Any]] = [
                {"role": "system", "content": self.role_spec.system_prompt},
                *self._conversation,
            ]
            response_text = await self._chat_with_tools(msgs)
        else:
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
        role_part = f", role={self.role_name!r}" if self.role_name else ""
        return (
            f"SubAgent(name={self.name!r}, workspace={self.workspace!r}"
            f"{role_part})"
        )


__all__ = ["SubAgent"]
