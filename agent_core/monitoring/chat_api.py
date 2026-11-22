"""Alternate browser chat for the dashboard.

Two interchangeable services sit behind ``/api/chat/*``.  They share the same
five-member surface (``send`` / ``history`` / ``reset`` / ``model_name`` /
``commands``), so ``dashboard_api`` never knows which one it is talking to:

``AgentTurnRunner`` (default)
    Runs the message through the REPL's OWN turn pipeline — ``Agent.chat_nlp``
    — so tools, the tool loop, plan mode, history trimming and persistence are
    the SAME code the terminal uses.  The dashboard adds a transport, not a
    second brain: there is exactly one implementation of "a chat turn".
    When a REPL session is live in this process the runner reuses THAT agent
    (``agent._shared_agent``), so the browser and the terminal share one
    conversation instead of forking it.

``ChatService`` (LLM-only)
    Drives ``LLMClient`` directly with its own tiny history.  No tool can run.
    Kept for embedders that want a read-only question box; the dashboard no
    longer uses it.

SAFETY CONTRACT
---------------
- The dashboard binds loopback ONLY and every POST must pass the CSRF/Host
  guard in ``dashboard_api`` BEFORE any work happens.
- ``AgentTurnRunner`` executes tools, exactly as the REPL does, and under the
  same allowlist/plan-mode rules.  A dashboard turn can therefore read, write
  or run what a REPL turn can — no more.  It is NOT a privilege escalation of
  the file editor, but it IS the reason the guard above must stay strict.
- The Agent and its LLM client are built LAZILY on the first message, so a
  dead or unconfigured backend can never break dashboard boot.
- Every failure becomes ``{"ok": False, "error": ...}``; nothing raises out of
  an HTTP handler.
- One ``threading.Lock`` serialises the whole send, so two browser tabs cannot
  interleave their history or hammer the provider concurrently.
- ``commands()`` exposes the REPL's OWN registry (never a second copy) so the
  page can show an opencode-style ``/`` menu.  It is strictly read-only: a
  browser may SEE the command list, but a slash command is still just text
  that the turn pipeline routes through the model — the menu is not a new way
  to execute a command, and no privilege is added by listing one.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Longest single user message accepted (characters).
MAX_MESSAGE_CHARS = 8000

#: Hard cap on stored history entries (user + assistant messages combined).
MAX_HISTORY = 40

#: Provider error prefixes that mean "no answer" (see llm/provider.py).
_ERROR_PREFIXES = ("[Error", "[LM Studio")

DEFAULT_SYSTEM_PROMPT = (
    "You are Agent1, a concise senior coding assistant. "
    "You are being used through a read-only chat panel that has NO tool access: "
    "you cannot read files, run commands or edit code. "
    "When a request needs repository facts, say briefly which tool the user "
    "should run instead of guessing. Never claim you performed an action."
)


class ChatService:
    """Tiny stateful chat wrapper: one model, one history, one lock."""

    def __init__(
        self,
        client_factory: Optional[Callable[[], Any]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_history: int = MAX_HISTORY,
    ) -> None:
        #: Injection seam for tests (and for a future tool-capable client).
        self._client_factory = client_factory
        self._system_prompt = system_prompt
        self._max_history = max(2, int(max_history))
        self._lock = threading.Lock()
        self._client: Optional[Any] = None
        self._history: List[Dict[str, str]] = []

    # -- client ---------------------------------------------------------
    def _get_client(self) -> Any:
        """Build the LLM client on first use (never at import/server start)."""
        if self._client is None:
            factory = self._client_factory
            self._client = factory() if factory is not None else self._default_client()
        return self._client

    @staticmethod
    def _default_client() -> Any:
        from agent_core.llm.client import LLMClient

        return LLMClient()

    @property
    def model_name(self) -> str:
        """Model name, or ``"unloaded"`` until the first message is sent."""
        client = self._client
        name = getattr(client, "model_name", None) if client is not None else None
        return str(name) if name else "unloaded"

    # -- history --------------------------------------------------------
    def history(self) -> List[Dict[str, str]]:
        """Return a copy of the stored conversation history."""
        with self._lock:
            return [dict(m) for m in self._history]

    def reset(self) -> None:
        """Forget the conversation (keeps the built client)."""
        with self._lock:
            self._history.clear()

    def commands(self) -> List[Dict[str, Any]]:
        """The REPL's command catalog, for the page's ``/`` slash menu.

        Read-only and agent-independent: the catalog comes from the registry
        builder, so the menu lists exactly what the terminal lists and cannot
        drift.  A client-side menu never executes anything.
        """
        from agent_core.commands.catalog import command_catalog

        try:
            from agent_core.cli.interactive import _build_registry

            return command_catalog(_build_registry())
        except Exception:  # a broken optional dep must not 500 the menu
            logger.debug("command catalog unavailable", exc_info=True)
            return []

    def models(self, fresh: bool = False) -> Dict[str, Any]:
        """The model catalogue for the page's picker.

        This transport has no agent, so there is no provider to ask: returning
        an empty catalogue says that plainly instead of inventing model names.
        """
        return {
            "current": self.model_name,
            "active_provider": "",
            "groups": [],
            "count": 0,
            "error": "this chat service has no agent, so no model catalogue",
        }

    def select_model(
        self, model: str, provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refuse a model switch: this transport cannot rebuild a provider."""
        return {
            "ok": False,
            "error": "this chat service has no agent, so it cannot switch models",
        }

    def _record(self, role: str, content: str) -> None:
        """Append one turn and trim the oldest entries beyond the cap."""
        self._history.append({"role": role, "content": content})
        overflow = len(self._history) - self._max_history
        if overflow > 0:
            del self._history[:overflow]

    # -- send -----------------------------------------------------------
    def send(self, text: str) -> Dict[str, Any]:
        """Send one user message and return ``{ok, reply, model}``.

        Never raises: provider failures come back as ``{"ok": False, "error"}``.
        """
        message = (text or "").strip()
        if not message:
            return {"ok": False, "error": "message is required"}
        if len(message) > MAX_MESSAGE_CHARS:
            return {
                "ok": False,
                "error": f"message too long (max {MAX_MESSAGE_CHARS} chars)",
            }
        # Serialise the entire turn: history append + provider call + record.
        # Two tabs then take strict turns instead of interleaving.
        with self._lock:
            try:
                client = self._get_client()
                messages = [{"role": "system", "content": self._system_prompt}]
                messages.extend(self._history)
                messages.append({"role": "user", "content": message})
                reply = asyncio.run(client.chat(messages))
            except Exception as exc:  # surface provider/transport errors to the UI
                logger.debug("dashboard chat failed", exc_info=True)
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "model": self.model_name,
                }
            answer = reply if isinstance(reply, str) else str(reply)
            if answer.startswith(_ERROR_PREFIXES):
                # Provider signalled failure in-band; do NOT store a dead turn.
                return {"ok": False, "error": answer, "model": self.model_name}
            self._record("user", message)
            self._record("assistant", answer)
            return {"ok": True, "reply": answer, "model": self.model_name}


def _run_coro(coro: Any) -> Any:
    """Run *coro* to completion from a synchronous caller.

    ``asyncio.run`` raises when this thread already owns a running loop, so in
    that case the coroutine is handed to a short-lived worker thread instead.
    One helper for both services: a dashboard turn must behave identically
    whatever thread the HTTP handler runs on.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    outcome: Dict[str, Any] = {}

    def _worker() -> None:
        try:
            outcome["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - re-raised in caller
            outcome["error"] = exc

    thread = threading.Thread(target=_worker, name="dashboard-chat-turn", daemon=True)
    thread.start()
    thread.join()
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _message_text(content: Any) -> str:
    """Flatten one chat message's content to plain text.

    Multimodal turns store a list of blocks (``[{"type": "text", ...},
    {"type": "image_url", ...}]``) and tool turns store non-string content, so
    a naive ``content`` read would put a repr in the transcript.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


#: Roles that make up the visible transcript (system/tool rows are internal).
_TRANSCRIPT_ROLES = ("user", "assistant")


class AgentTurnRunner:
    """Dashboard chat backed by the REPL's OWN turn pipeline.

    ``send`` calls :meth:`Agent.chat_nlp` — the exact function the terminal
    calls — so tools, the tool loop, plan mode, history trimming, memory and
    persistence are one implementation, not two.  Nothing about a turn is
    re-implemented here; this class is only the transport adapter:

    * serialises turns (one ``threading.Lock``) so two browser tabs queue;
    * reads the outcome from ``agent._last_turn_answer`` /
      ``agent._last_turn_error``, which ``_finish_turn`` fills in — the REPL
      ignores them because it already printed the answer;
    * reuses a live REPL agent when this process has one
      (``agent._shared_agent``), so the browser and the terminal share ONE
      conversation instead of forking it.

    ``agent`` / ``agent_factory`` are injection seams for tests.  Without
    them the agent and its LLM client are built on the FIRST message, so a
    dead backend cannot break dashboard boot.
    """

    #: Advertised to the UI (see ``_chat_history``): this surface can act.
    tool_capable = True

    def __init__(
        self,
        agent: Optional[Any] = None,
        agent_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self._agent = agent
        self._agent_factory = agent_factory
        self._lock = threading.Lock()

    # -- agent ----------------------------------------------------------
    def _resolve_agent(self) -> Any:
        """Return the agent to run the turn on, building it on first use."""
        if self._agent is not None:
            return self._agent
        shared = _shared_agent()
        if shared is not None:
            # A live REPL owns the session: run on it so both surfaces see the
            # same conversation, and never build a second Agent beside it.
            return shared
        factory = self._agent_factory
        self._agent = factory() if factory is not None else self._default_agent()
        return self._agent

    @staticmethod
    def _default_agent() -> Any:
        from agent_core.agent import Agent

        return Agent()

    @property
    def model_name(self) -> str:
        """Model name, or ``"unloaded"`` until an agent exists.

        Never builds the agent: reading a badge must not boot an LLM client.
        """
        for candidate in (self._agent, _shared_agent()):
            if candidate is None:
                continue
            name = getattr(candidate, "model_name", None)
            if name:
                return str(name)
        return "unloaded"

    # -- history --------------------------------------------------------
    def history(self) -> List[Dict[str, str]]:
        """Return the visible transcript (user/assistant rows only)."""
        agent = self._agent or _shared_agent()
        if agent is None:
            return []
        with self._lock:
            raw = list(getattr(agent, "_chat_history", []) or [])
        transcript: List[Dict[str, str]] = []
        for message in raw:
            role = message.get("role") if isinstance(message, dict) else None
            if role not in _TRANSCRIPT_ROLES:
                continue
            text = _message_text(message.get("content"))
            if text:
                transcript.append({"role": role, "content": text})
        return transcript

    def reset(self) -> None:
        """Forget the conversation using the REPL's own ``clear_history``."""
        with self._lock:
            agent = self._agent or _shared_agent()
            if agent is not None:
                agent.clear_history()

    def commands(self) -> List[Dict[str, Any]]:
        """The REPL's command catalog, for the page's ``/`` slash menu.

        Reads the SAME registry the terminal dispatches on
        (``interactive._register_commands``), so a command added to the REPL
        appears in the dashboard menu with no second list to maintain.
        Never builds the agent: listing commands must not boot an LLM client.

        Listing is not executing.  Nothing here widens what a browser turn can
        do — the registry is already reachable through any turn's tools — so
        this only makes the REPL's surface *visible* where the user is typing.
        """
        from agent_core.commands.catalog import command_catalog

        try:
            from agent_core.cli.interactive import _build_registry

            return command_catalog(_build_registry())
        except Exception:  # a broken optional dep must not 500 the menu
            logger.debug("command catalog unavailable", exc_info=True)
            return []

    def models(self, fresh: bool = False) -> Dict[str, Any]:
        """The model catalogue for the page's picker (shared with ``/models``).

        Built by :mod:`agent_core.commands.model_catalog` from the REPL's own
        fetchers, so the browser shows exactly what ``model list`` prints.  An
        unreachable provider is an empty group, never an exception.  Resolving
        the agent here is what loads the session's pinned model, which is why
        the catalog is only fetched when the user opens the picker.
        """
        try:
            agent = self._resolve_agent()
        except Exception as exc:  # surface a broken agent instead of 500ing
            logger.debug("model catalog: agent unavailable", exc_info=True)
            return {
                "current": "",
                "active_provider": "",
                "groups": [],
                "count": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
        from agent_core.commands.model_catalog import model_catalog

        catalog = model_catalog(agent, fresh=fresh)
        catalog["model"] = self.model_name
        return catalog

    def select_model(
        self, model: str, provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Switch this session's model (the picker's action, REPL logic)."""
        try:
            agent = self._resolve_agent()
        except Exception as exc:
            logger.debug("model switch: agent unavailable", exc_info=True)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        from agent_core.commands.slash import switch_model

        result = switch_model(agent, model, provider)
        result.setdefault("model", self.model_name)
        return result

    # -- send -----------------------------------------------------------
    def send(self, text: str) -> Dict[str, Any]:
        """Run one turn and return ``{ok, reply, model, tools}``.

        Never raises: failures come back as ``{"ok": False, "error"}``.
        """
        message = (text or "").strip()
        if not message:
            return {"ok": False, "error": "message is required"}
        if len(message) > MAX_MESSAGE_CHARS:
            return {
                "ok": False,
                "error": f"message too long (max {MAX_MESSAGE_CHARS} chars)",
            }
        # Serialise the whole turn: history append + tool loop + persistence.
        with self._lock:
            agent = None
            try:
                agent = self._resolve_agent()
                # A leading "/name" whose answer is a fact we already hold
                # (model list, help, kg show) is answered HERE, without the
                # model: asking an LLM to describe the model list made it
                # invent one.  None means "not an offline command" - the
                # normal turn runs unchanged.
                from agent_core.commands.slash import run_slash

                offline = run_slash(agent, message)
                if offline is not None:
                    offline.setdefault("model", self.model_name)
                    offline.setdefault("tools", True)
                    return offline
                # Clear the previous outcome first: a turn that dies before
                # _finish_turn must not return the PREVIOUS answer as its own.
                agent._last_turn_answer = ""
                agent._last_turn_error = None
                _run_coro(agent.chat_nlp(message))
            except Exception as exc:  # surface any failure to the UI
                logger.debug("dashboard agent turn failed", exc_info=True)
                return {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "model": self.model_name,
                    "tools": True,
                }
            answer = str(getattr(agent, "_last_turn_answer", "") or "")
            error = getattr(agent, "_last_turn_error", None)
            if error:
                return {
                    "ok": False,
                    "error": str(error),
                    "model": self.model_name,
                    "tools": True,
                }
            if not answer:
                # The pipeline finished without text (e.g. an empty final
                # answer); say so instead of rendering a blank bubble.
                return {
                    "ok": False,
                    "error": "the turn produced no answer",
                    "model": self.model_name,
                    "tools": True,
                }
            return {
                "ok": True,
                "reply": answer,
                "model": self.model_name,
                "tools": True,
            }


def _shared_agent() -> Optional[Any]:
    """Return the live REPL agent of this process, if it has one."""
    try:
        import agent
    except Exception:  # pragma: no cover - root import must not break a request
        return None
    return getattr(agent, "_shared_agent", None)


__all__: List[str] = [
    "AgentTurnRunner",
    "ChatService",
    "MAX_HISTORY",
    "MAX_MESSAGE_CHARS",
]
