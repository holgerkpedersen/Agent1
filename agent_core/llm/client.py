"""LLMClient — thin LLM client wrapper for backward compatibility.

Extracted from agent.py (step t08).  All real LLM logic lives in
:mod:`agent_core.llm.provider` / the provider implementations.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator

from agent_core.colors import magenta

if TYPE_CHECKING:
    from agent_core.config import AgentSettings
    from agent_core.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


@contextmanager
def _suppress_and_log(label: str) -> Iterator[None]:
    """Run *body*, log any exception as *label*, never propagate it."""
    try:
        yield
    except Exception:
        logger.exception(label)


class LLMClient:
    """Thin wrapper around an LLM provider for backward compatibility.

    Delegates to the provider selected by :func:`build_provider`
    (decision #007 — one abstraction; LM Studio default, opencode-go selectable).
    """

    def __init__(
        self, model_name: str | None = None, api_key: str | None = None,
    ) -> None:
        from agent_core.config import load_agent_settings
        from agent_core.constants import resolve_model
        from agent_core.llm.provider import build_provider

        self._model_name: str = resolve_model(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        try:
            settings = load_agent_settings()
        except Exception:
            settings = None
        self._provider: LLMProvider = build_provider(settings, self._model_name)
        self._profile_name: str | None = None
        #: True once the user pins a profile this session (``model profile
        #: use``) or one is restored from model.json — suppresses the
        #: meta-policy profile suggestion (plan B2; a restored profile
        #: counts as pinned).
        self._profile_pinned: bool = False

        # Restore active profile from model.json on startup
        try:
            from agent_core.constants import load_model_json

            data = load_model_json()
            prof_name = data.get("profile")
            if prof_name:
                from agent_core.llm.model_profiles import get_profile
                profile = get_profile(prof_name)
                self._profile_name = prof_name
                self._profile_pinned = True
                self._provider.apply_profile(
                    prof_name, profile.temperature, profile.max_tokens,
                )
        except Exception as _prof_err:
            logger.exception('Failed to restore active profile from model.json:\n')
            self._profile_name = None

        # Make the llama-server actually serve the requested model
        self._reconcile_llama_model(settings)

    # ------------------------------------------------------------------
    # Llama server reconciliation
    # ------------------------------------------------------------------

    def _reconcile_llama_model(self, settings: AgentSettings | None) -> None:
        """Best-effort: ensure the running llama-server serves self._model_name.

        No-op unless the active provider is llama.  On any failure we log
        and continue — chat still works against whatever the server serves.
        """
        try:
            from agent_core.llm import llama_server
            from agent_core.llm.provider import provider_for
        except Exception:
            return

        if not self._model_name.startswith("llama/"):
            return

        from agent_core.constants import load_model_json

        persisted = load_model_json()
        persisted_provider = str(persisted.get("provider") or "")
        provider_setting = (
            getattr(settings, "llm_provider", "lmstudio") if settings
            else "lmstudio"
        )
        if provider_for(self._model_name, provider_setting, persisted_provider) != "llama":
            return

        provider = self._provider
        # FailoverProvider may wrap LlamaProvider — unwrap to reach the concrete one
        if type(provider).__name__ == "FailoverProvider":
            for wrapped in getattr(provider, "providers", []):
                if type(wrapped).__name__ == "LlamaProvider":
                    provider = wrapped
                    break

        if type(provider).__name__ != "LlamaProvider":
            return

        # Diagnostics go to STDERR so stdout (REPL / test captures) stays clean
        def _info(message: str) -> None:
            sys.stderr.write(message + "\n")

        with _suppress_and_log("llama model reconciliation failed"):
            api_url = getattr(provider, "api_url", None)
            if not api_url:
                return

            bare_name = self._model_name.removeprefix("llama/")
            _info(f"  [llama] ensuring '{self._model_name}' is served by llama-server...")

            served = llama_server.list_served_models(api_url)
            if served:
                current_served_model = served[0]
                if current_served_model == bare_name:
                    _info(f"  [llama] Model '{bare_name}' already served by server.")
                    provider._cached_server_model_id = current_served_model
                else:
                    _info(
                        f"  [llama] Server serving '{current_served_model}', "
                        f"attempting to ensure '{bare_name}' is served..."
                    )
                    ok, msg = llama_server.ensure_model_served(api_url, self._model_name)
                    if ok:
                        served_after = llama_server.list_served_models(api_url)
                        if served_after and served_after[0] == bare_name:
                            provider._cached_server_model_id = served_after[0]
                            _info(f"  [llama] Successfully ensured '{bare_name}' is served.")
                        else:
                            _info(
                                f"  [llama] WARNING: ensure_model_served reported "
                                f"success, but model not found in list: {msg}"
                            )
                    else:
                        _info(f"  [llama] WARNING: could not ensure model served: {msg}")
            else:
                _info(
                    f"  [llama] No model currently served. "
                    f"Attempting to ensure '{self._model_name}' is served."
                )
                ok, msg = llama_server.ensure_model_served(api_url, self._model_name)
                if ok:
                    _info(f"  [llama] Successfully ensured model served: {msg}")
                else:
                    _info(f"  [llama] WARNING: could not ensure model served: {msg}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def model_name(self, value: str) -> None:
        self._model_name = value
        self._provider.model_name = value

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send chat request to LLM via the configured provider."""
        return await self._provider.chat(messages, tools, **kwargs)

    async def chat_stream(self, messages: list[dict[str, Any]]) -> str:
        """Chat with real-time token streaming to console."""
        return await self._provider.chat_stream(messages)

    async def chat_with_continuation(
        self,
        messages: list[dict[str, Any]],
        max_continues: int = 3,
        max_tokens: int | None = None,
    ) -> str:
        """Chat with auto-resume if response gets truncated at the token limit."""
        full_response = ""
        current_messages = [dict(m) for m in messages]

        for i in range(max_continues):
            result = await self.chat(current_messages, max_tokens=max_tokens)

            if result.startswith("[Error") or result.startswith("[LM Studio"):
                return full_response or result

            full_response += result

            stripped = full_response.rstrip()
            if stripped and not stripped.endswith(
                ('```', '}', ')', ']', '"', "'", '.', '\n')
            ):
                print(magenta(
                    f"\n[auto-resume] Truncated ({len(result)} chars), "
                    f"continuing ({i+1}/{max_continues})..."
                ))
                current_messages.append({"role": "assistant", "content": result})
                current_messages.append({
                    "role": "user",
                    "content": (
                        "Continue exactly where you stopped. "
                        "Output the remaining code without repeating anything."
                    ),
                })
                continue
            else:
                break

        return full_response

    async def analyze_code(self, code: str) -> str:
        """Analyze code using LLM."""
        return await self._provider.analyze_code(code)
