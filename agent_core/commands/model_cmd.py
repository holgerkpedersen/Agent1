"""Model command — list, switch, load, and unload LLM models via LM Studio API."""

from __future__ import annotations

import difflib
import re

from .base import Command
from agent_core.config import lmstudio_base_url
from agent_core.constants import KNOWN_MODELS, DEFAULT_MODEL, persist_model_choice
from agent_core.llm import lmstudio as _lms
from agent_core.llm import model_profiles as _profiles

from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from agent import Agent


def _format_size(bytes_val: int) -> str:
    """Return a human-readable size string."""
    gb = bytes_val / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.1f} GB"
    mb = bytes_val / (1024 ** 2)
    if mb >= 1:
        return f"{mb:.0f} MB"
    return f"{bytes_val} B"


class ModelCommand(Command):
    """Manage LLM models via the LM Studio REST API."""

    @property
    def name(self) -> str:
        return "model"

    @property
    def help_text(self) -> str:
        return "model [list|load|unload|reload|provider|profile] - Manage LLM models and providers"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        sub = args[0].strip().lower() if args else ""
        rest = args[1:] if len(args) > 1 else []

        if not args:
            self._list_models(agent, interactive=True)
            return True

        if sub in ("list", "ls"):
            self._list_models(agent)
            return True

        if sub == "provider":
            await self._handle_provider(rest, agent)
            return True

        if sub == "load":
            await self._load_model(rest, agent)
            return True

        if sub == "profile":
            await self._handle_profile(rest, agent)
            return True

        if sub == "unload":
            await self._unload_model(rest, agent)
            return True

        if sub == "reload":
            # Resolve what is actually loaded and sync
            self._sync_with_lmstudio(agent)
            return True

        # Subcommand not matched — treat as model name
        await self._switch_model(args, agent)
        return True

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _fetch_models(self) -> tuple[list[dict[str, Any]], list[str]]:
        """Return (all_models, loaded_instance_ids) from the LM Studio API."""
        models = _lms.get_models_status()
        loaded_ids = [m["instance_id"] for m in models if m["loaded"] and m["instance_id"]]
        return models, loaded_ids

    def _opencode_catalog(self, agent: "Agent") -> tuple[list[str], list[str], bool]:
        """Return (opencode-go ids, opencode-zen free ids, api_mode).

        The keyed opencode-go catalog uses the agent's real provider when it
        is one (same API-key resolution: OPENCODE_API_KEY / opencode's
        auth.json), or a freshly built provider from settings.  The keyless
        opencode-zen FREE catalog is fetched directly from ZEN_API_BASE and is
        always available without a key.
        """
        from agent_core.llm.opencode_provider import OpencodeProvider

        go_models: list[str] = []
        api_mode = False
        provider = getattr(getattr(agent, "llm", None), "_provider", None)
        # Only reuse the agent's provider for the GO catalog when it is itself
        # a go-mode provider.  A zen-mode provider (current model is e.g.
        # opencode-zen/hy3-free) would call list_models() against ZEN_API_BASE
        # and return opencode-zen/* ids, which must NOT appear under the
        # [opencode] header — that is what caused the duplicate listing.
        if (
            provider is not None
            and type(provider).__name__ == "OpencodeProvider"
            and not getattr(provider, "zen_mode", False)
        ):
            try:
                go_models = list(provider.list_models())
                api_mode = bool(getattr(provider, "api_mode", False))
            except Exception:
                go_models, api_mode = [], bool(getattr(provider, "api_mode", False))
        else:
            try:
                from agent_core.config import load_agent_settings
                s = load_agent_settings()
                # Force a go-mode provider (never zen) so the [opencode] block
                # always reflects the keyed opencode-go catalog, even when the
                # active model is a keyless opencode-zen free model.
                oc = OpencodeProvider(
                    model_name="opencode-go/placeholder",
                    server_url=getattr(s, "opencode_server_url", "http://127.0.0.1:4096"),
                    password=getattr(s, "opencode_password", ""),
                    api_url=getattr(s, "opencode_api_url", "https://opencode.ai/zen/go/v1"),
                    api_key=getattr(s, "opencode_api_key", ""),
                )
                go_models = list(oc.list_models())
                api_mode = bool(oc.api_mode)
            except Exception:
                go_models, api_mode = [], False

        # Keyless free tier — always reachable without a key.
        zen_models = self._zen_free_catalog()
        return go_models, zen_models, api_mode

    def _zen_free_catalog(self) -> list[str]:
        """Return the keyless opencode-zen FREE model ids (no API key needed).

        These are fetched live from ZEN_API_BASE; on any failure we return an
        empty list so listing never crashes (the rest of `model list` still
        works).  Free models carry a ``-free`` suffix.
        """
        try:
            from agent_core.llm.opencode_provider import OpencodeProvider
            prov = OpencodeProvider("opencode-zen/hy3-free", read_store=False)
            return list(prov.list_models())
        except Exception:
            return []

    def _get_vram_display(self, models: list[dict[str, Any]]) -> str:
        """Build a one-line VRAM summary string."""
        loaded = [m for m in models if m["loaded"]]
        total_bytes = sum(m["size_bytes"] for m in loaded)
        if not loaded:
            return "VRAM: 0 GB — no models loaded"
        return f"VRAM: {_format_size(total_bytes)} — {len(loaded)} model(s) loaded"

    # ------------------------------------------------------------------
    #  List
    # ------------------------------------------------------------------

    def _list_models(self, agent: "Agent", interactive: bool = False) -> None:
        """Show available models per LLM provider (LM Studio + opencode).

        READ-ONLY by design (multi-shell safety): listing must never switch
        the session's model or persist a new choice — another ``agent.py``
        shell (or the LM Studio GUI) may have loaded something else, and
        silently adopting it hijacked the running session.  This session
        keeps its pinned model; ``LMStudioProvider`` auto-reloads it on
        demand when a request finds it missing from VRAM.  Adoption of what
        LM Studio currently has loaded is explicit: ``model reload`` or
        ``model <name>``.
        """
        current = agent.llm.model_name
        from agent_core.llm.provider import provider_for
        from agent_core.config import load_agent_settings

        try:
            settings = load_agent_settings()
            from agent_core.constants import load_model_json
            persisted_provider = str(load_model_json().get("provider") or "")
            active_provider = provider_for(current, settings.llm_provider, persisted_provider)
        except Exception:
            active_provider = "lmstudio"

        # ---- opencode provider models ----
        # Use the agent's REAL opencode provider when it is one, so `model
        # list` shows exactly the catalog chat is using (same API-key
        # resolution: OPENCODE_API_KEY / opencode's auth.json).  The old code
        # built a placeholder with an empty key, so api_mode was forced off
        # and the hosted catalog was unreachable — opencode models vanished
        # from the list and could not be selected by name.  The keyless
        # opencode-zen FREE tier is always fetched separately (no key).
        opencode_models, zen_models, oc_api_mode = self._opencode_catalog(agent)

        print(f"\n  Providers: lmstudio (active: {'*' if active_provider == 'lmstudio' else ' '})"
              f"  opencode (active: {'*' if active_provider == 'opencode' else ' '})")
        if opencode_models:
            print(f"  [opencode] {len(opencode_models)} model(s) — needs API key:\n")
            for key in opencode_models:
                is_current = key == current
                marker = "  *" if is_current else "   "
                print(f"{marker} {key}")
        else:
            hint = ("'opencode serve --port 4096'" if not oc_api_mode
                    else "set OPENCODE_API_KEY to reach the hosted opencode-go API")
            print(f"  [opencode] unreachable — start {hint}")
            print("            (models appear once the connection works)\n")

        # ---- opencode-zen FREE tier (no API key required) ----
        if zen_models:
            print(f"\n  [opencode-zen] {len(zen_models)} FREE model(s) — no API key:\n")
            for key in zen_models:
                is_current = key == current
                marker = "  *" if is_current else "   "
                print(f"{marker} {key}  (provider=opencode-zen, free)")
        else:
            print("\n  [opencode-zen] free tier unreachable — check network\n")

        # ---- LM Studio models ----
        models, loaded_ids = self._fetch_models()

        if not models:
            print(f"  [lmstudio] LM Studio server not reachable at {lmstudio_base_url()}")
            print("  [lmstudio] Developer tab > Start Server, or 'lms server start'")
            self._list_known_only(agent)
            return

        print(f"\n  [lmstudio] {self._get_vram_display(models)}\n")
        for i, m in enumerate(models, 1):
            key = m["key"]
            name = m["display_name"]
            params = m["params_string"] or "?"
            size = _format_size(m["size_bytes"]) if m["size_bytes"] else "?"
            loaded = m["loaded"]
            is_current = key == current

            markers = []
            if loaded:
                markers.append("loaded")
            if is_current:
                markers.append("current")

            marker_str = f"  [{', '.join(markers)}]" if markers else ""
            prefix = " *" if is_current else "  "
            print(f" {prefix}{i:>3}. {name:<40} {params:<10} {size:<8}  [lmstudio]{marker_str}")

        kinfo = KNOWN_MODELS.get(current, {})
        active_profile = agent.llm._provider._profile_name
        profile_info = f" | profile={active_profile}" if active_profile else ""
        print(f"\n  Current model: {current}{profile_info}  ({kinfo.get('desc', '')})")
        print(f"  {len(models)} models available from LM Studio API")

        # Advisory only — never switch or persist here (multi-shell safety).
        # A concurrent agent.py shell may have loaded another model; stealing
        # it (and overwriting model.json) broke the running session.  The
        # pinned model auto-reloads on demand at request time; adopting what
        # LM Studio has loaded is an explicit `model reload` / `model <name>`.
        if active_provider == "lmstudio":
            loaded_keys = [m["key"] for m in models if m["loaded"]]
            if loaded_keys and current not in loaded_keys:
                print(
                    f"\n  ⚠ {current} is not in VRAM right now "
                    f"(loaded: {', '.join(loaded_keys)})."
                )
                print(
                    f"  This session keeps {current} — it will be reloaded "
                    "automatically on the next request."
                )
                print(
                    "  To adopt a loaded model instead, run: "
                    f"model {loaded_keys[0]}   (or: model reload)"
                )
            elif not loaded_keys and current:
                print(
                    "\n  ⚠ No models loaded in LM Studio. This session stays "
                    f"pinned to: {current} (auto-reloads on next request)."
                )
        elif current:
            print(f"\n  [{active_provider}] {current} — no LM Studio sync needed.")

    def _list_known_only(self, agent: "Agent") -> None:
        """Fallback: show the hardcoded KNOWN_MODELS."""
        current = agent.llm.model_name
        for name, info in sorted(KNOWN_MODELS.items()):
            marker = " <- current" if name == current else ""
            print(f"    {name}{marker}\n      {info['desc']}")
        print()

    # ------------------------------------------------------------------
    #  Switch
    # ------------------------------------------------------------------

    async def _switch_model(self, args: list[str], agent: "Agent") -> None:
        """Switch the model — provider-aware.

        ``opencode-go/...`` names select the opencode provider directly;
        everything else goes through the LM Studio fuzzy match.

        An explicit ``--provider <lmstudio|opencode>`` (or ``-p <p>``) flag
        overrides the model-name-based routing, so e.g. ``model laguna-s-2.1
        --provider opencode`` routes a name that would normally be LM Studio
        to the opencode provider, and ``model opencode-zen/laguna-s-2.1-free
        --provider lmstudio`` does the reverse.  Without the flag, routing is
        unchanged (prefix-based + fuzzy match).
        """
        args, provider_override = self._parse_provider_flag(args)
        if provider_override is not None:
            await self._switch_model_with_provider(args, agent, provider_override)
            return

        query = " ".join(args).strip()
        lowered = query.lower()

        if lowered.startswith("opencode-go/") or lowered.startswith("opencode/"):
            old = agent.llm.model_name
            if query == old:
                print(f"  Already using: {query}")
                return
            from agent_core.config import load_agent_settings
            from agent_core.llm.provider import build_provider
            settings = load_agent_settings()
            agent.llm._provider = build_provider(settings, query)
            agent.llm.model_name = query
            persist_model_choice(query, provider="opencode")
            print(f"  Switched: {old} -> {query}  (provider=opencode)")
            return

        # Keyless opencode-zen FREE tier (e.g. "opencode-zen/hy3-free",
        # "model nemotron-3.5-lightning-free").  No API key needed.
        if lowered.startswith("opencode-zen/") or lowered.startswith("zen/"):
            q = query if lowered.startswith("opencode-zen/") else f"opencode-zen/{query}"
            old = agent.llm.model_name
            if q == old:
                print(f"  Already using: {q}")
                return
            from agent_core.config import load_agent_settings
            from agent_core.llm.provider import build_provider
            settings = load_agent_settings()
            agent.llm._provider = build_provider(settings, q)
            agent.llm.model_name = q
            persist_model_choice(q, provider="opencode")
            print(f"  Switched: {old} -> {q}  (provider=opencode-zen, free)")
            return

        models, loaded_ids = self._fetch_models()

        # Opencode catalog models (go tier + keyless zen free tier) are fetched
        # lazily — only needed if the query does NOT resolve to an LM Studio
        # model.  The opencode catalog uses the agent's real provider (with the
        # resolved API key), so it reflects what chat uses.
        #
        # Precedence for a bare query (no opencode-go/ or opencode-zen/ prefix):
        #   1. A leading number index ("8" / "8.") selects the Nth LM Studio
        #      model shown by `model list` — unambiguous, never hijacked by a
        #      catalog substring.
        #   2. A unique strong LM Studio match (exact / substring on key,
        #      display name, params).  This prevents "model 8. Laguna S 2.1 UD"
        #      or "model laguna" from being hijacked to an opencode-zen
        #      substring such as "opencode-zen/laguna-s-2.1-free".
        #   3. Fall back to the opencode catalogs (zen first, then go) so
        #      partial names like "nemotron-3.5-lightning-free" still resolve
        #      to opencode-zen — difflib fuzz is NOT allowed to hijack them to
        #      a similarly-named LM Studio model.
        #   4. As a last resort, difflib against LM Studio models.
        lmstudio_match = self._resolve_lmstudio(query, models)

        if lmstudio_match is None and not models:
            print("  LM Studio not reachable — trying hardcoded list.")
            await self._switch_known(query, agent)
            return

        if lmstudio_match:
            current = agent.llm.model_name
            if lmstudio_match == current:
                print(f"  Already using: {lmstudio_match}")
                return

            target = next((m for m in models if m["key"] == lmstudio_match), None)
            if target and not target["loaded"]:
                print(f"  Loading {lmstudio_match} ...")
                ok, msg = self._try_load_or_unload(lmstudio_match)
                if not ok:
                    print(f"  Could not load: {msg}")
                    return
                print(f"  {msg}")

            info = KNOWN_MODELS.get(lmstudio_match, {})
            old = agent.llm.model_name
            agent.llm.model_name = lmstudio_match
            persist_model_choice(lmstudio_match, provider="lmstudio")
            # Rebuild the provider: a previously selected opencode provider must
            # not keep receiving LM Studio models (it would 401 on the hosted
            # API).
            self._rebuild_provider(agent, lmstudio_match)
            print(f"  Switched: {old} -> {lmstudio_match}  ({info.get('desc', '')})")
            return

        # No LM Studio match — try the opencode catalogs.
        opencode_models, zen_models, _ = self._opencode_catalog(agent)
        # The keyless free tier is checked first so "model nemotron-3.5-
        # lightning-free" resolves to opencode-zen, not a paid opencode-go
        # substring or an LM Studio model.
        zen_match = self._resolve_opencode_match(query, zen_models)
        if zen_match:
            old = agent.llm.model_name
            if zen_match == old:
                print(f"  Already using: {zen_match}")
                return
            from agent_core.config import load_agent_settings
            from agent_core.llm.provider import build_provider
            settings = load_agent_settings()
            agent.llm._provider = build_provider(settings, zen_match)
            agent.llm.model_name = zen_match
            persist_model_choice(zen_match, provider="opencode")
            print(f"  Switched: {old} -> {zen_match}  (provider=opencode-zen, free)")
            return
        oc_match = self._resolve_opencode_match(query, opencode_models)
        if oc_match:
            old = agent.llm.model_name
            if oc_match == old:
                print(f"  Already using: {oc_match}")
                return
            from agent_core.config import load_agent_settings
            from agent_core.llm.provider import build_provider
            settings = load_agent_settings()
            agent.llm._provider = build_provider(settings, oc_match)
            agent.llm.model_name = oc_match
            persist_model_choice(oc_match, provider="opencode")
            print(f"  Switched: {old} -> {oc_match}  (provider=opencode)")
            return

        # Last resort: difflib against LM Studio models (no catalog match).
        lmstudio_fuzzy = self._resolve_lmstudio_fuzzy(query, models)
        if lmstudio_fuzzy:
            current = agent.llm.model_name
            if lmstudio_fuzzy == current:
                print(f"  Already using: {lmstudio_fuzzy}")
                return
            target = next((m for m in models if m["key"] == lmstudio_fuzzy), None)
            if target and not target["loaded"]:
                print(f"  Loading {lmstudio_fuzzy} ...")
                ok, msg = self._try_load_or_unload(lmstudio_fuzzy)
                if not ok:
                    print(f"  Could not load: {msg}")
                    return
                print(f"  {msg}")
            info = KNOWN_MODELS.get(lmstudio_fuzzy, {})
            old = agent.llm.model_name
            agent.llm.model_name = lmstudio_fuzzy
            persist_model_choice(lmstudio_fuzzy, provider="lmstudio")
            self._rebuild_provider(agent, lmstudio_fuzzy)
            print(f"  Switched: {old} -> {lmstudio_fuzzy}  ({info.get('desc', '')})")
            return

        # No LM Studio match and no opencode match — nothing resolves.
        print(f"  No model matching '{query}'")
        self._list_models(agent)
        return

    def _parse_provider_flag(self, args: list[str]) -> tuple[list[str], str | None]:
        """Strip an optional ``--provider <p>`` / ``-p <p>`` flag from *args*.

        Returns ``(remaining_args, provider_override)`` where *provider_override*
        is ``"lmstudio"`` or ``"opencode"`` (lower-cased) when present, else
        ``None``.  Validation of the value happens in the caller so we can
        print a friendly error.
        """
        remaining: list[str] = []
        override: str | None = None
        i = 0
        while i < len(args):
            token = args[i]
            if token in ("--provider", "-p"):
                if i + 1 < len(args):
                    override = args[i + 1].strip().lower()
                    i += 2
                    continue
                # Flag with no value — leave it in `remaining` so the caller
                # reports the missing value, and don't set an override.
                remaining.append(token)
                i += 1
                continue
            # Also accept the joined form ``--provider=opencode``.
            if token.startswith("--provider="):
                override = token.split("=", 1)[1].strip().lower()
                i += 1
                continue
            remaining.append(token)
            i += 1
        return remaining, override

    async def _switch_model_with_provider(
        self, args: list[str], agent: "Agent", provider: str
    ) -> None:
        """Switch model with an explicitly chosen provider.

        Bypasses the model-name prefix routing: the user said which provider
        to use, so we honor it instead of inferring from the name.  The model
        name is matched against the appropriate catalog (LM Studio API for
        ``lmstudio``, opencode catalogs for ``opencode``) and persisted with
        the explicit provider.
        """
        query = " ".join(args).strip()
        if not query:
            print("  Specify a model name, e.g. `model <name> --provider <lmstudio|opencode>`.")
            return

        if provider not in ("lmstudio", "opencode"):
            print(f"  Unknown provider '{provider}'. Options: lmstudio, opencode.")
            return

        from agent_core.config import load_agent_settings
        from agent_core.llm.provider import build_provider
        settings = load_agent_settings()

        if provider == "opencode":
            # Match against the opencode catalogs (zen free first, then go).
            opencode_models, zen_models, _ = self._opencode_catalog(agent)
            zen_match = self._resolve_opencode_match(query, zen_models)
            if zen_match:
                old = agent.llm.model_name
                if zen_match == old:
                    print(f"  Already using: {zen_match}")
                    return
                agent.llm._provider = build_provider(settings, zen_match)
                agent.llm.model_name = zen_match
                persist_model_choice(zen_match, provider="opencode")
                print(f"  Switched: {old} -> {zen_match}  (provider=opencode-zen, free)")
                return
            oc_match = self._resolve_opencode_match(query, opencode_models)
            if oc_match:
                old = agent.llm.model_name
                if oc_match == old:
                    print(f"  Already using: {oc_match}")
                    return
                agent.llm._provider = build_provider(settings, oc_match)
                agent.llm.model_name = oc_match
                persist_model_choice(oc_match, provider="opencode")
                print(f"  Switched: {old} -> {oc_match}  (provider=opencode)")
                return
            # Bare name — treat as an opencode-go model id directly.
            q = query if query.startswith("opencode-go/") else f"opencode-go/{query}"
            old = agent.llm.model_name
            if q == old:
                print(f"  Already using: {q}")
                return
            agent.llm._provider = build_provider(settings, q, provider_override="opencode")
            agent.llm.model_name = q
            persist_model_choice(q, provider="opencode")
            print(f"  Switched: {old} -> {q}  (provider=opencode)")
            return

        # provider == "lmstudio" — match against the LM Studio API.
        models, _ = self._fetch_models()
        if not models:
            print("  LM Studio not reachable — trying hardcoded list.")
            await self._switch_known(query, agent)
            return

        lmstudio_match = self._resolve_lmstudio(query, models)
        if lmstudio_match is None:
            lmstudio_match = self._resolve_lmstudio_fuzzy(query, models)
        if not lmstudio_match:
            print(f"  No LM Studio model matching '{query}'")
            self._list_models(agent)
            return

        current = agent.llm.model_name
        if lmstudio_match == current:
            print(f"  Already using: {lmstudio_match}")
            return

        target = next((m for m in models if m["key"] == lmstudio_match), None)
        if target and not target["loaded"]:
            print(f"  Loading {lmstudio_match} ...")
            ok, msg = self._try_load_or_unload(lmstudio_match)
            if not ok:
                print(f"  Could not load: {msg}")
                return
            print(f"  {msg}")

        info = KNOWN_MODELS.get(lmstudio_match, {})
        old = agent.llm.model_name
        agent.llm.model_name = lmstudio_match
        persist_model_choice(lmstudio_match, provider="lmstudio")
        self._rebuild_provider(agent, lmstudio_match)
        print(f"  Switched: {old} -> {lmstudio_match}  ({info.get('desc', '')})")
        return

    def _rebuild_provider(self, agent: "Agent", model_name: str) -> None:
        """Rebuild the agent's LLM provider for *model_name* (provider-aware)."""
        from agent_core.config import load_agent_settings
        from agent_core.llm.provider import build_provider

        agent.llm._provider = build_provider(load_agent_settings(), model_name)

    async def _handle_provider(self, args: list[str], agent: "Agent") -> None:
        """`model provider [lmstudio|opencode]` — show or switch the provider."""
        if not args:
            from agent_core.config import load_agent_settings
            from agent_core.llm.provider import provider_for
            from agent_core.constants import load_model_json
            settings = load_agent_settings()
            persisted_provider = str(load_model_json().get("provider") or "")
            current = provider_for(agent.llm.model_name, settings.llm_provider, persisted_provider)
            print(f"  Provider: {current}  (model: {agent.llm.model_name})")
            print(f"  Persisted provider: {persisted_provider or '(auto)'}")
            print("  Options: model provider lmstudio | model provider opencode")
            print("  Pick a model + provider explicitly: model <name> --provider <lmstudio|opencode>")
            return

        target = args[0].strip().lower()
        if target not in ("lmstudio", "opencode"):
            print(f"  Unknown provider '{target}'. Options: lmstudio, opencode.")
            return

        from agent_core.constants import load_model_json, persist_model_choice
        from agent_core.config import load_agent_settings
        from agent_core.llm.provider import build_provider, provider_for

        settings = load_agent_settings()
        from agent_core.constants import load_model_json
        persisted_provider = str(load_model_json().get("provider") or "")
        current = provider_for(agent.llm.model_name, settings.llm_provider, persisted_provider)
        if target == current:
            print(f"  Already on provider '{target}'.")
            return

        if target == "opencode":
            model = settings.opencode_model
        else:
            from agent_core.constants import DEFAULT_MODEL
            persisted = load_model_json()
            model = str(persisted.get("model") or DEFAULT_MODEL)
            if model.startswith("opencode"):
                model = DEFAULT_MODEL

        agent.llm._provider = build_provider(settings, model)
        agent.llm.model_name = model
        persist_model_choice(model, provider=target)
        print(f"  Provider switched: {current} -> {target}  (model: {model})")

    async def _switch_known(self, query: str, agent: "Agent") -> None:
        """Fallback switch using hardcoded KNOWN_MODELS."""
        current = agent.llm.model_name
        if query == current:
            print(f"  Already using: {current}")
            return
        if query in KNOWN_MODELS:
            agent.llm.model_name = query
            self._persist_model(query)
            self._rebuild_provider(agent, query)
            print(f"  Switched: {current} -> {query}")
            return
        sub_matches = [m for m in KNOWN_MODELS if query.lower() in m.lower()]
        if sub_matches:
            best = sub_matches[0]
            agent.llm.model_name = best
            self._persist_model(best)
            self._rebuild_provider(agent, best)
            print(f"  Switched: {current} -> {best}")
            return
        close = difflib.get_close_matches(query, KNOWN_MODELS.keys(), n=1, cutoff=0.3)
        if close:
            agent.llm.model_name = close[0]
            self._persist_model(close[0])
            self._rebuild_provider(agent, close[0])
            print(f"  Switched: {current} -> {close[0]}")
            return
        print(f"  No match for '{query}'")

    # ------------------------------------------------------------------
    #  Sync (reload)
    # ------------------------------------------------------------------

    def _sync_with_lmstudio(self, agent: "Agent") -> None:
        """Check what LM Studio has loaded and align agent's model_name.

        Only meaningful when the active provider is lmstudio: an opencode
        model never needs LM Studio loading, so sync must not replace the
        user's chosen model with the LM Studio-loaded one.
        """
        current = agent.llm.model_name
        try:
            from agent_core.config import load_agent_settings
            from agent_core.constants import load_model_json
            from agent_core.llm.provider import provider_for

            settings = load_agent_settings()
            persisted_provider = str(load_model_json().get("provider") or "")
            active_provider = provider_for(current, settings.llm_provider, persisted_provider)
        except Exception:
            active_provider = "lmstudio"
        if active_provider != "lmstudio":
            print(f"  {current} is an {active_provider} model — no LM Studio sync needed.")
            return

        models, loaded_ids = self._fetch_models()
        current = agent.llm.model_name

        if not models:
            print("  LM Studio not reachable.")
            return

        loaded = [m for m in models if m["loaded"]]
        if not loaded:
            print(f"  No models loaded in LM Studio. Agent is set to: {current}")
            return

        active = loaded[0]
        active_key = active["key"]

        if active_key == current:
            print(f"  Agent matches LM Studio: {current}  ({_format_size(active['size_bytes'])})")
        else:
            print(f"  Agent: {current}  |  LM Studio has: {active_key}")
            print("  Syncing agent to match LM Studio...")
            agent.llm.model_name = active_key
            self._persist_model(active_key)
            self._rebuild_provider(agent, active_key)
            print(f"  Done: {active_key}")

    def _persist_model(self, model_name: str) -> None:
        """Persist the active model choice to disk (LM Studio sync path)."""
        persist_model_choice(model_name, provider="lmstudio")

    # ------------------------------------------------------------------
    #  Load / Unload
    # ------------------------------------------------------------------

    def _unload_current_if_needed(self, models: list[dict[str, Any]], new_key: str) -> None:
        """Unload currently loaded models to free VRAM before loading a new one."""
        loaded = [m for m in models if m["loaded"] and m["key"] != new_key]
        for m in loaded:
            ok, msg = _lms.unload_model(m["instance_id"])
            if ok:
                print(f"    Freed {_format_size(m['size_bytes'])} — {msg}")

    def _try_load_or_unload(self, model_key: str) -> tuple[bool, str]:
        """Try to load *model_key*.  If it fails with a space/memory error,
        unload current models and retry once.
        """
        ok, msg = _lms.load_model(model_key)
        if ok:
            return True, msg

        if any(kw in msg.lower() for kw in ("space", "memory", "vram", "failed to load", "allocation")):
            models, _ = self._fetch_models()
            loaded = [m for m in models if m["loaded"] and m["key"] != model_key]
            if loaded:
                print(f"    Not enough VRAM — unloading {len(loaded)} model(s) first")
                self._unload_current_if_needed(models, model_key)
                return _lms.load_model(model_key)

        return False, msg

    async def _load_model(self, rest: list[str], agent: "Agent") -> None:
        """Load a model into LM Studio and optionally switch to it."""
        query = " ".join(rest).strip()
        if not query:
            print("  Usage: model load <name>")
            return

        resolved = _lms.resolve_model_name(query)
        if not resolved:
            # Try hardcoded
            matches = difflib.get_close_matches(query, KNOWN_MODELS.keys(), n=1, cutoff=0.3)
            resolved = matches[0] if matches else query

        # Check if already loaded — skip API call if so
        models, loaded_ids = self._fetch_models()
        loaded_keys = [m["key"] for m in models if m["loaded"]]
        if resolved in loaded_keys:
            print(f"  Already loaded: {resolved}")
            agent.llm.model_name = resolved
            self._persist_model(resolved)
            print(f"  Switched to: {resolved}")
            return

        print(f"  Loading: {resolved} ...")
        ok, msg = self._try_load_or_unload(resolved)
        if ok:
            print(f"  {msg}")
            # Auto-switch to the loaded model
            agent.llm.model_name = resolved
            self._persist_model(resolved)
            print(f"  Switched to: {resolved}")
        else:
            print(f"  Error: {msg}")

    async def _handle_profile(self, rest: list[str], agent: "Agent") -> None:
        """model profile [list|save|delete|name] — manage model profiles."""
        sub = rest[0].strip().lower() if rest else "list"
        args = rest[1:] if len(rest) > 1 else []

        if sub == "list":
            profiles = _profiles.list_profiles()
            if not profiles:
                print("  No profiles.")
                return
            current_profile = getattr(agent.llm, "_profile_name", None)
            print(f"\n  Profiles ({len(profiles)}):")
            for p in sorted(profiles, key=lambda x: x.name):
                marker = " *" if p.name == current_profile else "  "
                print(f" {marker} {p.name:<20} temp={p.temperature}  max_tok={p.max_tokens}  {p.description}")
            print()
            return

        if sub == "save":
            name = args[0].strip().lower() if args else ""
            if not name:
                print("  Usage: model profile save <name> [--temp 0.3] [--max-tokens 8000] [--desc \"text\"]")
                return
            model = agent.llm.model_name
            temp = 0.7
            max_tok = 50000
            desc = ""
            for i, a in enumerate(args):
                if a == "--temp" and i + 1 < len(args):
                    temp = float(args[i + 1])
                if a == "--max-tokens" and i + 1 < len(args):
                    max_tok = int(args[i + 1])
                if a == "--desc" and i + 1 < len(args):
                    desc = args[i + 1]
            profile = _profiles.ProfileMetadata(
                name=name, description=desc, model=model,
                temperature=temp, max_tokens=max_tok,
            )
            _profiles.save_profile(profile)
            print(f"  Saved: {name} (model={model}, temp={temp}, max_tok={max_tok})")
            return

        if sub == "delete":
            name = args[0].strip().lower() if args else ""
            if not name:
                print("  Usage: model profile delete <name>")
                return
            if _profiles.delete_profile(name):
                print(f"  Deleted: {name}")
            else:
                print(f"  Cannot delete built-in profile: {name}")
            return

        if sub == "use":
            name = args[0].strip().lower() if args else ""
            if not name:
                print("  Usage: model profile use <name>")
                return
            try:
                profile = _profiles.get_profile(name)
            except KeyError:
                print(f"  No profile: {name}")
                return
            agent.llm._profile_name = name
            agent.llm._provider.apply_profile(
                name, profile.temperature, profile.max_tokens,
            )
            persist_model_choice(agent.llm.model_name)
            # Persist profile name in model.json
            from agent_core.constants import load_model_json, save_model_json
            data = load_model_json()
            data["profile"] = name
            save_model_json(data)
            print(f"  Profile active: {name} ({profile.description})")
            return

        # Default: show specific profile
        try:
            profile = _profiles.get_profile(sub)
            print(f"  {profile.name}: {profile.description}")
            print(f"    model={profile.model or '(any)'}  temp={profile.temperature}  max_tok={profile.max_tokens}")
        except KeyError:
            print(f"  No profile: {sub}")

    async def _unload_model(self, rest: list[str], agent: "Agent") -> None:
        """Unload a model from LM Studio."""
        query = " ".join(rest).strip()
        models, loaded_ids = self._fetch_models()

        if not loaded_ids:
            print("  No models loaded in LM Studio.")
            return

        if not query or query in ("--all", "-a"):
            print(f"  Unloading all ({len(loaded_ids)} model(s)) ...")
            for lid in loaded_ids:
                ok, msg = _lms.unload_model(lid)
                print(f"    {msg}")
            return

        # Fuzzy-match against loaded instance IDs
        matches = difflib.get_close_matches(query, loaded_ids, n=1, cutoff=0.3)
        if not matches:
            sub = [lid for lid in loaded_ids if query.lower() in lid.lower()]
            if sub:
                matches = [sub[0]]
        if not matches:
            # Try matching model keys
            keys = [m["key"] for m in models if m["loaded"]]
            key_match = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
            if key_match:
                target = next((m for m in models if m["key"] == key_match[0]), None)
                if target and target["instance_id"]:
                    matches = [target["instance_id"]]

        if not matches:
            print(f"  No loaded model matching '{query}'")
            print(f"  Loaded: {', '.join(loaded_ids)}")
            return

        print(f"  Unloading: {matches[0]} ...")
        ok, msg = _lms.unload_model(matches[0])
        print(f"  {msg}")

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _resolve_opencode_match(self, query: str, opencode_models: list[str]) -> str | None:
        """Fuzzy-match *query* against the opencode model catalog.

        Mirrors the LM Studio fuzzy logic (exact, substring, then difflib) but
        against the opencode id list so partial names like
        ``nemotron-3.5-lightning-free`` resolve to ``opencode-go/...`` instead
        of being silently switched to an LM Studio model.
        """
        if not query or not opencode_models:
            return None
        qlo = query.lower()

        # Exact (also accept an unprefixed id, e.g. "deepseek-v4-flash")
        for m in opencode_models:
            if qlo == m.lower() or qlo == m.lower().split("/")[-1]:
                return str(m)
        # Substring on the full id
        sub = [m for m in opencode_models if qlo in m.lower()]
        if len(sub) == 1:
            return str(sub[0])
        # Substring on the unprefixed tail
        sub_tail = [m for m in opencode_models if qlo in m.lower().split("/")[-1]]
        if len(sub_tail) == 1:
            return str(sub_tail[0])
        # difflib on full ids
        matches = difflib.get_close_matches(query, opencode_models, n=1, cutoff=0.3)
        if matches:
            return str(matches[0])
        # difflib on unprefixed tails
        tails = [m.split("/")[-1] for m in opencode_models]
        matches = difflib.get_close_matches(query, tails, n=1, cutoff=0.3)
        if matches:
            for m in opencode_models:
                if m.split("/")[-1] == matches[0]:
                    return str(m)
        return None

    def _resolve_match(self, query: str, models: list[dict[str, Any]]) -> str | None:
        """Fuzzy-match *query* against model keys and display names."""
        if not query:
            return None
        # Strong matches first (exact / substring on key, name, params).
        strong = self._resolve_match_strong(query, models)
        if strong:
            return strong
        qlo = query.lower()

        # difflib on keys
        keys = [m["key"] for m in models]
        matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
        if matches:
            return str(matches[0])

        # difflib on display names
        names = [m["display_name"] for m in models]
        matches = difflib.get_close_matches(query, names, n=1, cutoff=0.3)
        if matches:
            for m in models:
                if m["display_name"] == matches[0]:
                    return str(m["key"])

        return None

    def _resolve_match_strong(self, query: str, models: list[dict[str, Any]]) -> str | None:
        """Exact / substring match against LM Studio model keys, display names
        and params — no difflib fuzz.  Used to decide LM Studio precedence
        over the opencode catalogs without spurious fuzzy collisions."""
        if not query or not models:
            return None
        qlo = query.lower()

        # Search keys and display names (return the key)
        for m in models:
            if qlo == m["key"].lower() or qlo == m["display_name"].lower():
                return str(m["key"])

        # Substring match on keys
        sub_keys = [m for m in models if qlo in m["key"].lower()]
        if len(sub_keys) == 1:
            return str(sub_keys[0]["key"])

        # Substring match on display names
        sub_disp = [m for m in models if qlo in m["display_name"].lower()]
        if len(sub_disp) == 1:
            return str(sub_disp[0]["key"])

        # Substring match on params (e.g. "9b", "27b")
        sub_params = [m for m in models if m["params_string"] and qlo in m["params_string"].lower()]
        if len(sub_params) == 1:
            return str(sub_params[0]["key"])

        return None

    def _resolve_lmstudio(self, query: str, models: list[dict[str, Any]]) -> str | None:
        """Resolve *query* to an LM Studio model key, or ``None``.

        A bare query has two kinds of LM Studio references:

        * A leading number index — ``8`` or ``8.`` — selects the Nth model in
          the ``model list`` output.  This is the unambiguous, position-based
          selector and is checked FIRST so that e.g.
          ``model 8. Laguna S 2.1 UD`` selects LM Studio item #8 rather than
          being hijacked by an opencode-zen substring match
          (``opencode-zen/laguna-s-2.1-free``).
        * Otherwise the query is matched (exact / substring only, no difflib)
          against LM Studio keys / display names / params.  Difflib fuzz is
          intentionally excluded here so a partial opencode name like
          ``nemotron-3.5-lightning-free`` is not hijacked to a similarly-named
          LM Studio model — it should resolve to the opencode catalog instead.

        Returns ``None`` when the query is not a strong LM Studio reference.
        """
        if not query or not models:
            return None

        stripped = query.strip()
        m = re.match(r"^(\d+)\.?\s*(.*)$", stripped)
        if m and not m.group(2).strip():
            # Bare number index: "8", "8.", " 8." — 1-based position in the
            # `model list` output.  An out-of-range bare number is not an LM
            # Studio reference.
            idx = int(m.group(1))
            if 1 <= idx <= len(models):
                return str(models[idx - 1]["key"])
            return None

        # Full-query strong match (exact / substring on key, display, params).
        # This catches "9b" (params), "laguna" (display name), etc. BEFORE the
        # number-stripping below, so a query like "9b" is never read as
        # "model #9".
        full = self._resolve_match_strong(query, models)
        if full:
            return full

        # Numbered reference: "8. Laguna S 2.1 UD" — the "8." is the listing
        # position, the remainder is the model name.  A numbered reference is
        # an LM Studio listing reference, so the remainder is matched only
        # against LM Studio models (never the opencode catalogs).
        if m:
            rest = m.group(2).strip()
            if rest:
                return self._resolve_match_strong(rest, models)

        return None

    def _resolve_lmstudio_fuzzy(self, query: str, models: list[dict[str, Any]]) -> str | None:
        """Difflib-only fallback for LM Studio — used only when neither a strong
        LM Studio match nor an opencode catalog match was found."""
        if not query or not models:
            return None
        # difflib on keys
        keys = [m["key"] for m in models]
        matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
        if matches:
            return str(matches[0])
        # difflib on display names
        names = [m["display_name"] for m in models]
        matches = difflib.get_close_matches(query, names, n=1, cutoff=0.3)
        if matches:
            for m in models:
                if m["display_name"] == matches[0]:
                    return str(m["key"])
        return None