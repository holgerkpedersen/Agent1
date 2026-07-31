"""Model command — list, switch, load, and unload LLM models via LM Studio API."""

import os
import difflib

from .base import Command
from agent_core.constants import KNOWN_MODELS, DEFAULT_MODEL, persist_model_choice
from agent_core.llm import lmstudio as _lms
from agent_core.llm import model_profiles as _profiles

from typing import TYPE_CHECKING
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
        return "model [list|load|unload|reload|profile|name] — Manage LLM models"

    async def execute(self, args: list[str], agent: "Agent") -> bool:
        sub = args[0].strip().lower() if args else ""
        rest = args[1:] if len(args) > 1 else []

        if not args:
            self._list_models(agent, interactive=True)
            return True

        if sub in ("list", "ls"):
            self._list_models(agent)
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

    def _fetch_models(self) -> tuple[list[dict], list[str]]:
        """Return (all_models, loaded_instance_ids) from the LM Studio API."""
        models = _lms.get_models_status()
        loaded_ids = [m["instance_id"] for m in models if m["loaded"] and m["instance_id"]]
        return models, loaded_ids

    def _get_vram_display(self, models: list[dict]) -> str:
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
        """Show available models from the LM Studio API."""
        models, loaded_ids = self._fetch_models()

        if not models:
            print("  (LM Studio server not reachable at localhost:1234)")
            print("  Is the LM Studio server running? Developer tab > Start Server, or 'lms server start'")
            print("  Showing cached model list instead:\n")
            self._list_known_only(agent)
            return

        print(f"\n  {self._get_vram_display(models)}\n")
        current = agent.llm.model_name

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
            print(f" {prefix}{i:>3}. {name:<40} {params:<10} {size:<8}{marker_str}")

        kinfo = KNOWN_MODELS.get(current, {})
        profile_info = f" | profile={agent.llm._provider._profile_name}" if agent.llm._provider._profile_name else ""
        print(f"\n  Current model: {current}{profile_info}  ({kinfo.get('desc', '')})")
        print(f"  {len(models)} models available from LM Studio API")

        # Auto-sync: if agent's model isn't loaded in LM Studio, switch to what is
        loaded_keys = [m["key"] for m in models if m["loaded"]]
        if loaded_keys and current not in loaded_keys:
            print(f"\n  ⚠ {current} not loaded, switching to {loaded_keys[0]}")
            agent.llm.model_name = loaded_keys[0]
            persist_model_choice(loaded_keys[0])
        elif not loaded_keys and current:
            print(f"\n  ⚠ No models loaded in LM Studio. Agent is set to: {current}")

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
        """Fuzzy-match args against real API model keys and switch."""
        query = " ".join(args).strip()
        models, loaded_ids = self._fetch_models()

        if not models:
            print("  LM Studio not reachable — trying hardcoded list.")
            await self._switch_known(query, agent)
            return

        current = agent.llm.model_name
        matched = self._resolve_match(query, models)

        if not matched:
            print(f"  No model matching '{query}'")
            self._list_models(agent)
            return

        if matched == current:
            print(f"  Already using: {matched}")
            return

        target = next((m for m in models if m["key"] == matched), None)

        if target and not target["loaded"]:
            print(f"  Loading {matched} ...")
            ok, msg = self._try_load_or_unload(matched)
            if not ok:
                print(f"  Could not load: {msg}")
                return
            print(f"  {msg}")

        # Update the agent's model name
        info = KNOWN_MODELS.get(matched, {})
        old = agent.llm.model_name
        agent.llm.model_name = matched
        persist_model_choice(matched)
        print(f"  Switched: {old} -> {matched}  ({info.get('desc', '')})")

    async def _switch_known(self, query: str, agent: "Agent") -> None:
        """Fallback switch using hardcoded KNOWN_MODELS."""
        current = agent.llm.model_name
        if query == current:
            print(f"  Already using: {current}")
            return
        if query in KNOWN_MODELS:
            agent.llm.model_name = query
            self._persist_model(query)
            print(f"  Switched: {current} -> {query}")
            return
        sub_matches = [m for m in KNOWN_MODELS if query.lower() in m.lower()]
        if sub_matches:
            best = sub_matches[0]
            agent.llm.model_name = best
            self._persist_model(best)
            print(f"  Switched: {current} -> {best}")
            return
        close = difflib.get_close_matches(query, KNOWN_MODELS.keys(), n=1, cutoff=0.3)
        if close:
            agent.llm.model_name = close[0]
            self._persist_model(close[0])
            print(f"  Switched: {current} -> {close[0]}")
            return
        print(f"  No match for '{query}'")

    # ------------------------------------------------------------------
    #  Sync (reload)
    # ------------------------------------------------------------------

    def _sync_with_lmstudio(self, agent: "Agent") -> None:
        """Check what LM Studio has loaded and align agent's model_name."""
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
        active_id = active.get("instance_id", active_key)

        if active_key == current:
            print(f"  Agent matches LM Studio: {current}  ({_format_size(active['size_bytes'])})")
        else:
            print(f"  Agent: {current}  |  LM Studio has: {active_key}")
            print(f"  Syncing agent to match LM Studio...")
            agent.llm.model_name = active_key
            self._persist_model(active_key)
            print(f"  Done: {active_key}")

    # ------------------------------------------------------------------
    #  Load / Unload
    # ------------------------------------------------------------------

    def _unload_current_if_needed(self, models: list[dict], new_key: str) -> None:
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
            agent.llm._provider._profile_name = name
            agent.llm._provider.temperature = profile.temperature
            agent.llm._provider.max_tokens = profile.max_tokens
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

    def _resolve_match(self, query: str, models: list[dict]) -> str | None:
        """Fuzzy-match *query* against model keys and display names."""
        if not query:
            return None
        qlo = query.lower()

        # Search keys and display names (return the key)
        for m in models:
            if qlo == m["key"].lower() or qlo == m["display_name"].lower():
                return m["key"]

        # Substring match on keys
        sub_keys = [m for m in models if qlo in m["key"].lower()]
        if len(sub_keys) == 1:
            return sub_keys[0]["key"]

        # Substring match on display names
        sub_disp = [m for m in models if qlo in m["display_name"].lower()]
        if len(sub_disp) == 1:
            return sub_disp[0]["key"]

        # Substring match on params (e.g. "9b", "27b")
        sub_params = [m for m in models if m["params_string"] and qlo in m["params_string"].lower()]
        if len(sub_params) == 1:
            return sub_params[0]["key"]

        # difflib on keys
        keys = [m["key"] for m in models]
        matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
        if matches:
            return matches[0]

        # difflib on display names
        names = [m["display_name"] for m in models]
        matches = difflib.get_close_matches(query, names, n=1, cutoff=0.3)
        if matches:
            for m in models:
                if m["display_name"] == matches[0]:
                    return m["key"]

        return None
