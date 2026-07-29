"""Model command — list, switch, load, and unload LLM models via LM Studio API."""

import os
import difflib

from .base import Command
from agent_core.constants import KNOWN_MODELS, DEFAULT_MODEL, load_model_json, save_model_json
from agent_core.llm import lmstudio as _lms

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
        return "model [list|load|unload|reload|name] — Manage LLM models"

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
            print("  (LM Studio not reachable — showing cached list)")
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
        print(f"\n  Current model: {current}  ({kinfo.get('desc', '')})")
        print(f"  {len(models)} models available from LM Studio API")

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
        keys = [m["key"] for m in models]
        matched = self._resolve_match(query, keys)

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
            ok, msg = _lms.load_model(matched)
            if not ok:
                print(f"  Could not load: {msg}")
                return
            print(f"  {msg}")

        # Update the agent's model name
        info = KNOWN_MODELS.get(matched, {})
        old = agent.llm.model_name
        agent.llm.model_name = matched
        self._persist_model(matched)
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

        print(f"  Loading: {resolved} ...")
        ok, msg = _lms.load_model(resolved)
        if ok:
            print(f"  {msg}")
            # Auto-switch to the loaded model
            agent.llm.model_name = resolved
            self._persist_model(resolved)
            print(f"  Switched to: {resolved}")
        else:
            print(f"  Error: {msg}")

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

    def _resolve_match(self, query: str, keys: list[str]) -> str | None:
        """Fuzzy-match *query* against a list of model keys."""
        if not query:
            return None
        if query in keys:
            return query
        sub = [k for k in keys if query.lower() in k.lower()]
        if len(sub) == 1:
            return sub[0]
        matches = difflib.get_close_matches(query, keys, n=1, cutoff=0.3)
        return matches[0] if matches else None

    def _persist_model(self, model_name: str) -> None:
        """Write the current model to model.json and .env."""
        data = load_model_json()
        data["model"] = model_name
        save_model_json(data)

        env_path = ".env"
        lines = []
        found = False
        if os.path.exists(env_path):
            with open(env_path, "r") as ef:
                lines = ef.readlines()
        with open(env_path, "w") as ef:
            for line in lines:
                if line.startswith("AGENT_MODEL="):
                    ef.write(f"AGENT_MODEL={model_name}\n")
                    found = True
                else:
                    ef.write(line)
            if not found:
                ef.write(f"\nAGENT_MODEL={model_name}\n")
