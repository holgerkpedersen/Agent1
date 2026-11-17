#!/usr/bin/env python3
"""Seed the agent's knowledge graph with curated Agent1 entries.

The graph lives in ``agent_memory.json`` under the key ``knowledge_graph``
and uses the schema the ``kg`` command writes:

    {section: {key: {"value": <str|list>, "ts": <iso-utc>}}}

Sections are top-level categories (facts, files, tags, symbols); keys are
entity names; values are plain strings or JSON lists.  This script MERGES:
an existing key keeps its stored value unless ``--force`` is given, so
re-running the seed never clobbers entries the user edited via ``kg``.

Budget: the whole graph is rendered into the system prompt on every turn and
that block is capped three ways (``agent_core.run_utils``): 6000 chars for the
block, 200 chars per entry, and 40 rendered lines.  A seed that outgrows any of
them is worse than useless — the tail silently never reaches the model and long
values are cut off mid-sentence.  So keep values short and run
``python scripts/seed_kg.py --budget`` after editing; it exits non-zero when
the seed no longer fits.

Usage::

    python scripts/seed_kg.py            # add missing curated entries
    python scripts/seed_kg.py --dry-run  # show what would be added
    python scripts/seed_kg.py --force    # overwrite existing keys
    python scripts/seed_kg.py --budget   # report prompt-budget usage
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_PATH = os.path.join(REPO_ROOT, "agent_memory.json")

SECTIONS: tuple[str, ...] = ("facts", "files", "tags", "symbols")

#: Prompt caps, mirrored from agent_core.run_utils.  Held as literals so this
#: standalone script stays importable from a bare checkout (no package on
#: sys.path); tests/test_kg_seed.py asserts the mirror still matches the real
#: constants, so it cannot drift silently.
MAX_KG_BLOCK_CHARS = 6000
MAX_KG_ENTRY_CHARS = 200
MAX_KG_BLOCK_LINES = 40

#: Characters the rendered block spends on its own marker + `kg add` steering
#: before the first entry.  Measured against a rendered block and rounded up;
#: tests/test_kg_seed.py re-measures it, so an edit to the prompt text shows up
#: as a test failure instead of a quietly over-budget seed.
_BLOCK_OVERHEAD = 470

# Curated knowledge about Agent1 itself.  Section -> key -> value.
# Kept to <= MAX_KG_BLOCK_LINES entries in total: one entry is one prompt line.
SEED: dict[str, dict[str, Any]] = {
    "facts": {
        # -- the knowledge graph itself -----------------------------------
        "kg-command": (
            "kg (commands/kg_cmd.py) adds/shows/removes "
            "agent._knowledge_graph entries; each mutation calls _save_memory()."
        ),
        "clear-command": (
            "clear wipes history, knowledge graph, working memory, semantic "
            "index, files-read set and goal; it leaves session metadata alone "
            "(use 'session')."
        ),
        "kg-schema": (
            "Schema {section: {key: {value, ts}}}; sections are "
            "facts/files/tags/symbols; an emptied section is dropped."
        ),
        # -- memory & persistence ------------------------------------------
        "memory-store": (
            "MemoryStore (memory/memory_store.py) is a thread-safe cache over "
            "SQLiteStorage; its embedding cache is keyed per entry TEXT, so an "
            "append costs one embedding."
        ),
        "semantic-memory-loop": (
            "memory/distill.py record_turn writes one ledger entry per turn; "
            "types.semantic_memory_block reads it back ranked, fail-open to "
            "unranked when the embedder is down."
        ),
        # -- REPL & command registry ----------------------------------------
        "command-registry": (
            "CommandRegistry (commands/registry.py) does register/get/names/"
            "execute/print_help; interactive._register_commands is the single "
            "registration point for banner and dispatch."
        ),
        # -- tool loop & turn loop -------------------------------------------
        "tool-loop": (
            "ToolLoopRunner (llm/tool_loop.py) runs the LLM<->tool loop (max "
            "150 iterations) with in-loop compaction and a forced final answer."
        ),
        "turn-loop": (
            "turn_loop.py _run_chained_tool_loop runs one REPL turn; "
            "_finish_turn records it; check_stale_files tracks on-disk changes."
        ),
        # -- modes -------------------------------------------------------------
        "plan-mode": (
            "modes.py is_plan_mode/filter_tool_schemas/check_tool_allowed gate "
            "plan mode: mutating tools blocked, read-only allowed."
        ),
        # -- parallel & subagents -----------------------------------------------
        "parallel-delegate": (
            "llm/parallel.py + llm/tool_reply.py handle parallel delegation; "
            "subagent.py and subagent_roles.py hold the sub-agent machinery."
        ),
        # -- autonomous decisions -----------------------------------------------
        "decision-ledger": (
            "decisions.py records/searches/resolves .decisions.json; "
            "decision_improvement.py flags drift and writes accepted proposals "
            "back to the ledger."
        ),
        # -- jev decision engine -------------------------------------------------
        "jev-engine": (
            "jev_engine.py answers typed yesno/choice/score questions with "
            "probabilities on a small model; jev_strategy.py adds "
            "cascade/triage/verify."
        ),
        # -- sessions ------------------------------------------------------------
        "session-isolation": (
            "session/ holds identities, the metadata store, workspace locks and "
            "parse_and_activate; each session owns its memory and chat-history "
            "paths."
        ),
        # -- skills --------------------------------------------------------------
        "skills": (
            "skills.py loads on-demand SKILL.md bodies and injects a hint "
            "block; the read_skill NLP tool returns the body."
        ),
        # -- kanban bridge -------------------------------------------------------
        "kanban-bridge": (
            "commands/kanban_cmd.py works a card across both repos and the "
            "card<->issue keys; issue_router.py routes issues back to the board."
        ),
        # -- reply-language policy ------------------------------------------------
        "reply-language": (
            "language.py solely owns answer-language selection: "
            "detect_language/language_directive/language_block build the "
            "instruction; override_language honours a setting."
        ),
        # -- implement filename safety ---------------------------------------------
        "implement-safety": (
            "commands/implement_safety.py _is_dangerous_filename/"
            "_shadowing_stdlib_dir/_find_safe_subpackage stop a write that "
            "shadows stdlib or escapes the package."
        ),
    },
    "files": {
        "agent_core/agent.py": (
            "AgentCore (~428): state stores (~471), tool-call execution, "
            "chat_nlp; Agent = AgentCore (~1767)."
        ),
        "agent_core/cli/interactive.py": (
            "REPL registry + dispatch; registers kg (~241); run_interactive "
            "(~340) is a module function, not an Agent method."
        ),
        "agent_core/commands/kg_cmd.py": (
            "kg command: add/show/remove over agent._knowledge_graph."
        ),
        "agent_core/commands/registry.py": (
            "CommandRegistry: register/get/names/execute/print_help — the "
            "dispatch core."
        ),
        "agent_core/memory/persistence.py": (
            "Memory JSON load/save, clear_history, memory_stats."
        ),
        "agent_core/memory/memory_store.py": (
            "MemoryStore: thread-safe cache-over-backend store with optional "
            "semantic search."
        ),
        "agent_core/llm/tool_loop.py": (
            "ToolLoopRunner: the LLM<->tool loop with compaction and forced "
            "final answer."
        ),
        "agent_core/turn_loop.py": (
            "One REPL turn: chained tool loop, failover reporting, _finish_turn, "
            "stale files."
        ),
    },
    "symbols": {
        "run_interactive": (
            "interactive.py (~340): module-level REPL entry point; NOT an Agent "
            "method (the class is AgentCore)."
        ),
        "CommandRegistry.register": (
            "Registers REPL commands; KGCommand is registered from "
            "interactive.py (~241)."
        ),
        "add_entry": (
            "kg_cmd.add_entry(agent, section, key, value): creates the section, "
            "stores {value, ts}."
        ),
        "remove_entry": (
            "kg_cmd.remove_entry(agent, section, key): prunes empty sections."
        ),
        "_save_memory": (
            "persistence.py (~216): persists all memory stores atomically to "
            "agent_memory.json."
        ),
        "clear_history": (
            "persistence.py (~81): clears history, chat, files read, knowledge "
            "graph, working memory, semantic index; removes state files."
        ),
        "memory_stats": (
            "persistence.py (~62): counts chat_history, files_read, "
            "stale_files, history, working_memory, semantic_index, "
            "knowledge_graph."
        ),
        "ToolLoopRunner.run": (
            "tool_loop.py (~949; class ~813): drives messages through "
            "llm_chat_fn/execute_tool_fn until answer or budget."
        ),
        "_run_chained_tool_loop": (
            "turn_loop.py (~238): async runner for one chained REPL turn; "
            "returns text, messages, llm error, loop."
        ),
    },
    "tags": {
        "memory-subsystem": (
            "Memory: persistence.py, memory_store.py, types.py "
            "(StorageBackend/SQLiteStorage)."
        ),
        # Generated from the live registry at seed time (see
        # _live_repl_command_tag) so the count can never drift from the code.
        "repl-commands": "",
        "llm-subsystem": (
            "LLM layer: client, engines, providers (lemonade, llama, lmstudio, "
            "opencode, openrouter), orchestrator, parallel, tool_loop, retry."
        ),
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _live_repl_command_tag() -> str:
    """Return the ``repl-commands`` tag, built from the live command registry.

    The old value was a hand-maintained list of ~40 command names.  It rotted
    (commands were added without a matching edit) and, at ~400 chars, broke
    ``MAX_KG_ENTRY_CHARS``, so the prompt showed a list cut off mid-name.
    Deriving the count removes both failures: the tag stays correct and short.

    Falls back to an empty string when ``agent_core`` cannot be imported, so
    seeding stays usable from a bare checkout.
    """
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    try:
        from agent_core.cli.interactive import _build_registry
    except Exception:  # pragma: no cover - optional dependency for seeding
        return ""
    try:
        count = len(_build_registry().names())
    except Exception:  # pragma: no cover - a broken registry must not block
        return ""
    if not count:
        return ""
    return (
        f"REPL commands ({count}): all registered in one place by "
        "interactive._register_commands; 'help' lists them."
    )


def build_seed() -> dict[str, dict[str, Any]]:
    """Return :data:`SEED` with the generated entries filled in."""
    seed = {section: dict(items) for section, items in SEED.items()}
    generated = _live_repl_command_tag()
    if generated:
        seed.setdefault("tags", {})["repl-commands"] = generated
    else:
        seed.get("tags", {}).pop("repl-commands", None)
    return seed


def _one_line(value: Any) -> str:
    """Collapse a value the way the prompt renderer does."""
    return " ".join(str(value).split())


def rendered_entry(section: str, key: str, value: Any) -> str:
    """Return one prompt line for an entry: ``section/key: value``."""
    return f"{section}/{key}: {_one_line(value)}"


def rendered_chars(seed: dict[str, dict[str, Any]] | None = None) -> int:
    """Prompt cost of *seed* in characters, at the prompt's own granularity.

    Counts each entry line plus the block marker/steering, so the result is
    directly comparable to :data:`MAX_KG_BLOCK_CHARS`.
    """
    seed = build_seed() if seed is None else seed
    total = _BLOCK_OVERHEAD
    for section in SECTIONS:
        for key, value in seed.get(section, {}).items():
            total += len(rendered_entry(section, key, value)) + 1
    return total


def entry_count(seed: dict[str, dict[str, Any]] | None = None) -> int:
    """Number of prompt lines *seed* renders (one entry per line)."""
    seed = build_seed() if seed is None else seed
    return sum(len(seed.get(section, {})) for section in SECTIONS)


def oversized_entries(
    seed: dict[str, dict[str, Any]] | None = None,
) -> list[tuple[str, int]]:
    """Entries the prompt would cut off as ``(label, rendered_chars)``."""
    seed = build_seed() if seed is None else seed
    out: list[tuple[str, int]] = []
    for section in SECTIONS:
        for key, value in seed.get(section, {}).items():
            line = rendered_entry(section, key, value)
            if len(line) > MAX_KG_ENTRY_CHARS:
                out.append((f"{section}/{key}", len(line)))
    return out


def load_graph(memory_path: str | None = None) -> dict[str, dict[str, Any]]:
    """Read the knowledge graph from the memory file (empty dict if absent)."""
    if memory_path is None:
        memory_path = MEMORY_PATH
    try:
        with open(memory_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    kg = data.get("knowledge_graph")
    return kg if isinstance(kg, dict) else {}


def merge(seed: dict[str, dict[str, Any]], graph: dict[str, dict[str, Any]],
          force: bool = False) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge seed entries into *graph*; return (merged_graph, added_keys)."""
    added: list[str] = []
    merged = dict(graph)
    for section in SECTIONS:
        seed_items = seed.get(section, {})
        existing = merged.get(section)
        items: dict[str, Any] = existing if isinstance(existing, dict) else {}
        for key, value in seed_items.items():
            if key in items and not force:
                continue
            items[key] = {"value": value, "ts": _now()}
            added.append(f"{section}/{key}")
        if items:
            merged[section] = items
    return merged, added


def save_graph(
    graph: dict[str, dict[str, Any]], memory_path: str | None = None
) -> None:
    """Atomically write the merged graph back into the memory file."""
    if memory_path is None:
        memory_path = MEMORY_PATH
    data: dict[str, Any] = {}
    if os.path.exists(memory_path):
        try:
            with open(memory_path, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    data["knowledge_graph"] = graph
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(memory_path) or ".",
                                       prefix=".agent_memory", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, memory_path)
    except OSError:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def count_seed(seed: dict[str, dict[str, Any]] = None) -> int:
    """Total number of curated seed entries across all sections."""
    seed = SEED if seed is None else seed
    return sum(len(items) for items in seed.values())


def budget_report(seed: dict[str, dict[str, Any]] | None = None) -> tuple[int, bool]:
    """Return (rendered_chars, fits) for *seed* against every prompt cap."""
    total = rendered_chars(seed)
    fits = (
        total <= MAX_KG_BLOCK_CHARS
        and entry_count(seed) <= MAX_KG_BLOCK_LINES
        and not oversized_entries(seed)
    )
    return total, fits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite existing keys")
    parser.add_argument(
        "--budget",
        action="store_true",
        help="report system-prompt budget usage and exit",
    )
    args = parser.parse_args(argv)

    seed = build_seed()
    if args.budget:
        total, fits = budget_report(seed)
        entries = entry_count(seed)
        print(f"seed entries: {entries} / {MAX_KG_BLOCK_LINES} prompt lines")
        print(f"prompt cost:  {total} / {MAX_KG_BLOCK_CHARS} chars")
        for label, size in oversized_entries(seed):
            print(f"  TOO LONG: {label} renders {size} chars "
                  f"(cap {MAX_KG_ENTRY_CHARS}) — the prompt cuts it off")
        if entries > MAX_KG_BLOCK_LINES:
            print(f"  TOO MANY: {entries - MAX_KG_BLOCK_LINES} entries past the "
                  "line cap never reach the model")
        print("fits the prompt caps" if fits else
              "OVER BUDGET — part of the graph never reaches the model")
        return 0 if fits else 1

    graph = load_graph()
    merged, added = merge(seed, graph, force=args.force)
    if args.dry_run:
        print(f"dry run: {len(added)} entries would be added: "
              + (", ".join(added) if added else "(none)"))
        return 0
    save_graph(merged)
    total = sum(len(v) for v in merged.values() if isinstance(v, dict))
    print(f"knowledge graph seeded: {len(added)} added, {total} entries total "
          f"({len(merged)} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
