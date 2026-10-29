#!/usr/bin/env python3
"""Seed the agent's knowledge graph with curated Agent1 entries.

The graph lives in ``agent_memory.json`` under the key ``knowledge_graph``
and uses the schema the ``kg`` command writes:

    {section: {key: {"value": <str|list>, "ts": <iso-utc>}}}

Sections are top-level categories (facts, files, tags, symbols); keys are
entity names; values are plain strings or JSON lists.  This script MERGES:
an existing key keeps its stored value unless ``--force`` is given, so
re-running the seed never clobbers entries the user edited via ``kg``.

Usage::

    python scripts/seed_kg.py            # add missing curated entries
    python scripts/seed_kg.py --dry-run  # show what would be added
    python scripts/seed_kg.py --force    # overwrite existing keys
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

# Curated knowledge about Agent1 itself.  Section -> key -> value.
SEED: dict[str, dict[str, Any]] = {
    "facts": {
        # -- the knowledge graph itself -----------------------------------
        "kg-command": (
            "kg (commands/kg_cmd.py) adds/shows/removes entries in "
            "agent._knowledge_graph; every mutation calls _save_memory()."
        ),
        "clear-command": (
            "clear resets conversation history and clears the knowledge "
            "graph, working memory and session log."
        ),
        "kg-schema": (
            "Schema: {section: {key: {value, ts}}}; sections are "
            "facts/files/tags/symbols; an emptied section is removed."
        ),
        "kg-values": (
            "kg add joins words after <key> into one value; a value that "
            "parses as JSON list/dict is stored parsed, else plain string."
        ),
        "tags-section": (
            "The tags section is a free-form user category; it starts "
            "empty and fills via 'kg add tag <key> <value>'."
        ),
        # -- memory & persistence ------------------------------------------
        "memory-persistence": (
            "Memory persists atomically (tmp + os.replace) to "
            "agent_memory.json: files_read, semantic_index, knowledge_graph, "
            "working_memory, history, original_goal."
        ),
        "corrupt-file-quarantine": (
            "A corrupt memory/chat-history JSON is quarantined as "
            "<file>.bad-<timestamp> before falling back to empty defaults, "
            "so bad bytes stay inspectable instead of being clobbered by "
            "the next save."
        ),
        "memory-store": (
            "memory/memory_store.py: MemoryStore is a thread-safe "
            "cache-over-backend store (SQLiteStorage backend) with optional "
            "semantic search; the semantic-index cache is keyed by "
            "(embedding-service id, vector-db id, ledger fingerprint)."
        ),
        # -- REPL & command registry ----------------------------------------
        "command-registry": (
            "commands/registry.py: CommandRegistry.register/get/names/"
            "execute/print_help. interactive._register_commands is the "
            "single registration point — the banner listing and dispatch "
            "both read from it."
        ),
        "repl-line-editing": (
            "interactive._start_line_editing enables arrow keys and "
            "command recall; history lives in <workspace>/.agent_history."
        ),
        # -- tool loop & turn loop -------------------------------------------
        "tool-loop": (
            "llm/tool_loop.py: ToolLoopRunner drives the iterative "
            "LLM<->tool loop (default max 150 iterations) with in-loop "
            "context compaction, oversized-arg stripping, path-miss "
            "recovery and a forced final answer when the budget runs out."
        ),
        "turn-loop": (
            "turn_loop.py: _run_chained_tool_loop runs one REPL turn; "
            "_finish_turn records the experience; check_stale_files/"
            "invalidate_stale track files changed on disk since they were read."
        ),
        # -- modes -------------------------------------------------------------
        "plan-mode": (
            "modes.py: is_plan_mode/filter_tool_schemas/check_tool_allowed "
            "+ plan-mode prompt suffixes. Plan mode blocks mutating tools; "
            "read-only tools stay allowed."
        ),
        # -- parallel & subagents -----------------------------------------------
        "parallel-delegate": (
            "llm/parallel.py + llm/tool_reply.py: parallel delegation of "
            "tool calls and tool-reply handling, extracted from the monolith "
            "during the turn-loop refactor. Sub-agent machinery lives in "
            "subagent.py."
        ),
    },
    "files": {
        "agent_core/agent.py": (
            "Agent class: state stores (~line 451), tool-call loop, "
            "run/run_interactive entry points."
        ),
        "agent_core/cli/interactive.py": (
            "REPL: command registry and dispatch; registers built-in "
            "commands including kg (~line 235)."
        ),
        "agent_core/commands/kg_cmd.py": (
            "kg command: add/show/remove over agent._knowledge_graph."
        ),
        "agent_core/commands/clear_cmd.py": (
            "clear command: shows memory stats (incl. knowledge-graph count) "
            "and clears history/memory."
        ),
        "agent_core/commands/registry.py": (
            "CommandRegistry: register/get/names/execute/print_help — the "
            "REPL command dispatch core."
        ),
        "agent_core/memory/persistence.py": (
            "load/save of agent memory JSON, clear_history, memory_stats."
        ),
        "agent_core/memory/memory_store.py": (
            "MemoryStore: thread-safe cache-over-backend store with "
            "optional semantic search over the memory ledger."
        ),
        "agent_core/turn_loop.py": (
            "One REPL turn: chained tool loop, failover reporting, "
            "_finish_turn bookkeeping, stale-file tracking."
        ),
        "agent_core/llm/tool_loop.py": (
            "ToolLoopRunner: the iterative LLM<->tool loop with context "
            "compaction and forced-final-answer logic."
        ),
        "agent_core/modes.py": (
            "Plan-mode gating: is_plan_mode, filter_tool_schemas, "
            "check_tool_allowed, plan-mode prompt suffixes."
        ),
    },
    "symbols": {
        "Agent.run_interactive": (
            "Interactive REPL entry point in agent_core/agent.py."
        ),
        "CommandRegistry.register": (
            "Registers REPL commands; KGCommand is registered from "
            "interactive.py."
        ),
        "add_entry": (
            "kg_cmd.add_entry(agent, section, key, value) - creates the "
            "section dict if missing, stores {value, ts}; None for unknown "
            "section."
        ),
        "remove_entry": (
            "kg_cmd.remove_entry(agent, section, key) - prunes empty "
            "sections after removal."
        ),
        "_save_memory": (
            "memory/persistence.py: persists all memory stores atomically "
            "to agent_memory.json."
        ),
        "clear_history": (
            "memory/persistence.py (~line 75): clears history, chat, files "
            "read, knowledge graph, working memory and semantic index; "
            "removes the persisted state files."
        ),
        "memory_stats": (
            "memory/persistence.py (~line 61): counts chat_history, "
            "files_read, stale_files, history, working_memory, "
            "semantic_index and knowledge_graph sections — the numbers "
            "clear prints."
        ),
        "ToolLoopRunner.run": (
            "llm/tool_loop.py (~line 795): entry point of the tool loop; "
            "runs messages through llm_chat_fn/execute_tool_fn until a "
            "final answer or the iteration budget."
        ),
        "_run_chained_tool_loop": (
            "turn_loop.py (~line 52): async runner for one chained REPL "
            "turn; returns final text, messages, llm error and the loop."
        ),
    },
    "tags": {
        "core-subsystems": (
            "Core subsystems: agent loop, REPL commands, memory "
            "persistence, plan execution. Add more with 'kg add tag "
            "<key> <value>'."
        ),
        "repl-commands": (
            "Registry-based REPL commands (read, write, search, clear, "
            "model, analyze, plan, entities, taskplan, cleanup, git, "
            "implement, fix, workflow, optimize, perf, habits, paste, "
            "display, decide, review, issue, run, self_heal, reconstruct, "
            "multillm, demo_data, mode, subagent, mcp, propose, speculate, "
            "jev, memory, kg, help) — all registered in one place by "
            "_register_commands."
        ),
        "memory-subsystem": (
            "Memory: persistence.py (load/save/clear/stats), "
            "memory_store.py (MemoryStore + semantic search), types.py "
            "(StorageBackend/SQLiteStorage/vector DB)."
        ),
        "llm-subsystem": (
            "LLM layer: client, engines, providers (lemonade, llama, "
            "lmstudio, opencode, openrouter), orchestrator, parallel, "
            "tool_loop, tool_reply, retry, caching and metrics modules."
        ),
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite existing keys")
    args = parser.parse_args(argv)

    graph = load_graph()
    merged, added = merge(SEED, graph, force=args.force)
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
