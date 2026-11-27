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
    python scripts/seed_kg.py --force --prune  # also drop orphaned keys
    python scripts/seed_kg.py --budget   # report prompt-budget usage

Merge never deletes by default, so a renamed or merged seed key leaves an
orphan behind that still costs a prompt line and can push the graph over its
cap.  ``--prune`` removes those; it deletes EVERY key in the four canonical
sections that the seed does not define, hand-added ones included, so it is
opt-in and reports what it removed.

Re-run the seed while Agent1 is NOT running.  ``persistence._save_memory``
rewrites the whole ``knowledge_graph`` key from the live agent's in-memory
dict at the end of every turn (``turn_loop``), so a seed that lands mid-session
is not visible to that session and the session's own view is what it will
write back.  A live REPL reads the seeded graph back when it switches session
(``session <name>`` and back), because a session switch calls ``_load_memory``.
A session that started before the seed keeps the new entries only thanks to the
merge described below; reloading makes the session authoritative about them.

The save path now MERGES the session's graph with the one on disk, entry by
entry, so a seed is no longer lost to a live session.  An entry on both sides
keeps the newer ``ts``.  An entry on disk that this session loaded and deleted
stays deleted.  An entry on disk that the session never saw — the seed case —
is kept.  The one case that still keeps the on-disk copy wholesale is a session
whose in-memory graph is EMPTY (it started before the seed and holds no entries
at all): such a session has no evidence about any entry, so the save keeps the
on-disk graph and prints a ``[memory]`` line naming the divergence.  That is
what used to empty this graph silently.

The write path still WARNS on stderr when a live Agent1 session owns the memory
file, naming the session and PID so the operator can judge whether to exit
first and re-run.  It does not refuse: seeding from inside a running REPL is
the common way this script is invoked, and an unverified verdict would block
that flow.  ``--budget`` and ``--dry-run`` write nothing and never warn.
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
            "kg (kg_cmd.py) adds/shows/removes _knowledge_graph entries "
            "{section: {key: {value, ts}}}; each mutation calls "
            "_save_memory(); prompts/context_blocks.py renders it."
        ),
        "clear-command": (
            "clear wipes history, KG, working memory, semantic index, "
            "files-read and goal; session metadata survives (use 'session')."
        ),
        # -- memory & persistence ------------------------------------------
        "memory-store": (
            "MemoryStore (memory/memory_store.py) is a thread-safe cache over "
            "SQLiteStorage; embedding cache keyed per entry TEXT."
        ),
        "semantic-memory-loop": (
            "memory/distill.py record_turn writes a ledger entry per turn; "
            "memory/scoring.rank_memories ranks it (fail-open); "
            "memory/types.semantic_memory_block reads it back."
        ),
        # -- REPL & command registry ----------------------------------------
        "command-registry": (
            "CommandRegistry (commands/registry.py) is the dispatch core; "
            "interactive._register_commands is the single registration point."
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
        "dashboard-chat": (
            "AgentTurnRunner (chat_api.py) runs /chat via chat_nlp; slash.py "
            "serves /commands offline; monitoring/dashboard_api.py serves "
            "chat.html, wired by agent_dashboard.py."
        ),
        "model-picker": (
            "model_catalog.py is the ONE catalogue for the REPL 'model' and the "
            "dashboard /model; a blank '/' is the no-LLM command menu."
        ),
        # -- modes -------------------------------------------------------------
        "plan-mode": (
            "modes.py is_plan_mode/filter_tool_schemas/check_tool_allowed gate "
            "plan mode: mutating tools blocked, read-only allowed."
        ),
        # -- parallel & subagents -----------------------------------------------
        "parallel-delegate": (
            "llm/parallel.py + llm/tool_reply.py do parallel delegation; "
            "subagent.py + subagent_roles.py hold the sub-agent machinery."
        ),
        # -- autonomous decisions -----------------------------------------------
        "decision-ledger": (
            "decisions.py records/searches/resolves .decisions.json; "
            "discrepancy.py checks compliance; decision_improvement.py proposes; "
            "decisions_index.py reads."
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
            "parse_and_activate; each session owns its memory and history paths."
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
            "language.py owns answer-language selection via "
            "detect_language/language_directive/language_block; "
            "override_language honours a setting."
        ),
        # -- implement filename safety ---------------------------------------------
        "implement-safety": (
            "commands/implement_safety.py _is_dangerous_filename/"
            "_shadowing_stdlib_dir/_find_safe_subpackage stop writes that "
            "shadow stdlib or escape the package."
        ),
        # -- real-change detection ------------------------------------------------
        "pytest-gate": (
            "pytest_gate.py is_scratch_path + FullRunGate + "
            "full_suite_timeout decide real changes, scale the budget."
        ),
    },
    "files": {
        "agent_core/agent.py": (
            "AgentCore (~462): state stores (~520: _knowledge_graph, "
            "_working_memory), tool-call execution, chat_nlp (~1336); "
            "Agent = AgentCore (~1837)."
        ),
        "agent_core/decision_collector.py": (
            "DecisionCollector.collect_turn_decisions re-reads a turn's "
            "choices (read-only); aggregate_periodic folds the ledgers into "
            "counts."
        ),
        "agent_core/rich_render.py": (
            "render_markdown() previews LLM markdown via rich; falls back to "
            "plain print when rich is missing (HAS_RICH)."
        ),
        "agent_core/file_searcher.py": (
            "FileSearcher.search(query, path, include) backs the search tool: "
            "include is an fnmatch glob, path scopes the walk."
        ),
        "agent_core/cli/interactive.py": (
            "REPL registry + dispatch; registers kg (~241); run_interactive "
            "(~359) is a module function, not an Agent method."
        ),
        "agent_core/commands/kg_cmd.py": (
            "kg command: add/show/remove over agent._knowledge_graph."
        ),
        "agent_core/memory/persistence.py": (
            "Memory JSON load/save, clear_history, memory_stats."
        ),
        "agent_core/turn_loop.py": (
            "One REPL turn: chained tool loop, failover reporting, _finish_turn "
            "outcome, stale files."
        ),
        "agent_core/monitoring/chat_api.py": (
            "Dashboard chat: AgentTurnRunner -> Agent.chat_nlp; ChatService "
            "is the LLM-only fallback."
        ),
        "agent_core/commands/catalog.py": (
            "command_catalog(registry) turns the live registry into JSON for "
            "the dashboard slash menu; needs_terminal_input marks "
            "terminal-only ones."
        ),
    },
    "symbols": {
        "run_interactive": (
            "interactive.py (~359): module-level REPL entry point; NOT an "
            "Agent method (the class is AgentCore)."
        ),
        "CommandRegistry.register": (
            "Registers REPL commands; KGCommand is registered from "
            "interactive.py (~241)."
        ),
        "_save_memory": (
            "persistence.py (~336): persists stores atomically; merges the KG "
            "per entry so a seeded entry survives."
        ),
        "clear_history": (
            "persistence.py (~81): clears history, chat, files read, knowledge "
            "graph, working memory, semantic index; removes state files."
        ),
        "memory_stats": (
            "persistence.py (~62): counts every store, "
            "knowledge_graph_entries and stale_files."
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
    return f"REPL commands ({count}); 'help' lists them all."


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
          force: bool = False,
          prune: bool = False) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Merge seed entries into *graph*; return (merged_graph, added_keys).

    Keys already present keep their stored value unless *force* is given.
    With *prune*, any key in a canonical section that the seed does not define
    is dropped first, so renamed or merged seed keys stop costing prompt lines.
    Section order and non-dict sections are preserved untouched.
    """
    added: list[str] = []
    merged = dict(graph)
    for section in SECTIONS:
        seed_items = seed.get(section, {})
        existing = merged.get(section)
        items: dict[str, Any] = existing if isinstance(existing, dict) else {}
        if prune:
            items = {k: v for k, v in items.items() if k in seed_items}
        for key, value in seed_items.items():
            if key in items and not force:
                continue
            items[key] = {"value": value, "ts": _now()}
            added.append(f"{section}/{key}")
        if items:
            merged[section] = items
    return merged, added


def orphans(
    seed: dict[str, dict[str, Any]] | None = None,
    graph: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    """Return ``section/key`` labels present in *graph* but absent from *seed*.

    These are the entries ``--prune`` would remove: keys renamed or merged in
    :data:`SEED` leave their old entry behind, still costing a prompt line.
    """
    seed = build_seed() if seed is None else seed
    graph = load_graph() if graph is None else graph
    out: list[str] = []
    for section in SECTIONS:
        live = graph.get(section)
        if not isinstance(live, dict):
            continue
        seeded = seed.get(section, {})
        out.extend(f"{section}/{k}" for k in live if k not in seeded)
    return sorted(out)


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


def live_agent_owners(memory_path: str | None = None) -> list[dict[str, Any]]:
    """Owners of live sessions whose agent memory file is *memory_path*.

    This is the condition behind the silent-revert hazard: a session that holds
    the workspace lock rewrites the whole ``knowledge_graph`` key from its
    in-memory dict at the end of every turn, so a seed written underneath it is
    discarded on that session's next turn.

    Fail-open by design — an unimportable ``agent_core`` (bare checkout) or any
    session-layer error returns ``[]`` so this stays a standalone script.
    """
    if memory_path is None:
        memory_path = MEMORY_PATH
    try:  # local import: the script must stay importable without the package
        from agent_core.session.lock import probe_owner
        from agent_core.session.store import list_sessions, paths_for
    except Exception:
        return []
    target = os.path.normcase(os.path.abspath(memory_path))
    owners: list[dict[str, Any]] = []
    try:
        sessions = list_sessions()
    except Exception:
        return []
    for meta in sessions:
        try:
            paths = paths_for(meta.id)
            if os.path.normcase(
                os.path.abspath(paths.agent_memory_path)
            ) != target:
                continue
            owner = probe_owner(paths)
        except Exception:
            continue
        if not owner or owner.get("pid") == os.getpid():
            continue
        owner.setdefault("session", meta.id)
        owners.append(owner)
    return owners


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="report, write nothing")
    parser.add_argument("--force", action="store_true", help="overwrite existing keys")
    parser.add_argument(
        "--prune",
        action="store_true",
        help="drop keys the seed no longer defines (hand-added ones included)",
    )
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
    stale = orphans(seed, graph) if args.prune else []
    merged, added = merge(seed, graph, force=args.force, prune=args.prune)
    if args.dry_run:
        removed = f"; {len(stale)} would be pruned: " + ", ".join(stale) \
            if stale else ""
        print(f"dry run: {len(added)} entries would be added: "
              + (", ".join(added) if added else "(none)") + removed)
        return 0
    save_graph(merged)
    owners = live_agent_owners()
    if owners:
        detail = ", ".join(
            f"session '{o.get('session')}' pid {o.get('pid')}" for o in owners
        )
        print(
            f"warning: {len(owners)} live Agent1 session(s) own this memory "
            f"file ({detail}).\n"
            "  A session that holds KG entries rewrites the whole "
            "knowledge_graph key from its in-memory dict at the end of every "
            "turn, so if it never reloads, this seed is lost on its next turn.\n"
            "  A session with an EMPTY in-memory graph no longer clobbers the "
            "file (the save keeps the on-disk graph and reports it).\n"
            "  A running REPL picks it up on 'session <name>' (that path calls "
            "_load_memory).",
            file=sys.stderr,
        )
    total = sum(len(v) for v in merged.values() if isinstance(v, dict))
    pruned = f", {len(stale)} pruned ({', '.join(stale)})" if stale else ""
    print(f"knowledge graph seeded: {len(added)} added{pruned}, "
          f"{total} entries total ({len(merged)} sections)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
