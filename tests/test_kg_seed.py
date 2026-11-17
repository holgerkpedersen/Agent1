"""Tests for scripts/seed_kg.py — seeding the knowledge graph.

Paths are resolved lazily (``None`` -> module-level ``MEMORY_PATH``) so
tests can patch ``seed_kg.MEMORY_PATH`` to a temp file; the real
agent_memory.json is never touched.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_spec = importlib.util.spec_from_file_location(
    "seed_kg",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scripts",
        "seed_kg.py",
    ),
)
seed_kg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(seed_kg)

from agent_core.memory import persistence  # noqa: E402  (after sys.path setup)


class _FakeAgent:
    def __init__(self):
        self._knowledge_graph: dict = {}
        self._files_read: set = set()
        self._semantic_index: dict = {}
        self._working_memory: list = []
        self._history: list = []
        self._original_goal: str = ""


class SeedKgTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.memory = os.path.join(self.tmpdir.name, "agent_memory.json")

    def tearDown(self):
        self.tmpdir.cleanup()

    # -- merge semantics -------------------------------------------------

    def test_merge_adds_all_seed_entries_to_empty_graph(self):
        merged, added = seed_kg.merge(seed_kg.SEED, {})
        self.assertEqual(len(added), seed_kg.count_seed())
        for section in seed_kg.SECTIONS:
            self.assertIn(section, merged)

    def test_merge_is_idempotent(self):
        merged1, added1 = seed_kg.merge(seed_kg.SEED, {})
        merged2, added2 = seed_kg.merge(seed_kg.SEED, merged1)
        self.assertEqual(added2, [])
        self.assertEqual(merged2, merged1)

    def test_merge_keeps_existing_values_without_force(self):
        graph = {"facts": {"kg-command": {"value": "user's own text", "ts": "x"}}}
        merged, added = seed_kg.merge(seed_kg.SEED, graph)
        self.assertEqual(merged["facts"]["kg-command"]["value"], "user's own text")
        self.assertNotIn("facts/kg-command", added)

    def test_force_overwrites_existing_values(self):
        graph = {"facts": {"kg-command": {"value": "old", "ts": "x"}}}
        merged, _ = seed_kg.merge(seed_kg.SEED, graph, force=True)
        self.assertNotEqual(merged["facts"]["kg-command"]["value"], "old")

    def test_seed_sections_are_the_canonical_four(self):
        self.assertEqual(set(seed_kg.SEED), set(seed_kg.SECTIONS))

    # -- save/load round-trip through persistence -------------------------

    def test_save_then_load_roundtrips(self):
        merged, _ = seed_kg.merge(seed_kg.SEED, {})
        seed_kg.save_graph(merged, self.memory)
        reloaded = seed_kg.load_graph(self.memory)
        self.assertEqual(reloaded, merged)

    def test_save_preserves_other_memory_keys(self):
        with open(self.memory, "w", encoding="utf-8") as f:
            json.dump({"files_read": ["a.py"], "knowledge_graph": {}}, f)
        merged, _ = seed_kg.merge(seed_kg.SEED, {})
        seed_kg.save_graph(merged, self.memory)
        with open(self.memory, encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["files_read"], ["a.py"])
        self.assertEqual(data["knowledge_graph"], merged)

    def test_load_missing_file_is_empty(self):
        self.assertEqual(seed_kg.load_graph(self.memory), {})

    # -- CLI ---------------------------------------------------------------

    def test_dry_run_writes_nothing(self):
        with patch.object(seed_kg, "MEMORY_PATH", self.memory):
            rc = seed_kg.main(["--dry-run"])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(self.memory))

    def test_main_seeds_file(self):
        with patch.object(seed_kg, "MEMORY_PATH", self.memory):
            rc = seed_kg.main([])
        self.assertEqual(rc, 0)
        graph = seed_kg.load_graph(self.memory)
        self.assertEqual(
            sum(len(v) for v in graph.values()), seed_kg.count_seed()
        )

    # -- integration with the real persistence helpers ---------------------

    def test_seeded_graph_loads_via_persistence_and_counts(self):
        import agent as agent_mod
        from agent_core.memory import persistence

        merged, _ = seed_kg.merge(seed_kg.SEED, {})
        seed_kg.save_graph(merged, self.memory)
        agent = _FakeAgent()
        with patch.object(agent_mod, "AGENT_MEMORY_JSON_PATH", self.memory):
            persistence._load_memory(agent)
        stats = persistence.memory_stats(_FakeStatsAgent(agent._knowledge_graph))
        self.assertEqual(stats["knowledge_graph"], len(seed_kg.SECTIONS))

    def test_memory_stats_counts_knowledge_graph_entries(self) -> None:
        """``knowledge_graph_entries`` sums entries across sections; it is
        the number the ``clear`` command prints (issue: 'always showing 0
        entries')."""
        graph = {
            "facts": {"a": {"value": 1}, "b": {"value": 2}},
            "files": {"x.py": {"value": ""}},
            "junk": "not-a-dict",  # tolerated: counts as 0, never crashes
        }
        stats = persistence.memory_stats(_FakeStatsAgent(graph))
        self.assertEqual(stats["knowledge_graph"], 3)
        self.assertEqual(stats["knowledge_graph_entries"], 3)


class SeedAccuracyTests(unittest.TestCase):
    """Guard the curated TEXT against drift from the real code base.

    The rest of this module pins the merge/save *mechanics*; nothing checked
    that the seed still DESCRIBES Agent1, so stale facts survived for weeks
    (a removed ``Agent.run_interactive``, a "session log" ``clear`` never
    touched, a ledger-fingerprint cache key that became per-entry).  These
    checks are structural on purpose — they assert names, paths and
    registrations, never prose.
    """

    #: Dotted seed keys name a class attribute -> the class to resolve.
    #: A new dotted key must be registered here, which keeps the check honest.
    _CLASS_ATTRIBUTES = {
        "CommandRegistry": "agent_core.commands.registry.CommandRegistry",
        "ToolLoopRunner": "agent_core.llm.tool_loop.ToolLoopRunner",
    }

    # -- the command list is the most drift-prone fact ---------------------

    def test_repl_commands_tag_matches_the_live_registry(self):
        """The tag is GENERATED from the registry, so it cannot list stale names.

        It used to be a hand-maintained list of ~40 names; it rotted (commands
        were added with no matching edit) and, at ~400 chars, broke the 200-char
        prompt entry cap so the model only ever saw a list cut off mid-name.
        Now ``build_seed`` derives it from ``_build_registry()``, so this test
        guards the derivation instead of comparing two copies of the same list.
        """
        from agent_core.cli.interactive import _build_registry

        tag = seed_kg.build_seed()["tags"]["repl-commands"]
        live = set(_build_registry().names())
        self.assertTrue(tag, "build_seed must generate tags/repl-commands")
        self.assertIn("(", tag, "the tag must name the command count in ()")
        self.assertEqual(
            int(tag.split("(", 1)[1].split(")", 1)[0]),
            len(live),
            "seed_kg tags/repl-commands count is out of date with "
            "_register_commands",
        )
        # A generated tag must also respect the prompt's per-entry cap.
        self.assertLessEqual(
            len(tag), seed_kg.MAX_KG_ENTRY_CHARS,
            "the generated repl-commands tag exceeds the prompt entry cap",
        )

    # -- every path the seed mentions must still exist ---------------------

    def test_seed_file_entries_exist(self):
        for path in seed_kg.SEED["files"]:
            with self.subTest(path=path):
                self.assertTrue(
                    os.path.isfile(os.path.join(seed_kg.REPO_ROOT, path)),
                    f"seed mentions {path}, which no longer exists",
                )

    # -- every symbol the seed names must still be defined -----------------

    def test_seed_symbols_are_still_defined(self):
        source = self._agent_core_source()
        for key in seed_kg.SEED["symbols"]:
            with self.subTest(symbol=key):
                if "." in key:
                    self._assert_class_attribute(key)
                    continue
                defined = re.compile(
                    rf"^\s*(?:async )?def {re.escape(key)}\b"
                    rf"|^{re.escape(key)}\s*=",
                    re.MULTILINE,
                )
                self.assertRegex(
                    source, defined, f"{key} is no longer defined in agent_core"
                )

    def test_no_symbol_is_attributed_to_a_missing_agent_method(self):
        """Regression: the seed claimed ``Agent.run_interactive`` existed.

        It never did — the entry point is the module function
        ``agent_core.cli.interactive.run_interactive`` and the class is
        ``AgentCore``.
        """
        from agent_core.agent import Agent

        self.assertFalse(hasattr(Agent, "run_interactive"))
        self.assertFalse(any(k.startswith("Agent.") for k in seed_kg.SEED["symbols"]))

    # -- facts that were wrong / went stale --------------------------------

    def test_clear_command_fact_does_not_claim_a_session_log(self):
        """``clear`` clears memory, never session metadata — it used to say so."""
        value = seed_kg.SEED["facts"]["clear-command"]
        self.assertNotIn("session log", value)
        self.assertNotIn(
            "session", inspect.getsource(persistence.clear_history).lower()
        )
        self.assertIn("session", value, "the fact should point at the session command")

    def test_memory_store_fact_describes_the_per_entry_cache(self):
        """The cache key is ``(id(service), text)``, not a ledger fingerprint."""
        from agent_core.memory import memory_store

        value = seed_kg.SEED["facts"]["memory-store"]
        self.assertNotIn("ledger fingerprint", value)
        self.assertTrue(callable(getattr(memory_store, "_vector_for", None)))
        self.assertIn("TEXT", memory_store._vector_for.__doc__ or "")

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _agent_core_source() -> str:
        chunks: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(
            os.path.join(seed_kg.REPO_ROOT, "agent_core")
        ):
            for name in sorted(filenames):
                if name.endswith(".py"):
                    with open(
                        os.path.join(dirpath, name), encoding="utf-8", errors="replace"
                    ) as f:
                        chunks.append(f.read())
        return "\n".join(chunks)

    def _assert_class_attribute(self, key: str) -> None:
        prefix, _, attr = key.partition(".")
        target = self._CLASS_ATTRIBUTES.get(prefix)
        self.assertIsNotNone(
            target,
            f"register {prefix!r} in SeedAccuracyTests._CLASS_ATTRIBUTES "
            f"so {key} can be verified",
        )
        module_path, _, class_name = target.rpartition(".")
        cls = getattr(importlib.import_module(module_path), class_name)
        self.assertTrue(hasattr(cls, attr), f"{class_name}.{attr} does not exist")


class _KgStubAgent:
    """Minimal stand-in for the agent attribute the KG block reads."""

    def __init__(self, graph):
        self._knowledge_graph = graph


class SeedPromptBudgetTests(unittest.TestCase):
    """The seed must survive the real system-prompt renderer, not just look tidy.

    Regression: the seed grew to 52 entries / 8243 rendered chars against a
    6000-char block cap, so the ``symbols`` and ``tags`` sections were silently
    truncated away — the model only ever saw ~32 of them, and 12 facts were cut
    off mid-sentence at the 200-char entry cap.  Nothing caught it because the
    seed was only ever tested as a dict, never through the prompt.
    """

    def _graph(self, seed=None):
        seed = seed or seed_kg.build_seed()
        graph = {section: {} for section in seed_kg.SECTIONS}
        for section, entries in seed.items():
            for key, value in entries.items():
                graph[section][key] = {"value": value, "ts": 0}
        return graph

    def _block(self, seed=None):
        from agent_core.prompts.context_blocks import _knowledge_graph_block

        return _knowledge_graph_block(_KgStubAgent(self._graph(seed)))

    # -- the caps the seed must respect -----------------------------------

    def test_mirrored_caps_match_the_real_constants(self):
        """The literals in seed_kg must not drift from agent_core.run_utils."""
        from agent_core import run_utils

        self.assertEqual(seed_kg.MAX_KG_BLOCK_CHARS, run_utils.MAX_KG_BLOCK_CHARS)
        self.assertEqual(seed_kg.MAX_KG_ENTRY_CHARS, run_utils.MAX_KG_ENTRY_CHARS)
        self.assertEqual(seed_kg.MAX_KG_BLOCK_LINES, run_utils.MAX_KG_BLOCK_LINES)

    def test_seed_never_exceeds_the_rendered_block_cap(self):
        """A bigger graph is silently truncated: the tail never reaches the model."""
        block = self._block()
        self.assertTrue(block, "the seed must render a KG block")
        self.assertLessEqual(
            len(block),
            seed_kg.MAX_KG_BLOCK_CHARS,
            "the seed outgrows MAX_KG_BLOCK_CHARS, so part of it is invisible; "
            "run 'python scripts/seed_kg.py --budget'",
        )
        self.assertNotIn(
            "run 'kg show'",
            block,
            "the block hit the line cap and dropped entries behind an overflow "
            "marker — the seed renders more lines than MAX_KG_BLOCK_LINES",
        )

    def test_every_seed_entry_reaches_the_prompt_intact(self):
        """Every key must appear, and no value may be cut off with an ellipsis."""
        block = self._block()
        seed = seed_kg.build_seed()
        for section in seed_kg.SECTIONS:
            for key, value in seed.get(section, {}).items():
                with self.subTest(entry=f"{section}/{key}"):
                    rendered = seed_kg.rendered_entry(section, key, value)
                    self.assertLessEqual(
                        len(rendered),
                        seed_kg.MAX_KG_ENTRY_CHARS,
                        f"{section}/{key} exceeds the per-entry cap and is cut "
                        "off mid-sentence in the prompt",
                    )
                    self.assertIn(rendered, block)

    def test_block_overhead_estimate_is_not_an_underestimate(self):
        """``rendered_chars`` gates the budget, so it must not under-report."""
        block = self._block(seed={section: {} for section in seed_kg.SECTIONS})
        self.assertGreaterEqual(
            seed_kg._BLOCK_OVERHEAD,
            len(block),
            "_BLOCK_OVERHEAD is lower than the real header/steering text, so "
            "rendered_chars under-reports the prompt cost",
        )

    def test_budget_report_flags_an_oversized_seed(self):
        """``--budget`` must fail loudly rather than quietly accept an overrun."""
        heavy = {
            "facts": {
                f"k{i}": "x" * seed_kg.MAX_KG_ENTRY_CHARS
                for i in range(seed_kg.MAX_KG_BLOCK_LINES + 5)
            }
        }
        total, fits = seed_kg.budget_report(heavy)
        self.assertFalse(fits)
        self.assertGreater(total, seed_kg.MAX_KG_BLOCK_CHARS)

    def test_curated_seed_fits_the_budget(self):
        """The shipped seed must pass its own gate."""
        total, fits = seed_kg.budget_report()
        self.assertTrue(
            fits,
            f"seed renders {total} chars / {seed_kg.entry_count()} lines; "
            f"caps are {seed_kg.MAX_KG_BLOCK_CHARS} chars / "
            f"{seed_kg.MAX_KG_BLOCK_LINES} lines",
        )


class _FakeStatsAgent:
    """Minimal stand-in for the fields memory_stats reads."""

    def __init__(self, knowledge_graph):
        self._knowledge_graph = knowledge_graph
        self._files_read: set = set()
        self._history: list = []
        self._working_memory: list = []
        self._semantic_index: dict = {}
        self._chat_history: list = []

    def check_stale_files(self):
        return []


if __name__ == "__main__":
    unittest.main()
