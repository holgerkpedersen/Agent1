"""Tests for scripts/seed_kg.py — seeding the knowledge graph.

Paths are resolved lazily (``None`` -> module-level ``MEMORY_PATH``) so
tests can patch ``seed_kg.MEMORY_PATH`` to a temp file; the real
agent_memory.json is never touched.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
