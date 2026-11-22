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

    # -- orphan pruning ---------------------------------------------------

    def test_orphans_reports_keys_the_seed_no_longer_defines(self):
        """A renamed/merged seed key leaves its old entry behind; that
        leftover still costs a prompt line (issue: merge() never deletes)."""
        graph = {
            "symbols": {
                "remove_entry": {"value": "old split symbol", "ts": "x"},
                "_save_memory": {"value": "still seeded", "ts": "x"},
            }
        }
        self.assertEqual(seed_kg.orphans(seed_kg.SEED, graph),
                         ["symbols/remove_entry"])

    def test_orphans_ignores_non_dict_and_unknown_sections(self):
        graph = {"facts": "not-a-dict", "custom": {"keep.me": {"value": 1}}}
        self.assertEqual(seed_kg.orphans(seed_kg.SEED, graph), [])

    def test_merge_without_prune_keeps_orphans(self):
        graph = {"symbols": {"remove_entry": {"value": "orphan", "ts": "x"}}}
        merged, _ = seed_kg.merge(seed_kg.SEED, graph)
        self.assertIn("remove_entry", merged["symbols"])

    def test_prune_drops_orphans_but_keeps_seeded_values(self):
        graph = {
            "symbols": {"remove_entry": {"value": "orphan", "ts": "x"}},
            "facts": {"kg-command": {"value": "user's own text", "ts": "x"}},
        }
        merged, _ = seed_kg.merge(seed_kg.SEED, graph, prune=True)
        self.assertNotIn("remove_entry", merged["symbols"])
        # prune must not silently reset a value the user edited
        self.assertEqual(merged["facts"]["kg-command"]["value"],
                         "user's own text")

    def test_prune_leaves_exactly_the_seed_keys(self):
        graph = {"files": {"ghost.py": {"value": "", "ts": "x"}}}
        merged, _ = seed_kg.merge(seed_kg.SEED, graph, force=True, prune=True)
        self.assertEqual(seed_kg.orphans(seed_kg.SEED, merged), [])
        self.assertEqual(
            sum(len(merged[s]) for s in seed_kg.SECTIONS), seed_kg.count_seed()
        )

    def test_cli_prune_removes_the_orphan_from_the_file(self):
        seed_kg.save_graph({"symbols": {"remove_entry": {"value": "old"}}},
                           self.memory)
        with patch.object(seed_kg, "MEMORY_PATH", self.memory):
            rc = seed_kg.main(["--force", "--prune"])
        self.assertEqual(rc, 0)
        graph = seed_kg.load_graph(self.memory)
        self.assertNotIn("remove_entry", graph.get("symbols", {}))
        self.assertEqual(seed_kg.orphans(seed_kg.build_seed(), graph), [])

    def test_cli_dry_run_reports_would_be_pruned(self):
        seed_kg.save_graph({"files": {"ghost.py": {"value": ""}}}, self.memory)
        with patch.object(seed_kg, "MEMORY_PATH", self.memory):
            with patch("builtins.print") as printed:
                rc = seed_kg.main(["--dry-run", "--prune"])
        self.assertEqual(rc, 0)
        out = " ".join(str(c.args[0]) for c in printed.call_args_list if c.args)
        self.assertIn("files/ghost.py", out)

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

    # -- live-agent warning (silent-revert hazard) --------------------------

    def test_live_agent_warning_names_session_and_pid(self):
        """A live session means the next turn's _save_memory rewrites the
        whole knowledge_graph key from its in-memory dict, so the seed can be
        discarded.  The write still happens, but the operator must be told."""
        owner = {"session": "default", "pid": 4242, "workspace": self.tmpdir.name}
        with patch.object(seed_kg, "MEMORY_PATH", self.memory), \
                patch.object(seed_kg, "live_agent_owners", return_value=[owner]), \
                patch("sys.stderr") as err:
            rc = seed_kg.main([])
        self.assertEqual(rc, 0, "a live session must warn, not fail the seed")
        self.assertTrue(
            os.path.exists(self.memory), "the seed is still written"
        )
        text = " ".join(str(c.args[0]) for c in err.write.call_args_list if c.args)
        self.assertIn("default", text)
        self.assertIn("4242", text)

    def test_no_warning_without_a_live_session(self):
        with patch.object(seed_kg, "MEMORY_PATH", self.memory), \
                patch.object(seed_kg, "live_agent_owners", return_value=[]), \
                patch("sys.stderr") as err:
            rc = seed_kg.main([])
        self.assertEqual(rc, 0)
        self.assertFalse(err.write.called, "silent when nothing is running")

    def test_budget_and_dry_run_never_probe_live_sessions(self):
        """Read-only modes write nothing, so they must not pay for (or warn
        about) a lock probe — and must not fail on one."""
        with patch.object(seed_kg, "MEMORY_PATH", self.memory), \
                patch.object(seed_kg, "live_agent_owners",
                             side_effect=AssertionError("probed")):
            self.assertEqual(seed_kg.main(["--budget"]), 0)
            self.assertEqual(seed_kg.main(["--dry-run"]), 0)

    def test_live_agent_owners_ignores_other_sessions_and_self(self):
        """Only a live session holding THIS memory file counts, and a probe
        that reports our own PID is not a peer."""
        from agent_core.session import lock as lock_mod
        from agent_core.session import store as store_mod
        from agent_core.session.models import SessionMeta, SessionPaths
        from pathlib import Path

        target = os.path.abspath(self.memory)
        other = os.path.join(self.tmpdir.name, "other_session.json")

        def paths(session_id):
            mem = target if session_id == "default" else other
            return SessionPaths(
                session_id=session_id,
                state_dir=Path(self.tmpdir.name),
                chat_history_path=os.path.join(self.tmpdir.name, "chat.json"),
                agent_memory_path=mem,
                turn_log_path=os.path.join(self.tmpdir.name, "turns.log"),
            )

        metas = [SessionMeta(id="default"), SessionMeta(id="elsewhere")]
        owners = {
            "default": {"session": "default", "pid": 9999},
            "elsewhere": {"session": "elsewhere", "pid": 8888},
        }
        with patch.object(store_mod, "list_sessions", return_value=metas), \
                patch.object(store_mod, "paths_for", side_effect=paths), \
                patch.object(lock_mod, "probe_owner",
                             side_effect=lambda p: owners.get(p.session_id)):
            found = seed_kg.live_agent_owners(target)
        self.assertEqual([o["pid"] for o in found], [9999])

    def test_live_agent_owners_fails_open_without_the_package(self):
        """A bare checkout has no agent_core on sys.path: the guard is a
        warning, so the probe must degrade to [] instead of raising."""
        with patch.dict(sys.modules, {"agent_core.session.lock": None}):
            self.assertEqual(seed_kg.live_agent_owners(self.memory), [])

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

    # -- modules named in the fact prose must still resolve ----------------

    #: The fact/tag prose names modules and callables directly.  The other
    #: accuracy tests check the ``files`` paths and ``symbols`` keys, so a
    #: renamed module inside a FACT drifted silently: it still read correctly
    #: while pointing at nothing.  Dotted names resolve as attributes.
    _FACT_MODULES = {
        "agent_core.commands.kg_cmd": ("add_entry", "remove_entry"),
        "agent_core.commands.registry": (
            "CommandRegistry.register",
            "CommandRegistry.get",
            "CommandRegistry.names",
            "CommandRegistry.execute",
            "CommandRegistry.print_help",
        ),
        "agent_core.commands.implement_safety": (
            "_is_dangerous_filename",
            "_shadowing_stdlib_dir",
            "_find_safe_subpackage",
        ),
        "agent_core.memory.memory_store": ("MemoryStore",),
        "agent_core.memory.types": ("SQLiteStorage", "StorageBackend"),
        "agent_core.memory.distill": ("record_turn",),
        "agent_core.llm.tool_loop": ("ToolLoopRunner",),
        "agent_core.monitoring.chat_api": ("AgentTurnRunner", "ChatService"),
        "agent_core.commands.model_catalog": (
            "model_catalog",
            "select_model",
            "ModelsCommand",
        ),
        "agent_core.modes": (
            "is_plan_mode",
            "filter_tool_schemas",
            "check_tool_allowed",
        ),
        "agent_core.language": (
            "detect_language",
            "language_directive",
            "language_block",
            "override_language",
        ),
        "agent_core.cli.interactive": ("_register_commands",),
        "agent_core.turn_loop": ("_run_chained_tool_loop", "_finish_turn"),
    }

    def test_modules_named_in_fact_prose_still_exist(self):
        for dotted, names in self._FACT_MODULES.items():
            with self.subTest(module=dotted):
                try:
                    module = importlib.import_module(dotted)
                except ModuleNotFoundError:
                    self.fail(
                        f"the seed prose names {dotted}, which no longer "
                        "imports — update the fact text in scripts/seed_kg.py"
                    )
                    continue
                for name in names:
                    obj = module
                    for part in name.split("."):
                        obj = getattr(obj, part, None)
                        if obj is None:
                            self.fail(
                                f"the seed prose names {dotted}.{name}, which is "
                                "gone — update the fact text in scripts/seed_kg.py"
                            )

    def test_tool_loop_fact_iteration_cap_matches_the_code(self):
        """The prose says 'max 150 iterations'; guard the literal."""
        from agent_core.llm.tool_loop import ToolLoopRunner

        default = inspect.signature(ToolLoopRunner.__init__).parameters[
            "max_iterations"
        ].default
        self.assertIn(
            str(default),
            seed_kg.SEED["facts"]["tool-loop"],
            "facts/tool-loop claims an iteration cap the code no longer uses",
        )

    def test_line_pointers_stay_near_their_anchors(self):
        """The ``~NNN`` line numbers are the seed's most rot-prone detail.

        No test verified them, so ``Agent = AgentCore (~1767)`` still read
        perfectly while the alias had moved to 1808.  Each pointer is paired
        with the definition it names and must stay within ``_LINE_SLACK``.
        """
        from agent_core import agent as agent_mod
        from agent_core import turn_loop as turn_loop_mod
        from agent_core.cli import interactive as interactive_mod
        from agent_core.llm.tool_loop import ToolLoopRunner
        from agent_core.memory import persistence

        agent_src = self._source_lines("agent_core/agent.py")
        iface_src = self._source_lines("agent_core/cli/interactive.py")

        def attr(obj):
            return lambda: inspect.getsourcelines(obj)[1]

        anchors = [
            (("files", "agent_core/agent.py"), attr(agent_mod.AgentCore)),
            (
                ("files", "agent_core/agent.py"),
                lambda: self._line_of(agent_src, "self._history: list[Any] = []"),
            ),
            (
                ("files", "agent_core/agent.py"),
                lambda: self._line_of(agent_src, "Agent = AgentCore"),
            ),
            (
                ("files", "agent_core/cli/interactive.py"),
                lambda: self._line_of(iface_src, "registry.register(KGCommand())"),
            ),
            (
                ("files", "agent_core/cli/interactive.py"),
                attr(interactive_mod.run_interactive),
            ),
            (("symbols", "run_interactive"), attr(interactive_mod.run_interactive)),
            (
                ("symbols", "CommandRegistry.register"),
                lambda: self._line_of(iface_src, "registry.register(KGCommand())"),
            ),
            (("symbols", "_save_memory"), attr(persistence._save_memory)),
            (("symbols", "clear_history"), attr(persistence.clear_history)),
            (("symbols", "memory_stats"), attr(persistence.memory_stats)),
            (("symbols", "ToolLoopRunner.run"), attr(ToolLoopRunner.run)),
            (("symbols", "ToolLoopRunner.run"), attr(ToolLoopRunner)),
            (
                ("symbols", "_run_chained_tool_loop"),
                attr(turn_loop_mod._run_chained_tool_loop),
            ),
        ]

        for (section, key), resolve in anchors:
            with self.subTest(entry=f"{section}/{key}"):
                actual = resolve()
                self.assertIsNotNone(actual, f"{section}/{key} anchor not found")
                pointers = [
                    int(n)
                    for n in re.findall(r"~(\d+)", seed_kg.SEED[section][key])
                ]
                self.assertTrue(pointers, f"{section}/{key} has no ~line pointer")
                self.assertTrue(
                    any(abs(p - actual) <= self._LINE_SLACK for p in pointers),
                    f"{section}/{key} points at ~{pointers} but its anchor is at "
                    f"line {actual}; update the number in scripts/seed_kg.py",
                )

    #: How far a hand-maintained ``~NNN`` may drift before the test complains.
    _LINE_SLACK = 15

    @staticmethod
    def _source_lines(rel_path: str) -> list[str]:
        path = os.path.join(seed_kg.REPO_ROOT, rel_path)
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()

    @staticmethod
    def _line_of(lines: list[str], exact: str) -> int | None:
        for index, line in enumerate(lines, start=1):
            if line.strip() == exact:
                return index
        return None

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


class PersistedGraphDriftTests(unittest.TestCase):
    """The graph ON DISK must match the seed, not just the seed constant.

    Regression: every other check here reads ``seed_kg.build_seed()``, so the
    seed was kept perfect while ``agent_memory.json`` kept serving stale facts
    (``AgentCore (~428) ... Agent = AgentCore (~1767)`` against a real 459/1808),
    one orphan, and a rendered block of 7332 chars against the 6000-char cap —
    so part of the graph silently stopped reaching the model.  Seeding is a
    manual step; nothing forced the stored file to follow the curated text.
    """

    @staticmethod
    def _as_seed(graph):
        """Re-shape the stored graph (``{value, ts}``) into seed form."""
        out = {}
        for section, items in graph.items():
            if isinstance(items, dict):
                out[section] = {
                    key: (value.get("value") if isinstance(value, dict) else value)
                    for key, value in items.items()
                }
        return out

    @staticmethod
    def _norm(text):
        """Collapse whitespace so a reflowed seed value is not called stale."""
        return " ".join(str(text).split())

    @classmethod
    def _drifted(cls, seed, graph):
        """Return (stale_keys, orphan_keys) for *graph* against *seed*."""
        live = cls._as_seed(graph)
        stale = [
            f"{section}/{key}"
            for section, entries in seed.items()
            for key, value in entries.items()
            if cls._norm(live.get(section, {}).get(key, "")) != cls._norm(value)
        ]
        orphans = [
            f"{section}/{key}"
            for section, entries in live.items()
            for key in entries
            if key not in seed.get(section, {})
        ]
        return stale, orphans

    def test_detector_flags_a_stale_value_and_an_orphan(self):
        """The drift check must actually FAIL on drift, or it guards nothing."""
        seed = seed_kg.build_seed()
        graph = {
            section: {key: {"value": value, "ts": 0} for key, value in entries.items()}
            for section, entries in seed.items()
        }
        self.assertEqual(self._drifted(seed, graph), ([], []))

        section = next(iter(seed))
        key = next(iter(seed[section]))
        graph[section][key]["value"] = "description from an older layout"
        graph[section]["zz-not-in-seed"] = {"value": "removed seed key", "ts": 0}
        stale, orphans = self._drifted(seed, graph)
        self.assertEqual(stale, [f"{section}/{key}"])
        self.assertEqual(orphans, [f"{section}/zz-not-in-seed"])

    def _live_hint(self):
        """Explain a reverted store: a running REPL saves from memory each turn."""
        owners = seed_kg.live_agent_owners()
        if not owners:
            return ""
        who = ", ".join(
            f"session '{o.get('session')}' pid {o.get('pid')}" for o in owners
        )
        return (
            f" NOTE: {who} is running and rewrites knowledge_graph from its "
            "in-memory copy at the end of every turn — restart it (or run "
            "'session <name>' to reload) before re-seeding, or the seed is lost."
        )

    def test_stored_graph_matches_the_seed(self):
        """A stale stored graph must not pass silently.

        Regression: this drift went unnoticed because every earlier accuracy
        test compared the SEED against itself, never against agent_memory.json.
        A running Agent1 then reverted the seeded graph on its next turn and
        nothing failed, so the model read ``~428`` pointers for a 459-line class.
        """
        if not os.path.exists(seed_kg.MEMORY_PATH):
            self.skipTest("no agent_memory.json in this workspace")
        stale, orphans = self._drifted(seed_kg.build_seed(), seed_kg.load_graph())
        self.assertEqual(
            stale,
            [],
            "agent_memory.json holds values that differ from scripts/seed_kg.py: "
            f"{', '.join(stale)} — re-seed with "
            "`python scripts/seed_kg.py --force --prune`" + self._live_hint(),
        )
        self.assertEqual(
            orphans,
            [],
            "agent_memory.json holds keys the seed no longer defines: "
            f"{', '.join(orphans)} — drop them with "
            "`python scripts/seed_kg.py --prune`" + self._live_hint(),
        )

    def test_stored_graph_stays_inside_the_prompt_caps(self):
        """A stored graph over the cap truncates silently in the prompt."""
        if not os.path.exists(seed_kg.MEMORY_PATH):
            self.skipTest("no agent_memory.json in this workspace")
        live = self._as_seed(seed_kg.load_graph())
        total, fits = seed_kg.budget_report(live)
        self.assertTrue(
            fits,
            f"stored graph renders {total} chars / {seed_kg.entry_count(live)} "
            f"lines; caps are {seed_kg.MAX_KG_BLOCK_CHARS} chars / "
            f"{seed_kg.MAX_KG_BLOCK_LINES} lines — run "
            "`python scripts/seed_kg.py --force --prune`",
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
