"""Regression tests for episodic-memory retrieval over the trace corpus.

Pins harnessfix/{embed,episodes,retrieval}.py:
  - a successful trace compiles to exactly one Episode with the right
    problem/actions/outcome,
  - failed variants are excluded (tool_error kind, interrupted, guard without
    answer) while an answer-delivering guard run is KEPT (corpus._is_failed_trace),
  - EpisodeIndex.retrieve does hybrid filter+rank with a deterministic
    dependency-free embedder (no model download),
  - format_episodic_notes is "" when disabled or corpus-empty (prompt contract).

The real-corpus behaviour is covered by the disabled-default invariant; these
unit tests use synthetic traces so they run anywhere.
"""
import json
import os

import pytest

from harnessfix.episodes import Episode, clear_episode_cache, extract_episode, successful_episodes
from harnessfix.embed import HashEmbedder, build_embedder
from harnessfix.retrieval import EpisodeIndex, clear_index_cache, format_episodic_notes

TRACE_DIR = "reports"
TRACE_SUB = "traces"


def _trace(tmp_path, name, events):
    d = tmp_path / TRACE_DIR / TRACE_SUB
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
    return p


def _begin(task, text):
    return {
        "task_id": task, "ts": 1.0, "kind": "task_begin", "layer": "context",
        "user_input": text, "model": "m", "profile": "p",
    }


def _llm(task, text, n=1):
    return {
        "task_id": task, "ts": 2.0, "kind": "llm_response", "layer": "observability",
        "iteration": n, "text": text, "tool_calls_requested": 0,
    }


def _tool(task, tool, args, kind="tool_result", result="", exc="", msg=""):
    ev = {
        "task_id": task, "ts": 3.0, "kind": kind, "layer": "tool_interface",
        "tool": tool, "args_hash": json.dumps(args),
    }
    if kind == "tool_error":
        ev["exception"] = exc
        ev["message"] = msg
    else:
        ev["result"] = result
        ev["affected_files"] = args.get("path") and [args["path"]] or []
    return ev


def _loop_end(task, outcome):
    return {"task_id": task, "ts": 9.0, "kind": "loop_end", "layer": "lifecycle", "outcome": outcome}


def _successful_events(task, problem, file="agent_core/foo.py", err=None):
    events = [
        _begin(task, problem),
        _tool(task, "write", {"path": file}, result="Written"),
        _tool(task, "run", {"command": "python foo.py"}, result=(err or "ok")),
        _llm(task, "Fixed " + (err or "it") + " by adjusting the import path and re-exporting the symbol.", n=3),
        _loop_end(task, "completed"),
    ]
    return events


class TestEpisodeExtraction:
    def test_successful_trace_yields_one_episode(self):
        from harnessfix.htir import compile_trace

        events = _successful_events("t1", "ImportError: cannot import name 'keyboard' from 'ursina'", file="agent_core/foo.py", err="ImportError: cannot import name 'keyboard'")
        import tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp())
        p = _trace(d, "t1.jsonl", events)
        graph = compile_trace(p)
        ep = extract_episode(graph)
        assert isinstance(ep, Episode)
        assert ep.task_id == "t1"
        assert "ImportError" in ep.problem
        assert "foo" in ep.file_stems
        assert "ImportError" in ep.error_classes
        assert "write" in ep.actions_summary
        assert ep.outcome == "completed"
        assert len(ep.final_answer) >= 80

    def test_tool_error_trace_excluded(self, tmp_path):
        events = [
            _begin("t2", "fix the crash"),
            _tool("t2", "run", {"command": "x"}, kind="tool_error", exc="ImportError", msg="no module"),
            _loop_end("t2", "error"),
        ]
        _trace(tmp_path, "t2.jsonl", events)
        eps = successful_episodes(str(tmp_path))
        assert eps == []

    def test_interrupted_trace_excluded(self, tmp_path):
        events = [
            _begin("t3", "task"),
            _tool("t3", "read", {"path": "a.py"}),
            _tool("t3", "edit", {"path": "a.py"}),
            _tool("t3", "run", {"path": "a.py"}),
        ]
        _trace(tmp_path, "t3.jsonl", events)
        assert successful_episodes(str(tmp_path)) == []

    def test_guard_without_answer_excluded(self, tmp_path):
        events = [
            _begin("t4", "task"),
            _tool("t4", "write", {"path": "b.py"}),
            _llm("t4", "short", n=1),
            _loop_end("t4", "stuck"),
        ]
        _trace(tmp_path, "t4.jsonl", events)
        assert successful_episodes(str(tmp_path)) == []

    def test_guard_with_answer_kept(self, tmp_path):
        events = [
            _begin("t5", "task"),
            _tool("t5", "write", {"path": "c.py"}),
            _llm("t5", "I delivered the final answer with a full explanation of the fix and why it works.", n=2),
            _loop_end("t5", "stuck"),
        ]
        _trace(tmp_path, "t5.jsonl", events)
        eps = successful_episodes(str(tmp_path))
        assert len(eps) == 1
        assert eps[0].task_id == "t5"


class TestEmbedder:
    def test_hash_embedder_deterministic_and_normalized(self):
        e = HashEmbedder()
        a = e.from_string("ImportError cannot import name keyboard from ursina")
        b = e.from_string("ImportError cannot import name keyboard from ursina")
        assert a.shape == (384,)
        assert abs(float(a.dot(b)) - 1.0) < 1e-9
        import numpy as np
        assert abs(float(np.linalg.norm(a)) - 1.0) < 1e-9

    def test_similar_texts_rank_above_dissimilar(self):
        e = HashEmbedder()
        import numpy as np
        q = e.from_string("ImportError: cannot import name 'keyboard' from 'ursina'")
        sim = e.from_string("ImportError cannot import name keyboard from ursina module")
        diff = e.from_string("the weather today is sunny and warm outside")
        assert float(q.dot(sim)) > float(q.dot(diff))

    def test_build_embedder_default_is_hash(self):
        assert isinstance(build_embedder(None), HashEmbedder)
        assert isinstance(build_embedder(""), HashEmbedder)


class TestEpisodeIndex:
    def _index(self, tmp_path):
        ev_a = _successful_events("a", "ImportError: cannot import name 'keyboard' from 'ursina'", file="agent_core/foo.py", err="ImportError: cannot import name 'keyboard'")
        ev_b = _successful_events("b", "TypeError: unsupported operand type for +: 'int' and 'str'", file="agent_core/bar.py", err="TypeError: unsupported operand")
        _trace(tmp_path, "a.jsonl", ev_a)
        _trace(tmp_path, "b.jsonl", ev_b)
        eps = successful_episodes(str(tmp_path))
        return EpisodeIndex(eps, embedder=HashEmbedder()), eps

    def test_retrieve_fix_filters_by_error_class(self, tmp_path):
        idx, eps = self._index(tmp_path)
        out = idx.retrieve("ImportError: cannot import name 'keyboard' from 'ursina'", mode="fix", k=3)
        assert out
        assert all("ImportError" in ec for ec in (e.error_classes for e in out))
        assert out[0].task_id == "a"

    def test_retrieve_fix_filters_by_file_stem(self, tmp_path):
        idx, eps = self._index(tmp_path)
        out = idx.retrieve("fix foo.py which has an ImportError", mode="fix", k=3)
        assert out[0].task_id == "a"

    def test_retrieve_implement_pure_semantic(self, tmp_path):
        idx, eps = self._index(tmp_path)
        out = idx.retrieve("TypeError about operand types on bar", mode="implement", k=3)
        assert out[0].task_id == "b"

    def test_retrieve_empty_corpus(self):
        idx = EpisodeIndex([], embedder=HashEmbedder())
        assert idx.retrieve("anything", mode="fix") == []


class TestFormatNotes:
    def test_disabled_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.delenv("AGENT_RAG_EPISODES", raising=False)
        _trace(tmp_path, "a.jsonl", _successful_events("a", "ImportError in foo", file="agent_core/foo.py"))
        assert format_episodic_notes("ImportError in foo", mode="fix", workspace=str(tmp_path)) == ""

    def test_enabled_returns_block(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_RAG_EPISODES", "1")
        clear_index_cache()
        _trace(tmp_path, "a.jsonl", _successful_events("a", "ImportError: cannot import name 'keyboard' from 'ursina'", file="agent_core/foo.py", err="ImportError: cannot import name 'keyboard'"))
        block = format_episodic_notes("ImportError: cannot import name 'keyboard' from 'ursina'", mode="fix", workspace=str(tmp_path), embedder=HashEmbedder())
        assert block.startswith("\n## SUCCESSFUL RUN EXAMPLES")
        assert "keyboard" in block
        clear_index_cache()

    def test_empty_corpus_returns_empty_even_if_enabled(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENT_RAG_EPISODES", "1")
        clear_index_cache()
        assert format_episodic_notes("x", mode="fix", workspace=str(tmp_path), embedder=HashEmbedder()) == ""
        clear_index_cache()

    def test_command_shaped_multiline_query(self, tmp_path, monkeypatch):
        # Mirrors the exact shape fix_cmd.py passes: error msg + rel path +
        # traceback text, all on one multi-line string.
        monkeypatch.setenv("AGENT_RAG_EPISODES", "1")
        clear_index_cache()
        _trace(tmp_path, "a.jsonl", _successful_events("a", "ImportError: cannot import name 'keyboard' from 'ursina'", file="agent_core/foo.py", err="ImportError: cannot import name 'keyboard'"))
        query = (
            "ImportError: cannot import name 'keyboard' from 'ursina'\n"
            "agent_core/foo.py\n"
            "Traceback (most recent call last):\n  File \"agent_core/foo.py\", line 3\n"
            "ImportError: cannot import name 'keyboard' from 'ursina'"
        )
        block = format_episodic_notes(query, mode="fix", workspace=str(tmp_path), embedder=HashEmbedder())
        assert block.startswith("\n## SUCCESSFUL RUN EXAMPLES")
        assert "keyboard" in block
        clear_index_cache()


class TestRealCorpusIntegration:
    """Guarded against the live reports/traces corpus.

    Skipped unless AGENT_RAG_EPISODES is set, because it reads the real
    (large) trace directory.  Proves the feature works end-to-end on real
    data, not just synthetic traces.
    """

    def test_fix_retrieval_surfaces_import_error_episodes(self, monkeypatch):
        if not os.environ.get("AGENT_RAG_EPISODES"):
            pytest.skip("set AGENT_RAG_EPISODES=1 to run real-corpus integration")
        clear_episode_cache()
        clear_index_cache()
        eps = successful_episodes(".")
        assert eps, "expected at least some successful episodes in reports/traces"
        idx = EpisodeIndex(eps, embedder=HashEmbedder())
        out = idx.retrieve(
            "ImportError: cannot import name 'keyboard' from 'ursina'",
            mode="fix", k=3,
        )
        assert out, "fix retrieval must return episodes for a real ImportError query"
        assert all("ImportError" in ec for ec in (e.error_classes for e in out))
        # disabled flag must never alter the prompt contract
        monkeypatch.delenv("AGENT_RAG_EPISODES", raising=False)
        assert format_episodic_notes("ImportError in foo", mode="fix", workspace=".", embedder=HashEmbedder()) == ""
        clear_index_cache()
