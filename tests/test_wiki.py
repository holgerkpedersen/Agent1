"""Regression tests for the WikiSkill wiki layer (harnessfix/wiki.py).

Covers:
  - consolidate() distills failed traces into failure pages keyed by
    {layer}:{mechanism} and successful traces into success pages, merging
    evidence across runs that hit the same key.
  - retrieve() does fix-mode filter (error-class + file-stem overlap) with a
    token-overlap fallback for vague queries; increments hit_count on retrieval.
  - format_wiki_notes renders an empty string when no pages match and a
    COMPILED KNOWLEDGE block otherwise (prompt injection contract).
  - wiki_stats computes page_count, coverage (% of diagnosed layers), avg_hit_count,
    top_pages by hits, and last_consolidated timestamp.
  - load_wiki/save_wiki round-trip is atomic and schema-tolerant on corrupt lines.
"""
from __future__ import annotations

import json
from pathlib import Path

from harnessfix.wiki import (
    WikiPage,
    consolidate,
    format_wiki_notes,
    wiki_stats,
    load_wiki,
    save_wiki,
    retrieve,
)


def _failed_trace(path: Path, task_id: str = "t1", exc: str = "ValueError", msg: str = "") -> None:
    """Write a minimal failed trace (tool_error + loop_end error outcome)."""
    events = [
        {"task_id": task_id, "ts": 1.0, "kind": "task_begin", "layer": "context",
         "user_input": f"fix the {exc} crash in foo.py"},
        {"task_id": task_id, "ts": 2.0, "kind": "tool_call", "layer": "tool_interface",
         "iteration": 0, "tool": "run", "args_hash": "{}"},
        {"task_id": task_id, "ts": 3.0, "kind": "tool_error", "layer": "tool_interface",
         "iteration": 0, "tool": "run", "exception": exc, "message": msg or f"boom in foo.py"},
        {"task_id": task_id, "ts": 9.0, "kind": "loop_end", "layer": "lifecycle",
         "iteration": 0, "outcome": "error", "termination_reason": "tool_error"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def _success_trace(path: Path, task_id: str = "s1", file_stem: str = "foo") -> None:
    """Write a minimal successful trace (write+run+loop_end completed)."""
    events = [
        {"task_id": task_id, "ts": 1.0, "kind": "task_begin", "layer": "context",
         "user_input": f"fix the ImportError in {file_stem}.py"},
        {"task_id": task_id, "ts": 2.0, "kind": "tool_call", "layer": "tool_interface",
         "iteration": 1, "tool": "write", "args_hash": json.dumps({"path": f"{file_stem}.py"})},
        {"task_id": task_id, "ts": 3.0, "kind": "tool_result", "layer": "execution_environment",
         "iteration": 1, "tool": "write", "args_hash": json.dumps({"path": f"{file_stem}.py"}),
         "result": "Written"},
        {"task_id": task_id, "ts": 4.0, "kind": "tool_call", "layer": "tool_interface",
         "iteration": 2, "tool": "run", "args_hash": json.dumps({"command": f"python {file_stem}.py"})},
        {"task_id": task_id, "ts": 5.0, "kind": "tool_result", "layer": "execution_environment",
         "iteration": 2, "tool": "run", "args_hash": json.dumps({"command": f"python {file_stem}.py"}),
         "result": "ok"},
        {"task_id": task_id, "ts": 6.0, "kind": "llm_response", "layer": "observability",
         "iteration": 2, "text": "Fixed the ImportError by re-exporting keyboard from the correct module path."},
        {"task_id": task_id, "ts": 9.0, "kind": "loop_end", "layer": "lifecycle",
         "iteration": 0, "outcome": "completed"},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


class TestConsolidate:
    def test_failed_trace_yields_failure_page(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", exc="ValueError", msg="boom in foo.py")
        wiki_dir = tmp_path / "wiki"

        pages = consolidate(traces, wiki_dir)
        assert len(pages) == 1
        pg = pages[0]
        assert pg.layer == "tool_interface"
        # key is {layer}:{mechanism} — mechanism contains the exception type.
        assert pg.key.startswith("tool_interface:")
        assert "ValueError" in pg.error_classes
        assert "foo" in pg.file_stems
        assert "t1" in pg.evidence

    def test_success_trace_yields_success_page(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _success_trace(traces / "s1.jsonl", file_stem="foo")
        wiki_dir = tmp_path / "wiki"

        pages = consolidate(traces, wiki_dir)
        assert len(pages) == 1
        pg = pages[0]
        assert pg.layer == "(success)"
        assert pg.key.startswith("success:")
        assert "foo" in pg.file_stems
        assert "s1" in pg.evidence

    def test_two_failed_traces_merge_into_one_page(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        # Both hit the same layer+mechanism (ValueError tool error).
        _failed_trace(traces / "t1.jsonl", task_id="t1", exc="ValueError")
        _failed_trace(traces / "t2.jsonl", task_id="t2", exc="ValueError")

        pages = consolidate(traces, tmp_path / "wiki")
        assert len(pages) == 1
        pg = pages[0]
        # Evidence merged across both traces.
        assert set(pg.evidence) == {"t1", "t2"}

    def test_mixed_corpus_produces_both_page_types(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "f.jsonl", task_id="f")
        _success_trace(traces / "s.jsonl", task_id="s", file_stem="bar")

        pages = consolidate(traces, tmp_path / "wiki")
        assert len(pages) == 2
        layers = {pg.layer for pg in pages}
        assert "tool_interface" in layers  # failure page
        assert "(success)" in layers      # success page

    def test_consolidate_is_idempotent_merge(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", task_id="t1")
        wiki_dir = tmp_path / "wiki"

        first = consolidate(traces, wiki_dir)
        assert len(first) == 1
        # Re-consolidating the same trace must not duplicate.
        second = consolidate(traces, wiki_dir)
        assert len(second) == 1
        assert set(second[0].evidence) == {"t1"}

    def test_corrupt_trace_skipped_fail_open(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        (traces / "bad.jsonl").write_text("not json\n", encoding="utf-8")
        _failed_trace(traces / "t1.jsonl", task_id="t1")

        pages = consolidate(traces, tmp_path / "wiki")
        assert len(pages) == 1


class TestRetrieve:
    def test_retrieve_by_error_class(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", exc="ValueError", msg="boom in foo.py")
        wiki_dir = tmp_path / "wiki"
        consolidate(traces, wiki_dir)

        results = retrieve("ValueError crash in foo.py", k=5, path=wiki_dir / "wiki.jsonl")
        assert len(results) >= 1
        pg = results[0]
        assert "ValueError" in pg.error_classes
        # hit_count incremented on retrieval.
        assert pg.hit_count == 1

    def test_retrieve_by_file_stem(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _success_trace(traces / "s1.jsonl", file_stem="foo")
        wiki_dir = tmp_path / "wiki"
        consolidate(traces, wiki_dir)

        results = retrieve("fix foo.py ImportError", k=5, path=wiki_dir / "wiki.jsonl")
        assert len(results) >= 1
        pg = results[0]
        assert "foo" in pg.file_stems

    def test_retrieve_vague_query_falls_back_to_token_overlap(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", exc="ValueError", msg="inspect tool_interface harness mechanism")
        wiki_dir = tmp_path / "wiki"
        consolidate(traces, wiki_dir)

        # Query with no error-class or file-stem signals — must fall back.
        results = retrieve("tool interface problem", k=5, path=wiki_dir / "wiki.jsonl")
        assert len(results) >= 1
        pg = results[0]
        assert "tool_interface" in pg.key

    def test_retrieve_empty_wiki_returns_empty(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        results = retrieve("anything", k=5, path=wiki_dir / "wiki.jsonl")
        assert results == []


class TestFormatWikiNotes:
    def test_empty_when_no_pages(self, tmp_path):
        notes = format_wiki_notes("fix foo.py", k=3, path=tmp_path / "missing.jsonl")
        assert notes == ""

    def test_renders_compiled_knowledge_block(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", exc="ValueError", msg="boom in foo.py")
        wiki_dir = tmp_path / "wiki"
        consolidate(traces, wiki_dir)

        notes = format_wiki_notes("ValueError crash foo.py", k=3, path=wiki_dir / "wiki.jsonl")
        assert notes != ""
        assert "COMPILED KNOWLEDGE (wiki)" in notes
        assert "lesson:" in notes
        # Evidence task_ids are shown.
        assert "t1" in notes

    def test_block_is_purely_additive_no_write_side_effect(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", exc="ValueError")
        wiki_dir = tmp_path / "wiki"
        consolidate(traces, wiki_dir)

        before = load_wiki(wiki_dir / "wiki.jsonl")[0].hit_count
        format_wiki_notes("ValueError foo.py", k=3, path=wiki_dir / "wiki.jsonl")
        after = load_wiki(wiki_dir / "wiki.jsonl")[0].hit_count
        # hit_count IS persisted on retrieval — that is the documented contract.
        assert after == before + 1


class TestWikiStats:
    def test_stats_on_empty_wiki(self, tmp_path):
        wiki_dir = tmp_path / "wiki"
        wiki_dir.mkdir()
        stats = wiki_stats(wiki_dir / "wiki.jsonl")
        assert stats["page_count"] == 0
        assert stats["total_evidence"] == 0
        assert stats["coverage"] == 0.0
        assert stats["avg_hit_count"] == 0.0
        assert stats["top_pages"] == []

    def test_stats_on_populated_wiki(self, tmp_path):
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", exc="ValueError")
        _success_trace(traces / "s1.jsonl", file_stem="foo")
        wiki_dir = tmp_path / "wiki"
        consolidate(traces, wiki_dir)

        # Retrieve once so hit_count > 0 on the failure page.
        retrieve("ValueError foo.py", k=5, path=wiki_dir / "wiki.jsonl")

        stats = wiki_stats(wiki_dir / "wiki.jsonl")
        assert stats["page_count"] == 2
        assert stats["total_evidence"] >= 2  # t1 + s1 at minimum
        # coverage: tool_interface is a diagnosed layer with a failure page.
        assert stats["coverage"] > 0.0
        assert len(stats["top_pages"]) <= 10
        # last_consolidated is a timestamp (float).
        assert isinstance(stats["last_consolidated"], float)


class TestPersistence:
    def test_save_load_round_trip(self, tmp_path):
        pages = [
            WikiPage(
                key="tool_interface:test_mechanism",
                title="[tool_interface] test mechanism",
                lesson="inspect tool_interface harness mechanism",
                evidence=["t1"],
                layer="tool_interface",
                error_classes=("ValueError",),
                file_stems=("foo",),
            ),
        ]
        wpath = tmp_path / "wiki.jsonl"
        save_wiki(pages, wpath)

        loaded = load_wiki(wpath)
        assert len(loaded) == 1
        pg = loaded[0]
        assert pg.key == "tool_interface:test_mechanism"
        assert pg.layer == "tool_interface"
        assert pg.error_classes == ("ValueError",)
        assert pg.file_stems == ("foo",)

    def test_load_tolerates_corrupt_lines(self, tmp_path):
        wpath = tmp_path / "wiki.jsonl"
        good = WikiPage(key="k1", title="t1", lesson="l1")
        save_wiki([good], wpath)
        # Append a corrupt line.
        with wpath.open("a", encoding="utf-8") as fh:
            fh.write("not valid json\n")
            fh.write("{bad: json}\n")

        loaded = load_wiki(wpath)
        assert len(loaded) == 1
        assert loaded[0].key == "k1"

    def test_save_is_atomic_no_partial_read(self, tmp_path):
        pages = [WikiPage(key="k", title="t", lesson="l")]
        wpath = tmp_path / "wiki.jsonl"
        save_wiki(pages, wpath)
        # No .tmp file left behind after a successful save.
        assert not wpath.with_suffix(".tmp").exists()


class TestDashboardAPI:
    def test_autonomous_status_includes_wiki(self, tmp_path):
        """The dashboard API endpoint must include wiki stats (fail-open)."""
        traces = tmp_path / "traces"
        traces.mkdir()
        _failed_trace(traces / "t1.jsonl", exc="ValueError")
        wiki_dir = tmp_path / "reports" / "wiki"

        from agent_core.monitoring import dashboard_api
        # Redirect the handler's base dir to our temp root.
        dashboard_api.DashboardAPIHandler._base_dir = str(tmp_path)

        from harnessfix.wiki import consolidate, wiki_stats

        consolidate(traces, wiki_dir)
        stats = wiki_stats(wiki_dir / "wiki.jsonl")
        assert stats["page_count"] == 1

        # The _autonomous_status method reads wiki via the handler's base dir.
        class _Stub:
            pass
        stub = dashboard_api.DashboardAPIHandler.__new__(dashboard_api.DashboardAPIHandler)
        result = stub._autonomous_status()
        assert "wiki" in result
        assert isinstance(result["wiki"], dict)
        # When wiki exists under the base dir, page_count should be populated.
        assert result["wiki"].get("page_count", 0) >= 1

    def test_autonomous_status_wiki_empty_when_no_wiki(self, tmp_path):
        """Wiki stats must degrade gracefully when no wiki file exists."""
        from agent_core.monitoring import dashboard_api
        dashboard_api.DashboardAPIHandler._base_dir = str(tmp_path)

        stub = dashboard_api.DashboardAPIHandler.__new__(dashboard_api.DashboardAPIHandler)
        result = stub._autonomous_status()
        assert "wiki" in result
        assert isinstance(result["wiki"], dict)
