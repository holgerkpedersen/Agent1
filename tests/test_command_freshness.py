"""Regression tests for stale-code detection (2026-08-19 incident).

A paste-session fix to ``workflow_cmd.py`` landed on disk at 10:13:57 but
the running REPL kept executing the old in-memory module — the 10:15:40
run still showed the duplicated ``## 7.`` sections. The REPL loop now warns
when loaded module files change on disk; these tests pin the helpers.
"""
import os
import sys

from agent_core.commands.freshness import (
    diff_snapshots,
    format_stale_warning,
    loaded_module_mtimes,
)


class TestLoadedModuleMtimes:
    def test_collects_watched_module_files_only(self, monkeypatch, tmp_path):
        watched = tmp_path / "agent_core" / "commands" / "demo_cmd.py"
        watched.parent.mkdir(parents=True)
        watched.write_text("x = 1", encoding="utf-8")
        outside = tmp_path / "other" / "mod.py"
        outside.parent.mkdir(parents=True)
        outside.write_text("x = 2", encoding="utf-8")

        class _M:
            def __init__(self, f: str) -> None:
                self.__file__ = f

        monkeypatch.setattr(
            sys,
            "modules",
            {**sys.modules,
             "agent_core.commands.demo_cmd": _M(str(watched)),
             "other.mod": _M(str(outside))},
        )
        m = loaded_module_mtimes()
        assert any(p.endswith("demo_cmd.py") for p in m)
        assert not any("other" in p.replace("\\", "/").split("/") for p in m)

    def test_pyc_file_maps_to_source(self, monkeypatch, tmp_path):
        src = tmp_path / "agent_core" / "compiled.py"
        src.parent.mkdir(parents=True)
        src.write_text("x = 1", encoding="utf-8")

        class _M:
            __file__ = str(src) + "c"  # "compiled.pyc" -> source "compiled.py"

        monkeypatch.setattr(sys, "modules", {**sys.modules, "agent_core.compiled": _M()})
        m = loaded_module_mtimes()
        assert any(p.endswith("compiled.py") and not p.endswith(".pyc") for p in m)

    def test_missing_files_skipped(self, monkeypatch):
        class _M:
            __file__ = "Z:\\agent_core\\does_not_exist.py"

        monkeypatch.setattr(sys, "modules", {**sys.modules, "agent_core.missing": _M()})
        m = loaded_module_mtimes()
        assert not any("does_not_exist" in p for p in m)

    def test_entry_script_included(self, monkeypatch, tmp_path):
        entry = tmp_path / "agent.py"
        entry.write_text("x = 1", encoding="utf-8")
        monkeypatch.setattr(sys, "modules", {**sys.modules})
        m = loaded_module_mtimes(entry_script=str(entry))
        assert any(p.endswith("agent.py") for p in m)


class TestDiffSnapshots:
    def test_unchanged_returns_empty(self, tmp_path):
        f = tmp_path / "agent_core" / "stable.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1", encoding="utf-8")
        snap = {str(f): os.path.getmtime(str(f))}
        assert diff_snapshots(snap) == []

    def test_changed_file_reported(self, tmp_path):
        f = tmp_path / "agent_core" / "edited.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1", encoding="utf-8")
        snap = {str(f): os.path.getmtime(str(f))}
        f.write_text("x = 2", encoding="utf-8")
        os.utime(f, (snap[str(f)] + 10, snap[str(f)] + 10))
        assert diff_snapshots(snap) == [str(f)]

    def test_deleted_file_reported(self, tmp_path):
        f = tmp_path / "agent_core" / "gone.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1", encoding="utf-8")
        snap = {str(f): os.path.getmtime(str(f))}
        f.unlink()
        assert diff_snapshots(snap) == [str(f)]

    def test_new_module_files_not_reported(self, tmp_path):
        f = tmp_path / "agent_core" / "old.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1", encoding="utf-8")
        snap = {str(f): os.path.getmtime(str(f))}
        g = tmp_path / "agent_core" / "brand_new.py"
        g.write_text("x = 2", encoding="utf-8")
        assert diff_snapshots(snap) == []


class TestFormatStaleWarning:
    def test_lists_paths_and_mentions_stale(self, tmp_path):
        text = format_stale_warning([str(tmp_path / "a.py")])
        assert "a.py" in text
        assert "STALE" in text
        assert "Restart the REPL" in text

    def test_caps_at_limit(self):
        paths = [f"p{i}.py" for i in range(7)]
        text = format_stale_warning(paths, limit=5)
        assert "p4.py" in text
        assert "p5.py" not in text
        assert "2 more" in text

    def test_empty_paths(self):
        text = format_stale_warning([])
        assert "0 loaded module" in text
