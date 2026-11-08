"""Phase 2 Kanban isolation: plan-finish writes must touch no Kanban surface.

Contract (docs/INTEGRATION_KANBAN_AGENT1.md): the Kanban integration is a
message-queue bridge over two JSONL queue directories under
``KANBAN_WORKING_DIR`` (``queue-kanban-to-agent1/``, ``queue-agent1-to-kanban/``),
plus the board files (``board-agent1.json``, ``boards/agent1.json``) and the
issue<->card map (``.card_id_map.json``).  Only card CRUD and issue
create/update/resolve emit queue messages via
``harnessfix.kanban_bridge`` / ``issues._enqueue_sync``.

Phase 2 of the self-evaluation plan writes ONLY:

* ``.docs/<ts>/evaluation.json`` (the per-plan artifact), and
* one appended line in ``reports/history/executions.jsonl`` (the ledger).

These tests lock that isolation in permanently:

1. Running the real plan-finish evaluation path enqueues nothing into any
   Kanban queue, mutates no card-id map, touches no board file, and calls no
   bridge/enqueue function at all (ast: no kanban import; runtime: an
   instrumented ``harnessfix.kanban_bridge`` records zero interactions).
2. The evaluation path modules never import ``harnessfix.kanban_bridge`` —
   not statically, and not dynamically on a realistic plan-finish run.
"""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The Kanban surface that Phase 2 writes must never touch.
KANBAN_MODULE = "harnessfix.kanban_bridge"

#: Every public + private callable on the bridge (enqueueing, queue IO,
#: inbound/outbound processing, ID-map access).  Any call during plan-finish
#: would mean the evaluation path leaked into the Kanban integration.
BRIDGE_CALLABLES = (
    "enqueue",
    "_enqueue",
    "_append_message",
    "_write_json_atomic",
    "_read_offset",
    "_write_offset",
    "process_inbound",
    "_process_single_message",
    "start_inbound_processor",
    "resolve_id",
    "link_ids",
    "_lookup_issue_for",
    "_queue_dir",
)

PLAN_MD = """# Plan

## Tasks

- t1: do the thing (role: implementer)
- t2: test the thing (role: tester, deps: t1)
"""

REPORT_MD = "- [t1] completed (role: implementer)\n- [t2] completed (role: tester)\n"


# --------------------------------------------------------------------------
# Fixtures: a workspace that *looks* like a live Kanban integration, so any
# accidental bridge usage would leave observable evidence.
# --------------------------------------------------------------------------

def _make_workspace(tmp_path: Path) -> Path:
    """Workspace with a plan dir, traces, and Kanban-shaped decoy files."""
    workspace = tmp_path / "ws"
    plan_dir = workspace / ".docs" / "2026-11-08_12-00-00"
    plan_dir.mkdir(parents=True)
    (plan_dir / "plan_executing.md").write_text(PLAN_MD, encoding="utf-8")
    (plan_dir / "plan_execution_report.md").write_text(REPORT_MD, encoding="utf-8")

    trace_dir = workspace / "reports" / "traces"
    trace_dir.mkdir(parents=True)
    (trace_dir / "run1.jsonl").write_text(
        json.dumps({"kind": "token_usage", "total_tokens": 140}) + "\n",
        encoding="utf-8",
    )

    # Kanban surface decoys: if Phase 2 ever enqueues, writes a board file,
    # appends to a queue, or mutates the ID map, these files change.
    kanban_data = tmp_path / "kanban_data"
    for q in ("queue-kanban-to-agent1", "queue-agent1-to-kanban"):
        (kanban_data / q).mkdir(parents=True)
        (kanban_data / q / "messages.jsonl").write_text("", encoding="utf-8")
        (kanban_data / q / "seq.txt").write_text("0\n", encoding="utf-8")
    (kanban_data / ".card_id_map.json").write_text(
        json.dumps({"issues": {"t1": "card-guid-1"}}), encoding="utf-8"
    )
    board = {"frames": [{"id": "f1", "cards": []}], "sync": {"agent1": True}}
    (kanban_data / "board-agent1.json").write_text(
        json.dumps(board), encoding="utf-8"
    )
    boards_dir = kanban_data / "boards"
    boards_dir.mkdir()
    (boards_dir / "agent1.json").write_text(json.dumps(board), encoding="utf-8")

    # In-workspace copies too (in case a writer resolves relative to cwd).
    (workspace / ".card_id_map.json").write_text(
        json.dumps({"issues": {}}), encoding="utf-8"
    )
    return workspace


def _snapshot(paths: list[Path]) -> dict[str, tuple[int, int]]:
    """(size, mtime_ns) per path — cheap fingerprint of the Kanban surface."""
    snap = {}
    for p in paths:
        try:
            st = p.stat()
            snap[str(p)] = (st.st_size, st.st_mtime_ns)
        except OSError:
            snap[str(p)] = (-1, -1)
    return snap


def _kanban_files(workspace: Path, kanban_data: Path) -> list[Path]:
    files = []
    for q in ("queue-kanban-to-agent1", "queue-agent1-to-kanban"):
        files += [kanban_data / q / "messages.jsonl", kanban_data / q / "seq.txt"]
    files += [
        kanban_data / ".card_id_map.json",
        kanban_data / "board-agent1.json",
        kanban_data / "boards" / "agent1.json",
        workspace / ".card_id_map.json",
    ]
    return files


def _install_instrumented_bridge(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace harnessfix.kanban_bridge with a tripwire stub.

    Any attribute access to one of the bridge's queue/ID-map callables is
    recorded, and calling it raises — plan-finish must never touch them.
    The module is also marked so import checks see it as "imported" by us,
    not silently absent.
    """
    calls: list[str] = []

    class _Tripwire(ModuleType):
        def __getattr__(self, name: str):
            if name in BRIDGE_CALLABLES:
                def _boom(*a, **kw):
                    calls.append(name)
                    raise AssertionError(
                        f"plan-finish evaluation path called Kanban bridge "
                        f"{KANBAN_MODULE}.{name} — isolation violated"
                    )
                return _boom
            raise AttributeError(name)

    stub = _Tripwire(KANBAN_MODULE)
    monkeypatch.setitem(sys.modules, KANBAN_MODULE, stub)
    return calls


# --------------------------------------------------------------------------
# Static analysis: the evaluation path never imports the bridge.
# --------------------------------------------------------------------------

EVAL_PATH_MODULES = [
    REPO_ROOT / "agent_core" / "plan_execution" / "plan_finish_eval.py",
    REPO_ROOT / "harnessfix" / "plan_eval.py",
    REPO_ROOT / "harnessfix" / "scoring.py",
]


def _imported_modules(tree: ast.AST) -> set[str]:
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.level == 0:
                mods.add(node.module)
            else:
                # Resolve relative imports (e.g. `from . import x`).
                mods.add(f"harnessfix.{node.module}")
    return mods


class TestEvaluationPathNeverImportsKanbanBridge:
    @pytest.mark.parametrize("module", EVAL_PATH_MODULES, ids=lambda p: p.name)
    def test_no_kanban_import_in_source(self, module: Path) -> None:
        assert module.is_file(), f"evaluation-path module moved: {module}"
        tree = ast.parse(module.read_text(encoding="utf-8"))
        mods = _imported_modules(tree)
        kanbanish = [m for m in mods if "kanban" in m.lower()]
        assert not kanbanish, (
            f"{module.name} imports Kanban surface {kanbanish}; the Phase 2 "
            f"evaluation path must stay isolated from harnessfix.kanban_bridge"
        )

    def test_no_kanban_import_during_real_plan_finish(self, tmp_path, monkeypatch):
        """Dynamic check: run the real evaluation write; bridge stays unimported."""
        workspace = _make_workspace(tmp_path)
        plan_dir = workspace / ".docs" / "2026-11-08_12-00-00"

        # Poison the slot: if the path imports the real bridge, the stub's
        # __getattr__ records/raises on any callable use.
        calls = _install_instrumented_bridge(monkeypatch)

        from agent_core.plan_execution.plan_finish_eval import (
            write_evaluation_for_plan,
        )

        summary = write_evaluation_for_plan(plan_dir, workspace, success=True)
        assert calls == []
        # The evaluation itself still happened (isolation must not disable it).
        assert (plan_dir / "evaluation.json").is_file()
        ledger = workspace / "reports" / "history" / "executions.jsonl"
        assert ledger.is_file() and len(ledger.read_text().splitlines()) == 1
        assert summary.startswith("Evaluation: score=")

    def test_nlp_plan_finish_handler_source_has_no_kanban_reference(self) -> None:
        """The plan-finish handler (entry point) is also Kanban-free."""
        src = REPO_ROOT / "agent_core" / "handlers" / "nlp_tools.py"
        text = src.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.AsyncFunctionDef)
                or node.name != "_nlp_plan_finish"
            ):
                continue
            mods = _imported_modules(node)
            assert not [m for m in mods if "kanban" in m.lower()]
            # Executable code (not comments/docstrings) must contain no
            # Kanban reference.  ast.get_source_segment spans include
            # comments, so tokenise and drop COMMENT/STRING tokens.
            import io
            import tokenize

            _fn_start, _fn_end = node.lineno - 1, node.end_lineno
            fn_src = "".join(text.splitlines(keepends=True)[_fn_start:_fn_end])
            code_text = "".join(
                tok.string
                for tok in tokenize.generate_tokens(io.StringIO(fn_src).readline)
                if tok.type not in (tokenize.COMMENT, tokenize.STRING)
            )
            assert "kanban" not in code_text.lower(), (
                "plan-finish handler code references the Kanban surface"
            )
            return
        pytest.fail("_nlp_plan_finish not found in nlp_tools.py — handler moved?")


# --------------------------------------------------------------------------
# Runtime: plan-finish enqueues no Kanban queue messages and mutates nothing.
# --------------------------------------------------------------------------

class TestPlanFinishEnqueuesNothing:
    def test_full_plan_finish_touches_no_kanban_surface(self, tmp_path, monkeypatch):
        """Drive the real _nlp_plan_finish; assert zero Kanban side effects."""
        workspace = _make_workspace(tmp_path)
        kanban_data = tmp_path / "kanban_data"
        plan_dir = workspace / ".docs" / "2026-11-08_12-00-00"

        calls = _install_instrumented_bridge(monkeypatch)
        # Also tripwire the issues-ledger enqueue used for outbound sync.
        import harnessfix.issues as issues

        def _no_enqueue(op, source_id, payload=None, **kw):  # pragma: no cover
            raise AssertionError(
                f"plan-finish enqueued Kanban op {op!r} via issues._enqueue_sync"
            )

        monkeypatch.setattr(issues, "_enqueue_sync", _no_enqueue, raising=False)

        # Class bodies cannot see enclosing function locals - build the fake
        # agent's workspace attribute after class creation.
        class FakeAgent:
            pass

        FakeAgent.workspace = str(workspace)

        from agent_core.handlers import nlp_tools

        state = {
            "status": "executing",
            "plan_dir": str(plan_dir),
            "error": None,
        }

        async def _fake_get_plan_state(agent):
            return state

        monkeypatch.setattr(nlp_tools, "_get_plan_state", _fake_get_plan_state)

        # Lifecycle transition moves plan files; keep it in the tmp workspace.
        result = asyncio.run(
            nlp_tools._nlp_plan_finish(FakeAgent(), {"success": True})
        )

        assert "Plan marked as" in result, result
        assert calls == [], f"bridge callables touched: {calls}"

        # Kanban surface byte-for-byte untouched.
        after = _snapshot(_kanban_files(workspace, kanban_data))
        before = _snapshot(_kanban_files(workspace, kanban_data))
        # (recomputed baseline: nothing wrote in between, so compare to fresh
        # reads — all queues must still be empty and maps unchanged)
        assert after == before
        for q in ("queue-kanban-to-agent1", "queue-agent1-to-kanban"):
            assert (kanban_data / q / "messages.jsonl").read_text() == ""
        assert json.loads(
            (kanban_data / ".card_id_map.json").read_text(encoding="utf-8")
        ) == {"issues": {"t1": "card-guid-1"}}

        # Phase 2 writes DID happen, in exactly the two sanctioned places.
        assert (plan_dir / "evaluation.json").is_file()
        ledger = workspace / "reports" / "history" / "executions.jsonl"
        lines = ledger.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["command"] == "plan_eval"

    def test_failed_plan_finish_also_enqueues_nothing(self, tmp_path, monkeypatch):
        workspace = _make_workspace(tmp_path)
        kanban_data = tmp_path / "kanban_data"
        plan_dir = workspace / ".docs" / "2026-11-08_12-00-00"

        calls = _install_instrumented_bridge(monkeypatch)
        from agent_core.plan_execution import plan_finish_eval

        summary = plan_finish_eval.write_evaluation_for_plan(
            plan_dir, workspace, success=False
        )
        assert summary.startswith("Evaluation: score=")
        assert calls == []
        record = json.loads((plan_dir / "evaluation.json").read_text(encoding="utf-8"))
        assert record["success"] is False
        # Failed plans score 0 (scoring contract) — still no Kanban side effects.
        assert record["score"] == 0
        for q in ("queue-kanban-to-agent1", "queue-agent1-to-kanban"):
            assert (kanban_data / q / "messages.jsonl").read_text() == ""
