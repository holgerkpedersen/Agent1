"""Tests for the single-handler queue lock and the queue de-duplication tool.

Two defects are covered:

1. **No cross-process mutex** — several handlers (boot poller, a second REPL,
   the ``--process-in`` CLI) each drained the same queue directory at once, so
   every message was applied two to four times.  ``handling_lock`` must let
   exactly one process in and make ``process_inbound`` step aside otherwise.
2. **Duplicated history** — the queue file accumulated repeated lines.  The
   repair tool must collapse them WITHOUT breaking the cursor protocol: a
   consumed message stays consumed, an unconsumed one stays unconsumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _queue(tmp_path: Path, lines: list[dict], offset: int = 0) -> Path:
    """Create a queue dir holding *lines* with the given reader cursor."""
    qdir = tmp_path / "queue-kanban-to-agent1"
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "messages.jsonl").write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in lines),
        encoding="utf-8",
    )
    (qdir / "offset.txt").write_text(str(offset), encoding="utf-8")
    (qdir / "seq.txt").write_text(str(len(lines)), encoding="utf-8")
    return qdir


def _lines(qdir: Path) -> list[dict]:
    return [
        json.loads(ln)
        for ln in (qdir / "messages.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def _msg(seq: int, op: str, sid: str, **extra) -> dict:
    base = {"seq": seq, "op": op, "source_id": sid, "payload": {"source_id": sid}}
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# handling_lock — exclusive ACROSS processes, re-entrant within one
# ---------------------------------------------------------------------------


class TestHandlingLock:
    def test_second_process_cannot_take_the_lock(self, tmp_path):
        """The real defect: a second handler must not enter for the same queue."""
        import harnessfix.kanban_bridge as kb

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")])

        holder = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys, time\n"
                "from harnessfix.kanban_bridge import handling_lock\n"
                "with handling_lock(sys.argv[1]) as owned:\n"
                "    print('owned' if owned else 'busy', flush=True)\n"
                "    time.sleep(5)\n",
                str(qdir),
            ],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert holder.stdout is not None
            assert holder.stdout.readline().strip() == "owned"
            with kb.handling_lock(qdir) as owned:
                assert owned is False, "a second process must not get the lock"
        finally:
            holder.kill()
            holder.wait(timeout=10)

    def test_lock_is_reentrant_inside_one_process(self, tmp_path):
        """Nested calls (CLI -> process_inbound) must not report themselves busy."""
        import harnessfix.kanban_bridge as kb

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")])
        with kb.handling_lock(qdir) as outer:
            assert outer is True
            with kb.handling_lock(qdir) as inner:
                assert inner is True

    def test_lock_is_released_after_the_block(self, tmp_path):
        """A finished handler must leave the queue usable for the next one."""
        import harnessfix.kanban_bridge as kb

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")])
        with kb.handling_lock(qdir):
            pass
        with kb.handling_lock(qdir) as again:
            assert again is True

    def test_missing_queue_directory_is_not_created(self, tmp_path):
        """Polling a mistyped path must not silently invent an empty queue."""
        import harnessfix.kanban_bridge as kb

        missing = tmp_path / "nope"
        with kb.handling_lock(missing) as owned:
            assert owned is False
        assert not missing.exists()

    def test_process_inbound_steps_aside_when_the_lock_is_taken(
        self, tmp_path, monkeypatch
    ):
        """The guard that makes the duplication impossible."""
        import harnessfix.kanban_bridge as kb

        qdir = _queue(tmp_path, [_msg(1, "bogus_op", "abc")])
        monkeypatch.setattr(kb, "handling_lock", lambda *_a, **_k: _busy())
        monkeypatch.setattr(kb, "QUEUE_KANBAN_TO_AGENT1", qdir)

        assert kb.process_inbound(qdir) == 0
        # Untouched: the cursor did not move and nothing was re-queued.
        assert (qdir / "offset.txt").read_text(encoding="utf-8") == "0"
        assert len(_lines(qdir)) == 1


class _busy:
    """Context manager that reports the queue as owned by somebody else."""

    def __enter__(self) -> bool:
        return False

    def __exit__(self, *_exc) -> bool:
        return False


# ---------------------------------------------------------------------------
# repair — collapse duplicates without breaking the cursor
# ---------------------------------------------------------------------------


class TestRepairCollapsesDuplicates:
    def test_exact_duplicates_are_reduced_to_one(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(
            tmp_path,
            [
                _msg(1, "card_update", "abc"),
                _msg(2, "card_update", "abc"),
                _msg(3, "card_update", "abc"),
            ],
        )
        result = repair_mod.repair(qdir, apply=True)
        assert result["applied"] is True
        assert result["duplicate_lines"] == 2
        assert len(_lines(qdir)) == 1

    def test_retry_count_does_not_hide_a_duplicate(self, tmp_path):
        """Copies differ only in bookkeeping — they are still the same message."""
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(
            tmp_path,
            [
                _msg(1, "card_update", "abc"),
                _msg(2, "card_update", "abc", retry_count=1, processed=False),
            ],
        )
        result = repair_mod.repair(qdir, apply=True)
        assert result["duplicate_lines"] == 1
        assert len(_lines(qdir)) == 1

    def test_distinct_messages_are_all_kept(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(
            tmp_path,
            [
                _msg(1, "card_create", "abc"),
                _msg(2, "card_move", "abc"),
                _msg(3, "card_update", "abc"),
            ],
        )
        result = repair_mod.repair(qdir, apply=True)
        assert result["duplicate_lines"] == 0
        assert len(_lines(qdir)) == 3

    def test_order_is_preserved(self, tmp_path):
        """A create must never end up after the move that depends on it."""
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(
            tmp_path,
            [
                _msg(1, "card_create", "abc"),
                _msg(2, "card_move", "abc"),
                _msg(3, "card_create", "abc"),
                _msg(4, "card_delete", "abc"),
            ],
        )
        repair_mod.repair(qdir, apply=True)
        assert [m["op"] for m in _lines(qdir)] == [
            "card_create",
            "card_move",
            "card_delete",
        ]


class TestRepairKeepsTheCursorHonest:
    def test_consumed_messages_stay_consumed(self, tmp_path):
        """The bug this guards: a rewrite that resets the cursor replays history.

        Three identical lines were all applied (offset 3).  After compaction one
        line remains — and it must still count as consumed, so the cursor has to
        become 1, not stay at 3 and not drop to 0.
        """
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(
            tmp_path,
            [
                _msg(1, "card_update", "abc"),
                _msg(2, "card_update", "abc"),
                _msg(3, "card_update", "abc"),
            ],
            offset=3,
        )
        result = repair_mod.repair(qdir, apply=True)
        assert result["offset_after"] == 1
        assert (qdir / "offset.txt").read_text(encoding="utf-8") == "1"
        assert result["pending_after"] == 0, "nothing may become pending again"

    def test_unconsumed_messages_stay_unconsumed(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(
            tmp_path,
            [
                _msg(1, "card_update", "abc"),
                _msg(2, "card_move", "abc"),
                _msg(3, "card_move", "abc"),
            ],
            offset=1,
        )
        result = repair_mod.repair(qdir, apply=True)
        assert result["pending_after"] == 1
        pending, _next = __import__(
            "harnessfix.kanban_bridge", fromlist=["read_queue"]
        ).read_queue(qdir)
        assert [m["op"] for m in pending] == ["card_move"]

    def test_seq_counter_matches_the_remaining_lines(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(7, "card_update", "abc")] * 3)
        repair_mod.repair(qdir, apply=True)
        assert int((qdir / "seq.txt").read_text(encoding="utf-8")) == 7


class TestRepairSafety:
    def test_dry_run_writes_nothing(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")] * 2)
        before = (qdir / "messages.jsonl").read_text(encoding="utf-8")
        result = repair_mod.repair(qdir, apply=False)
        assert result["applied"] is False
        assert result["duplicate_lines"] == 1
        assert (qdir / "messages.jsonl").read_text(encoding="utf-8") == before
        assert not list(tmp_path.glob("**/messages.jsonl.bak-repair-*"))

    def test_backup_is_written_and_kept(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")] * 2)
        result = repair_mod.repair(qdir, apply=True)
        backup = Path(result["backup"])
        assert backup.exists()
        assert len(backup.read_text(encoding="utf-8").splitlines()) == 2

    def test_refuses_when_a_handler_owns_the_queue(self, tmp_path, monkeypatch):
        """Rewriting under a live reader is how the damage started."""
        import harnessfix.kanban_bridge as kb
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")] * 2)
        monkeypatch.setattr(kb, "handling_lock", lambda *_a, **_k: _busy())
        result = repair_mod.repair(qdir, apply=True)
        assert result["applied"] is False
        assert "another handler" in result["refused"]
        assert len(_lines(qdir)) == 2

    def test_undecodable_line_is_preserved_for_the_bridge(
        self, tmp_path
    ):
        """A malformed line is the bridge's to quarantine, not the tool's to eat."""
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")])
        with open(qdir / "messages.jsonl", "a", encoding="utf-8") as f:
            f.write("{not json}\n")
        result = repair_mod.repair(qdir, apply=True)
        assert result["duplicate_lines"] == 0
        assert (qdir / "messages.jsonl").read_text(encoding="utf-8").count(
            "{not json}"
        ) == 1


class TestRepairSupersededPass:
    def _ledger(self, tmp_path: Path, ledger: list[dict] | None = None) -> Path:
        """Write a ledger holding only *ledger* — never the live one.

        Passing ``ledger=`` explicitly matters: the default resolves to the
        committed ``.issues.json`` of this repo, so a test that omitted it would
        assert against real project data and drift as the ledger grows.
        """
        ledger_path = tmp_path / "ledger.json"
        ledger_path.write_text(
            json.dumps({"issues": ledger or []}), encoding="utf-8"
        )
        return ledger_path

    def test_delete_only_history_is_dropped_when_asked(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(
            tmp_path,
            [
                _msg(1, "card_delete", "gone"),
                _msg(2, "card_create", "live"),
                _msg(3, "card_delete", "live"),
            ],
        )
        ledger = self._ledger(tmp_path)
        result = repair_mod.repair(
            qdir, apply=True, drop_superseded=True, ledger=ledger
        )
        assert result["superseded_lines"] == 1
        assert [m["source_id"] for m in _lines(qdir)] == ["live", "live"]

    def test_superseded_pass_is_off_by_default(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_delete", "gone")])
        ledger = self._ledger(tmp_path)
        result = repair_mod.repair(qdir, apply=True, ledger=ledger)
        assert result["superseded_lines"] == 0
        assert len(_lines(qdir)) == 1

    def test_card_still_linked_in_the_ledger_is_kept(self, tmp_path):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_delete", "abc123")])
        ledger = self._ledger(
            tmp_path, [{"id": "iss-1", "locations": ["kanban:abc123"]}]
        )
        result = repair_mod.repair(
            qdir, apply=True, drop_superseded=True, ledger=ledger
        )
        assert result["superseded_lines"] == 0
        assert len(_lines(qdir)) == 1

    def test_card_still_on_a_board_is_kept(self, tmp_path):
        """Boards are read from the queue's parent dir — the Kanban data dir."""
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_delete", "abc123")])
        ledger = self._ledger(tmp_path)
        (qdir.parent / "board-agent1.json").write_text(
            json.dumps({"frames": [{"cards": [{"id": "abc123"}]}]}), encoding="utf-8"
        )
        result = repair_mod.repair(
            qdir, apply=True, drop_superseded=True, ledger=ledger
        )
        assert result["superseded_lines"] == 0
        assert len(_lines(qdir)) == 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_dry_run_exits_zero_and_reports(self, tmp_path, capsys):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")] * 2)
        assert repair_mod._main(["--queue-dir", str(qdir)]) == 0
        out = capsys.readouterr().out
        assert "duplicates       : 1" in out
        assert "applied          : False" in out

    def test_json_output_is_parseable(self, tmp_path, capsys):
        from harnessfix import kanban_queue_repair as repair_mod

        qdir = _queue(tmp_path, [_msg(1, "card_update", "abc")] * 2)
        repair_mod._main(["--queue-dir", str(qdir), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["lines_after"] == 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
