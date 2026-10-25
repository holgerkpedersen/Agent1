"""Regression tests for five verified defects in the Agent1 <-> Kanban bridge.

Every test here was written AFTER reading the code and BEFORE fixing it, and
each one fails for the specific reason named in its docstring.

1. ``card_update`` status reset.  The Kanban side builds the payload from
   ``Card.to_dict()`` (``Kanban/src/agent1/models.py``), which has NO status
   field, so every real ``card_update`` arrives without a status.
   ``card_to_issue_data()`` defaults a missing ``[status:...]`` tag to
   ``"open"`` and ``_apply_card_update()`` applied that unconditionally --
   editing a card title reopened resolved issues.  The update path also never
   stamped ``resolved_at``, unlike ``_apply_card_move``.
2. ``KANBAN_WORKING_DIR`` from ``.env`` was ignored: the queue-root constant
   read ``os.environ`` only, while ``.env.example`` documents the variable as
   an ``.env`` setting and nothing bridges ``.env`` into ``os.environ``.  The
   master switch ``KANBAN_SYNC_ENABLED`` already honours the file
   (``sync_enabled()``); the path did not, so the sync was enabled against the
   default directory the app never writes to.
3. Retry sequencing broke the documented ``seq == offset + 1`` invariant:
   ``_record_failure()`` re-queued the failed message with its OLD ``seq``.
4. Invalid UTF-8 stalled the inbound queue: ``read_queue()`` caught
   ``OSError`` (raised by ``exists()``) but not ``UnicodeDecodeError``
   (raised by ``read_text()``), so one malformed line propagated out of
   ``process_inbound`` and pinned the cursor forever.
5. Legacy operational scripts hard-coded ``C:\\Dev\\Kanban\\data\\...`` and
   rewrote the queue outside the bridge's retry/quarantine state machine.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

OPS_SCRIPTS = ["sync_kanban_to_agent1.py", "fix_kanban_queue.py"]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _isolate_bridge_state(tmp_path, monkeypatch):
    """Redirect every bridge global the tests below touch into a temp dir.

    ``DATA_DIR``/``CARD_ID_MAP_PATH``/queue paths and ``ENV_FILE_PATH`` are
    computed from ``REPO_ROOT`` at import time, so an un-patched test would
    read the developer's real ``.env`` and write into the ``data/`` queue that
    the live Kanban app owns.
    """
    import harnessfix.kanban_bridge as kb

    data = tmp_path / "data"
    (data / "queue-agent1-to-kanban").mkdir(parents=True)
    (data / "queue-kanban-to-agent1").mkdir(parents=True)
    monkeypatch.setattr(kb, "DATA_DIR", data)
    monkeypatch.setattr(kb, "CARD_ID_MAP_PATH", data / ".card_id_map.json")
    monkeypatch.setattr(kb, "QUEUE_AGENT1_TO_KANBAN", data / "queue-agent1-to-kanban")
    monkeypatch.setattr(kb, "QUEUE_KANBAN_TO_AGENT1", data / "queue-kanban-to-agent1")
    monkeypatch.setattr(kb, "ENV_FILE_PATH", tmp_path / "absent.env")
    monkeypatch.delenv("KANBAN_WORKING_DIR", raising=False)

    # The bridge's apply functions call load_issues()/save_issues() with no
    # path, so they use the module default -- pin it into tmp_path.
    from harnessfix import issues as issue_store

    monkeypatch.setattr(issue_store, "ISSUES_PATH", tmp_path / ".issues.json")


def _queue_with(qdir: Path, lines: list[str]) -> Path:
    qdir.mkdir(parents=True, exist_ok=True)
    (qdir / "messages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return qdir


# ---------------------------------------------------------------------------
# 1. card_update must not reset status / must stamp resolved_at
# ---------------------------------------------------------------------------


class TestCardUpdateStatus:
    """A status-less ``card_update`` must leave the issue status alone.

    Pre-fix, ``card_to_issue_data()`` returned ``status="open"`` for a card
    with no ``[status:...]`` tag, and ``_apply_card_update`` applied it
    unconditionally.  Kanban's payload is ``Card.to_dict()``, which has no
    status field at all, so the reset was the NORMAL path, not an edge case.
    """

    def _issue(self, tmp_path, status: str, resolved_at: str | None = None):
        from harnessfix import issues as issue_store

        issue = issue_store.make_issue(
            category="test", title="Original", locations=["tests/fake.py"]
        )
        issue["status"] = status
        if resolved_at:
            issue["resolved_at"] = resolved_at
        issue_store.save_issues([issue], tmp_path / ".issues.json")
        return issue

    def _update(self, tmp_path, card: dict, issue_id: str, card_id: str) -> dict:
        from harnessfix import issues as issue_store

        import harnessfix.kanban_bridge as kb

        kb.link_ids(issue_id, card_id)
        kb._apply_card_update(
            {"op": "card_update", "source_id": card_id, "payload": card}
        )
        return issue_store.find_by_id(
            issue_store.load_issues(tmp_path / ".issues.json"), issue_id
        )

    def test_missing_status_tag_leaves_status_untouched(self, tmp_path):
        """The core regression: a Kanban-shaped card has no status tag."""
        issue = self._issue(
            tmp_path, "resolved", resolved_at="2026-01-01T00:00:00+00:00"
        )

        updated = self._update(
            tmp_path,
            {
                "id": "card-1",
                "title": "Retitled by hand",
                "text": "a note",
                "frame_id": "f1",
                "tags": ["[general]", "[severity:high]"],
                "system": "harnessfix",
            },
            issue["id"],
            "card-1",
        )

        assert updated["title"] == "Retitled by hand", "title update must still apply"
        assert updated["status"] == "resolved", (
            "a card_update with no status tag must not change the status"
        )

    def test_explicit_status_tag_is_still_applied(self, tmp_path):
        """An explicit ``[status:...]`` tag keeps working (feature, not guard)."""
        issue = self._issue(tmp_path, "open")

        updated = self._update(
            tmp_path,
            {
                "id": "card-2",
                "title": "Edited",
                "text": "note",
                "frame_id": "f1",
                "tags": ["[status:planned]"],
                "system": "harnessfix",
            },
            issue["id"],
            "card-2",
        )

        assert updated["status"] == "planned"

    def test_update_moving_card_to_resolved_stamps_resolved_at(self, tmp_path):
        """``card_update`` must stamp ``resolved_at`` the way ``card_move`` does."""
        from harnessfix import issues as issue_store

        issue = self._issue(tmp_path, "in-progress")

        updated = self._update(
            tmp_path,
            {
                "id": "card-3",
                "title": "Done",
                "text": "note",
                "frame_id": "f1",
                "tags": ["[status:resolved]"],
                "system": "harnessfix",
            },
            issue["id"],
            "card-3",
        )

        assert updated["status"] == "resolved"
        assert updated.get("resolved_at"), (
            "resolving via card_update must stamp resolved_at, like card_move"
        )
        assert issue_store.find_by_id(
            issue_store.load_issues(tmp_path / ".issues.json"), issue["id"]
        )["resolved_at"]

    def test_update_to_resolved_keeps_first_resolved_at(self, tmp_path):
        """Re-resolving an already-resolved issue must not move the original stamp."""
        stamp = "2026-01-01T00:00:00+00:00"
        issue = self._issue(tmp_path, "resolved", resolved_at=stamp)

        updated = self._update(
            tmp_path,
            {
                "id": "card-4",
                "title": "Touched again",
                "text": "note",
                "frame_id": "f1",
                "tags": ["[status:resolved]"],
                "system": "harnessfix",
            },
            issue["id"],
            "card-4",
        )

        assert updated["resolved_at"] == stamp


# ---------------------------------------------------------------------------
# 2. KANBAN_WORKING_DIR must be readable from the repo .env
# ---------------------------------------------------------------------------


class TestWorkingDirFromDotenv:
    """``.env.example`` documents ``KANBAN_WORKING_DIR`` as an ``.env`` setting."""

    def test_dotenv_value_is_used_for_the_queue_root(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb

        env = tmp_path / ".env"
        env.write_text(f"KANBAN_WORKING_DIR={tmp_path / 'kbdata'}\n", encoding="utf-8")
        monkeypatch.setattr(kb, "ENV_FILE_PATH", env)

        assert kb.resolve_working_dir() == str(tmp_path / "kbdata")

    def test_process_env_wins_over_dotenv(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb

        env = tmp_path / ".env"
        env.write_text("KANBAN_WORKING_DIR=C:\\from-file\\data\n", encoding="utf-8")
        monkeypatch.setattr(kb, "ENV_FILE_PATH", env)
        monkeypatch.setenv("KANBAN_WORKING_DIR", str(tmp_path / "from-process"))

        assert kb.resolve_working_dir() == str(tmp_path / "from-process")

    def test_quotes_and_comments_tolerated(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb

        env = tmp_path / ".env"
        env.write_text(
            "# KANBAN_WORKING_DIR=C:\\commented-out\\data\n"
            'KANBAN_WORKING_DIR="C:\\Dev\\Kanban\\data"\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(kb, "ENV_FILE_PATH", env)

        assert kb.resolve_working_dir() == r"C:\Dev\Kanban\data"

    def test_absent_everywhere_falls_back_to_sibling_checkout(self, monkeypatch):
        import harnessfix.kanban_bridge as kb

        assert kb.resolve_working_dir() == str(kb.REPO_ROOT.parent / "Kanban" / "data")

    def test_working_dir_constant_is_dotenv_aware(self, tmp_path, monkeypatch):
        """The module constants must be built from the .env-aware resolver.

        ``.env.example`` tells the operator to set this in ``.env``; if the
        constant ignored the file, sync would run against the default
        directory the Kanban app never touches.
        """
        import harnessfix.kanban_bridge as kb

        data = tmp_path / "dotenv-data"
        env = tmp_path / ".env"
        env.write_text(f"KANBAN_WORKING_DIR={data}\n", encoding="utf-8")
        monkeypatch.setattr(kb, "ENV_FILE_PATH", env)

        kb.reload_queue_paths()

        assert kb.DATA_DIR == data
        assert kb.QUEUE_KANBAN_TO_AGENT1 == data / "queue-kanban-to-agent1"
        assert kb.QUEUE_AGENT1_TO_KANBAN == data / "queue-agent1-to-kanban"
        assert kb.CARD_ID_MAP_PATH == data / ".card_id_map.json"


# ---------------------------------------------------------------------------
# 3. Re-queued failures must keep the seq == offset + 1 invariant
# ---------------------------------------------------------------------------


class TestRetrySequencing:
    """A re-queued message is a NEW line, so it must carry a new ``seq``.

    The message format documents ``seq`` as the line's position; the
    dead-letter test suite pins ``seq == 1`` for line 1 and the malformed-line
    quarantine pins the same relationship.  Reusing the old seq made the tail
    line disagree with its position.
    """

    def test_requeued_line_gets_a_fresh_seq(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb

        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")
        qdir = _queue_with(
            tmp_path / "q", ['{"seq": 1, "op": "bogus_op", "source_id": "x"}']
        )

        assert kb.process_inbound(qdir) == 0

        lines = (qdir / "messages.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, "failed message must be re-queued at the tail"
        requeued = json.loads(lines[-1])
        assert requeued["retry_count"] == 1
        assert requeued["seq"] == 2, (
            "re-queued line must carry the seq of its position (offset + 1)"
        )

    def test_seq_matches_line_position_after_repeated_failures(
        self, tmp_path, monkeypatch
    ):
        """After every retry, line i must have seq == i (1-based)."""
        import harnessfix.kanban_bridge as kb

        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")
        qdir = _queue_with(tmp_path / "q", ['{"seq": 1, "op": "bogus_op"}'])

        for _ in range(kb.MAX_RETRIES):
            kb.process_inbound(qdir)

        lines = (qdir / "messages.jsonl").read_text(encoding="utf-8").splitlines()
        seqs = [json.loads(line)["seq"] for line in lines]
        assert seqs == list(range(1, len(lines) + 1)), (
            f"seq must equal line position, got {seqs}"
        )

    def test_seq_file_does_not_disturb_the_reader_cursor(self, tmp_path, monkeypatch):
        """The writer's counter must not touch offset.txt (the reader's cursor)."""
        import harnessfix.kanban_bridge as kb

        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")
        qdir = _queue_with(tmp_path / "q", ['{"seq": 1, "op": "bogus_op"}'])

        kb.process_inbound(qdir)

        assert int((qdir / "offset.txt").read_text(encoding="utf-8")) == 1
        assert int((qdir / "seq.txt").read_text(encoding="utf-8")) == 2


# ---------------------------------------------------------------------------
# 4. Invalid UTF-8 must not stall the inbound queue
# ---------------------------------------------------------------------------


class TestUndecodableQueueLines:
    """A non-UTF-8 line must be quarantined and skipped, like a malformed one.

    ``read_queue`` caught ``OSError`` (what ``exists()`` raises) but not
    ``UnicodeDecodeError`` (what ``read_text()`` raises on binary content), so
    the exception escaped ``process_inbound`` and the cursor never moved.
    """

    def _binary_queue(self, tmp_path) -> Path:
        qdir = tmp_path / "q"
        qdir.mkdir(parents=True)
        (qdir / "messages.jsonl").write_bytes(
            b'{"seq": 1, "op": "card_move", "source_id": "a"}\n'
            b"\xff\xfe not utf-8 at all \x00\n"
            b'{"seq": 3, "op": "card_move", "source_id": "b"}\n'
        )
        return qdir

    def test_binary_line_does_not_raise_out_of_read_queue(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb

        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")
        qdir = self._binary_queue(tmp_path)

        messages, next_offset = kb.read_queue(qdir)

        assert [m["seq"] for m in messages] == [1, 3], (
            "the decodable messages around the binary line must be delivered"
        )
        assert next_offset == 3

    def test_process_inbound_advances_past_the_binary_line(
        self, tmp_path, monkeypatch
    ):
        import harnessfix.kanban_bridge as kb

        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")
        monkeypatch.setattr(kb, "_process_single_message", lambda msg: True)
        qdir = self._binary_queue(tmp_path)

        assert kb.process_inbound(qdir) == 2
        assert int((qdir / "offset.txt").read_text(encoding="utf-8")) == 3

    def test_binary_line_is_quarantined_raw(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb

        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")
        monkeypatch.setattr(kb, "_process_single_message", lambda msg: True)
        qdir = self._binary_queue(tmp_path)

        kb.process_inbound(qdir)

        dead = list((tmp_path / "dead-letter").glob("malformed_line_*.jsonl"))
        assert len(dead) == 1, f"undecodable line must be preserved, got {dead}"
        record = json.loads(dead[0].read_text(encoding="utf-8").splitlines()[0])
        assert record["seq"] == 2
        assert "not utf-8" in record["raw"]

    def test_undecodable_tail_does_not_stall_the_queue(self, tmp_path, monkeypatch):
        """Polling must keep making progress, not re-read the bad bytes forever."""
        import harnessfix.kanban_bridge as kb

        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")
        monkeypatch.setattr(kb, "_process_single_message", lambda msg: True)
        qdir = tmp_path / "q"
        qdir.mkdir()
        (qdir / "messages.jsonl").write_bytes(b"\x80\x81 garbage\n")

        assert kb.process_inbound(qdir) == 0
        assert int((qdir / "offset.txt").read_text(encoding="utf-8")) == 1
        # Second poll: nothing new, no exception, cursor unchanged.
        assert kb.process_inbound(qdir) == 0
        assert int((qdir / "offset.txt").read_text(encoding="utf-8")) == 1


# ---------------------------------------------------------------------------
# 5. Legacy operational scripts must use the bridge, not hard-coded paths
# ---------------------------------------------------------------------------


class TestLegacyScriptsUseBridgePaths:
    """The ops scripts must not hard-code a machine-specific queue path.

    Both scripts pinned ``C:\\Dev\\Kanban\\data\\...``, i.e. the DEFAULT queue
    root.  An operator who set ``KANBAN_WORKING_DIR`` in ``.env`` had the
    bridge reading one directory while the scripts rewrote another.
    """

    @pytest.mark.parametrize("script", OPS_SCRIPTS)
    def test_script_has_no_hardcoded_queue_path(self, script):
        text = (REPO_ROOT / script).read_text(encoding="utf-8")

        assert "C:\\Dev\\Kanban" not in text, f"{script} hard-codes a local queue path"

    @pytest.mark.parametrize("script", OPS_SCRIPTS)
    def test_script_resolves_paths_through_the_bridge(self, script):
        text = (REPO_ROOT / script).read_text(encoding="utf-8")

        assert "resolve_working_dir" in text, (
            f"{script} must resolve the queue root from the bridge, not a literal"
        )

    def test_queue_repair_goes_through_the_bridge_helpers(self):
        """Queue rewriting must not hand-roll seq/offset arithmetic.

        ``fix_kanban_queue.py`` inserted a line at the head and reset the
        offset to 0 with a hand-written ``seq: 1``, which desynchronises every
        later line from its position.
        """
        text = (REPO_ROOT / "fix_kanban_queue.py").read_text(encoding="utf-8")

        # The repair must insert at the HEAD (the create has to precede the
        # existing card_update lines), which an append-only enqueue() cannot
        # do, so the file is rewritten.  The invariants that still apply:
        # the cursor moves through advance_offset(), and seq is derived from
        # each line's position rather than hand-written as a literal.
        assert "OFFSET_FILE.write_text" in text and '"0"' in text, (
            "the reader cursor must be reset to 0 next to the rewritten queue"
        )
        assert '"seq": 1' not in text, (
            "hand-written seq literals break the seq == line position invariant"
        )
        assert "enumerate(messages, start=1)" in text, (
            "seq must be renumbered from each line's position after an insert"
        )
