"""Tests for Agent1 kanban_bridge.py — disk queue, ID mapping, conversion."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers — create a temp data dir and return (data_dir, tmp_path)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_kb_globals(tmp_path: Path):
    """Redirect the bridge's module-level paths into a temp dir for EVERY test.

    Regression guard: ``CARD_ID_MAP_PATH`` defaults to ``REPO_ROOT`` (fixed at
    import time), so any test that calls ``link_ids``/``_save_id_map`` without
    passing a mapping wrote ``.card_id_map.json`` into the repository root.
    Isolating the globals here keeps the workspace clean regardless of which
    helper a test happens to use.
    """
    import harnessfix.kanban_bridge as kb

    saved = (kb.DATA_DIR, kb.CARD_ID_MAP_PATH, kb.QUEUE_AGENT1_TO_KANBAN, kb.QUEUE_KANBAN_TO_AGENT1)
    data = tmp_path / "data"
    (data / "queue-agent1-to-kanban").mkdir(parents=True)
    (data / "queue-kanban-to-agent1").mkdir(parents=True)
    kb.DATA_DIR = data
    kb.CARD_ID_MAP_PATH = tmp_path / ".card_id_map.json"
    kb.QUEUE_AGENT1_TO_KANBAN = data / "queue-agent1-to-kanban"
    kb.QUEUE_KANBAN_TO_AGENT1 = data / "queue-kanban-to-agent1"
    try:
        yield
    finally:
        kb.DATA_DIR, kb.CARD_ID_MAP_PATH, kb.QUEUE_AGENT1_TO_KANBAN, kb.QUEUE_KANBAN_TO_AGENT1 = saved


@pytest.fixture()
def kb_data(tmp_path: Path):
    """Return (data_dir, tmp_path) for tests that need the paths explicitly."""
    data = tmp_path / "data"
    (data / "queue-agent1-to-kanban").mkdir(parents=True, exist_ok=True)
    (data / "queue-kanban-to-agent1").mkdir(parents=True, exist_ok=True)
    return data, tmp_path


# ---------------------------------------------------------------------------
# enqueue / read_queue round-trip
# ---------------------------------------------------------------------------

class TestEnqueueReadQueue:
    def test_enqueue_appends_json_line(self, kb_data):
        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        msg = {"op": "issue_create", "source_id": "iss-foo"}
        seq = kb.enqueue(msg)
        assert isinstance(seq, int) and seq >= 1

        lines = (data_dir / "queue-agent1-to-kanban" / "messages.jsonl").read_text().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["op"] == "issue_create"
        assert data["source_id"] == "iss-foo"
        assert data["seq"] == seq

    def test_enqueue_increments_sequence(self, kb_data):
        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        msg = {"op": "issue_update"}
        s1 = kb.enqueue(msg)
        s2 = kb.enqueue(msg)
        assert s2 > s1

    def test_read_queue_returns_new_lines(self, kb_data):
        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        msg = {"op": "issue_create", "source_id": "x"}
        kb.enqueue(msg)
        # enqueue() must NOT touch offset.txt (the reader's cursor), so the
        # message it just wrote is visible to the reader with no reset dance.
        messages, offset = kb.read_queue(kb.QUEUE_AGENT1_TO_KANBAN)
        assert len(messages) == 1
        assert messages[0]["source_id"] == "x"
        assert offset == 1

    def test_read_queue_skips_processed(self, kb_data):
        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        msg = {"op": "issue_create", "source_id": "a"}
        kb.enqueue(msg)
        # The reader owns offset.txt; enqueue() leaves it at 0 so the message
        # is visible on the first read.
        messages1, next_offset = kb.read_queue(kb.QUEUE_AGENT1_TO_KANBAN)
        assert len(messages1) == 1
        assert next_offset == 1

        # Advance offset to mark it processed
        kb.advance_offset(kb.QUEUE_AGENT1_TO_KANBAN, 1)
        messages2, _ = kb.read_queue(kb.QUEUE_AGENT1_TO_KANBAN)
        assert len(messages2) == 0

    def test_read_queue_empty_returns_zero(self, kb_data):
        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        messages, offset = kb.read_queue(kb.QUEUE_AGENT1_TO_KANBAN)
        assert messages == []
        assert offset == 0

    def test_advance_offset(self, kb_data):
        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        kb.advance_offset(kb.QUEUE_AGENT1_TO_KANBAN, 42)
        val = int((data_dir / "queue-agent1-to-kanban" / "offset.txt").read_text().strip())
        assert val == 42


# ---------------------------------------------------------------------------
# ID mapping
# ---------------------------------------------------------------------------

class TestIdMapping:
    def test_link_and_resolve(self, kb_data):
        import harnessfix.kanban_bridge as kb
        _, tmp_path = kb_data
        m = {}
        linked = kb.link_ids("iss-foo", "card-bar", m)
        assert linked["iss-foo"] == "card-bar"
        assert linked["card-bar"] == "iss-foo"

        resolved = kb.resolve_id("iss-foo", linked)
        assert resolved == "card-bar"
        resolved_rev = kb.resolve_id("card-bar", linked)
        assert resolved_rev == "iss-foo"

    def test_link_ids_persists_to_disk(self, kb_data):
        import harnessfix.kanban_bridge as kb
        data_dir, tmp_path = kb_data
        kb.DATA_DIR = data_dir
        # CARD_ID_MAP_PATH is derived from REPO_ROOT at import time, so redirect
        # it into the temp dir instead of expecting it to follow DATA_DIR.
        kb.CARD_ID_MAP_PATH = tmp_path / ".card_id_map.json"
        m = {}
        kb.link_ids("iss-a", "card-b", m)
        data = json.loads((tmp_path / ".card_id_map.json").read_text())
        assert data["iss-a"] == "card-b"

    def test_resolve_missing_returns_none(self, kb_data):
        import harnessfix.kanban_bridge as kb
        resolved = kb.resolve_id("nonexistent")
        assert resolved is None

    def test_link_ids_default_path_never_writes_repo_root(self, kb_data):
        """Regression: link_ids() with no mapping must not touch REPO_ROOT.

        The pre-fix default wrote ``.card_id_map.json`` into the repository
        root, so merely running the test suite left an untracked artifact
        behind. The autouse isolation fixture redirects CARD_ID_MAP_PATH; this
        test pins that the default call path honours it.
        """
        import harnessfix.kanban_bridge as kb
        _, tmp_path = kb_data
        repo_root_artifact = kb.REPO_ROOT / ".card_id_map.json"
        existed_before = repo_root_artifact.exists()

        kb.link_ids("iss-default", "card-default")

        assert kb.CARD_ID_MAP_PATH == tmp_path / ".card_id_map.json"
        assert kb.CARD_ID_MAP_PATH.exists()
        assert kb.resolve_id("iss-default") == "card-default"
        assert repo_root_artifact.exists() == existed_before


# ---------------------------------------------------------------------------
# Message conversion (no filesystem needed)
# ---------------------------------------------------------------------------

class TestConversion:
    def test_issue_to_card_data(self):
        import harnessfix.kanban_bridge as kb
        issue = {
            "title": "Fix bug",
            "category": "best-effort-except",
            "severity": "high",
            "evidence": "Found in line 42",
            "suggested_approach": "Use _suppress_and_log",
            "status": "open",
        }
        data = kb.issue_to_card_data(issue)
        assert data["title"] == "Fix bug"
        assert "line 42" in data["text"]
        assert "Approach:" in data["text"]
        tags = [t for t in data["tags"] if t.startswith("[best-effort-except]")]
        assert len(tags) == 1

    def test_card_to_issue_data(self):
        import harnessfix.kanban_bridge as kb
        card = {
            "id": "abc123",
            "title": "Fix bug",
            "text": "Found in line 42\n\n---\nApproach:\nUse _suppress_and_log",
            "tags": ["[best-effort-except]", "[severity:high]"],
        }
        data = kb.card_to_issue_data(card)
        assert data["title"] == "Fix bug"
        assert "line 42" in data["evidence"]
        assert "_suppress_and_log" in data["suggested_approach"]


# ---------------------------------------------------------------------------
# QueueProcessor thread (lightweight — no real queue needed)
# ---------------------------------------------------------------------------

class TestQueueProcessor:
    def test_run_once_returns_zero_no_queue(self):
        import harnessfix.kanban_bridge as kb
        proc = kb.QueueProcessor()
        result = proc.run_once()
        assert isinstance(result, int)

    def test_stop_event_works(self):
        import harnessfix.kanban_bridge as kb
        proc = kb.QueueProcessor(poll_interval=0.1)
        proc.stop()
        assert proc._stop_event.is_set()


# ---------------------------------------------------------------------------
# Integration: Agent1 make_issue with sync enabled
# ---------------------------------------------------------------------------

class TestMakeIssueSync:
    def test_make_issue_enqueues_when_enabled(self):
        import harnessfix.kanban_bridge as kb
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}), \
             patch.object(kb, 'enqueue') as mock_enqueue, \
             patch('harnessfix.issues._get_kanban_bridge', return_value=kb):
            from harnessfix import issues as issue_store
            issue_store.make_issue(
                category="test", title="Test Issue", locations=["file.py:42"],
            )
            assert mock_enqueue.called
            call_args = mock_enqueue.call_args[0][0]
            assert call_args["op"] == "issue_create"
            assert call_args["source_id"].startswith("iss-test")

    def test_make_issue_no_op_when_disabled(self):
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "0"}, clear=False), \
             patch('harnessfix.issues._get_kanban_bridge', return_value=None):
            from harnessfix import issues as issue_store
            result = issue_store.make_issue(
                category="test", title="Test Issue", locations=["file.py:42"],
            )
            assert "title" in result

    def test_make_issue_no_crash_when_bridge_missing(self):
        """If kanban_bridge can't be imported, make_issue still works."""
        with patch('harnessfix.issues._get_kanban_bridge', side_effect=ImportError), \
             patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}):
            from harnessfix import issues as issue_store
            result = issue_store.make_issue(
                category="test", title="Test Issue", locations=["file.py:42"],
            )
            assert "title" in result

    def test_resolve_enqueues_when_enabled(self):
        import harnessfix.kanban_bridge as kb
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}), \
             patch.object(kb, 'enqueue') as mock_enqueue, \
             patch('harnessfix.issues._get_kanban_bridge', return_value=kb):
            from harnessfix import issues as issue_store
            issues = [issue_store.make_issue("test", "Test", ["f.py"])]
            result = issue_store.resolve(issues, issues[0]["id"], "resolved")
            assert result is True
            assert mock_enqueue.called
            call_args = mock_enqueue.call_args[0][0]
            assert call_args["op"] == "issue_resolve"

    def test_promote_enqueues_issue_update(self):
        """promote() must push the new autonomy_level to Kanban.

        Pre-fix ``promote()`` mutated ``autonomy_level`` and returned without
        enqueueing, so the card kept showing the level it had at creation time
        and a promoted issue looked un-promoted on the board.  Kanban already
        implements the ``issue_update`` handler, so the op was dead code from
        this side.
        """
        import harnessfix.kanban_bridge as kb
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}), \
             patch.object(kb, 'enqueue') as mock_enqueue, \
             patch('harnessfix.issues._get_kanban_bridge', return_value=kb):
            from harnessfix import issues as issue_store
            issues = [issue_store.make_issue("test", "Test", ["f.py"])]
            mock_enqueue.reset_mock()

            ok, _msg = issue_store.promote(issues, issues[0]["id"], 2)

            assert ok is True
            assert mock_enqueue.called, "promote() must enqueue an issue_update"
            payload = mock_enqueue.call_args[0][0]
            assert payload["op"] == "issue_update"
            assert payload["source_id"] == issues[0]["id"]
            assert payload["payload"]["autonomy_level"] == 2

    def test_promote_invalid_level_does_not_enqueue(self):
        """An out-of-range level must be rejected before any sync side-effect."""
        import harnessfix.kanban_bridge as kb
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}), \
             patch.object(kb, 'enqueue') as mock_enqueue:
            from harnessfix import issues as issue_store
            issues = [issue_store.make_issue("test", "Test", ["f.py"])]
            mock_enqueue.reset_mock()

            ok, msg = issue_store.promote(issues, issues[0]["id"], 99)

            assert ok is False
            assert "invalid autonomy_level" in msg
            assert not mock_enqueue.called
            assert issues[0]["autonomy_level"] == issue_store.DEFAULT_AUTONOMY_LEVEL

    def test_promote_no_op_when_sync_disabled(self):
        """Sync off -> promote() still works and enqueues nothing."""
        import harnessfix.kanban_bridge as kb
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "0"}, clear=False), \
             patch.object(kb, 'enqueue') as mock_enqueue:
            from harnessfix import issues as issue_store
            issues = [issue_store.make_issue("test", "Test", ["f.py"])]
            mock_enqueue.reset_mock()

            ok, _msg = issue_store.promote(issues, issues[0]["id"], 2)

            assert ok is True
            assert not mock_enqueue.called
            assert issues[0]["autonomy_level"] == 2

    def test_promote_survives_enqueue_raising(self):
        """A broken bridge must not abort the promotion itself."""
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}), \
             patch('harnessfix.issues._get_kanban_bridge') as mock_get:
            mock_get.return_value.enqueue.side_effect = OSError("disk full")
            from harnessfix import issues as issue_store
            issue = issue_store.make_issue("test", "T", ["f.py:1"])
            ok, _msg = issue_store.promote([issue], issue["id"], 2)
            assert ok is True
            assert issue["autonomy_level"] == 2


# ---------------------------------------------------------------------------
# Regression tests for bugs fixed in the kanban bridge rewrite
# ---------------------------------------------------------------------------

class TestEnqueueConcurrency:
    """Regression: concurrent enqueue() must not corrupt the queue or deadlock.

    The pre-rewrite implementation used a single global lock with a nested
    _lock_for() helper that could deadlock (non-reentrant) and drop the
    per-message ``seq`` field. Both must hold under thread contention.
    """

    def test_concurrent_enqueue_unique_contiguous_seqs(self, kb_data):
        import threading

        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        n_threads, n_each = 6, 25
        errors: list[str] = []

        def worker():
            try:
                for i in range(n_each):
                    kb.enqueue({"op": "issue_create", "source_id": f"t{i}"})
            except Exception as exc:  # noqa: BLE001
                errors.append(repr(exc))

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not any(t.is_alive() for t in threads), "enqueue() deadlocked"
        assert errors == [], f"enqueue() raised under contention: {errors}"

        total = n_threads * n_each
        lines = (data_dir / "queue-agent1-to-kanban" / "messages.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        seqs = sorted(json.loads(line)["seq"] for line in lines)
        assert len(lines) == total
        assert seqs == list(range(1, total + 1)), "seqs not unique+contiguous"

        # The writer's counter is seq.txt.  offset.txt is the reader's cursor
        # and must be untouched by enqueue() — otherwise every write silently
        # consumes the line it just produced and nothing is ever delivered.
        seq_counter = int(
            (data_dir / "queue-agent1-to-kanban" / "seq.txt").read_text(encoding="utf-8").strip()
        )
        assert seq_counter == total

        offset_file = data_dir / "queue-agent1-to-kanban" / "offset.txt"
        assert not offset_file.exists(), "enqueue() must not move the reader's cursor"

    def test_enqueue_writes_seq_field(self, kb_data):
        """Regression: enqueue() must persist its sequence number in the line."""
        import harnessfix.kanban_bridge as kb
        data_dir, _ = kb_data
        kb.DATA_DIR = data_dir
        kb.QUEUE_AGENT1_TO_KANBAN = data_dir / "queue-agent1-to-kanban"

        seq = kb.enqueue({"op": "issue_create", "source_id": "s"})
        line = (data_dir / "queue-agent1-to-kanban" / "messages.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()[0]
        assert json.loads(line)["seq"] == seq


class TestBestEffortSync:
    """Regression: Kanban sync is a side-effect and must never break the ledger.

    Pre-fix, _should_sync()/make_issue() called _get_kanban_bridge() unguarded,
    so a broken bridge (raising ImportError) aborted issue creation.
    """

    def test_make_issue_survives_bridge_raising(self):
        with patch('harnessfix.issues._get_kanban_bridge', side_effect=ImportError), \
             patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}):
            from harnessfix import issues as issue_store
            result = issue_store.make_issue(
                category="test", title="T", locations=["f.py:1"],
            )
            assert result["title"] == "T"
            assert result["id"].startswith("iss-test")

    def test_make_issue_survives_enqueue_raising(self):
        """A failing enqueue() (e.g. disk full) must not abort make_issue."""
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}), \
             patch('harnessfix.issues._get_kanban_bridge') as mock_get:
            mock_get.return_value.enqueue.side_effect = OSError("disk full")
            from harnessfix import issues as issue_store
            result = issue_store.make_issue(
                category="test", title="T", locations=["f.py:1"],
            )
            assert result["title"] == "T"

    def test_resolve_survives_enqueue_raising(self):
        with patch.dict(os.environ, {"KANBAN_SYNC_ENABLED": "1"}), \
             patch('harnessfix.issues._get_kanban_bridge') as mock_get:
            mock_get.return_value.enqueue.side_effect = OSError("disk full")
            from harnessfix import issues as issue_store
            issue = issue_store.make_issue("test", "T", ["f.py:1"])
            assert issue_store.resolve([issue], issue["id"], "resolved") is True


# ---------------------------------------------------------------------------
# Dead-lettering: durable retries + offset always advances
# ---------------------------------------------------------------------------

class TestDeadLettering:
    """A poison message must not pin the offset and stall the whole queue.

    Pre-fix, ``retry_count`` was only ever incremented in a local variable that
    was thrown away (or, worse, written back by matching a line *index* against
    ``seq``, which is ``offset + 1`` — so it rewrote the wrong line), and the
    offset advanced only ``if applied > 0``.  Net effect: ``retry_count >= 3``
    was unreachable and one bad message blocked every later message forever.
    """

    def _queue_with(self, qdir: Path, lines: list[str]) -> Path:
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "messages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return qdir

    def test_offset_advances_even_when_every_message_fails(self, tmp_path):
        import harnessfix.kanban_bridge as kb
        qdir = self._queue_with(
            tmp_path / "q", ['{"seq": 1, "op": "bogus_op", "source_id": "x"}'],
        )
        assert kb.process_inbound(qdir) == 0
        # The offset must move, or the poison message is re-read forever.
        assert int((qdir / "offset.txt").read_text(encoding="utf-8")) == 1

    def test_failed_message_is_requeued_with_incremented_retry_count(self, tmp_path):
        import harnessfix.kanban_bridge as kb
        qdir = self._queue_with(
            tmp_path / "q", ['{"seq": 1, "op": "bogus_op", "source_id": "x"}'],
        )
        kb.process_inbound(qdir)

        lines = (qdir / "messages.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2, "failed message must be re-queued at the tail"
        requeued = json.loads(lines[-1])
        assert requeued["retry_count"] == 1
        assert requeued["processed"] is False

    def test_dead_letter_dir_is_module_level_and_patchable(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb
        # The regression: DEAD_LETTER_DIR used to be a local inside
        # process_inbound, so this setattr raised AttributeError.
        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")

        qdir = self._queue_with(
            tmp_path / "q", ['{"seq": 1, "op": "bogus_op", "source_id": "x"}'],
        )
        for _ in range(kb.MAX_RETRIES):
            kb.process_inbound(qdir)

        dead = list((tmp_path / "dead-letter").glob("seq_*.jsonl"))
        assert len(dead) == 1, f"expected one dead-letter file, got {dead}"
        assert json.loads(dead[0].read_text(encoding="utf-8").splitlines()[0])["seq"] == 1

    def test_poison_message_does_not_block_following_good_message(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")

        applied_ops: list[str] = []
        monkeypatch.setattr(
            kb, "_process_single_message",
            lambda msg: applied_ops.append(msg.get("op")) or (msg.get("op") == "good_op"),
        )

        qdir = self._queue_with(
            tmp_path / "q",
            ['{"seq": 1, "op": "bad_op"}', '{"seq": 2, "op": "good_op"}'],
        )
        assert kb.process_inbound(qdir) == 1
        assert applied_ops == ["bad_op", "good_op"], "good message must still be reached"


class TestMalformedQueueLines:
    """An unparseable line must be quarantined, not dropped and not re-read.

    ``read_queue`` used to log a warning and move on while the caller advanced
    the offset past the line, so the only copy of whatever the sender wrote was
    lost to a log message.  Worse, when the malformed line was the *only*
    content, ``process_inbound`` guarded the offset update behind
    ``if new_messages:`` — which is empty in that case — so the cursor never
    moved and the bad line was re-read and re-logged on every poll forever.
    """

    def _queue_with(self, qdir: Path, lines: list[str]) -> Path:
        qdir.mkdir(parents=True, exist_ok=True)
        (qdir / "messages.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return qdir

    def test_malformed_only_queue_quarantines_and_advances(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")

        qdir = self._queue_with(tmp_path / "q", ["{not json"])
        assert kb.process_inbound(qdir) == 0

        # The cursor must move even though nothing parsed, or this line is
        # re-read (and re-quarantined) on every single poll.
        assert int((qdir / "offset.txt").read_text(encoding="utf-8")) == 1

        dead = list((tmp_path / "dead-letter").glob("malformed_line_*.jsonl"))
        assert len(dead) == 1, f"malformed line must be preserved, got {dead}"
        record = json.loads(dead[0].read_text(encoding="utf-8").splitlines()[0])
        assert record["raw"] == "{not json"
        assert record["seq"] == 1

    def test_malformed_line_is_not_silently_dropped(self, tmp_path, monkeypatch):
        """A good line must not carry the offset past an unparseable one."""
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")

        good = '{"seq": 1, "op": "card_update", "source_id": "c1", "payload": {}}'
        qdir = self._queue_with(tmp_path / "q", [good, "{not json"])
        kb.process_inbound(qdir)

        dead = list((tmp_path / "dead-letter").glob("malformed_line_*.jsonl"))
        assert len(dead) == 1, "the bad line must be quarantined, not discarded"
        assert json.loads(
            dead[0].read_text(encoding="utf-8").splitlines()[0]
        )["raw"] == "{not json"

    def test_malformed_line_is_read_once(self, tmp_path, monkeypatch):
        import harnessfix.kanban_bridge as kb
        monkeypatch.setattr(kb, "DEAD_LETTER_DIR", tmp_path / "dead-letter")

        qdir = self._queue_with(tmp_path / "q", ["{not json"])
        kb.process_inbound(qdir)
        kb.process_inbound(qdir)
        kb.process_inbound(qdir)

        dead = list((tmp_path / "dead-letter").glob("malformed_line_*.jsonl"))
        assert len(dead) == 1, (
            f"a quarantined line must not be re-processed on later polls: {dead}"
        )


# ---------------------------------------------------------------------------
# card_move → issue status: FRAME_TO_STATUS is keyed by column TITLE
# ---------------------------------------------------------------------------

class TestResolveFrameStatus:
    """``_resolve_frame_status`` maps a moved card's column to an issue status.

    ``FRAME_TO_STATUS`` is keyed by the human-readable column *title*
    ("Finished" -> resolved), while the Kanban app identifies a frame by an
    opaque uuid. A ``card_move`` message that carried only ``frame_id`` could
    therefore never hit the table: every move fell back to "open" and silently
    rewrote a resolved issue back to open.

    The Kanban side now sends ``frame_title`` alongside ``frame_id``; these
    tests pin the reader's half of that contract, including the fallback that
    keeps pre-fix messages (id only) from crashing.
    """

    def test_title_resolves_to_its_status(self):
        import harnessfix.kanban_bridge as kb

        assert kb._resolve_frame_status(
            {"payload": {"frame_id": "uuid-hex", "frame_title": "Finished"}}
        ) == "resolved"

    def test_every_mapped_column_title_resolves(self):
        """The table is the contract: each key must be reachable by title."""
        import harnessfix.kanban_bridge as kb

        for title, status in kb.FRAME_TO_STATUS.items():
            got = kb._resolve_frame_status(
                {"payload": {"frame_id": "opaque", "frame_title": title}}
            )
            assert got == status, f"{title!r} must map to {status!r}, got {got!r}"

    def test_opaque_id_alone_falls_back_to_open(self):
        """Pre-fix messages carry only the uuid; they must not crash."""
        import harnessfix.kanban_bridge as kb

        assert kb._resolve_frame_status(
            {"payload": {"frame_id": "6f1c-uuid-not-a-title"}}
        ) == "open"

    def test_empty_title_does_not_shadow_the_id(self):
        """An empty ``frame_title`` must not stop a usable id being tried."""
        import harnessfix.kanban_bridge as kb

        # frame_title is blank, so resolution must move on to the next key.
        assert kb._resolve_frame_status(
            {"payload": {"frame_title": "", "frame_id": "Finished"}}
        ) == "resolved"

    def test_unwrapped_payload_is_accepted(self):
        """A bare payload (no outer ``payload`` key) resolves the same way."""
        import harnessfix.kanban_bridge as kb

        assert kb._resolve_frame_status(
            {"frame_id": "x", "frame_title": "On-going"}
        ) == "in-progress"

    def test_unknown_title_is_open(self):
        import harnessfix.kanban_bridge as kb

        assert kb._resolve_frame_status(
            {"payload": {"frame_title": "Not A Column"}}
        ) == "open"
