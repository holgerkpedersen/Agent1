"""Tests for Agent1's per-board Kanban sync gate (the fail-closed side).

``issues._should_sync()`` is a two-layer gate:

* layer 1 — the process-level master switch ``KANBAN_SYNC_ENABLED`` (process
  env first, then the repo ``.env``); and
* layer 2 — the Kanban app's ACTIVE board must have opted in to agent1 sync
  via its on-disk ``sync`` map (see ``kanban_bridge.active_board_allows_agent1``).

Both layers fail closed: missing files, malformed JSON or a non-allowing
board all keep every outbound enqueue shut. The regression this exists for:
with only the env switch on, no issue could reach Kanban until the active
board explicitly opted in — and it must stay that way.
"""

import json

import pytest

import harnessfix.kanban_bridge as kb
from harnessfix import issues as issue_store


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Point the bridge at a throwaway data dir with no board files."""
    data = tmp_path / "kb-data"
    data.mkdir()
    env_file = tmp_path / ".env"
    env_file.write_text("KANBAN_SYNC_ENABLED=0\n", encoding="utf-8")
    monkeypatch.setattr(kb, "DATA_DIR", data)
    monkeypatch.setattr(kb, "ENV_FILE_PATH", env_file)
    monkeypatch.delenv("KANBAN_SYNC_ENABLED", raising=False)
    return data


def _write_board(data, *, board_id="demo", agent1=True):
    """Write an active registry entry plus the board's working copy."""
    (data / "active.json").write_text(
        json.dumps({"active": board_id}), encoding="utf-8"
    )
    (data / f"board-{board_id}.json").write_text(
        json.dumps({"frames": [], "cards": [], "sync": {"agent1": agent1}}),
        encoding="utf-8",
    )


# ---- active_board_allows_agent1 -------------------------------------------------


class TestActiveBoardAllowsAgent1:
    def test_no_registry_fails_closed(self, _isolate):
        assert kb.active_board_allows_agent1() is False

    def test_malformed_registry_fails_closed(self, _isolate):
        (_isolate / "active.json").write_text("{not json", encoding="utf-8")
        assert kb.active_board_allows_agent1() is False

    def test_missing_active_id_fails_closed(self, _isolate):
        (_isolate / "active.json").write_text("{}", encoding="utf-8")
        assert kb.active_board_allows_agent1() is False

    def test_non_string_active_id_fails_closed(self, _isolate):
        (_isolate / "active.json").write_text(
            json.dumps({"active": 7}), encoding="utf-8"
        )
        assert kb.active_board_allows_agent1() is False

    def test_missing_board_file_fails_closed(self, _isolate):
        (_isolate / "active.json").write_text(
            json.dumps({"active": "ghost"}), encoding="utf-8"
        )
        assert kb.active_board_allows_agent1() is False

    def test_malformed_board_fails_closed(self, _isolate):
        (_isolate / "active.json").write_text(
            json.dumps({"active": "demo"}), encoding="utf-8"
        )
        (_isolate / "board-demo.json").write_text("nope", encoding="utf-8")
        assert kb.active_board_allows_agent1() is False

    def test_no_sync_map_fails_closed(self, _isolate):
        (_isolate / "active.json").write_text(
            json.dumps({"active": "demo"}), encoding="utf-8"
        )
        (_isolate / "board-demo.json").write_text(
            json.dumps({"frames": [], "cards": []}), encoding="utf-8"
        )
        assert kb.active_board_allows_agent1() is False

    def test_sync_map_without_agent1_fails_closed(self, _isolate):
        (_isolate / "active.json").write_text(
            json.dumps({"active": "demo"}), encoding="utf-8"
        )
        (_isolate / "board-demo.json").write_text(
            json.dumps({"sync": {"other-system": True}}), encoding="utf-8"
        )
        assert kb.active_board_allows_agent1() is False

    def test_agent1_false_fails_closed(self, _isolate):
        _write_board(_isolate, agent1=False)
        assert kb.active_board_allows_agent1() is False

    def test_agent1_true_allows(self, _isolate):
        _write_board(_isolate)
        assert kb.active_board_allows_agent1() is True

    def test_path_like_active_id_rejected(self, _isolate):
        (_isolate / "active.json").write_text(
            json.dumps({"active": "../evil"}), encoding="utf-8"
        )
        assert kb.active_board_allows_agent1() is False


# ---- issues._should_sync two-layer gating ---------------------------------------


class TestShouldSyncGating:
    def test_env_on_and_board_allows(self, _isolate, monkeypatch):
        _write_board(_isolate)
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")
        assert issue_store._should_sync() is True

    def test_env_on_but_no_board_fails_closed(self, _isolate, monkeypatch):
        """The regression this feature exists for: the env switch alone must
        not open outbound sync while no board has opted in."""
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")
        assert issue_store._should_sync() is False

    def test_env_on_board_opted_out(self, _isolate, monkeypatch):
        _write_board(_isolate, agent1=False)
        monkeypatch.setenv("KANBAN_SYNC_ENABLED", "1")
        assert issue_store._should_sync() is False

    def test_env_off_board_allows_still_closed(self, _isolate):
        """Layer 1 still dominates: master switch off keeps sync closed even
        for an opted-in board."""
        _write_board(_isolate)
        # .env fixture says KANBAN_SYNC_ENABLED=0 and process env is unset.
        assert issue_store._should_sync() is False
