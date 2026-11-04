"""The inspector's CLI surface — the commands a human/agent types at a prompt.

``tests/test_kanban_bridge_content.py`` covers the inspector's library calls.
This file covers ``main()``, because that is what the REPL's ``kanban`` command
and the skill tell people to run: a subcommand that parses but prints nothing,
or one that raises on a missing board, is a broken integration command even
while every library function passes its own test.

Everything runs against temp files — never the real ``data/`` queue.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

AGENT1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AGENT1_ROOT))

from harnessfix import issues as issue_store  # noqa: E402
from harnessfix import kanban_bridge as kb  # noqa: E402
from harnessfix import kanban_inspect as ki  # noqa: E402


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    """Isolate the ledger, the queue, the id map and the board in one temp dir."""
    data = tmp_path / "data"
    data.mkdir()
    for mod in (kb,):
        monkeypatch.setattr(mod, "DATA_DIR", data)
        monkeypatch.setattr(
            mod, "QUEUE_AGENT1_TO_KANBAN", data / "queue-agent1-to-kanban"
        )
        monkeypatch.setattr(
            mod, "QUEUE_KANBAN_TO_AGENT1", data / "queue-kanban-to-agent1"
        )
        monkeypatch.setattr(mod, "CARD_ID_MAP_PATH", data / ".card_id_map.json")
        monkeypatch.setattr(mod, "DEAD_LETTER_DIR", data / "dead-letter")
    monkeypatch.setattr(issue_store, "ISSUES_PATH", tmp_path / "issues.json")
    # Isolate the .env master switch: the status test asserts sync is enabled,
    # so provide an enabled .env in the temp dir instead of relying on the
    # (gitignored, CI-absent) real repo .env.
    env_file = tmp_path / ".env"
    env_file.write_text("KANBAN_SYNC_ENABLED=1\n", encoding="utf-8")
    monkeypatch.setattr(kb, "ENV_FILE_PATH", env_file)
    monkeypatch.delenv("KANBAN_SYNC_ENABLED", raising=False)

    (data / "active.json").write_text(
        json.dumps({"active": "b1"}), encoding="utf-8"
    )
    (data / "board-b1.json").write_text(
        json.dumps(
            {
                "frames": [
                    {"id": "f1", "title": "Issues"},
                    {"id": "f2", "title": "On-going"},
                    {"id": "f3", "title": "Finished"},
                ],
                "cards": [
                    {
                        "id": "ccli",
                        "title": "CLI card under test",
                        "text": "the description the bridge used to drop",
                        "frame_id": "f2",
                        "tags": [],
                        "status": "in-progress",
                    }
                ],
                "sync": {"agent1": True},
            }
        ),
        encoding="utf-8",
    )
    return data


def _run(capsys, argv: list[str]) -> dict:
    """Run the CLI and parse what it printed."""
    assert ki.main(argv) == 0
    return json.loads(capsys.readouterr().out)


# ── subcommands ──────────────────────────────────────────────────────────────


def test_keys_subcommand_documents_the_key_map(sandbox, capsys) -> None:
    """``keys`` is the answer to "how do the keys work between the repos"."""
    payload = _run(capsys, ["keys"])
    keys = {item["key"]: item for item in payload}

    assert any("source_id" in k for k in keys)
    assert ".card_id_map.json" in keys
    assert "locations: ['kanban:<card-id>']" in keys
    for item in payload:
        assert item["meaning"]
        assert item["where"]


def test_card_subcommand_shows_both_sides_and_the_description(
    sandbox, capsys
) -> None:
    """The card view is the entry point for "work on card X"."""
    kb._apply_card_create(
        {
            "id": "ccli",
            "title": "CLI card under test",
            "text": "the description the bridge used to drop",
            "frame_id": "f2",
            "tags": [],
        }
    )

    view = _run(capsys, ["card", "ccli"])

    assert view["link"]["card_id"] == "ccli"
    assert view["link"]["issue_id"]
    assert view["card"]["frame_title"] == "On-going"
    # The default view is compact; the full ledger record needs --full.
    assert set(view["issue"]) <= {
        "id",
        "title",
        "status",
        "severity",
        "autonomy_level",
    }
    assert "the description the bridge used to drop" in view["description"]

    full = _run(capsys, ["card", "ccli", "--full"])
    assert "locations" in full["issue"]


def test_card_subcommand_resolves_a_title_the_user_typed(sandbox, capsys) -> None:
    """A human types the card's words, not its guid."""
    view = _run(capsys, ["card", "cli card under"])
    assert view["link"]["card_id"] == "ccli"
    assert view["issue"] is None
    assert any("never linked" in w for w in view["link"]["warnings"])


def test_card_subcommand_on_an_unknown_key_explains_the_miss(
    sandbox, capsys
) -> None:
    """No match is a normal outcome and must not raise — the agent reads this."""
    view = _run(capsys, ["card", "nosuchkey"])
    assert not view["link"].get("card_id")
    assert not view["link"].get("issue_id")
    assert view["link"]["warnings"]


def test_queue_subcommand_lists_pending_with_issue_ids(sandbox, capsys) -> None:
    kb._apply_card_create(
        {"id": "ccli", "title": "CLI card under test", "text": "t", "frame_id": "f2"}
    )
    kb.enqueue(
        {"op": "card_update", "source_id": "ccli", "payload": {"title": "renamed"}},
        sandbox / "queue-kanban-to-agent1",
    )

    payload = _run(capsys, ["queue", "--limit", "5"])
    inbound = next(q for q in payload["queues"] if q["direction"] == "kanban→agent1")
    assert inbound["pending"] >= 1
    assert any(m.get("resolved_issue") for m in inbound["messages"])


def test_board_subcommand_maps_columns_to_issue_statuses(sandbox, capsys) -> None:
    payload = _run(capsys, ["board"])
    assert payload["board_id"] == "b1"
    finished = next(f for f in payload["frames"] if f["title"] == "Finished")
    assert finished["issue_status"] == "resolved"
    ongoing = next(f for f in payload["frames"] if f["title"] == "On-going")
    assert ongoing["cards"][0]["id"] == "ccli"


def test_status_subcommand_reports_map_health_and_sync_gate(sandbox, capsys) -> None:
    payload = _run(capsys, ["status"])
    assert payload["active_board"] == "b1"
    assert payload["sync_enabled"] is True
    assert payload["active_board_allows_agent1"] is True
    assert payload["map"]["entries"] == 0
    assert len(payload["queues"]) == 2


def test_sync_descriptions_subcommand_is_dry_run_by_default_safe(
    sandbox, capsys
) -> None:
    """The write path requires an explicit flag; the preview writes nothing."""
    kb._apply_card_create(
        {"id": "ccli", "title": "CLI card under test", "text": "", "frame_id": "f2"}
    )

    preview = _run(capsys, ["sync-descriptions", "--dry-run"])
    assert preview["dry_run"] is True
    assert preview["updated"] >= 1
    issue = issue_store.load_issues()[0]
    assert issue["suggested_approach"] == "", "the dry run wrote to the ledger"

    applied = _run(capsys, ["sync-descriptions"])
    assert applied["dry_run"] is False
    assert applied["updated"] >= 1
    issue = issue_store.load_issues()[0]
    assert (
        issue["suggested_approach"]
        == "the description the bridge used to drop"
    )


def test_main_requires_a_subcommand(sandbox) -> None:
    """A bare invocation is a usage error, not a crash with a traceback."""
    with pytest.raises(SystemExit):
        ki.main([])
