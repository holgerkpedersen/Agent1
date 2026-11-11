"""Regression tests: session identity, claims, and outbound payload stamping.

Covers the Kanban card-work-visibility contract from plan
``.docs/2026-11-10_23-06-43/plan_proposed.md`` (tasks 1-6):

* ``agent_core.session.identity`` — session id resolution, owner label,
  first-writer-wins claims, TTL expiry, owner-only release.
* ``harnessfix.issues`` — every outbound Kanban payload carries the identity
  fields; claim-before-work refuses a foreign live claim; resolving releases
  the owning claim and never steals a foreign one; sync-off is a no-op.

All tests are hermetic: env vars, the claims mirror path, the issues ledger
path and the bridge module are injected — nothing touches the real repo state.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from agent_core.session import identity


@pytest.fixture()
def claims_file(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the claims mirror to a temp file and clear session env vars."""
    path = tmp_path / ".card_claims.json"
    monkeypatch.setattr(identity, "CLAIMS_PATH", path)
    monkeypatch.delenv("AGENT1_SESSION_ID", raising=False)
    monkeypatch.setenv("AGENT1_SESSION_ID", "tester1")
    return path


@pytest.fixture()
def issues_env(tmp_path: Path, monkeypatch):
    """Hermetic harnessfix.issues env: temp ledger, fake bridge, sync on/off.

    Yields ``(issues_mod, recorder, setter)`` where *recorder* is the list of
    enqueued messages and *setter* toggles sync (True/False).
    """
    from harnessfix import issues as im

    ledger = tmp_path / ".issues.json"
    monkeypatch.setattr(im, "ISSUES_PATH", ledger)

    recorder: list[dict] = []

    class _FakeBridge:
        def __init__(self) -> None:
            self.sync_on = True

        def sync_enabled(self) -> bool:
            return self.sync_on

        def active_board_allows_agent1(self) -> bool:
            return self.sync_on

        def enqueue(self, msg: dict) -> None:
            recorder.append(msg)

    fake = _FakeBridge()
    monkeypatch.setattr(im, "_get_kanban_bridge", lambda: fake)
    monkeypatch.delenv("AGENT1_SESSION_ID", raising=False)
    monkeypatch.setenv("AGENT1_SESSION_ID", "tester1")

    def setter(on: bool) -> None:
        fake.sync_on = on

    return im, recorder, setter


# ---------------------------------------------------------------------------
# Task 1 — session identity helper
# ---------------------------------------------------------------------------

def test_session_id_prefers_env(monkeypatch):
    monkeypatch.setenv("AGENT1_SESSION_ID", "Tester-One")
    assert identity.session_id() == "tester-one"  # slugified, not raw


def test_session_id_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("AGENT1_SESSION_ID", raising=False)
    monkeypatch.setattr(identity, "_read_env_file_value", lambda key: None)
    monkeypatch.setattr(
        identity,
        "_read_env_file_value",
        lambda key: None,  # no repo .env value
    )
    # No env value at all -> the legacy single-session default.
    import agent_core.session.context as ctx

    monkeypatch.setattr(ctx, "active_session", lambda: None)
    assert identity.session_id() == "default"


def test_agent_label_is_slug_at_host(monkeypatch):
    monkeypatch.setenv("AGENT1_SESSION_ID", "tester1")
    monkeypatch.setattr(identity, "_host", lambda: "BOX42")
    assert identity.agent_label() == "tester1@BOX42"


def test_identity_payload_has_contract_fields():
    payload = identity.identity_payload()
    assert set(payload) == {"agent1_session", "agent1_label", "claimed_at"}
    assert payload["agent1_session"]  # non-empty slug
    # claimed_at must parse as ISO-8601 (the Kanban side renders it).
    from datetime import datetime

    datetime.fromisoformat(payload["claimed_at"])


# ---------------------------------------------------------------------------
# Task 3 — first-writer-wins claims
# ---------------------------------------------------------------------------

def test_claim_is_first_writer_wins(claims_file):
    acquired, rec = identity.claim_card("cardA")
    assert acquired and rec["owner"] == "tester1"
    # Same session re-claiming is idempotent (owner match).
    again, rec2 = identity.claim_card("cardA")
    assert not again and rec2["owner"] == "tester1"


def test_foreign_live_claim_is_refused(claims_file):
    claims_file.write_text(
        json.dumps(
            {
                "cardA": {
                    "owner": "other",
                    "label": "other@BOX42",
                    "claimed_at": identity._iso(),
                    "host": identity._host(),
                }
            }
        ),
        encoding="utf-8",
    )
    # Foreign claim, fresh (inside TTL) -> live -> refused.
    acquired, rec = identity.claim_card("cardA")
    assert not acquired and rec["owner"] == "other"


def test_expired_foreign_claim_is_takeable(claims_file):
    expired = identity._now() - timedelta(seconds=identity.CLAIM_TTL_SECONDS + 60)
    old = identity._iso(expired)
    claims_file.write_text(
        json.dumps(
            {
                "cardA": {
                    "owner": "ghost",
                    "label": "ghost@BOX42",
                    "claimed_at": old,
                    "host": "",
                }
            }
        ),
        encoding="utf-8",
    )
    acquired, rec = identity.claim_card("cardA")
    assert acquired and rec["owner"] == "tester1"


def test_release_is_owner_only(claims_file):
    identity.claim_card("cardA")
    # A foreign session's release must not steal our claim.
    claims = json.loads(claims_file.read_text(encoding="utf-8"))
    claims["cardB"] = {
        "owner": "other",
        "label": "other@BOX42",
        "claimed_at": identity._iso(),
        "host": identity._host(),
    }
    claims_file.write_text(json.dumps(claims), encoding="utf-8")
    assert identity.release_card("cardA") is True
    assert not (claims_file.read_text(encoding="utf-8")).count("cardA")
    assert identity.release_card("cardB") is False  # not ours
    assert "cardB" in json.loads(claims_file.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Task 2 — identity stamped on every outbound payload
# ---------------------------------------------------------------------------

def test_issue_create_payload_carries_identity(issues_env):
    im, recorder, _ = issues_env
    im.make_issue("bug", "boom", ["src/x.py"], parent_card_id="cardA")
    ops = [m for m in recorder if m["op"] == "issue_create"]
    assert ops, "issue_create must be enqueued"
    payload = ops[0]["payload"]
    assert payload["agent1_session"] == "tester1"
    assert payload["agent1_label"].startswith("tester1")
    assert payload["claimed_at"]


def test_child_card_payload_carries_identity_and_badge(issues_env):
    im, recorder, _ = issues_env
    im.make_issue(
        "bug",
        "boom",
        ["src/x.py"],
        parent_card_id="cardA",
        emit_child_card=True,
    )
    children = [m for m in recorder if m["op"] == "child_card_create"]
    assert len(children) == 1, "exactly one child card per work item"
    payload = children[0]["payload"]
    assert payload["parent_id"] == "cardA"
    assert payload["agent1_session"] == "tester1"
    assert "[tester1]" in payload["title"]
    assert "[card:cardA]" in payload["tags"]
    assert "[tester1]" in payload["tags"]


def test_sync_off_is_noop(issues_env):
    im, recorder, setter = issues_env
    setter(False)
    im.make_issue("bug", "boom", ["src/x.py"], parent_card_id="cardA")
    assert recorder == []


# ---------------------------------------------------------------------------
# Task 3/5 — claim-before-work and the resolve ownership guard
# ---------------------------------------------------------------------------

def _mk_card_issue(im, tmp_path: Path) -> str:
    issue = im.make_issue("bug", "boom", ["src/x.py"], parent_card_id="cardA")
    issues = im.load_issues()
    im.upsert(issues, issue)
    im.save_issues(issues)
    return issue["id"]


def test_claim_card_for_work_acquires_and_notifies(issues_env, claims_file):
    im, recorder, _ = issues_env
    issue_id = _mk_card_issue(im, claims_file.parent)
    ok, msg = im.claim_card_for_work(issue_id)
    assert ok and "cardA" in msg
    claims = json.loads(claims_file.read_text(encoding="utf-8"))
    assert claims["cardA"]["owner"] == "tester1"
    notified = [m for m in recorder if m["op"] == "card_claimed"]
    assert notified and notified[0]["payload"]["owner"] == "tester1"


def test_claim_card_for_work_refuses_foreign_live_claim(issues_env, claims_file):
    im, recorder, _ = issues_env
    issue_id = _mk_card_issue(im, claims_file.parent)
    claims_file.write_text(
        json.dumps(
            {
                "cardA": {
                    "owner": "other",
                    "label": "other@BOX42",
                    "claimed_at": identity._iso(),
                    "host": identity._host(),
                }
            }
        ),
        encoding="utf-8",
    )
    ok, msg = im.claim_card_for_work(issue_id)
    assert not ok and "claimed by" in msg
    # The board is told who owns it (the foreign owner), not us.
    notified = [m for m in recorder if m["op"] == "card_claimed"]
    assert notified and notified[0]["payload"]["owner"] == "other"


def test_claim_card_for_work_sync_off_is_success_noop(issues_env):
    im, recorder, setter = issues_env
    setter(False)
    ok, msg = im.claim_card_for_work("iss-whatever")
    assert ok and recorder == []


def test_resolve_releases_own_claim(issues_env, claims_file):
    im, recorder, _ = issues_env
    issue_id = _mk_card_issue(im, claims_file.parent)
    assert im.claim_card_for_work(issue_id)[0]
    issues = im.load_issues()
    assert im.resolve(issues, issue_id, "resolved", note="done", emit_child_card=True)
    im.save_issues(issues)
    claims = json.loads(claims_file.read_text(encoding="utf-8"))
    assert "cardA" not in claims, "resolving must end the owner's claim"


def test_resolve_never_steals_foreign_claim(issues_env, claims_file):
    im, recorder, _ = issues_env
    issue_id = _mk_card_issue(im, claims_file.parent)
    claims_file.write_text(
        json.dumps(
            {
                "cardA": {
                    "owner": "other",
                    "label": "other@BOX42",
                    "claimed_at": identity._iso(),
                    "host": identity._host(),
                }
            }
        ),
        encoding="utf-8",
    )
    issues = im.load_issues()
    assert im.resolve(issues, issue_id, "resolved")
    im.save_issues(issues)
    claims = json.loads(claims_file.read_text(encoding="utf-8"))
    assert claims["cardA"]["owner"] == "other", "foreign claim survives our resolve"
