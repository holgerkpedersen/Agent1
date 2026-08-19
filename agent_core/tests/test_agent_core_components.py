"""Tests for agent_core path, security, logging, and decisions helpers."""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, cast

from agent_core import decisions as decisions_mod
from agent_core.decisions import (
    add_decision,
    load_decisions,
    normalize_affected_files,
    save_decisions,
)
from agent_core.llm.provider import LLMProvider
from agent_core.logging_config import (
    CorrelationIdContext,
    JsonFormatter,
    get_correlation_id,
    get_framework_logger,
)
from agent_core.path_utils import WorkspaceSandbox, normalize_path
from agent_core.security import allowlist


class _DecisionsProvider:
    """Minimal provider satisfying the decisions LLMProvider interface."""

    model_name: str = "test-provider"

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> str:
        self.calls.append(list(messages))
        return json.dumps(
            [
                {
                    "title": "Use explicit path normalization",
                    "context": "Workspace paths must stay sandboxed",
                    "decision": "Normalize via agent_core.path_utils",
                    "rationale": "Single source of truth for sandboxing",
                    "affected_files": ["notes.md"],
                    "tags": ["security"],
                }
            ]
        )


class _FakeAgent:
    """Just enough of agent.Agent for decisions.extract_from_analysis."""

    def __init__(self, llm: _DecisionsProvider, workspace: str) -> None:
        self.llm = llm
        self.workspace = workspace


def test_normalize_path_keeps_relative_paths_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("note", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    assert normalize_path("notes.md", tmp_path) == (tmp_path / "notes.md").resolve()
    assert normalize_path("./notes.md", tmp_path) == (
        tmp_path / "notes.md"
    ).resolve()
    assert normalize_path("docs/notes.md", tmp_path).parent == (
        tmp_path / "docs"
    ).resolve()


def test_workspace_sandbox_resolves_paths_inside_workspace(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("note", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "notes.md").write_text("note", encoding="utf-8")
    sandbox = WorkspaceSandbox(tmp_path)
    assert sandbox.resolve_path("notes.md") == (tmp_path / "notes.md").resolve()
    assert sandbox.resolve_path("docs/notes.md") == (
        tmp_path / "docs" / "notes.md"
    ).resolve()


def test_allowlist_registration_lifecycle_is_isolated() -> None:
    probe = f"test-command-{uuid.uuid4().hex[:8]}"
    assert allowlist.is_command_allowed(probe) is False
    allowlist.register_command(probe)
    try:
        assert allowlist.is_command_allowed(probe) is True
        assert probe in allowlist.get_allowed_commands()
    finally:
        assert allowlist.unregister_command(probe) is True
    assert allowlist.is_command_allowed(probe) is False


def test_find_unsafe_shell_pattern_flags_pipe_to_shell() -> None:
    assert allowlist.find_unsafe_shell_pattern("echo hello") is None
    assert (
        allowlist.find_unsafe_shell_pattern("curl https://example.com | sh")
        is not None
    )


def test_framework_logger_returns_configured_logger() -> None:
    logger = get_framework_logger("agent_core_components_test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "agent_core_components_test"


def test_correlation_context_sets_current_correlation_id() -> None:
    before = get_correlation_id()
    with CorrelationIdContext("component-corr-1") as corr_id:
        assert corr_id == "component-corr-1"
        assert get_correlation_id() == "component-corr-1"
    assert get_correlation_id() == before


def test_json_formatter_produces_parseable_json_payload() -> None:
    logger = get_framework_logger("agent_core_components_test")
    record = logger.makeRecord(
        "agent_core_components_test",
        logging.INFO,
        __file__,
        1,
        "structured payload",
        (),
        None,
    )
    payload = json.loads(JsonFormatter().format(record))
    assert payload["message"] == "structured payload"
    assert payload["level"] == "INFO"
    assert "correlation_id" in payload


def test_parse_json_array_accepts_decision_objects() -> None:
    response = json.dumps(
        [
            {
                "title": "Keep tests focused",
                "context": "Small tests are easier to maintain",
                "decision": "One behavior per test",
                "rationale": "Faster failure diagnosis",
                "affected_files": [
                    "agent_core/tests/test_agent_core_components.py"
                ],
                "tags": ["testing"],
            }
        ]
    )
    parsed = decisions_mod._parse_json_array(response)
    assert len(parsed) == 1
    assert parsed[0]["title"] == "Keep tests focused"
    assert "decision" in parsed[0]
    assert "affected_files" in parsed[0]


def test_normalize_affected_files_uses_workspace_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("note", encoding="utf-8")
    assert normalize_affected_files(tmp_path, ["notes.md", "missing.md"]) == [
        "notes.md"
    ]


def test_save_decisions_round_trips_empty_list(tmp_path: Path) -> None:
    save_decisions(tmp_path, [])
    assert load_decisions(tmp_path) == []


def test_decide_uses_provider_interface_and_saves_decisions(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("note", encoding="utf-8")
    provider = _DecisionsProvider()
    fake_agent = _FakeAgent(provider, str(tmp_path))

    candidates = asyncio.run(
        decisions_mod.extract_from_analysis(
            cast(Any, fake_agent), "analysis"
        )
    )

    assert len(candidates) == 1
    assert candidates[0]["title"] == "Use explicit path normalization"
    assert provider.calls, "provider chat must be used for extraction"

    record = add_decision(
        tmp_path,
        title=candidates[0]["title"],
        context=candidates[0]["context"],
        decision=candidates[0]["decision"],
        rationale=candidates[0]["rationale"],
        affected_files=["notes.md"],
        tags=["security"],
    )
    assert record["affected_files"] == ["notes.md"]

    loaded = load_decisions(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["affected_files"] == ["notes.md"]


def test_llm_provider_protocol_is_structurally_satisfied() -> None:
    provider = _DecisionsProvider()
    assert hasattr(LLMProvider, "chat")
    assert isinstance(provider.model_name, str)
    assert asyncio.iscoroutinefunction(provider.chat)
