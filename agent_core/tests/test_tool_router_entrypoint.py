"""Entrypoint tests for tool_router typed validation and allowlist enforcement."""

import json

import pytest

import tool_router
from agent_core.security import allowlist
from tool_router import (
    RoutingError,
    ShellCommandArgs,
    ToolExecutionError,
    ToolRouter,
)


def _router() -> ToolRouter:
    router = ToolRouter()
    schemas = {schema["name"] for schema in router.get_schemas()}
    assert "run_command" in schemas, (
        "shell_command route should exist: " + ", ".join(sorted(schemas))
    )
    return router


def test_tool_router_typed_validation_accepts_valid_args() -> None:
    router = _router()
    name, args = router.parse_explicit_command("/tool:run_command command='echo ok'")
    assert name == "run_command"
    assert isinstance(args, ShellCommandArgs)
    assert args.command == "echo ok"


def test_tool_router_typed_validation_rejects_missing_args() -> None:
    router = _router()
    with pytest.raises(ToolExecutionError) as exc_info:
        router._validate_args("read_file", {})
    assert "Validation failed for read_file" in str(exc_info.value)


def test_tool_router_allowlist_enforces_registered_command_lifecycle() -> None:
    probe = "tool-router-allowlist-probe"
    try:
        allowlist.register_command(probe)
        assert allowlist.is_command_allowed(probe) is True
    finally:
        assert allowlist.unregister_command(probe) is True
    assert allowlist.is_command_allowed(probe) is False


def test_tool_router_allowlist_flags_unsafe_shell_pattern() -> None:
    assert allowlist.find_unsafe_shell_pattern("rm -rf /") is None
    assert allowlist.find_unsafe_shell_pattern("cat file.txt && rm -rf /") is not None


def test_tool_router_rejects_disallowed_shell_command() -> None:
    router = _router()
    with pytest.raises(ToolExecutionError) as exc_info:
        router.route_and_execute("/tool:run_command command='rm -rf /'")
    assert "not in the allowed command list" in str(exc_info.value)


def test_tool_router_rejects_unsafe_shell_command_even_if_binary_is_registered() -> None:
    probe = "tool-router-unsafe-probe"
    allowlist.register_command(probe)
    try:
        handler = tool_router.ShellCommandHandler()
        with pytest.raises(ToolExecutionError) as exc_info:
            handler.execute(ShellCommandArgs(command=f"{probe} && rm -rf /"))
    finally:
        allowlist.unregister_command(probe)
    assert "chaining" in str(exc_info.value)


def test_tool_router_levenshtein_distance_is_stable_for_typo_recovery() -> None:
    router = _router()
    assert router._resolve_tool_name("read_file") == "read_file"
    assert router._resolve_tool_name("write_file") == "write_file"
    assert router._resolve_tool_name("read_fil") == "read_file"
    for typo in ("shell_command", "shell_commands"):
        with pytest.raises(RoutingError):
            router._resolve_tool_name(typo)


def test_tool_router_get_schemas_returns_jsonable_definitions() -> None:
    router = _router()
    schemas = router.get_schemas()
    assert {"read_file", "write_file", "search_files", "run_command"} <= {
        schema["name"] for schema in schemas
    }
    for schema in schemas:
        assert set(schema) == {"name", "description", "parameters"}
        json.dumps(schema)
