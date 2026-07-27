from __future__ import annotations

import pytest

from tool_router import (
    ReadFileArgs,
    WriteFileArgs,
    SearchFilesArgs,
    ShellCommandArgs,
    ShellCommandHandler,
    ToolDefinition,
    ToolRouter,
    ToolExecutionError,
    RoutingError,
)


class TestPydanticValidation:
    def test_read_file_args_valid(self) -> None:
        args = ReadFileArgs(path="foo.py")
        assert args.path == "foo.py"

    def test_read_file_args_missing_path(self) -> None:
        with pytest.raises(ValueError):
            ReadFileArgs()

    def test_write_file_args_valid(self) -> None:
        args = WriteFileArgs(path="out.txt", content="hello")
        assert args.path == "out.txt"
        assert args.content == "hello"

    def test_write_file_args_missing_fields(self) -> None:
        with pytest.raises(ValueError):
            WriteFileArgs(path="x")

    def test_search_files_args_valid(self) -> None:
        args = SearchFilesArgs(query="pattern")
        assert args.query == "pattern"
        assert args.file_pattern is None

    def test_search_files_args_with_file_pattern(self) -> None:
        args = SearchFilesArgs(query="def", file_pattern="*.py")
        assert args.file_pattern == "*.py"

    def test_search_files_args_missing_query(self) -> None:
        with pytest.raises(ValueError):
            SearchFilesArgs()

    def test_shell_command_args_valid(self) -> None:
        args = ShellCommandArgs(command="ls -la")
        assert args.command == "ls -la"

    def test_shell_command_args_missing(self) -> None:
        with pytest.raises(ValueError):
            ShellCommandArgs()


class TestShellCommandHandler:
    def test_allowed_command(self) -> None:
        handler = ShellCommandHandler()
        result = handler.execute(ShellCommandArgs(command="echo hello"))
        assert isinstance(result, dict)
        assert result["returncode"] == 0

    def test_disallowed_command(self) -> None:
        handler = ShellCommandHandler()
        with pytest.raises(ToolExecutionError):
            handler.execute(ShellCommandArgs(command="rm -rf /"))


class TestToolRouter:
    @pytest.fixture
    def router(self) -> ToolRouter:
        r = ToolRouter()
        return r

    def test_router_init_with_defaults(self, router: ToolRouter) -> None:
        assert len(router._tools) == 4
        assert "read_file" in router._tools
        assert "write_file" in router._tools
        assert "search_files" in router._tools
        assert "run_command" in router._tools

    def test_run_command_handler_registered(self, router: ToolRouter) -> None:
        assert "run_command" in router._handlers

    def test_get_schemas(self, router: ToolRouter) -> None:
        schemas = router.get_schemas()
        assert len(schemas) == 4
        for s in schemas:
            assert "name" in s
            assert "description" in s
            assert "parameters" in s

    def test_parse_explicit_command(self, router: ToolRouter) -> None:
        tool_name, args = router.parse_explicit_command(
            "/tool:read_file path=agent.py"
        )
        assert tool_name == "read_file"
        assert isinstance(args, ReadFileArgs)
        assert args.path == "agent.py"

    def test_parse_explicit_command_invalid_format(self, router: ToolRouter) -> None:
        with pytest.raises(RoutingError):
            router.parse_explicit_command("read_file agent.py")

    def test_parse_openai_function_call(self, router: ToolRouter) -> None:
        tool_name, args = router.parse_openai_function_call({
            "function": {
                "name": "read_file",
                "arguments": '{"path": "agent.py"}',
            }
        })
        assert tool_name == "read_file"
        assert isinstance(args, ReadFileArgs)
        assert args.path == "agent.py"

    def test_parse_openai_missing_function_key(self, router: ToolRouter) -> None:
        with pytest.raises(RoutingError):
            router.parse_openai_function_call({})

    def test_register_new_tool(self, router: ToolRouter) -> None:
        from tool_router import _VALIDATION_REGISTRY as vr
        vr["custom_tool"] = ReadFileArgs

        td = ToolDefinition(
            name="custom_tool",
            description="a custom tool",
            parameters_schema={"type": "object"},
        )
        router.register_tool(td)
        assert "custom_tool" in router._tools

    def test_execute_unregistered_handler(self, router: ToolRouter) -> None:
        router._handlers.pop("run_command", None)
        with pytest.raises(ToolExecutionError):
            router.execute("run_command", ShellCommandArgs(command="echo"))

    def test_natural_language_read(self, router: ToolRouter) -> None:
        tool_name, args = router.parse_natural_language(
            "read the file agent.py"
        )
        assert tool_name == "read_file"
        assert isinstance(args, ReadFileArgs)
        assert args.path == "agent.py"

    def test_natural_language_no_match(self, router: ToolRouter) -> None:
        with pytest.raises(RoutingError):
            router.parse_natural_language("do something unknown")

    def test_route_and_execute_run_command(self, router: ToolRouter) -> None:
        tool_name, args, result = router.route_and_execute(
            "/tool:run_command command=echo test"
        )
        assert tool_name == "run_command"
        assert isinstance(args, ShellCommandArgs)
        assert isinstance(result, dict)
        assert result["returncode"] == 0

    def test_router_rejects_unknown_tool(self, router: ToolRouter) -> None:
        with pytest.raises(KeyError):
            router._validate_args("nonexistent", {"x": 1})
