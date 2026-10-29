"""Normalize provider replies into assistant messages with tool calls.

Providers return one of three shapes:

1. plain text — the model's answer;
2. a JSON envelope carrying ``tool_calls`` — the provider/server parsed a
   native tool call;
3. the model's native tool-call XML as TEXT — the serving stack had no
   matching parser, or the model used a name the request did not declare.

:func:`parse_tool_reply` turns all three into the ``(text, updated_messages)``
contract every tool loop uses.  Shape 3 is the live failure this module was
written for: Qwen3.8-Flash-Next served through Strata emitted

    <tool_call>
    <function=grep>
    <parameter=pattern>
    class Command
    </parameter>
    </function>
    </tool_call>

as CONTENT because ``grep`` was not a declared tool; the REPL printed the
block as the final answer instead of running it.  :func:`recover_text_tool_calls`
maps the common cross-harness tool names (``grep`` -> ``search``, ...) and
their argument names so the recovered call is executable.

Only complete, unambiguous blocks are recovered; malformed or unrelated text
is returned untouched.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

#: ``<tool_call>...</tool_call>`` wrapper (attributes tolerated).
_WRAPPED_TOOL_CALL_RE = re.compile(
    r"<tool_call\b[^>]*>(?P<body>.*?)</tool_call>",
    re.IGNORECASE | re.DOTALL,
)

#: Qwen3-Coder function block: ``<function=name>...</function>``.
_FUNCTION_RE = re.compile(
    r"<function\s*=\s*(?P<name>[A-Za-z_][\w.\-]*)\s*>(?P<body>.*?)</function>",
    re.IGNORECASE | re.DOTALL,
)

#: Qwen3-Coder parameter block: ``<parameter=key>value</parameter>``.
_PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*(?P<name>[A-Za-z_][\w.\-]*)\s*>(?P<value>.*?)</parameter>",
    re.IGNORECASE | re.DOTALL,
)

#: Tool names other agent harnesses (opencode, Hermes, ...) teach models to
#: use, mapped onto this agent's equivalents.  Without the mapping the
#: recovered call dies at dispatch with "Unknown tool" even though the intent
#: was perfectly clear (the live failure asked for ``grep``).
_TOOL_ALIASES: dict[str, str] = {
    "bash": "run",
    "shell": "run",
    "run_command": "run",
    "grep": "search",
    "list": "list_files",
    "ls": "list_files",
    "read_file": "read",
    "write_file": "write",
    "edit_file": "edit",
}


def _build_param_types() -> dict[str, dict[str, str]]:
    """``{tool: {parameter: json-schema type}}`` from the advertised schemas.

    Reading the types from the ONE schema source keeps argument coercion in
    sync with the tools instead of maintaining a second hand-written list.
    """
    types: dict[str, dict[str, str]] = {}
    for schema in NLP_TOOL_SCHEMAS:
        function = schema.get("function") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        parameters = function.get("parameters") or {}
        properties = parameters.get("properties") or {}
        types[name] = {
            str(key): str((spec or {}).get("type") or "string")
            for key, spec in properties.items()
        }
    return types


_PARAM_TYPES = _build_param_types()


def _canonical_name(name: str) -> str:
    return _TOOL_ALIASES.get(name.lower(), name)


def _coerce_value(tool_name: str, parameter: str, raw: str) -> Any:
    """Convert one parameter's text using the tool's declared JSON type.

    Text-heavy parameters (``content``, ``old_text``, ...) are strings in the
    schema, so values that merely LOOK like JSON/bools/numbers stay verbatim —
    a write must never turn ``"true"`` into ``True`` or reformat JSON.
    """
    text = raw.strip()
    param_type = _PARAM_TYPES.get(_canonical_name(tool_name), {}).get(
        parameter, "string"
    )
    if param_type == "integer":
        try:
            return int(text)
        except ValueError:
            return text
    if param_type == "number":
        try:
            return float(text)
        except ValueError:
            return text
    if param_type == "boolean":
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        return text
    if param_type in ("array", "object"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _parameter_args(tool_name: str, body: str) -> dict[str, Any]:
    """Parse every ``<parameter=...>`` block in *body*."""
    args: dict[str, Any] = {}
    for match in _PARAMETER_RE.finditer(body):
        parameter = match.group("name")
        args[parameter] = _coerce_value(tool_name, parameter, match.group("value"))
    return args


def _function_args(tool_name: str, body: str) -> dict[str, Any] | None:
    """Arguments of one ``<function>`` body: parameter tags or a JSON object."""
    if _PARAMETER_RE.search(body):
        return _parameter_args(tool_name, body)
    try:
        parsed = json.loads(body.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _json_call_args(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Hermes-style JSON call(s) inside a ``<tool_call>`` wrapper."""
    try:
        parsed = json.loads(body.strip())
    except (json.JSONDecodeError, TypeError):
        return []
    items = parsed if isinstance(parsed, list) else [parsed]
    calls: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        function = function if isinstance(function, dict) else {}
        name = item.get("name") or function.get("name")
        raw_args = item.get("arguments")
        if raw_args is None:
            raw_args = function.get("arguments")
        if raw_args is None:
            raw_args = item.get("parameters")
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                raw_args = {}
        if not name or not isinstance(raw_args, dict):
            continue
        calls.append((str(name), raw_args))
    return calls


def _normalize_args(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Map a foreign tool's argument names onto this agent's tool schema."""
    lowered = name.lower()
    if lowered == "grep":
        query = args.get("query")
        if query is None:
            query = args.get("pattern")
        if query is None:
            query = args.get("regex")
        out: dict[str, Any] = {"query": "" if query is None else str(query)}
        if args.get("path"):
            out["path"] = str(args["path"])
        return out
    if lowered in ("bash", "shell", "run_command"):
        out = dict(args)
        if "command" not in out:
            for key in ("cmd", "script"):
                if key in out:
                    out["command"] = out.pop(key)
                    break
        return out
    if lowered in ("read_file", "write_file", "edit_file"):
        out = dict(args)
        if "path" not in out:
            for key in ("file", "filename", "file_path"):
                if key in out:
                    out["path"] = out.pop(key)
                    break
        return out
    return dict(args)


def _build_call(name: str, args: dict[str, Any], index: int) -> dict[str, Any]:
    """One OpenAI-format tool call with a stable, loop-unique id."""
    return {
        "id": f"call_text_{index}",
        "type": "function",
        "function": {
            "name": _canonical_name(name),
            "arguments": json.dumps(
                _normalize_args(name, args), ensure_ascii=False
            ),
        },
    }


def _remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Return *text* with the given character spans removed."""
    pieces: list[str] = []
    last = 0
    for start, end in sorted(spans):
        pieces.append(text[last:start])
        last = end
    pieces.append(text[last:])
    cleaned = "".join(pieces)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


def recover_text_tool_calls(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Extract text-form tool calls from *text*.

    Returns ``(remaining_text, tool_calls)``.  ``remaining_text`` is *text*
    with every recovered call block removed (surrounding prose is kept), and
    ``tool_calls`` is in OpenAI format.  When nothing is recoverable the
    original text is returned with an empty list — never a partial rewrite.
    """
    if not text or "<" not in text:
        return text, []

    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []

    for wrapper in _WRAPPED_TOOL_CALL_RE.finditer(text):
        body = wrapper.group("body")
        parsed: list[tuple[str, dict[str, Any]]] = []
        for function in _FUNCTION_RE.finditer(body):
            name = function.group("name")
            args = _function_args(name, function.group("body"))
            if args is not None:
                parsed.append((name, args))
        if not parsed:
            parsed = _json_call_args(body)
        if not parsed:
            # Unreadable wrapper: keep it as text rather than deleting it.
            continue
        spans.append(wrapper.span())
        for name, args in parsed:
            calls.append(_build_call(name, args, len(calls)))

    if not calls:
        # Bare ``<function=...>`` blocks (no ``<tool_call>`` wrapper).  Require
        # at least one parameter tag so prose *about* the syntax is not run.
        for function in _FUNCTION_RE.finditer(text):
            body = function.group("body")
            if not _PARAMETER_RE.search(body):
                continue
            name = function.group("name")
            args = _function_args(name, body)
            if args is None:
                continue
            spans.append(function.span())
            calls.append(_build_call(name, args, len(calls)))

    if not calls:
        return text, []
    return _remove_spans(text, spans), calls


def parse_tool_reply(
    raw: str,
    messages: list[dict[str, Any]],
    *,
    repair_arguments: Callable[[str], dict[str, Any]] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Turn a provider reply into ``(text, messages + assistant message)``.

    Handles the JSON ``tool_calls`` envelope (the provider/server fast path)
    and falls back to recovering text-form calls.  *repair_arguments* is an
    optional hook that turns malformed JSON arguments into a dict (the agent's
    ``_repair_tool_arguments``); when omitted, malformed arguments are left
    untouched for the caller's provider to handle.
    """
    if raw.strip() == "(no output)":
        raw = ""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict) and parsed.get("tool_calls"):
        parsed.pop("role", None)
        for call in parsed["tool_calls"]:
            function = call.get("function") or {}
            arguments = function.get("arguments", "{}")
            try:
                json.loads(arguments)
            except (json.JSONDecodeError, TypeError):
                if repair_arguments is not None:
                    function["arguments"] = json.dumps(
                        repair_arguments(arguments), ensure_ascii=False
                    )
        updated = list(messages)
        updated.append(
            {"role": "assistant", "content": parsed.get("content") or "", **parsed}
        )
        return str(parsed.get("content") or ""), updated
    content_text = raw
    if isinstance(parsed, dict) and "content" in parsed:
        content_text = str(parsed.get("content") or "")
    recovered_text, text_calls = recover_text_tool_calls(content_text)
    if text_calls:
        updated = list(messages)
        updated.append(
            {
                "role": "assistant",
                "content": recovered_text,
                "tool_calls": text_calls,
            }
        )
        return recovered_text, updated
    updated = list(messages)
    updated.append({"role": "assistant", "content": raw})
    return raw, updated


__all__ = ["parse_tool_reply", "recover_text_tool_calls"]
