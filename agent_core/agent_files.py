"""File tool wrappers extracted from Agent (refactor step r02).

Each former ``Agent._tool_*`` method is a plain module-level function that
takes the owning :class:`Agent` instance as its first parameter.  Agent
methods that still live in ``agent.py`` (e.g. ``_resolve_nlp_path``,
``read_file``) are forwarded via ``getattr(agent, name)`` at call time so
monkeypatching and subclass overrides keep working.
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from agent_core.path_utils import resolve_path, safe_path

if TYPE_CHECKING:  # pragma: no cover - typing only
    from agent import Agent

logger = logging.getLogger(__name__)


async def _tool_read_file(agent: "Agent", path: str, **kwargs: Any) -> str:
    result = await agent.fs.read(path)
    if not result.startswith('File not found') and (not result.startswith('Error')):
        safe = agent.fs.safe_path(path)
        agent._files_read.add(safe)
        try:
            agent._file_mtimes[safe] = os.path.getmtime(safe)
        except OSError as e:
            logger.warning('Could not stat %s: %s', safe, e)
    return result


async def _tool_write_file(agent: "Agent", path: str, content: str, **kwargs: Any) -> str:
    return await agent.fs.write(path, content)


async def _tool_apply_patch(agent: "Agent", path: str, find: str, replace: str, **kwargs: Any) -> str:
    return await agent.fs.apply_patch(path, find, replace)


async def _tool_edit_file(agent: "Agent", path: str, content: str, **kwargs: Any) -> str:
    return await agent.fs.edit(path, content)


async def _tool_search(agent: "Agent", query: str, path: str = '.', **kwargs: Any) -> str:
    search_path = getattr(agent, '_resolve_nlp_path')(path or '.')
    return await agent.searcher.search(query, search_path)


async def _tool_list_files(agent: "Agent", path: str = '.', pattern: str = '*', **kwargs: Any) -> str:
    return await agent.fs.list_files(path, pattern)


async def _tool_delete_file(agent: "Agent", path: str, **kwargs: Any) -> str:
    return await agent.fs.delete(path)


async def _tool_analyze_file(agent: "Agent", path: str, **kwargs: Any) -> str:
    return await agent.llm.analyze_code(await getattr(agent, 'read_file')(path))


async def _tool_llm_analyze(agent: "Agent", path: str, **kwargs: Any) -> str:
    file_content = await getattr(agent, 'read_file')(path, track_read=False)
    if file_content.startswith('File not found:') or file_content.startswith('Error reading file:'):
        return f'Could not analyze: {file_content}'
    return await agent.llm.analyze_code(file_content)


async def execute_tool(agent: "Agent", tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name using the dispatcher."""
    return await agent.dispatcher.execute(tool_name, arguments)


def _normalize_path(agent: "Agent", path: str) -> str:
    """Normalize and validate paths with security checks."""
    return resolve_path(path)


def _safe_path(agent: "Agent", path: str) -> str:
    """Validate and normalize path in one step."""
    return safe_path(path)
