from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Final, Optional


_PREF_FILENAME: Final[str] = "agent_llm_prefs.json"


def _pref_path(workspace: pathlib.Path) -> pathlib.Path:
    return workspace / ".workspace" / _PREF_FILENAME


class WorkspacePrefsError(Exception):
    pass


def get_pref(workspace: pathlib.Path, key: str) -> Optional[Any]:
    if not isinstance(workspace, pathlib.Path):
        raise WorkspacePrefsError("workspace must be a pathlib.Path")

    pref_file = _pref_path(workspace)
    if not pref_file.exists():
        return None

    try:
        with pref_file.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise WorkspacePrefsError(f"Failed to read prefs file: {exc}") from exc

    return data.get(key)


def set_pref(workspace: pathlib.Path, key: str, value: Any) -> None:
    if not isinstance(workspace, pathlib.Path):
        raise WorkspacePrefsError("workspace must be a pathlib.Path")

    pref_file = _pref_path(workspace)
    workspace_dir = pref_file.parent

    try:
        workspace_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkspacePrefsError(f"Failed to create .workspace directory: {exc}") from exc

    if pref_file.exists():
        try:
            with pref_file.open("r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise WorkspacePrefsError(f"Failed to read existing prefs file: {exc}") from exc
    else:
        data = {}

    data[key] = value

    try:
        with pref_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError as exc:
        raise WorkspacePrefsError(f"Failed to write prefs file: {exc}") from exc