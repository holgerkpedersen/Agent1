"""JSON I/O helpers with quarantine on corruption.

Extracted from agent.py as the t00 JSON concern (step t01).
Provides safe JSON read/write with backup-on-corruption.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _read_json_quarantining(
    path: str | os.PathLike[str],
    label: str = "JSON data",
    default: Any = None,
) -> Any:
    """Read JSON from path, quarantining (moving aside) corrupted files.

    On success, returns the parsed Python object.
    On failure, logs a warning, quarantines the bad file, and returns *default*.
    """
    path = os.fspath(path)
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return default if default is not None else {}
    except json.JSONDecodeError as exc:
        logger.warning("JSON parse error in %s (%s): %s — quarantining", path, label, exc)
        _quarantine(path)
        return default if default is not None else {}
    except Exception as exc:
        logger.warning("Could not read %s (%s): %s", path, label, exc)
        return default if default is not None else {}


def _quarantine(path: str | os.PathLike[str]) -> None:
    """Move a corrupted file aside with a timestamp suffix."""
    path = os.fspath(path)
    if not os.path.exists(path):
        return
    base, ext = os.path.splitext(path)
    import time

    suffix = f".corrupt.{int(time.time() * 1000)}{ext}"
    try:
        os.rename(path, base + suffix)
    except Exception:
        pass
