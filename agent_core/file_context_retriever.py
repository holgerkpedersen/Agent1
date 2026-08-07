"""File context retriever — extracts file content snippets for /file and @file keywords."""

from __future__ import annotations

import re
from pathlib import Path


class FileContextRetriever:
    """Retrieves file content snippets referenced via ``/file`` or ``@file`` in messages.

    The retriever scans a message for ``/filename`` or ``@filename`` patterns and
    attempts to locate matching files relative to the working directory.
    """

    _KEYWORD_RE: re.Pattern[str] = re.compile(r"(?:/|@)(\S+\.\w+)")

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()

    def retrieve(self, filename: str) -> str | None:
        """Return the text content of *filename* if it exists, else ``None``."""
        target = self._base_dir / filename
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def extract_filenames(self, message: str) -> list[str]:
        """Return the filenames referenced in *message* via ``/file`` or ``@file``."""
        return self._KEYWORD_RE.findall(message)
