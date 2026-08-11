"""File context retriever — extracts file content snippets for /file and @file keywords."""


import re
from pathlib import Path

from agent_core.security.path_utils import normalize_path, SecurityViolationError


class FileContextRetriever:
    """Retrieves file content snippets referenced via ``/file`` or ``@file`` in messages.

    The retriever scans a message for ``/filename`` or ``@filename`` patterns and
    attempts to locate matching files relative to the working directory.
    """

    _KEYWORD_RE: re.Pattern[str] = re.compile(r"(?:/|@)(\S+\.\w+)")
    _FALLBACK_EXTENSIONS: tuple[str, ...] = (".py", ".txt", ".md", ".json", ".yaml", ".yml", ".toml")

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir).resolve() if base_dir is not None else Path.cwd().resolve()

    def retrieve(self, filename: str) -> str | None:
        """Return the text content of *filename* if it exists, else ``None``."""
        # Primary lookup with consolidated security normalization
        try:
            target = normalize_path(self._base_dir, filename)
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, SecurityViolationError):
            pass

        # Context-aware fallback 1: append common extensions when none present
        if not Path(filename).suffix:
            for ext in self._FALLBACK_EXTENSIONS:
                try:
                    target = normalize_path(self._base_dir, f"{filename}{ext}")
                    return target.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError, SecurityViolationError):
                    continue

        # Context-aware fallback 2: case-insensitive directory scan
        if self._base_dir.is_dir():
            for entry in self._base_dir.iterdir():
                if entry.name.lower() == filename.lower() and entry.is_file():
                    try:
                        return entry.read_text(encoding="utf-8")
                    except (OSError, UnicodeDecodeError):
                        break

        return None

    def extract_filenames(self, message: str) -> list[str]:
        """Return the filenames referenced in *message* via ``/file`` or ``@file``."""
        return self._KEYWORD_RE.findall(message)