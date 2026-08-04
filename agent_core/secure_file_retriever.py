"""Secure file retriever module with filename safety validation."""

from __future__ import annotations

import os
from pathlib import Path


class FilenameSafetyValidator:
    """Validates filenames to prevent directory traversal attacks and unsafe characters."""

    def _is_safe_filename(self, filename: str) -> bool:
        """Check if a filename is safe from directory traversal and unsafe characters.

        Args:
            filename: The filename string to validate.

        Returns:
            True if the filename is safe, False otherwise.
        """
        if not filename or len(filename) > 255:
            return False

        # Reject any path separators or traversal sequences
        forbidden_chars = ["/", "\\", "..", "~"]
        for char in forbidden_chars:
            if char in filename:
                return False

        # Reject null bytes and other unsafe characters
        if "\x00" in filename or "|" in filename or ";" in filename:
            return False

        # Ensure the basename matches (no path components)
        basename = os.path.basename(filename)
        if basename != filename:
            return False

        return True


class FileContextRetriever:
    """Retrieves file content context with secure filename validation and configurable limits."""

    def __init__(self, max_chars: int | None = 1000) -> None:
        """Initialize the retriever with an optional character limit.

        Args:
            max_chars: Maximum number of characters to read from a file. None means no limit.
        """
        self._validator = FilenameSafetyValidator()
        self._max_chars: int | None = max_chars if (max_chars is not None and max_chars >= 0) else None

    def retrieve_context(self, filename: str, base_dir: str | Path) -> str:
        """Retrieve file content context securely.

        Args:
            filename: The filename to read (must be safe).
            base_dir: The base directory within which the file must reside.

        Returns:
            The file content as a string, truncated if max_chars is set.

        Raises:
            ValueError: If the filename fails safety validation or escapes base_dir.
            OSError: On encoding/IO errors during file reading.
        """
        if not self._validator._is_safe_filename(filename):
            raise ValueError(f"Unsafe filename detected: {filename!r}")

        base_path = Path(base_dir).resolve()
        target_path = (base_path / filename).resolve()

        # Ensure the resolved path stays within base_dir
        try:
            target_path.relative_to(base_path)
        except ValueError as exc:
            raise ValueError(
                f"Filename escapes base directory: {filename!r}"
            ) from exc

        if not target_path.is_file():
            raise OSError(f"File does not exist or is not a regular file: {target_path}")

        try:
            content = target_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise OSError(f"Encoding error reading file: {exc.reason}") from exc
        except IOError as exc:
            raise OSError(f"I/O error reading file: {exc.strerror or str(exc)}") from exc

        if self._max_chars is not None and len(content) > self._max_chars:
            content = content[:self._max_chars]

        return content