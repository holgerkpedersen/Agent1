"""Strict UTF-8 detection for text files this project reads or writes.

The project invariant (see AGENTS.md, "Text encoding") is that every text
file is UTF-8 without BOM. Reading with ``errors="replace"`` hides a
violation: the bad bytes become U+FFFD and a later write-back silently
rewrites the file. This module makes a non-UTF-8 file an explicit,
reported condition instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class NonUtf8TextError(ValueError):
    """Raised when a file that must be UTF-8 text is not valid UTF-8."""

    def __init__(self, path: str, line: int, column: int, byte_value: int) -> None:
        self.path = path
        self.line = line
        self.column = column
        self.byte_value = byte_value
        super().__init__(
            f"{path} is not valid UTF-8: byte 0x{byte_value:02X} at "
            f"line {line}, column {column}. Re-save the file as UTF-8 "
            "(not cp1252/latin-1) before editing it."
        )


@dataclass(frozen=True)
class Utf8Check:
    """Result of a strict UTF-8 check of one file."""

    path: str
    ok: bool
    line: int = 0
    column: int = 0
    byte_value: int = 0
    message: str = ""


def check_utf8_bytes(data: bytes, path: str = "<bytes>") -> Utf8Check:
    """Strictly validate *data* as UTF-8; locate the first invalid byte."""
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        prefix = data[: exc.start]
        line = prefix.count(b"\n") + 1
        column = exc.start - (prefix.rfind(b"\n") + 1) + 1
        byte_value = data[exc.start]
        return Utf8Check(
            path=path,
            ok=False,
            line=line,
            column=column,
            byte_value=byte_value,
            message=NonUtf8TextError(path, line, column, byte_value).args[0],
        )
    return Utf8Check(path=path, ok=True)


def check_utf8_file(path: str | Path) -> Utf8Check:
    """Strictly validate one file as UTF-8. Raises OSError if unreadable."""
    p = Path(path)
    return check_utf8_bytes(p.read_bytes(), path=str(p))


#: Legacy single-byte encoding assumed for non-UTF-8 text (Windows default).
LEGACY_ENCODING = "cp1252"


class UnrepresentableTextError(ValueError):
    """Raised when new text cannot be encoded in a file's existing encoding."""


def read_text_preserving(path: str | Path) -> tuple[str, str]:
    """Read *path* and return ``(text, encoding)`` in the file's own encoding.

    A valid UTF-8 file is returned as UTF-8. A legacy file (not valid UTF-8)
    is decoded as cp1252 and returned with that encoding, so the caller can
    write it back in the same encoding. Raises NonUtf8TextError if the bytes
    are not decodable as cp1252 either, or if the decoded text would not
    re-encode to the identical bytes (the file cannot be round-tripped).
    """
    p = Path(path)
    data = p.read_bytes()
    try:
        text, encoding = data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        try:
            text, encoding = data.decode(LEGACY_ENCODING), LEGACY_ENCODING
        except UnicodeDecodeError as exc:
            prefix = data[: exc.start]
            line = prefix.count(b"\n") + 1
            column = exc.start - (prefix.rfind(b"\n") + 1) + 1
            raise NonUtf8TextError(str(p), line, column, data[exc.start]) from exc
    # Universal newlines, as text-mode open() does: CRLF -> LF. The writer
    # re-applies the platform newline convention, so this round-trips.
    return text.replace("\r\n", "\n"), encoding


def encode_in_encoding(text: str, encoding: str, path: str = "<text>") -> bytes:
    """Encode *text* in *encoding*; refuse characters it cannot represent."""
    try:
        return text.encode(encoding)
    except UnicodeEncodeError as exc:
        bad = text[exc.start]
        raise UnrepresentableTextError(
            f"{path}: character {bad!r} (U+{ord(bad):04X}) cannot be written "
            f"in the file's encoding {encoding}; the file was left unchanged."
        ) from exc


def read_text_strict(path: str | Path) -> str:
    """Read *path* as UTF-8; raise NonUtf8TextError if it is not valid UTF-8.

    Never substitutes replacement characters, so a caller that writes the
    text back cannot corrupt the original bytes.
    """
    p = Path(path)
    data = p.read_bytes()
    result = check_utf8_bytes(data, path=str(p))
    if not result.ok:
        raise NonUtf8TextError(str(p), result.line, result.column, result.byte_value)
    return data.decode("utf-8")
