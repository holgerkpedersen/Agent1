"""Optional rich-based markdown preview for LLM output.

The REPL's plain ``print`` path leaves markdown source (``# headings``,
``**bold**``, fenced code blocks, tables) as raw text.  When the ``rich``
package is available we render the same text through ``rich.markdown.Markdown``
so headings, code, lists and tables are previewed in the terminal.  When rich
is missing we fall back to the legacy plain-text print so the agent keeps
working on minimal installs.
"""
from __future__ import annotations

from typing import Optional

try:  # pragma: no cover - exercised only when rich is installed
    import rich
    from rich.console import Console
    from rich.markdown import Markdown

    HAS_RICH = True
except Exception:  # pragma: no cover - rich optional
    rich = None  # type: ignore[assignment]
    Console = None  # type: ignore[assignment, misc]
    Markdown = None  # type: ignore[assignment, misc]
    HAS_RICH = False


def render_markdown(text: str, *, style: Optional[str] = None) -> None:
    """Print *text* rendered as markdown when rich is available.

    Falls back to a plain ``print`` otherwise.  *style* is an optional rich
    style applied to the whole block (used to keep the green success tint).
    Never raises: any rendering failure degrades to plain text.
    """
    if not text:
        return
    if not HAS_RICH:
        print(text)
        return
    try:
        console = Console()
        md = Markdown(text)
        if style:
            console.print(md, style=style)
        else:
            console.print(md)
    except Exception:
        print(text)


__all__ = ["HAS_RICH", "render_markdown"]
