"""paste_image command — paste an image into the agent for vision-capable LLMs.

The image is read either from the system clipboard (Windows: an image copied
from a screenshot tool / browser / image viewer) or from a file path, encoded
as a base64 data URL, and handed to :meth:`Agent.chat_nlp` as a multimodal
user message.  Vision-capable models (e.g. Qwen-VL, Gemma-3, Pixtral, ...)
can then describe, transcribe, or reason about the image.
"""
import base64
import io
import os as _os
from typing import TYPE_CHECKING, Any

from .base import Command

if TYPE_CHECKING:
    from agent import Agent

#: File-extension → MIME type for image files loaded from disk.
_IMAGE_EXT_TYPES: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def encode_image_file(path: str) -> tuple[str, str]:
    """Read an image *file* and return ``(data_url, mime)``.

    Raises on read/decode failure so the caller can surface a clear error.
    """
    ext = _os.path.splitext(path)[1].lower()
    mime = _IMAGE_EXT_TYPES.get(ext, "image/png")
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}", mime


def encode_clipboard_image() -> tuple[str, str] | None:
    """Grab an image from the system clipboard.

    Returns ``(data_url, mime)`` when an image is available, otherwise ``None``.
    Uses Pillow's ``ImageGrab.grabclipboard()`` (works on Windows; on other
    platforms it may return ``None`` if no image is on the clipboard).
    """
    try:
        from PIL import Image, ImageGrab
    except Exception:
        return None
    try:
        img = ImageGrab.grabclipboard()
    except Exception:
        return None
    if img is None or not hasattr(img, "save"):
        # No image on the clipboard, or a list of file paths (not handled here).
        return None
    buf = io.BytesIO()
    try:
        img.save(buf, format="PNG")
    except Exception:
        return None
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}", "image/png"


class PasteImageCommand(Command):
    @property
    def name(self) -> str:
        return "paste_image"

    @property
    def help_text(self) -> str:
        return (
            "paste_image [path] [--prompt \"text\"] - Paste an image (clipboard "
            "or file) into the chat for vision-capable LLMs"
        )

    async def execute(self, args: list[str], agent: 'Agent') -> bool:
        parsed = list(args)
        prompt: str | None = None
        if "--prompt" in parsed:
            idx = parsed.index("--prompt")
            if idx + 1 < len(parsed):
                prompt = parsed[idx + 1]
                del parsed[idx:idx + 2]
            else:
                self.error("--prompt requires a quoted text argument")
                return True
        path = parsed[0] if parsed else None
        if path:
            original_path = path
            # Recursively strip surrounding quotes (handles ""path"" or 'path')
            while path and (path.startswith('"') or path.startswith("'")):
                path = path[1:-1].strip()
            
            abs_path = _os.path.abspath(path)
            print(f"  [debug] Path original: {original_path}")
            print(f"  [debug] Path stripped: {path}")
            print(f"  [debug] Absolute path: {abs_path}")

            if not _os.path.isfile(abs_path):
                self.error(f"Image file not found at absolute path: {abs_path}")
                return True
            try:
                data_url, mime = encode_image_file(path)
            except Exception as e:  # noqa: BLE001 - surface any read error clearly
                self.error(f"Could not read image: {e}")
                return True
            src = f"file {path}"
        else:
            grabbed = encode_clipboard_image()
            if grabbed is None:
                self.error(
                    "No image on the clipboard. Copy an image first, or pass a "
                    "file path: paste_image <path> [--prompt \"...\"]"
                )
                return True
            data_url, mime = grabbed
            src = "clipboard"

        size_kb = len(data_url) // 1024
        print(f"  [image] loaded from {src} ({mime}, ~{size_kb} KB base64)")

        await agent.chat_nlp(prompt or "", images=[data_url])
        return True
