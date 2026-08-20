"""paste_image command + multimodal chat_history handling (decision: vision).

Regression coverage for:
- ``PasteImageCommand`` encoding a file / clipboard and dispatching to chat_nlp
- ``Agent.chat_nlp`` building an OpenAI-format image_url content block
- ``_strip_image_blocks`` removing base64 blobs from persisted history
"""
import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from agent import Agent, _strip_image_blocks
from agent_core.commands.paste_image_cmd import (
    PasteImageCommand,
    encode_clipboard_image,
    encode_image_file,
)


def _make_png(tmp_path: Path) -> Path:
    """Write a tiny valid PNG and return its path."""
    from PIL import Image

    p = tmp_path / "shot.png"
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(p, format="PNG")
    return p


def test_encode_image_file_returns_data_url():
    import io

    from PIL import Image

    p = _make_png(Path("."))  # not used; build bytes directly instead
    p.unlink(missing_ok=True)
    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=(1, 2, 3)).save(buf, format="PNG")
    data = buf.getvalue()
    path = Path("tests/_tmp_enc.png")
    path.write_bytes(data)
    try:
        url, mime = encode_image_file(str(path))
        assert mime == "image/png"
        assert url.startswith("data:image/png;base64,")
        decoded = base64.b64decode(url.split(",", 1)[1])
        assert decoded == data
    finally:
        path.unlink(missing_ok=True)


def test_encode_image_file_reports_missing_file():
    with pytest.raises(FileNotFoundError):
        encode_image_file("tests/_does_not_exist_12345.png")


def test_strip_image_blocks_drops_pure_image_messages():
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
        ]},
        {"role": "assistant", "content": "ok"},
    ]
    stripped = _strip_image_blocks(messages)
    assert len(stripped) == 3
    # system preserved
    assert stripped[0]["role"] == "system"
    # mixed message keeps text, loses image block
    assert stripped[1]["content"] == [{"type": "text", "text": "look"}]
    # assistant preserved
    assert stripped[2]["role"] == "assistant"


def test_strip_image_blocks_leaves_plain_messages_untouched():
    messages = [
        {"role": "user", "content": "plain text"},
        {"role": "assistant", "content": "reply"},
    ]
    assert _strip_image_blocks(messages) == messages


def test_chat_nlp_builds_multimodal_user_message():
    """chat_nlp must attach image_url content blocks when images are passed."""
    agent = Agent(workspace=".")
    captured: dict = {}

    async def fake_llm_chat(messages, tools=None, **kwargs):
        captured["messages"] = messages
        return "image answer"

    agent.llm.chat = fake_llm_chat  # type: ignore[assignment]

    async def fake_run(self, messages, **kwargs):
        # Mirror the real ToolLoopRunner contract: return (text, messages).
        return "image answer", list(messages)

    with patch(
        "agent_core.llm.tool_loop.ToolLoopRunner.run",
        new=fake_run,
    ):
        import asyncio

        asyncio.run(agent.chat_nlp("describe", images=["data:image/png;base64,ZZZ"]))

    # The user turn we appended carries a multimodal content array.
    user_msgs = [m for m in agent._chat_history if m.get("role") == "user"]
    assert user_msgs
    last = user_msgs[-1]["content"]
    assert isinstance(last, list)
    types = [b.get("type") for b in last]
    assert "text" in types
    assert types.count("image_url") == 1
    assert last[-1]["image_url"]["url"] == "data:image/png;base64,ZZZ"


def test_paste_image_command_reads_file_and_dispatches():
    """paste_image <path> encodes the file and calls chat_nlp with images."""
    agent = Agent(workspace=".")
    img_path = _make_png(Path("tests"))
    sent: dict = {}

    async def fake_chat_nlp(user_input, images=None):
        sent["user_input"] = user_input
        sent["images"] = images

    with patch.object(agent, "chat_nlp", side_effect=fake_chat_nlp):
        import asyncio

        ok = asyncio.run(PasteImageCommand().execute([str(img_path)], agent))
    assert ok is True
    assert sent["images"] and sent["images"][0].startswith("data:image/png;base64,")
    img_path.unlink(missing_ok=True)


def test_paste_image_command_clipboard_fallback():
    """With no path arg, the command grabs the clipboard image."""
    agent = Agent(workspace=".")
    sent: dict = {}

    async def fake_chat_nlp(user_input, images=None):
        sent["images"] = images

    fake_url = "data:image/png;base64,CLIP"
    with patch.object(agent, "chat_nlp", side_effect=fake_chat_nlp), patch(
        "agent_core.commands.paste_image_cmd.encode_clipboard_image",
        return_value=(fake_url, "image/png"),
    ):
        import asyncio

        ok = asyncio.run(PasteImageCommand().execute([], agent))
    assert ok is True
    assert sent["images"] == [fake_url]


def test_paste_image_command_reports_empty_clipboard():
    """No clipboard image and no path → clear error, no dispatch."""
    agent = Agent(workspace=".")
    dispatched = {"called": False}

    async def fake_chat_nlp(user_input, images=None):
        dispatched["called"] = True

    with patch.object(agent, "chat_nlp", side_effect=fake_chat_nlp), patch(
        "agent_core.commands.paste_image_cmd.encode_clipboard_image",
        return_value=None,
    ):
        import asyncio

        ok = asyncio.run(PasteImageCommand().execute([], agent))
    assert ok is True
    assert dispatched["called"] is False
