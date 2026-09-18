"""Unit tests for agent_core.hue.bridge (HueBridge V1 API client).

All HTTP calls are mocked — no real Bridge required.
"""

from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import agent_core.hue.bridge as hue_bridge_mod
from agent_core.hue.bridge import HueBridge, HueBridgeError


@pytest.fixture(autouse=True)
def _no_real_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate from_env tests from the developer's real .env on disk."""
    monkeypatch.setattr(hue_bridge_mod, "_read_env_file", lambda: {})


# ── anyio backend fixture (required by project test convention) ──────────


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _mock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the two required env vars."""
    monkeypatch.setenv("HUE_BRIDGE_IP", "192.168.1.217")
    monkeypatch.setenv("HUE_API_KEY", "test-api-key-abc123")


# ── from_env ─────────────────────────────────────────────────────────────


def test_from_env_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    assert bridge._base_ip == "192.168.1.217"
    assert bridge._api_key == "test-api-key-abc123"
    # V1 API: username is embedded in the URL path
    assert bridge._base_url == "http://192.168.1.217/api/test-api-key-abc123"


def test_from_env_missing_ip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HUE_BRIDGE_IP", raising=False)
    monkeypatch.setenv("HUE_API_KEY", "key")
    with pytest.raises(HueBridgeError, match="HUE_BRIDGE_IP is not set"):
        HueBridge.from_env()


def test_from_env_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HUE_BRIDGE_IP", "192.168.1.1")
    monkeypatch.delenv("HUE_API_KEY", raising=False)
    with pytest.raises(HueBridgeError, match="HUE_API_KEY is not set"):
        HueBridge.from_env()


def test_from_env_falls_back_to_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: values present only in .env must be picked up (no export needed)."""
    monkeypatch.delenv("HUE_BRIDGE_IP", raising=False)
    monkeypatch.delenv("HUE_API_KEY", raising=False)
    monkeypatch.setattr(
        hue_bridge_mod, "_read_env_file",
        lambda: {"HUE_BRIDGE_IP": "10.0.0.5", "HUE_API_KEY": "filekey"},
    )
    bridge = HueBridge.from_env()
    assert bridge._base_ip == "10.0.0.5"
    assert bridge._api_key == "filekey"


def test_from_env_os_environ_wins_over_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit env vars override .env; missing ones still fall back per-var."""
    monkeypatch.setenv("HUE_BRIDGE_IP", "192.168.1.217")
    monkeypatch.delenv("HUE_API_KEY", raising=False)
    monkeypatch.setattr(
        hue_bridge_mod, "_read_env_file",
        lambda: {"HUE_BRIDGE_IP": "10.0.0.5", "HUE_API_KEY": "filekey"},
    )
    bridge = HueBridge.from_env()
    assert bridge._base_ip == "192.168.1.217"  # os.environ wins
    assert bridge._api_key == "filekey"        # missing key comes from file


def test_from_env_empty_file_degrades_to_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the .env fallback yields nothing, the original clear error is raised."""
    monkeypatch.delenv("HUE_BRIDGE_IP", raising=False)
    monkeypatch.setenv("HUE_API_KEY", "key")
    with pytest.raises(HueBridgeError, match="HUE_BRIDGE_IP is not set"):
        HueBridge.from_env()


# ── list_lights ──────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_list_lights_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()

    # V1 API returns a dict keyed by light ID
    bridge._get = AsyncMock(return_value={  # type: ignore[attr-defined]
        "1": {
            "name": "Lampe 1",
            "type": "Extended color light",
            "state": {"on": True, "bri": 203, "ct": 366},
        },
        "3": {
            "name": "Lampe 2",
            "type": "Extended color light",
            "state": {"on": False, "bri": 1, "ct": 230},
        },
    })

    lights = await bridge.list_lights()
    assert len(lights) == 2
    by_id = {l["id"]: l for l in lights}
    assert by_id["1"]["name"] == "Lampe 1"
    assert by_id["1"]["on"] is True
    # bri=203 → round(203/254*100, 1) = 79.9
    assert by_id["1"]["brightness"] == 79.9
    assert by_id["1"]["mirek"] == 366
    assert by_id["3"]["name"] == "Lampe 2"
    assert by_id["3"]["on"] is False


@pytest.mark.anyio
async def test_list_lights_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    bridge._get = AsyncMock(return_value={})  # type: ignore[attr-defined]
    lights = await bridge.list_lights()
    assert lights == []


@pytest.mark.anyio
async def test_list_lights_missing_state_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lights without bri/ct in state yield None brightness/mirek."""
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    bridge._get = AsyncMock(return_value={  # type: ignore[attr-defined]
        "5": {"name": "Minimal", "state": {"on": True}},
    })
    lights = await bridge.list_lights()
    assert len(lights) == 1
    assert lights[0]["brightness"] is None
    assert lights[0]["mirek"] is None


# ── get_light ────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_get_light_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    bridge._get = AsyncMock(return_value={  # type: ignore[attr-defined]
        "name": "Desk Lamp",
        "type": "Extended color light",
        "state": {
            "on": True,
            "bri": 114,
            "ct": 250,
            "xy": [0.4, 0.5],
        },
    })
    info = await bridge.get_light("3")
    assert info["id"] == "3"
    assert info["name"] == "Desk Lamp"
    assert info["on"] is True
    # bri=114 → round(114/254*100, 1) = 44.9
    assert info["brightness"] == 44.9
    assert info["mirek"] == 250
    assert info["color_xy"] == [0.4, 0.5]


@pytest.mark.anyio
async def test_get_light_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    # V1 API returns a list of error objects for missing lights
    bridge._get = AsyncMock(  # type: ignore[attr-defined]
        side_effect=HueBridgeError("Resource not found on Hue Bridge: /lights/99")
    )
    with pytest.raises(HueBridgeError, match="not found"):
        await bridge.get_light("99")


# ── set_light ────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_set_light_on_only(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    bridge._put = AsyncMock()  # type: ignore[attr-defined]
    await bridge.set_light("3", on=True)
    bridge._put.assert_awaited_once_with("/lights/3/state", {"on": True})


@pytest.mark.anyio
async def test_set_light_brightness_and_mirek(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    bridge._put = AsyncMock()  # type: ignore[attr-defined]
    await bridge.set_light("3", brightness=75, mirek=300)
    # bri = int(round(75/100*254)) = 190 (V1 uses 0-254 scale)
    bridge._put.assert_awaited_once_with("/lights/3/state", {"bri": 190, "ct": 300})


@pytest.mark.anyio
async def test_set_light_brightness_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    bridge._put = AsyncMock()  # type: ignore[attr-defined]

    await bridge.set_light("3", brightness=150)
    call_args = bridge._put.call_args
    assert call_args[0][1]["bri"] == 254  # clamped to max

    await bridge.set_light("3", brightness=-10)
    call_args2 = bridge._put.call_args
    assert call_args2[0][1]["bri"] == 0  # clamped to min


@pytest.mark.anyio
async def test_set_light_no_state_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    with pytest.raises(HueBridgeError, match="No state changes"):
        await bridge.set_light("3")


# ── Error handling ───────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_connection_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    import httpx

    # Mock at httpx level so the full _get error-handling path is exercised
    with patch("httpx.AsyncClient") as mock_cls:
        inst = AsyncMock()
        inst.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst
        with pytest.raises(HueBridgeError, match="Cannot reach Hue Bridge"):
            await bridge.list_lights()


@pytest.mark.anyio
async def test_403_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    """V1 API returns 403 for invalid credentials."""
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()

    resp = MagicMock()
    resp.status_code = 403
    resp.text = "Forbidden"

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_instance = AsyncMock()
        mock_instance.get = AsyncMock(return_value=resp)
        mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_instance.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_instance
        with pytest.raises(HueBridgeError, match="403 Unauthorized"):
            await bridge.list_lights()


@pytest.mark.anyio
async def test_api_error_in_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """V1 API returns errors as a list of {"error": {...}} with HTTP 200."""
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '[{"error": {"type": 5, "address": "/lights", "description": "Invalid request"}}]'
    mock_resp.json.return_value = [
        {"error": {"type": 5, "address": "/lights", "description": "Invalid request"}}
    ]
    with patch("httpx.AsyncClient") as mock_cls:
        inst = AsyncMock()
        inst.get = AsyncMock(return_value=mock_resp)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst
        with pytest.raises(HueBridgeError, match="Invalid request"):
            await bridge.list_lights()


@pytest.mark.anyio
async def test_put_error_in_body_with_http_200(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: V1 PUT rejects a bad write with HTTP 200 + an error body.

    Previously ``_put`` ignored the 2xx body entirely, so ``set_light`` to the
    wrong endpoint (``/lights/<id>`` instead of ``/lights/<id>/state``) printed
    success while the light never changed.
    """
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = '[{"error":{"type":6,"address":"/lights/3/on","description":"parameter, on, not available"}}]'
    mock_resp.json.return_value = [
        {"error": {"type": 6, "address": "/lights/3/on", "description": "parameter, on, not available"}}
    ]
    with patch("httpx.AsyncClient") as mock_cls:
        inst = AsyncMock()
        inst.put = AsyncMock(return_value=mock_resp)
        inst.__aenter__ = AsyncMock(return_value=inst)
        inst.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = inst
        with pytest.raises(HueBridgeError, match="not available"):
            await bridge.set_light("3", on=True)


# ── set_light_color / set_light_color_named ─────────────────────────────


def _bridge_with_mocked_io(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_bri: int | None = 100,
) -> HueBridge:
    """A bridge whose HTTP layer is mocked; returns the bridge for assertions."""
    _mock_env(monkeypatch)
    bridge = HueBridge.from_env()
    state = {} if current_bri is None else {"bri": current_bri}
    bridge._get = AsyncMock(return_value={"state": state})  # type: ignore[attr-defined]
    bridge._put = AsyncMock(return_value={})  # type: ignore[attr-defined]
    return bridge


def _sent_payload(bridge: HueBridge) -> tuple[str, dict]:
    """Extract the (path, payload) of the single PUT a bridge issued."""
    assert bridge._put.await_count == 1  # type: ignore[attr-defined]
    path, payload = bridge._put.await_args.args  # type: ignore[attr-defined]
    return path, payload


@pytest.mark.anyio
async def test_set_light_color_named_does_not_typeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: `set_light_color_named` passed `brightness` to `set_light_color`.

    `set_light_color` had no `brightness` parameter, so every named-colour call
    raised `TypeError` *before* contacting the Bridge. The NLP `set_color_named`
    action was therefore completely non-functional: it never changed a light.
    """
    bridge = _bridge_with_mocked_io(monkeypatch)

    await bridge.set_light_color_named("3", "red")

    path, payload = _sent_payload(bridge)
    assert path == "/lights/3/state"
    assert payload["xy"] == [0.64, 0.33]
    assert payload["bri"] == 254
    assert payload["on"] is True


@pytest.mark.anyio
async def test_set_light_color_named_default_is_full_brightness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented default (100%) must reach the Bridge as bri=254."""
    bridge = _bridge_with_mocked_io(monkeypatch)
    await bridge.set_light_color_named("3", "blue")
    _, payload = _sent_payload(bridge)
    assert payload["xy"] == [0.15, 0.06]
    assert payload["bri"] == 254


@pytest.mark.anyio
async def test_set_light_color_named_zero_brightness_is_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`brightness=0` means "off" and must not be coerced to full brightness.

    A `brightness or 100.0` default silently upgraded an explicit 0 to 100%.
    """
    bridge = _bridge_with_mocked_io(monkeypatch)

    await bridge.set_light_color_named("3", "red", brightness=0)

    _, payload = _sent_payload(bridge)
    assert payload["bri"] == 0
    assert payload["on"] is False


@pytest.mark.anyio
async def test_set_light_color_named_none_brightness_keeps_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`brightness=None` must leave the light's current brightness alone."""
    bridge = _bridge_with_mocked_io(monkeypatch, current_bri=77)

    await bridge.set_light_color_named("3", "red", brightness=None)

    _, payload = _sent_payload(bridge)
    assert payload["bri"] == 77
    assert payload["on"] is True


@pytest.mark.anyio
async def test_set_light_color_named_partial_brightness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mid-range brightness is converted to the 0-254 V1 scale."""
    bridge = _bridge_with_mocked_io(monkeypatch)
    await bridge.set_light_color_named("3", "red", brightness=50)
    _, payload = _sent_payload(bridge)
    assert payload["bri"] == 127


@pytest.mark.anyio
async def test_set_light_color_named_unknown_name_raises_bridge_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown colour name must surface as HueBridgeError, not ValueError.

    `_rgb_from_named_color` raises ValueError, which used to escape
    `set_light_color_named` and bypass the agent's HueBridgeError handler.
    """
    bridge = _bridge_with_mocked_io(monkeypatch)

    with pytest.raises(HueBridgeError, match="Invalid color name"):
        await bridge.set_light_color_named("3", "chartreuse")

    assert bridge._put.await_count == 0  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_set_light_color_named_non_string_name_raises_bridge_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-string name is a usage error, reported as HueBridgeError."""
    bridge = _bridge_with_mocked_io(monkeypatch)

    with pytest.raises(HueBridgeError, match="Invalid color name"):
        await bridge.set_light_color_named("3", 123)  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_set_light_color_explicit_brightness_overrides_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """set_light_color accepts an explicit brightness (RGB path)."""
    bridge = _bridge_with_mocked_io(monkeypatch, current_bri=10)

    await bridge.set_light_color("3", r=255, g=0, b=0, brightness=80)

    _, payload = _sent_payload(bridge)
    assert payload["bri"] == 203
    assert payload["xy"] == [0.64, 0.33]


@pytest.mark.anyio
async def test_set_light_color_without_brightness_keeps_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omitting brightness preserves the pre-existing behaviour."""
    bridge = _bridge_with_mocked_io(monkeypatch, current_bri=42)

    await bridge.set_light_color("3", r=255, g=0, b=0)

    _, payload = _sent_payload(bridge)
    assert payload["bri"] == 42


@pytest.mark.anyio
async def test_set_light_color_rejects_no_color_specification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling set_light_color with no colour at all is an error."""
    bridge = _bridge_with_mocked_io(monkeypatch)

    with pytest.raises(HueBridgeError, match="No color specification"):
        await bridge.set_light_color("3", brightness=50)

    assert bridge._put.await_count == 0  # type: ignore[attr-defined]


# ── agent-level entry point: _nlp_hue_control ────────────────────────────


def _agent_with_mocked_hue(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_bri: int = 100,
) -> tuple["Any", HueBridge]:
    """Return (agent, bridge) with the Bridge fully mocked.

    `agent.py` resolves ``HueBridge`` as a module-level name, so patching
    ``agent.HueBridge.from_env`` is enough to intercept the real entry point.
    """
    import agent as agent_mod

    bridge = HueBridge("1.2.3.4", "k")
    bridge._get = AsyncMock(return_value={"state": {"bri": current_bri}})  # type: ignore[attr-defined]
    bridge._put = AsyncMock(return_value={})  # type: ignore[attr-defined]
    monkeypatch.setattr(
        agent_mod.HueBridge, "from_env", staticmethod(lambda: bridge)
    )
    return agent_mod.Agent(workspace="."), bridge


@pytest.mark.anyio
async def test_nlp_hue_control_set_color_named_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: the `set_color_named` action never worked via the tool loop.

    `HueBridge.set_light_color_named` raised TypeError because it forwarded an
    unsupported `brightness` kwarg to `set_light_color`. This exercises the real
    entry point (`Agent._nlp_hue_control`) rather than the bridge in isolation.
    """
    bot, bridge = _agent_with_mocked_hue(monkeypatch)

    out = await bot._nlp_hue_control(
        {"action": "set_color_named", "light_id": "3", "name": "red"}
    )

    assert "color set to 'red'" in out
    path, payload = bridge._put.await_args.args  # type: ignore[attr-defined]
    assert path == "/lights/3/state"
    assert payload["xy"] == [0.64, 0.33]
    assert payload["bri"] == 254


@pytest.mark.anyio
async def test_nlp_hue_control_set_color_named_unknown_colour_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown colour yields a readable error string, not an exception.

    The ValueError used to escape the handler's `except HueBridgeError`.
    """
    bot, bridge = _agent_with_mocked_hue(monkeypatch)

    out = await bot._nlp_hue_control(
        {"action": "set_color_named", "light_id": "3", "name": "chartreuse"}
    )

    assert out.startswith("Hue Bridge error:")
    assert "Invalid color name" in out
    assert bridge._put.await_count == 0  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_nlp_hue_control_set_light_brightness_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V1 brightness=0 reaches the Bridge as bri=0 through the real handler.

    Note: the bridge sends only ``bri`` (no ``on`` key) — Hue itself treats
    ``bri=0`` as off, so the handler must not swallow the zero value.
    """
    bot, bridge = _agent_with_mocked_hue(monkeypatch)

    out = await bot._nlp_hue_control(
        {"action": "set_light", "light_id": "3", "brightness": 0}
    )

    assert "updated" in out
    path, payload = bridge._put.await_args.args  # type: ignore[attr-defined]
    assert path == "/lights/3/state"
    assert payload["bri"] == 0
    assert "on" not in payload


# ── schema reachability ──────────────────────────────────────────────────


def test_hue_control_schema_exposes_name_property() -> None:
    """Regression: `set_color_named` was advertised but had no `name` property.

    The `action` enum listed `set_color_named`, yet the parameters object
    defined no `name` field, so the model could never supply a colour name and
    the action was unreachable through the advertised tool contract.
    """
    from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

    hue = next(
        t for t in NLP_TOOL_SCHEMAS if t["function"]["name"] == "hue_control"
    )
    params = hue["function"]["parameters"]
    props = params["properties"]

    assert "set_color_named" in props["action"]["enum"]
    assert "name" in props, "set_color_named needs a 'name' parameter"
    assert props["name"]["type"] == "string"
    assert params["required"] == ["action"]


def test_hue_schema_advertised_colours_all_resolve() -> None:
    """Every colour the schema advertises must be accepted by the bridge.

    Guards against drift: an earlier revision of the schema description
    advertised 'warm white' in its ``e.g.`` examples, which
    ``_rgb_from_named_color`` rejects with ValueError. Anything mentioned in
    the tool contract must actually work, otherwise the model is guided into a
    guaranteed failure.

    Two independent checks, because the original bug hid in the *examples*
    rather than the ``Supported:`` list:
      1. each colour in ``Supported:`` is accepted by the bridge;
      2. each example in the ``e.g.`` clause is one of the supported colours.
    """
    import re

    from agent_core.hue.bridge import _rgb_from_named_color
    from agent_core.tool_schemas import NLP_TOOL_SCHEMAS

    hue = next(
        t for t in NLP_TOOL_SCHEMAS if t["function"]["name"] == "hue_control"
    )
    desc = hue["function"]["parameters"]["properties"]["name"]["description"]

    supported_match = re.search(r"Supported:\s*([^.]+)\.", desc)
    assert supported_match, f"schema lost its 'Supported:' list: {desc!r}"
    supported = [n.strip() for n in supported_match.group(1).split(",") if n.strip()]
    assert supported, "no colour names advertised"

    # 1. every advertised colour must actually resolve.
    unsupported = []
    for colour in supported:
        try:
            _rgb_from_named_color(colour)
        except ValueError:
            unsupported.append(colour)
    assert not unsupported, (
        f"schema 'Supported:' advertises colour(s) the bridge rejects: {unsupported}"
    )

    # 2. every example must be drawn from the supported set (this is the check
    #    that catches the original 'warm white' defect).
    example_match = re.search(r"e\.g\.\s*(.+?)\.", desc)
    assert example_match, f"schema lost its 'e.g.' examples: {desc!r}"
    examples = re.findall(r"'([^']+)'", example_match.group(1))
    assert examples, f"no examples parsed from: {example_match.group(1)!r}"

    not_supported = [e for e in examples if e not in supported]
    assert not not_supported, (
        f"schema examples advertise colour(s) absent from 'Supported:' "
        f"(and rejected by the bridge): {not_supported}"
    )
