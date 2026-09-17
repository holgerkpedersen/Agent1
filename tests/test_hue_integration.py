"""Unit tests for agent_core.hue.bridge (HueBridge V1 API client).

All HTTP calls are mocked — no real Bridge required.
"""

from __future__ import annotations

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
