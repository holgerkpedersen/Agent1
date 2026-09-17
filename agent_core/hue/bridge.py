"""Philips Hue Bridge V1 API client.

Configuration via environment variables:
    HUE_BRIDGE_IP   – IP address of the Hue Bridge (e.g. 192.168.1.217)
    HUE_API_KEY     – API key / username (obtained via link-button press + POST /api)

Usage::

    bridge = HueBridge.from_env()
    lights = await bridge.list_lights()
    info   = await bridge.get_light("3")
    await bridge.set_light("3", on=True, brightness=80, mirek=366)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("hue.bridge")


def _env_file_path() -> Path:
    """Project-root .env path (workspace root is two levels above this file)."""
    return Path(__file__).resolve().parents[2] / ".env"


def _read_env_file() -> dict[str, str]:
    """Best-effort read of project .env values for the from_env fallback.

    Reuses agent_core.config's parser so value handling (quotes etc.) stays
    identical to the rest of the app.  Never raises: on any failure it logs a
    warning and returns {} so callers degrade to the plain env-var error path.
    """
    try:
        from agent_core.config import _load_env_file

        return _load_env_file(_env_file_path())
    except Exception as exc:  # pragma: no cover - defensive degradation only
        logger.warning("Hue .env fallback unavailable: %s", exc)
        return {}


class HueBridgeError(Exception):
    """Raised on connection, auth, or API-level failures."""


class HueBridge:
    """Async client for the Philips Hue Bridge V1 API."""

    def __init__(self, base_ip: str, api_key: str) -> None:
        self._base_ip = base_ip
        self._api_key = api_key
        self._base_url = f"http://{base_ip}/api/{api_key}"

    @classmethod
    def from_env(cls) -> "HueBridge":
        """Create a HueBridge from HUE_BRIDGE_IP and HUE_API_KEY.

        Per-variable resolution order: os.environ first, then the project's
        .env file (so no manual export is needed once .env holds the values).
        """
        ip = os.environ.get("HUE_BRIDGE_IP", "").strip()
        key = os.environ.get("HUE_API_KEY", "").strip()
        if not ip or not key:
            file_vars = _read_env_file()
            ip = ip or (file_vars.get("HUE_BRIDGE_IP") or "").strip()
            key = key or (file_vars.get("HUE_API_KEY") or "").strip()
        if not ip:
            raise HueBridgeError(
                "HUE_BRIDGE_IP is not set. Set it in your .env or environment "
                "to the IP address of your Hue Bridge (e.g. 192.168.1.217)."
            )
        if not key:
            raise HueBridgeError(
                "HUE_API_KEY is not set. Press the link button on your Hue Bridge, "
                "then within 30 s run: "
                'curl -s -X POST http://{ip}/api -d \'{"devicetype":"agent1#agent"}\' '
                "and copy the returned username value."
            )
        return cls(base_ip=ip, api_key=key)

    # ------------------------------------------------------------------
    # GET helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str) -> dict[str, Any]:
        """Perform an authenticated GET and return the JSON body."""
        import httpx

        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)
        except httpx.ConnectError as exc:
            raise HueBridgeError(f"Cannot reach Hue Bridge at {self._base_ip}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HueBridgeError(f"HTTP error communicating with Hue Bridge: {exc}") from exc

        # V1 uses 403 (or 401) for bad credentials
        if resp.status_code in (401, 403):
            raise HueBridgeError(
                f"Hue Bridge returned {resp.status_code} Unauthorized. "
                "Check your HUE_API_KEY."
            )
        if resp.status_code == 404:
            raise HueBridgeError(f"Resource not found on Hue Bridge: {path}")

        try:
            data = resp.json()
        except ValueError:
            raise HueBridgeError(
                f"Hue Bridge returned non-JSON response ({resp.status_code}): "
                f"{resp.text[:200]}"
            )

        # V1 API returns errors as a list: [{"error": {"type": N, ...}}]
        if isinstance(data, list):
            for entry in data:
                err = entry.get("error") if isinstance(entry, dict) else None
                if err:
                    raise HueBridgeError(
                        f"Hue API error (type {err.get('type', '?')}): "
                        f"{err.get('description', str(err))}"
                    )

        return data

    async def _put(self, path: str, payload: dict[str, Any]) -> None:
        """Perform an authenticated PUT with the given JSON payload."""
        import httpx

        url = f"{self._base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.put(url, json=payload, timeout=10.0)
        except httpx.ConnectError as exc:
            raise HueBridgeError(f"Cannot reach Hue Bridge at {self._base_ip}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HueBridgeError(f"HTTP error communicating with Hue Bridge: {exc}") from exc

        # V1 uses 403 (or 401) for bad credentials
        if resp.status_code in (401, 403):
            raise HueBridgeError(
                f"Hue Bridge returned {resp.status_code} Unauthorized. "
                "Check your HUE_API_KEY."
            )

        # V1 PUT returns HTTP 200 even when the request failed: the body is a
        # list of {"success": {...}} and/or {"error": {...}} entries. Parse it
        # on both success and failure, otherwise rejected writes look like no-ops.
        if resp.status_code >= 300:
            try:
                data = resp.json()
            except ValueError:
                raise HueBridgeError(f"Hue API error ({resp.status_code}): {resp.text[:200]}")

            if isinstance(data, list):
                for entry in data:
                    err = entry.get("error") if isinstance(entry, dict) else None
                    if err:
                        raise HueBridgeError(
                            f"Hue API error (type {err.get('type', '?')}): "
                            f"{err.get('description', str(err))}"
                        )
            raise HueBridgeError(f"Hue API error ({resp.status_code}): {resp.text[:200]}")

        try:
            data = resp.json()
        except ValueError:
            return

        if isinstance(data, list):
            for entry in data:
                err = entry.get("error") if isinstance(entry, dict) else None
                if err:
                    raise HueBridgeError(
                        f"Hue API error (type {err.get('type', '?')}): "
                        f"{err.get('description', str(err))}"
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def list_lights(self) -> list[dict[str, Any]]:
        """Return a list of all lights from the V1 API.

        Each entry contains at minimum: id, name, on, brightness (0-254), mirek.
        """
        data = await self._get("/lights")
        result: list[dict[str, Any]] = []
        for rid, item in data.items():
            state = item.get("state", {})
            is_on: bool = state.get("on", False)
            bri_raw: int | None = state.get("bri")  # 0-254 in V1
            brightness: float | None = round((bri_raw / 254.0) * 100, 1) if bri_raw is not None else None
            mirek: int | None = state.get("ct")
            result.append({
                "id": rid,
                "name": item.get("name", f"Light {rid}"),
                "on": is_on,
                "brightness": brightness,
                "mirek": mirek,
            })
        return result

    async def get_light(self, light_id: str) -> dict[str, Any]:
        """Return detailed info for a single light."""
        data = await self._get(f"/lights/{light_id}")
        if not isinstance(data, dict):
            raise HueBridgeError(f"Light '{light_id}' not found on Hue Bridge.")
        state = data.get("state", {})
        is_on: bool = state.get("on", False)
        bri_raw: int | None = state.get("bri")
        brightness: float | None = round((bri_raw / 254.0) * 100, 1) if bri_raw is not None else None
        mirek: int | None = state.get("ct")
        return {
            "id": light_id,
            "name": data.get("name", f"Light {light_id}"),
            "type": data.get("type"),
            "on": is_on,
            "brightness": brightness,
            "mirek": mirek,
            "color_xy": state.get("xy"),
        }

    async def set_light(
        self,
        light_id: str,
        *,
        on: bool | None = None,
        brightness: float | None = None,
        mirek: int | None = None,
    ) -> None:
        """Set one or more state attributes on a light.

        Args:
            light_id: The Hue V1 ID of the light (e.g. "3").
            on:       True to turn on, False to turn off. None = leave unchanged.
            brightness: Brightness percentage 0.0–100.0 (0 = off). None = leave unchanged.
            mirek:    Color temperature in mired (153 = cool/blue, 500 = warm/orange).
                      None = leave unchanged.
        """
        payload: dict[str, Any] = {}
        if on is not None:
            payload["on"] = on
        if brightness is not None:
            clamped = max(0.0, min(100.0, float(brightness)))
            payload["bri"] = int(round((clamped / 100.0) * 254))
        if mirek is not None:
            clamped_m = max(153, min(500, int(mirek)))
            payload["ct"] = clamped_m

        if not payload:
            raise HueBridgeError("No state changes specified for set_light.")

        await self._put(f"/lights/{light_id}/state", payload)
        logger.info("Set light %s: %s", light_id, payload)
