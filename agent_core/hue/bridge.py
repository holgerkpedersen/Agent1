"""Philips Hue Bridge V1/V2 API client.

Configuration via environment variables:
    HUE_BRIDGE_IP   – IP address of the Hue Bridge (e.g. 192.168.1.217)
    HUE_API_KEY     – API key / username (obtained via link-button press + POST /api)

Usage::

    bridge = HueBridge.from_env()
    lights = await bridge.list_lights()
    info   = await bridge.get_light("3")
    await bridge.set_light("3", on=True, brightness=80, mirek=366)  # Warm white (V1 API)
    await bridge.set_light_color("3", r=255, g=0, b=0)              # Red (V2 API)

The library supports both V1 and V2 APIs:
- V1: on/off, brightness, color temperature (CT/mirek) – for white+CT bulbs  
- V2: Full RGB color control via xy or hs coordinates – for RGB/RGBA bulbs
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
    """Best-effort read of project .env values for the from_env fallback."""
    try:
        from agent_core.config import _load_env_file
        return _load_env_file(_env_file_path())
    except Exception as exc:  # pragma: no cover - defensive degradation only
        logger.warning("Hue .env fallback unavailable: %s", exc)
        return {}


class HueBridgeError(Exception):
    """Raised on connection, auth, or API-level failures."""


# ------------------------------------------------------------------
# sRGB gamma correction functions (module-level helpers)
# ------------------------------------------------------------------

def _sRGB_from_linear(c: float) -> float:
    """Convert linear RGB to sRGB using the standard gamma curve."""
    if c > 0.0031308:
        return 1.055 * (c ** (1 / 2.4)) - 0.055
    else:
        return 12.92 * c


def _rgb_from_named_color(name: str) -> tuple[int, int, int]:
    """Convert a named color string to RGB tuple."""
    name = name.lower().strip()

    # Common named colors (subset of CSS/HTML names)
    named_colors: dict[str, tuple[int, int, int]] = {
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "pink": (255, 192, 203),
        "lime": (0, 255, 0),
        "navy": (0, 0, 128),
        "teal": (0, 128, 128),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
    }

    # HSV-based color wheel colors for smoother transitions  
    hsv_colors: dict[str, tuple[float, float, float]] = {
        "red": (0.0, 1.0, 1.0),
        "green": (120.0, 1.0, 1.0),
        "blue": (240.0, 1.0, 1.0),
        "yellow": (60.0, 1.0, 1.0),
        "cyan": (180.0, 1.0, 1.0),
        "magenta": (300.0, 1.0, 1.0),
        "orange": (30.0, 1.0, 1.0),
        "purple": (280.0, 1.0, 1.0),
    }

    if name in named_colors:
        return named_colors[name]
    elif name in hsv_colors:
        h, s, v = hsv_colors[name]
        x, y = _hsv_to_xy(h, s, v)
        return _xy_to_rgb(x, y)

    raise ValueError(f"Unknown color name: {name}")


# RGB Color conversion helpers (module-level static methods)
def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (h: 0-360, s: 0-100, v: 0-100)."""
    r_norm, g_norm, b_norm = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r_norm, g_norm, b_norm)
    min_c = min(r_norm, g_norm, b_norm)
    delta = max_c - min_c

    if delta == 0:
        h = 0.0
    elif max_c == r_norm:
        h = (60 * ((g_norm - b_norm) / delta) % 360)
    elif max_c == g_norm:
        h = (60 * ((b_norm - r_norm) / delta) + 120)
    else:
        h = (60 * ((r_norm - g_norm) / delta) + 240)

    s = (delta / max_c) * 100.0 if max_c != 0 else 0.0
    v = max_c * 100.0

    return h, s, v


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (h: 0-360, s: 0-100, v: 0-100) to RGB (0-255)."""
    s_norm = s / 100.0
    v_norm = v / 100.0
    i = int(h / 60.0) % 6
    f = (h / 60.0) - int(h / 60.0)
    p = v_norm * (1 - s_norm)
    q = v_norm * (1 - f * s_norm)
    t = v_norm * (1 - (1 - f) * s_norm)

    rgb_map = {
        0: (v_norm, t, p),
        1: (q, v_norm, p),
        2: (p, v_norm, t),
        3: (p, q, v_norm),
        4: (t, p, v_norm),
        5: (v_norm, p, q),
    }
    r, g, b = rgb_map[i]
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


def _hsv_to_xy(h: float, s: float, v: float) -> tuple[float, float]:
    """Convert HSV to CIE xy coordinates via RGB."""
    r, g, b = _hsv_to_rgb(h, s, v)
    return _rgb_to_xy(r, g, b)


def _linear_from_srgb(c: float) -> float:
    """Convert sRGB (0-1) to linear RGB using inverse gamma curve."""
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_xy(r: int, g: int, b: int) -> tuple[float, float]:
    """Convert RGB (0-255) to CIE xy chromaticity coordinates."""
    # Linearize sRGB
    r_lin = _linear_from_srgb(r / 255.0)
    g_lin = _linear_from_srgb(g / 255.0)
    b_lin = _linear_from_srgb(b / 255.0)

    # sRGB to CIE XYZ (D65 illuminant, BT.709)
    X = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    Y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    Z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin

    total = X + Y + Z
    if total == 0:
        return (0.0, 0.0)

    return (round(X / total, 4), round(Y / total, 4))


def _xy_to_rgb(x: float, y: float, z: float | None = None) -> tuple[int, int, int]:
    """Convert CIE xy to RGB (0-255)."""
    Y = 1.0 if z is None else z

    # Convert xyY to XYZ using the standard observer function
    X = x / y * Y if y != 0 else 0.0
    Z = (1.0 - x - y) / y * Y if y != 0 else 0.0

    # Convert XYZ to RGB using the sRGB matrix
    r_prime = X - 0.576968 * Y + 0.142734 * Z
    g_prime = X + 1.086396 * Y - 0.528385 * Z
    b_prime = X - 0.292962 * Y + 0.258358 * Z

    # Apply gamma correction and clip to [0, 1] range
    r = _sRGB_from_linear(r_prime) if r_prime > 0 else 0.0
    g = _sRGB_from_linear(g_prime) if g_prime > 0 else 0.0
    b = _sRGB_from_linear(b_prime) if b_prime > 0 else 0.0

    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255))


# ------------------------------------------------------------------
# HueBridge class - V1 and V2 API client
# ------------------------------------------------------------------

class HueBridge:
    """Async client for the Philips Hue Bridge V1 and V2 APIs.

    Supports both legacy white+CT bulbs (V1) and full RGB color bulbs (V2).

    Methods:
        list_lights()           - List all lights (V1 API)
        get_light(light_id)     - Get light info (V1 API)  
        set_light(...)          - Set on/off/brightness/CT (V1 API)
        get_color_capabilities() - Get V2 color capabilities
        set_light_color(...)    - Set RGB/HSV/XY colors (V2 API)
        set_light_color_named()  - Set color by name (V2 API)
        set_light_brightness()   - Set brightness only (V1 API)

    Attributes:
        _base_ip: The Hue Bridge IP address.
        _api_key: The API key / username for authentication.
    """

    def __init__(self, base_ip: str, api_key: str) -> None:
        self._base_ip = base_ip
        self._api_key = api_key
        # V1 uses /api/{key} endpoint; V2 uses /api/v2 endpoints
        self._v1_base_url = f"http://{base_ip}/api/{api_key}"
        self._v2_base_url = f"http://{base_ip}/api"
        # For backward compatibility with existing tests  
        self._base_url = f"http://{base_ip}/api/{api_key}"

    @classmethod
    def from_env(cls) -> "HueBridge":
        """Create a HueBridge from HUE_BRIDGE_IP and HUE_API_KEY."""

    # ------------------------------------------------------------------
    # V1 API HTTP helpers (for legacy on/off/brightness/CT operations)
    # ------------------------------------------------------------------

    async def _get(self, path: str) -> dict[str, Any]:
        """Perform an authenticated GET via V1 API and return the JSON body."""
        import httpx

        url = f"{self._v1_base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)
        except httpx.ConnectError as exc:
            raise HueBridgeError(f"Cannot reach Hue Bridge at {self._base_ip}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HueBridgeError(f"Hue API error communicating with Hue Bridge: {exc}") from exc

        if resp.status_code in (401, 403):
            raise HueBridgeError(
                f"Hue Bridge returned {resp.status_code} Unauthorized. "
                "Check your HUE_API_KEY."
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise HueBridgeError(
                f"Hue Bridge returned non-JSON response ({resp.status_code}): "
                f"{resp.text[:200]}"
            ) from exc

        # Check for API error list in the body (Hue returns errors as [{"error": {...}}])
        if isinstance(data, list) and len(data) > 0:
            first_item = data[0]
            if isinstance(first_item, dict):
                # Hue V1 API wraps errors as {"error": {"type", "address", "description"}}
                nested_error = first_item.get("error")
                desc = ""
                if isinstance(nested_error, dict):
                    desc = nested_error.get("description", "")
                elif "description" in first_item:
                    desc = first_item["description"]
                # Handle cases where description is short (e.g., just "on")
                if not desc and isinstance(first_item, dict) and "address" in first_item:
                    desc = f"Hue API error on {first_item.get('address', 'unknown')}"
                if desc:
                    raise HueBridgeError(desc)

        return data

    async def _put(self, path: str, payload: dict[str, Any]) -> None:
        """Perform an authenticated PUT via V1 API with the given JSON payload."""
        import httpx

        url = f"{self._v1_base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.put(url, json=payload, timeout=10.0)
        except httpx.ConnectError as exc:
            raise HueBridgeError(f"Cannot reach Hue Bridge at {self._base_ip}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HueBridgeError(f"Hue API error communicating with Hue Bridge: {exc}") from exc

        if resp.status_code in (401, 403):
            raise HueBridgeError(
                f"Hue Bridge returned {resp.status_code} Unauthorized. "
                "Check your HUE_API_KEY."
            )

        # Check for API error list in the body (Hue returns errors as [{"error": {...}}])
        try:
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                first_item = data[0]
                if isinstance(first_item, dict):
                    # Hue V1 API wraps errors as {"error": {"type", "address", "description"}}
                    nested_error = first_item.get("error")
                    desc = ""
                    if isinstance(nested_error, dict):
                        desc = nested_error.get("description", "")
                    elif "description" in first_item:
                        desc = first_item["description"]
                    # Handle cases where description is short (e.g., just "on")
                    if not desc and isinstance(first_item, dict) and "address" in first_item:
                        desc = f"Hue API error on {first_item.get('address', 'unknown')}"
                    if desc:
                        raise HueBridgeError(desc)
        except ValueError:
            # Non-JSON response - ignore since we already handled errors above
            pass

    @classmethod
    def from_env(cls) -> "HueBridge":
        """Create a HueBridge from HUE_BRIDGE_IP and HUE_API_KEY."""
        ip = os.environ.get("HUE_BRIDGE_IP", "").strip()
        key = os.environ.get("HUE_API_KEY", "").strip()
        if not ip or not key:
            file_vars = _read_env_file()
            ip = ip or (file_vars.get("HUE_BRIDGE_IP") or "").strip()
            key = key or (file_vars.get("HUE_API_KEY") or "").strip()
        if not ip:
            raise HueBridgeError(
                "HUE_BRIDGE_IP is not set. Set it in your .env or environment "
                "to the IP address of the Hue Bridge (e.g. 192.168.1.217)."
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
    # V2 API HTTP helpers (for RGB color support)
    # ------------------------------------------------------------------

    async def _v2_get(self, path: str) -> dict[str, Any]:
        """Perform an authenticated GET via V2 API and return the JSON body."""
        import httpx

        url = f"{self._v2_base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=10.0)
        except httpx.ConnectError as exc:
            raise HueBridgeError(f"Cannot reach Hue Bridge at {self._base_ip}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HueBridgeError(f"Hue API error communicating with Hue Bridge: {exc}") from exc

        if resp.status_code in (401, 403):
            raise HueBridgeError(
                f"Hue Bridge returned {resp.status_code} Unauthorized. "
                "Check your HUE_API_KEY."
            )

        try:
            data = resp.json()
        except ValueError as exc:
            raise HueBridgeError(
                f"Hue Bridge returned non-JSON response ({resp.status_code}): "
                f"{resp.text[:200]}"
            ) from exc

        return data

    async def _v2_put(self, path: str, payload: dict[str, Any]) -> None:
        """Perform an authenticated PUT via V2 API with the given JSON payload."""
        import httpx

        url = f"{self._v2_base_url}{path}"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.put(url, json=payload, timeout=10.0)
        except httpx.ConnectError as exc:
            raise HueBridgeError(f"Cannot reach Hue Bridge at {self._base_ip}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise HueBridgeError(f"Hue API error communicating with Hue Bridge: {exc}") from exc

        if resp.status_code in (401, 403):
            raise HueBridgeError(
                f"Hue Bridge returned {resp.status_code} Unauthorized. "
                "Check your HUE_API_KEY."
            )

    # ------------------------------------------------------------------
    # Public API - V1 Methods (on/off/brightness/CT only)
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

        Raises:
            HueBridgeError: If no state changes are specified or if the light is invalid.
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
        logger.info("Set light %s (V1): %s", light_id, payload)


    # ------------------------------------------------------------------
    # Public API - V2 Methods (Full RGB Color Support)
    # ------------------------------------------------------------------

    async def get_color_capabilities(self, light_id: str) -> dict[str, Any]:
        """Get the current state and capabilities of a specific light via V2 API.

        Returns information about supported color properties (xy/hs).

        Args:
            light_id: The Hue V1 ID of the light.

        Returns:
            Dictionary containing:
            - id, name, on, brightness (standard state)
            - supported_features: list of capabilities like ['color', 'effect']
            - color_xy: Current xy coordinates if available
            - hs_hue: Current hue value if available
            - hs_saturation: Current saturation value if available

        Raises:
            HueBridgeError: If the light is not found.
        """
        # First get V1 state for basic info
        v1_data = await self._get(f"/lights/{light_id}")
        if not isinstance(v1_data, dict):
            raise HueBridgeError(f"Light '{light_id}' not found on Hue Bridge.")

        state = v1_data.get("state", {})
        is_on: bool = state.get("on", False)
        bri_raw: int | None = state.get("bri")
        brightness: float | None = round((bri_raw / 254.0) * 100, 1) if bri_raw is not None else None

        # Get V2 capabilities and color info
        v2_data = await self._v2_get(f"/lights/{light_id}")
        supported_features: list[str] = []

        if isinstance(v2_data, dict):
            supported_features = v2_data.get("supported_features", [])
            color_xy = v2_data.get("xy")
            hs_hue = v2_data.get("hs_hue")
            hs_saturation = v2_data.get("hs_saturation")

        return {
            "id": light_id,
            "name": v1_data.get("name", f"Light {light_id}"),
            "type": v1_data.get("type"),
            "on": is_on,
            "brightness": brightness,
            "supported_features": supported_features,
            "color_xy": color_xy,
            "hs_hue": hs_hue,
            "hs_saturation": hs_saturation,
        }

    async def set_light_color(
        self,
        light_id: str,
        *,
        r: int | None = None, g: int | None = None, b: int | None = None,
        x: float | None = None, y: float | None = None,
        h: float | None = None, s: float | None = None, v: float | None = None,
        brightness: float | None = None,
    ) -> None:
        """Set the color of a light via V2 API.

        Supports multiple ways to specify color:
        - RGB: r (0-255), g (0-255), b (0-255)
        - XY: x (0-1), y (0-1) - CIE xy chromaticity coordinates  
        - HSV: h (0-360 hue degrees), s (0-100 saturation%), v (0-100 value%)

        At least one color parameter must be specified.

        Args:
            light_id: The Hue V1 ID of the light (e.g. "3").
            r, g, b: RGB values (0-255). None = leave unchanged.
            x, y: CIE xy chromaticity coordinates (0-1). None = leave unchanged.
            h, s, v: HSV values. h in 0-360 degrees, s/v in 0-100%.
                      None = leave unchanged.
            brightness: Brightness percentage 0-100. None (default) keeps the
                      light's current brightness. Note that 0 is a real value
                      (light off) and is honoured rather than treated as unset.

        Raises:
            HueBridgeError: If no color parameters are specified or if the light is invalid.
        """
        payload: dict[str, Any] = {"on": True}  # Ensure light is on when setting color

        # Determine which color specification to use based on provided arguments
        has_rgb = r is not None and g is not None and b is not None
        has_xy = x is not None or y is not None
        has_hsv = h is not None or s is not None or v is not None

        if has_rgb:
            # Convert RGB to xy for V1 API
            xy = _rgb_to_xy(r, g, b)
            payload["xy"] = [round(xy[0], 4), round(xy[1], 4)]
        elif has_hsv:
            # Convert HSV to xy for V1 API
            h_norm = max(0.0, min(360.0, float(h or 0)))
            s_norm = max(0.0, min(100.0, float(s or 100))) if s is not None else 100.0
            v_norm = max(0.0, min(100.0, float(v or 100))) if v is not None else 100.0

            xy = _hsv_to_xy(h_norm, s_norm, v_norm)
            payload["xy"] = [round(xy[0], 4), round(xy[1], 4)]
        elif has_xy:
            # Use provided xy coordinates directly
            x_val = max(0.0, min(1.0, float(x or 0.5)))
            y_val = max(0.0, min(1.0, float(y or 0.5)))
            payload["xy"] = [round(x_val, 4), round(y_val, 4)]

        if "xy" not in payload:
            raise HueBridgeError("No color specification provided for set_light_color.")

        if brightness is not None:
            # Explicit brightness wins over the light's current level. 0 is a
            # real value (light off), so it must not be treated as "unset".
            clamped = max(0.0, min(100.0, float(brightness)))
            payload["bri"] = int(round((clamped / 100.0) * 254))
            payload["on"] = clamped > 0
        else:
            # Keep the light at its current brightness, as before.
            v1_data = await self._get(f"/lights/{light_id}")
            bri_raw: int | None = v1_data.get("state", {}).get("bri")
            if bri_raw is not None:
                payload["bri"] = bri_raw

        await self._put(f"/lights/{light_id}/state", payload)
        logger.info("Set light %s (V1 color): %s", light_id, payload)


    async def set_light_color_named(
        self,
        light_id: str,
        name: str,
        *,
        brightness: float | None = 100.0,
    ) -> None:
        """Set the color of a light using a named color string.

        Supports common CSS/HTML named colors plus HSV-based colors:
            red, green, blue, yellow, cyan, magenta, black, white,
            orange, purple, pink, lime, navy, teal, gray, grey,
            (HSV): h (0-360 degrees), s (0-100%), v (0-100%)

        Args:
            light_id: The Hue V1 ID of the light.
            name: Color name string or HSV specification like "red" or "h=240,s=100,v=80".
            brightness: Brightness percentage (default 100%). None = use current.

        Raises:
            HueBridgeError: If color name is invalid or light is not found.
        """
        # Parse named colour. _rgb_from_named_color raises ValueError for an
        # unknown name; translate it so callers only have to catch HueBridgeError.
        try:
            rgb = _rgb_from_named_color(name) if isinstance(name, str) else None
        except ValueError as exc:
            raise HueBridgeError(f"Invalid color name: {name}") from exc

        if not rgb:
            raise HueBridgeError(f"Invalid color name: {name}")

        x, y = _rgb_to_xy(*rgb)
        # Pass brightness through verbatim: `or` would turn an explicit 0
        # (light off) into full brightness.
        await self.set_light_color(light_id, x=x, y=y, brightness=brightness)


    async def set_light_brightness(self, light_id: str, percentage: float | None = None):
        """Set the brightness of a light without changing color.

        Args:
            light_id: The Hue V1 ID of the light.
            percentage: Brightness 0-100 (0=off). None = leave unchanged.

        Raises:
            HueBridgeError: If no changes specified or light invalid.
        """
        if percentage is None:
            raise HueBridgeError("No brightness change specified.")

        clamped = max(0.0, min(100.0, float(percentage)))
        payload = {"bri": int(round((clamped / 100.0) * 254))}

        # Ensure light is on when setting brightness (unless explicitly off)
        if percentage > 0:
            payload["on"] = True

        await self._put(f"/lights/{light_id}/state", payload)
        logger.info("Set light %s brightness (V1): %.1f%%", light_id, clamped)


# ------------------------------------------------------------------
# Module-level convenience functions (for quick access from tests)
# ------------------------------------------------------------------

def _get_bridge_for_testing(ip: str | None = None, key: str | None = None) -> HueBridge:
    """Convenience function for creating a bridge for testing."""
    return HueBridge(base_ip=ip or "127.0.0.1", api_key=key or "test-key")


def create_bridge(ip: str, key: str) -> HueBridge:
    """Create a HueBridge instance directly."""
    return HueBridge(base_ip=ip, api_key=key)
