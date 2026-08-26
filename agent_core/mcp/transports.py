"""Wire transports for the MCP consumer: stdio (local servers) and HTTP.

Safety contract shared by both transports:

* every ``request()`` is wall-clock capped - a stalled server surfaces as a
  :class:`TransportError` instead of pinning the caller (2026-08-25 stall
  post-mortem applied preemptively);
* stdio children are spawned from an argv LIST (never ``shell=True``), with
  an explicitly merged environment, and get killed on disconnect;
* responses are matched to requests by id; server-pushed notifications are
  consumed but never crash the reader.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import httpx

from . import jsonrpc

#: Hard floor/ceiling for per-request timeouts (seconds).
MIN_TIMEOUT_S = 1
MAX_TIMEOUT_S = 600

_STDIO_KILL_GRACE_S = 5.0
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class TransportError(RuntimeError):
    """A transport-level failure (spawn, IO, timeout, dead server)."""


def clamp_timeout(timeout_s: float) -> float:
    """Clamp *timeout_s* into ``[MIN_TIMEOUT_S, MAX_TIMEOUT_S]``."""
    return max(MIN_TIMEOUT_S, min(float(timeout_s), MAX_TIMEOUT_S))


class BaseTransport:
    """Interface both transports implement."""

    def request(self, method: str, params: dict[str, Any] | None,
                timeout_s: float) -> Any:
        raise NotImplementedError

    def notify(self, method: str) -> None:
        """Send a JSON-RPC notification; never waits for a reply."""

    def is_alive(self) -> bool:
        """Whether the underlying channel can carry another request."""
        return True

    def close(self) -> None:
        raise NotImplementedError


class StdioTransport(BaseTransport):
    """Talk JSON-RPC (newline-delimited) to a local MCP server subprocess.

    A single daemon reader thread feeds parsed messages into a queue; the
    request path matches replies by id under a serializing lock (one
    in-flight request per server keeps ordering and back-pressure simple).
    """

    def __init__(self, command: list[str], env: dict[str, str] | None = None,
                 cwd: str | None = None) -> None:
        self._command = list(command)
        self._extra_env = dict(env or {})
        self._cwd = cwd
        self._proc: subprocess.Popen[str] | None = None
        self._inbox: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._send_lock = threading.Lock()
        self._request_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.is_alive():
            return
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - argv list, no shell
                self._command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                env={**os.environ, **self._extra_env},
                cwd=self._cwd,
                creationflags=_CREATE_NO_WINDOW,
            )
        except OSError as exc:
            raise TransportError(
                f"cannot spawn {self._command[:1]}: {exc}"
            ) from exc
        assert self._proc.stdout is not None
        threading.Thread(
            target=self._reader, args=(self._proc.stdout,),
            name="mcp-stdio-reader", daemon=True,
        ).start()

    def _reader(self, stdout: Any) -> None:
        try:
            for line in stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    self._inbox.put(jsonrpc.parse_message(line))
                except jsonrpc.JsonRpcError:
                    continue  # malformed frame from the server - skip it
        except (ValueError, OSError):
            pass
        finally:
            self._inbox.put(None)  # EOF sentinel wakes pending waiters

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def notify(self, method: str) -> None:
        """Fire-and-forget notification (no id, no reply expected)."""
        if self._proc is None or self._proc.stdin is None:
            return
        note = jsonrpc.make_request(method)
        try:
            with self._send_lock:
                self._proc.stdin.write(
                    json.dumps(note, separators=(",", ":")) + "\n"
                )
                self._proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=_STDIO_KILL_GRACE_S)
        except (subprocess.TimeoutExpired, OSError):
            try:
                proc.kill()
            except OSError:
                pass

    # -- protocol ----------------------------------------------------------

    def request(self, method: str, params: dict[str, Any] | None,
                timeout_s: float) -> Any:
        timeout = clamp_timeout(timeout_s)
        with self._request_lock:
            self.start()
            assert self._proc is not None and self._proc.stdin is not None
            req = jsonrpc.make_request(method, params)
            payload = json.dumps(req, separators=(",", ":"), ensure_ascii=False)
            try:
                with self._send_lock:
                    self._proc.stdin.write(payload + "\n")
                    self._proc.stdin.flush()
            except (OSError, ValueError) as exc:
                raise TransportError(f"server pipe closed: {exc}") from exc
            deadline = time.monotonic() + timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TransportError(
                        f"no response to '{method}' within {timeout:.0f}s"
                    )
                try:
                    msg = self._inbox.get(timeout=remaining)
                except queue.Empty:
                    continue
                if msg is None:
                    raise TransportError(f"server exited during '{method}'")
                if jsonrpc.is_response(msg) and msg.get("id") == req["id"]:
                    return jsonrpc.result_of(msg)
                # notification / late reply to an abandoned id: keep waiting


class HttpTransport(BaseTransport):
    """JSON-RPC over HTTP POST (Streamable-HTTP style).

    Accepts a plain ``application/json`` response body, or an
    ``text/event-stream`` body whose SSE data lines carry the reply.
    Header values may reference the OS keyring via ``secret:<name>`` or the
    environment via ``${VAR}`` - resolution happens in :mod:`.config`
    before this class ever sees them.
    """

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        if not url.startswith(("http://", "https://")):
            raise TransportError(f"unsupported MCP URL scheme: {url!r}")
        self._url = url
        self._headers = {"Accept": "application/json, text/event-stream",
                         **(headers or {})}

    def request(self, method: str, params: dict[str, Any] | None,
                timeout_s: float) -> Any:
        timeout = clamp_timeout(timeout_s)
        req = jsonrpc.make_request(method, params)
        try:
            resp = httpx.post(
                self._url, json=req, headers=self._headers, timeout=timeout,
            )
        except httpx.HTTPError as exc:
            raise TransportError(f"MCP HTTP call failed: {exc}") from exc
        if resp.status_code >= 400:
            raise TransportError(f"MCP endpoint returned HTTP {resp.status_code}")
        ctype = resp.headers.get("content-type", "")
        if "text/event-stream" in ctype:
            return jsonrpc.result_of(_response_in_sse(resp.text, req["id"]))
        try:
            return jsonrpc.result_of(jsonrpc.parse_message(resp.text))
        except jsonrpc.JsonRpcError:
            raise
        except ValueError as exc:  # non-JSON body from a broken endpoint
            raise TransportError(f"unparseable MCP response: {exc}") from exc

    def notify(self, method: str) -> None:
        """POST the notification and discard whatever comes back."""
        try:
            httpx.post(self._url, json=jsonrpc.make_request(method),
                       headers=self._headers, timeout=5.0)
        except httpx.HTTPError:
            pass

    def close(self) -> None:  # stateless; nothing to tear down
        return


def _response_in_sse(body: str, req_id: Any) -> dict[str, Any]:
    """Scan an SSE stream for the data line that answers *req_id*."""
    for chunk in body.split("\n\n"):
        for line in chunk.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data:
                continue
            msg = jsonrpc.parse_message(data)
            if jsonrpc.is_response(msg) and msg.get("id") == req_id:
                return msg
    raise TransportError("SSE stream carried no response for our request")
