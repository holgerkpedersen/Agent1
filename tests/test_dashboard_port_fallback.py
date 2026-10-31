"""Dashboard port fallback for side-by-side sessions."""

from __future__ import annotations


import agent_dashboard


class _Holder:
    def __init__(self, port: int) -> None:
        self.port = port
        self._refresh = None

    def set_refresh(self, refresh) -> None:
        self._refresh = refresh

    def get_refresh(self):
        return self._refresh

    def start(self, **kwargs):
        if self.port == 8081:
            raise OSError("address already in use")
        return _Server(self.port)


class _Server:
    def __init__(self, port: int) -> None:
        self.port = port
        self.stopped = False

    def serve_forever(self) -> None:
        return None

    def shutdown(self) -> None:
        self.stopped = True


class _Alerts:
    def list_rules(self):
        return []

    def evaluate(self):
        return []


def test_default_port_falls_back_when_taken(monkeypatch) -> None:
    monkeypatch.setattr(agent_dashboard, "_dashboard_port_is_explicit", lambda: False)
    monkeypatch.setattr(
        agent_dashboard,
        "_build_dashboard",
        lambda collector, port: (_Holder(port), _Alerts()),
    )
    monkeypatch.setattr(agent_dashboard, "get_metrics_collector", lambda: object())
    httpd = agent_dashboard.start_dashboard_thread(8081)
    assert httpd is not None
    assert httpd.port == 8082
