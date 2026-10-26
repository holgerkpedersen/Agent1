"""Regression tests for tools/black_october_triage.py (Black October IoC checks).

Focus on the pure helpers: a broken classifier would either hide an exposed
local LLM endpoint (the Black October AI-server leg) or spam false positives.
"""

import pathlib
import sys

import pytest

_TOOL = (
    pathlib.Path(__file__).resolve().parents[1] / "tools" / "black_october_triage.py"
)


def _load():
    import importlib.util

    spec = importlib.util.spec_from_file_location("black_october_triage", _TOOL)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


bot = _load()


def test_tool_module_is_importable():
    assert hasattr(bot, "main")
    assert hasattr(bot, "check_listening_sockets")


def test_parse_listener_entries_reads_addr_port_process():
    text = "0.0.0.0:1234|lmstudio\n127.0.0.1:8080|python\n"
    assert bot._parse_listener_entries(text) == [
        ("0.0.0.0", "1234", "lmstudio"),
        ("127.0.0.1", "8080", "python"),
    ]


def test_parse_listener_entries_tolerates_missing_process_column():
    assert bot._parse_listener_entries("192.168.1.5:445\n") == [
        ("192.168.1.5", "445", "")
    ]


def test_parse_listener_entries_skips_junk_lines():
    text = "Active Connections\n  Proto  Local Address\nnot-a-socket\n127.0.0.1:1|x"
    assert bot._parse_listener_entries(text) == [("127.0.0.1", "1", "x")]


def test_parse_listener_entries_strips_ipv6_brackets():
    assert bot._parse_listener_entries("[::1]:5939|node") == [("::1", "5939", "node")]


NETSTAT_SAMPLE = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       2700
  TCP    0.0.0.0:61236          0.0.0.0:0              LISTENING       20444
  TCP    192.168.1.173:139      0.0.0.0:0              LISTENING       4
  TCP    127.0.0.1:9000         0.0.0.0:0              LISTENING       111
  TCP    10.0.0.5:50000         10.0.0.9:443           ESTABLISHED     999
"""


def test_parse_listener_entries_reads_netstat_rows():
    entries = bot._parse_listener_entries(NETSTAT_SAMPLE)
    assert ("0.0.0.0", "135", "") in entries
    assert ("192.168.1.173", "139", "") in entries
    assert ("127.0.0.1", "9000", "") in entries


def test_parse_listener_entries_ignores_established_connections():
    ports = [port for _addr, port, _proc in bot._parse_listener_entries(NETSTAT_SAMPLE)]
    assert "50000" not in ports


def test_pid_name_map_parses_tasklist_csv(monkeypatch):
    csv = (
        '"lmlink-connector.exe","20444","Console","1","51.744 K"\n'
        '"svchost.exe","2700","Services","0","8 K"\n'
    )
    monkeypatch.setattr(bot, "sh", lambda _cmd: csv)
    assert bot._pid_name_map() == {
        "20444": "lmlink-connector.exe",
        "2700": "svchost.exe",
    }


def test_lmlink_connector_counts_as_lmstudio_ai_process():
    buckets = bot._classify_listeners([("0.0.0.0", "61236", "lmlink-connector.exe")])
    assert buckets["ai_exposed"] == ["0.0.0.0:61236 (lmlink-connector.exe)"]


def test_classify_ignores_loopback():
    buckets = bot._classify_listeners(
        [("127.0.0.1", "1234", "lmstudio"), ("::1", "1234", "ollama")]
    )
    assert buckets == {"wildcard": [], "nonloopback": [], "ai_exposed": []}


def test_classify_flags_exposed_ai_process_as_its_own_bucket():
    buckets = bot._classify_listeners(
        [
            ("0.0.0.0", "1234", "lmstudio"),
            ("192.168.1.173", "11434", "ollama"),
            ("0.0.0.0", "445", "system"),
        ]
    )
    assert sorted(buckets["ai_exposed"]) == [
        "0.0.0.0:1234 (lmstudio)",
        "192.168.1.173:11434 (ollama)",
    ]
    assert buckets["wildcard"] == ["0.0.0.0:1234 (lmstudio)", "0.0.0.0:445 (system)"]
    assert buckets["nonloopback"] == ["192.168.1.173:11434 (ollama)"]


def test_classify_does_not_treat_lan_service_as_ai():
    buckets = bot._classify_listeners([("192.168.1.173", "445", "system")])
    assert buckets["ai_exposed"] == []
    assert buckets["nonloopback"] == ["192.168.1.173:445 (system)"]


def test_check_listening_sockets_flags_exposed_ai(monkeypatch):
    monkeypatch.setattr(
        bot, "_listener_scan", lambda: "0.0.0.0:1234|lmstudio\n127.0.0.1:9|x\n"
    )
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_listening_sockets()
    names = [name for _sev, name, _detail in bot.FINDINGS]
    assert "ai-server-exposed" in names
    assert all(name != "listener-nonloopback" for name in names)


def test_check_listening_sockets_quiet_when_loopback_only(monkeypatch):
    monkeypatch.setattr(bot, "_listener_scan", lambda: "127.0.0.1:1234|lmstudio\n")
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_listening_sockets()
    assert bot.FINDINGS == []


def test_check_listening_sockets_ignores_os_owned_wildcard(monkeypatch):
    """Windows' own services listen on 0.0.0.0 by default - that is not a finding."""
    monkeypatch.setattr(
        bot, "_listener_scan",
        lambda: (
            "0.0.0.0:135|svchost.exe\n"
            "0.0.0.0:445|system\n"
            "0.0.0.0:49664|lsass.exe\n"
        ),
    )
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_listening_sockets()
    assert bot.FINDINGS == []


def test_check_listening_sockets_flags_third_party_wildcard(monkeypatch):
    monkeypatch.setattr(bot, "_listener_scan", lambda: "0.0.0.0:5000|python.exe\n")
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_listening_sockets()
    flags = {name: detail for _sev, name, detail in bot.FINDINGS}
    assert "listener-wildcard" in flags
    assert "0.0.0.0:5000 (python.exe)" in flags["listener-wildcard"]


@pytest.mark.parametrize(
    "version,patched",
    [("v24.22.1", True), ("v24.15.0", False), ("v22.23.4", True),
     ("v26.12.1", True), ("v25.5.0", False), ("v24.23.0", True)],
)
def test_node_patch_gate_against_node2shell(version, patched):
    ok, _note = bot._node_is_patched(version)
    assert ok is patched


def test_node_gate_rejects_unparseable_version():
    ok, note = bot._node_is_patched("vNaN")
    assert ok is False
    assert "unparseable" in note


def test_main_returns_finding_count(monkeypatch):
    monkeypatch.setattr(bot, "FINDINGS", [])
    for fn in (
        "check_startup_hooks", "check_litellm", "check_run_keys",
        "check_startup_folders", "check_scheduled_tasks", "check_runtimes",
        "check_python_brotli", "check_wallets", "check_ca_trust",
        "check_ai_runtimes", "check_listening_sockets",
    ):
        monkeypatch.setattr(bot, fn, lambda: None)
    assert bot.main() == 0
