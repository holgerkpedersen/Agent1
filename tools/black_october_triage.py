"""Black October triage: local persistence + supply-chain IoC checks (read-only).

Checks the IoCs disclosed for the Black October wave (Oct 19-22 2026):
 - LiteLLM 1.107.0 supply chain: sitecustomize/usercustomize startup hooks,
   litellm presence, hourly RCE loader persistence.
 - Generic payload-drop persistence (Run keys, Startup folders, scheduled tasks).
 - Runtime exposure: PHP (CVE-2026-109618/19) and the bundled Brotli decoder
   (CVE-2026-111497) inside Node and other native runtimes.

Run: python tools/black_october_triage.py   (exit code = number of findings)
"""

import csv
import pathlib
import re
import subprocess
import sys

FINDINGS: list[tuple[str, str, str]] = []


def sh(cmd: str) -> str:
    """Run *cmd* and return stdout+stderr as text.

    ``encoding`` is pinned deliberately: with ``text=True`` alone Windows decodes
    child output using the locale codepage, which mojibakes or crashes on task
    names and paths containing non-ASCII characters.  See
    tests/test_subprocess_utf8_encoding.py, which enforces this repo-wide.
    """
    r = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    return (r.stdout or "") + (r.stderr or "")


def flag(sev: str, name: str, detail: str) -> None:
    FINDINGS.append((sev, name, detail))
    print(f"[{sev}] {name}: {detail}")


def check_startup_hooks() -> None:
    dirs = [p for p in sys.path if p and pathlib.Path(p).is_dir()]
    for d in dirs:
        for f in ("sitecustomize.py", "usercustomize.py"):
            p = pathlib.Path(d) / f
            if p.exists():
                sev = "HIGH" if f == "sitecustomize.py" else "MEDIUM"
                flag(sev, "python-startup-hook",
                     f"{p} exists (LiteLLM 1.107.0 stealer hid here) - inspect content")
    print(f"[ok] startup-hook scan done over {len(dirs)} site dirs")


def check_litellm() -> None:
    out = sh("python -m pip show litellm")
    if "Version:" in out:
        ver = [ln for ln in out.splitlines() if ln.startswith("Version:")][0]
        flag("CRITICAL", "litellm-installed",
             f"litellm present ({ver}) - Black October release compromise hit 1.107.0; "
             "verify version and provenance")
    else:
        print("[ok] litellm not installed")


def check_run_keys() -> None:
    for hive in ("HKCU", "HKLM"):
        run_key = f"{hive}\\Software\\Microsoft\\Windows\\CurrentVersion\\Run"
        out = sh(f'reg query "{run_key}"')
        if "ERROR" in out.upper() and "cannot find" in out.lower():
            print(f"[ok] {hive} Run key empty/absent")
            continue
        for line in out.splitlines():
            low = line.lower()
            keys = ("temp", "appdata", "roaming", "powershell", "pythonw", "mshta",
                    "rundll32", "certutil", "bitsadmin", "wscript", "cscript",
                    "schtasks", "svch0st", "update")
            if any(k in low for k in keys):
                flag("HIGH", f"{hive}-run-key-suspect", line.strip())
        print(f"[ok] {hive} Run key scanned")


def check_startup_folders() -> None:
    startup = "Microsoft/Windows/Start Menu/Programs/Startup"
    roots = (
        pathlib.Path.home() / "AppData/Roaming" / startup,
        pathlib.Path(f"C:/ProgramData/{startup}"),
    )
    for d in roots:
        if d.is_dir():
            entries = [e.name for e in d.iterdir()]
            if entries:
                flag("MEDIUM", "startup-folder", f"{d}: {entries}")
            else:
                print(f"[ok] startup folder empty: {d}")


#: Directories a dropper can write to without elevation.  A scheduled task whose
#: action lives here is the Black October persistence shape (payload drop + task).
#: ``programdata\\microsoft`` is deliberately absent: Defender's own platform
#: binaries live there and match every other heuristic in this file.
USER_WRITABLE = (
    "\\appdata\\",
    "\\temp\\",
    "\\users\\public\\",
    "\\downloads\\",
    "\\programdata\\",
    # Tasks often carry the unexpanded variable instead of a literal path, e.g.
    # "%localappdata%\Microsoft\OneDrive\...".  Matching only literal paths
    # silently missed those.
    "%appdata%",
    "%localappdata%",
    "%temp%",
    "%tmp%",
    "%public%",
)
#: Vendor sub-trees under ProgramData that belong to the OS, not to a dropper.
OS_PROGRAMDATA = ("\\programdata\\microsoft\\",)

#: LOLBins used by the wave's loader stages to hide the real payload.  Only an
#: LOLBin used OUTSIDE the Windows directory is a finding - Windows ships dozens
#: of rundll32-based maintenance tasks that would otherwise all flag.
LOLBINS = (
    "powershell",
    "pwsh",
    "cmd.exe /c",
    "cmd /c",
    "wscript",
    "cscript",
    "mshta",
    "rundll32",
    "regsvr32",
    "certutil",
    "bitsadmin",
    "curl ",
    "wget ",
)

#: Action prefixes (after variable expansion) that mean "this is an OS binary".
OS_ACTION_PREFIXES = ("%windir%", "%systemroot%", "%systemdrive%\\windows",
                      "c:\\windows\\")

#: Task actions that fetch or decode a second stage.
NETWORK_OR_ENCODE = ("http://", "https://", "-enc", "-encodedcommand", "frombase64",
                     "invoke-expression", "iex", "downloadstring", "bitsadmin")


def _parse_schtasks_verbose(text: str) -> list[dict[str, str]]:
    """Parse ``schtasks /query /fo csv /nh /v`` rows into task dicts.

    Columns (0-based) as emitted by Windows: 1 TaskName, 2 Next Run Time,
    3 Status, 7 Author, 8 Task To Run, 11 Scheduled Task State, 18 Schedule
    Type.  Rows shorter than 9 fields are skipped rather than indexed, so this
    cannot raise on localized output; the two trailing fields are read only
    when present and default to "" so a 9-field row still parses.
    """
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith('"'):
            continue
        fields = next(csv.reader([line]), [])
        if len(fields) < 9:
            continue
        rows.append(
            {
                "name": fields[1].strip(),
                "status": fields[3].strip(),
                "author": fields[7].strip(),
                "action": fields[8].strip(),
                "state": fields[11].strip() if len(fields) > 11 else "",
                "schedule_type": fields[18].strip() if len(fields) > 18 else "",
            }
        )
    return rows


#: Triggers that mean "this task brings the app back by itself" rather than
#: "this task does periodic maintenance".  Matched against the CSV's Schedule
#: Type column, so no per-task XML call is needed.
AUTOSTART_TRIGGERS = ("at logon", "at startup", "on workstation unlock")

#: Apps whose autostart has to be removed in *every* place it is registered.
#: OneDrive was found in three (Run key, AutoStartEnabled, logon task); removing
#: only the Run key looked like a fix and silently reverted, because the logon
#: task re-launched the client on the next sign-in.
AUTOSTART_APPS = {
    "onedrive": "OneDrive",
    "teams": "Teams",
    "lmlink": "LM Link connector",
    "lm studio": "LM Studio",
    "lemonade": "Lemonade Server",
}


def _autostart_restart_risk(task: dict[str, str]) -> tuple[str, str] | None:
    """Flag an *enabled* logon/startup task that relaunches a known autostart app.

    This is the gap that made the OneDrive removal look complete when it was not:
    ``check_run_keys`` reported the Run value, and nothing looked at the logon
    task that restarted the same client.  A vendor that ships both is not
    malicious, so the severity stays MEDIUM - it is an autostart-completeness
    finding, not a persistence finding.
    """
    if task.get("state", "").strip().lower() == "disabled":
        return None  # already handled; re-reporting it is noise
    raw_trigger = task.get("schedule_type", "").strip()
    trigger = next((t for t in AUTOSTART_TRIGGERS if t in raw_trigger.lower()), None)
    if trigger is None:
        return None
    blob = f"{task.get('name', '')} {task.get('action', '')}".lower()
    app = next((label for key, label in AUTOSTART_APPS.items() if key in blob), None)
    if app is None:
        return None
    return "MEDIUM", (
        f"{task.get('name', '')}: enabled task with trigger '{raw_trigger}' relaunches "
        f"{app} - removing the Run key alone will NOT stop it starting; disable the "
        f"task too | {task.get('action', '')[:120]}"
    )


def _task_risk(task: dict[str, str]) -> tuple[str, str] | None:
    """Return (severity, reason) for a suspicious task, else None.

    Two rounds of noise reduction are baked in here, both learned from real
    output on this host:

    * The first version substring-matched "helper"/"sync"/"updater" across the
      whole CSV row and flagged 37 healthy vendor tasks.
    * A naive "action mentions rundll32" rule then flagged ~20 of Windows' own
      maintenance tasks (Defender, Appx, Sysmain), and treating ProgramData as
      user-writable flagged Defender's platform binaries as CRITICAL.

    So an action that starts in ``%WINDIR%``/``%SystemRoot%`` is treated as an OS
    binary (its LOLBin use is baseline), and ProgramData\\Microsoft is not
    "user-writable".
    """
    name = task.get("name", "")
    author = task.get("author", "").lower()
    action = task.get("action", "")
    low_action = action.lower()

    os_action = low_action.startswith(OS_ACTION_PREFIXES)
    os_owned_dir = any(d in low_action for d in OS_PROGRAMDATA)

    droppable = next((d for d in USER_WRITABLE if d in low_action), None)
    if droppable and os_owned_dir:
        droppable = None
    fetch = next((n for n in NETWORK_OR_ENCODE if n in low_action), None)
    lolbin = next((b for b in LOLBINS if b in low_action), None)
    if lolbin and os_action:
        # rundll32 from %windir% is how Windows itself runs maintenance tasks.
        lolbin = None
    known_vendor = any(
        v in author
        for v in ("microsoft", "google", "adobe", "mozilla", "amd", "intel",
                  "nvidia", "apple", "dropbox", "spotify", "opera")
    )

    reasons = []
    if droppable:
        label = droppable.replace(chr(92), "")
        reasons.append(f"action in user-writable dir ({label})")
    if lolbin:
        reasons.append(f"LOLBin in action ({lolbin.strip()})")
    if fetch:
        reasons.append(f"fetches/decodes second stage ({fetch.strip()})")
    if not reasons:
        return None

    # The wave's persistence shape is a payload in user-writable space that is
    # launched by a shell and/or pulls a second stage.  One weak signal from a
    # known vendor stays MEDIUM so the CRITICALs keep meaning something.
    strong = bool(droppable and (fetch or lolbin)) or bool(fetch and not os_action)
    if strong:
        severity = "CRITICAL" if (droppable or not known_vendor) else "HIGH"
    elif known_vendor:
        severity = "MEDIUM"
    else:
        severity = "HIGH"
    return severity, f"{name}: " + ", ".join(reasons) + f" | {action[:160]}"


def check_scheduled_tasks() -> None:
    """Find tasks matching the wave's persistence shape, without vendor noise."""
    out = sh("schtasks /query /fo csv /nh /v")
    tasks = _parse_schtasks_verbose(out)
    for task in tasks:
        verdict = _task_risk(task)
        if verdict is None:
            verdict = _autostart_restart_risk(task)
        if verdict is None:
            continue
        severity, reason = verdict
        flag(severity, "scheduled-task-persistence", reason)
    # A zero-task parse is almost never "this box has no tasks": it is the probe
    # failing.  Reporting that as clean is the false negative that hid the
    # OneDrive logon task (`os.walk` of C:\Windows\System32\Tasks raises
    # Access denied and yields 0 hits).  Say so instead of printing [ok].
    if not tasks:
        flag("MEDIUM", "scheduled-task-scan-empty",
             "schtasks returned no parsable tasks - the task scan is BLIND, not "
             "clean; re-run elevated before trusting any absence of findings")
        return
    print(f"[ok] scheduled tasks scanned ({len(tasks)} task(s) parsed)")


FIXED_NODE = {"22": "22.23.4", "24": "24.22.1", "26": "26.12.1"}


def _node_is_patched(ver: str) -> tuple[bool, str]:
    """Return (patched, note) for a node version string like v24.15.0."""
    core = ver.lstrip("v").strip()
    try:
        major, minor, patch = (int(x) for x in core.split(".")[:3])
    except ValueError:
        return False, f"unparseable version {ver!r}"
    fixed = FIXED_NODE.get(str(major))
    if fixed is None:
        return False, (
            f"node {core}: major {major} is not a fixed release line - the advisory "
            f"patches only 22.23.4 / 24.22.1 / 26.12.1, so no fix exists on this "
            f"line; isolate or remove"
        )
    fx = tuple(int(x) for x in fixed.split("."))
    if (major, minor, patch) >= fx:
        return True, f"node {core} >= fixed {fixed}"
    return False, f"node {core} < fixed {fixed} - VULNERABLE to CVE-2026-111497"


#: CVE-2026-111497 is a Brotli *decoder* out-of-bounds write, not a Node-specific
#: flaw: Node inherits it by bundling Brotli 1.2.0.  Affected range is Brotli
#: 1.0.3 through 1.2.0; the fix landed in 1.0.10, 1.1.1 and 1.2.1.  Classifying
#: the Brotli version is therefore the real test - the Node version is only a proxy
#: and can disagree with reality (a backport may or may not have landed).
FIXED_BROTLI = {(1, 0, 10), (1, 1, 1), (1, 2, 1)}


def _brotli_is_patched(ver: str) -> tuple[bool, str]:
    """Return (patched, note) for a bundled Brotli version string like '1.2.0'."""
    core = (ver or "").strip().lstrip("v")
    parts = core.split(".")[:3]
    try:
        t = tuple(int(x) for x in parts)
    except ValueError:
        return False, f"unparseable brotli version {ver!r}"
    t = t + (0,) * (3 - len(t))
    if t in FIXED_BROTLI:
        return True, f"brotli {core} is a fixed release"
    if t < (1, 0, 3):
        return True, f"brotli {core} predates the affected range 1.0.3-1.2.0"
    if t <= (1, 2, 0):
        return False, (
            f"brotli {core} is in the affected range 1.0.3-1.2.0 "
            "(CVE-2026-111497, fixed in 1.2.1)"
        )
    return True, f"brotli {core} is newer than the affected range"


def _brotli_evidence(exe: str) -> str:
    """Return '; bundled <verdict>' read from a node binary, or '' if unreadable.

    Node exposes the exact packaged decoder version as process.versions.brotli,
    which is the fact that decides CVE-2026-111497 - not the Node version.

    The output is only trusted when it actually looks like a Brotli version.
    A Brotli release is major 0 or 1 (current fixed line is 1.2.1); anything else
    means we read a Node version or an error message, and inventing a verdict
    from that would be worse than saying nothing.
    """
    raw = sh(f'"{exe}" -p process.versions.brotli 2>&1').strip()
    if not raw:
        return ""
    val = raw.splitlines()[0].strip()
    if not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", val):
        return ""
    if int(val.split(".")[0]) > 1:
        return ""
    _ok, verdict = _brotli_is_patched(val)
    if "unparseable" in verdict:
        return ""
    return f"; bundled {verdict}"


def _bundled_node() -> list[str]:
    roots = (
        pathlib.Path.home() / "AppData/Local/Programs",
        pathlib.Path("C:/Program Files/nodejs"),
    )
    found: list[str] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("node.exe"):
            found.append(str(p))
            if len(found) >= 10:
                break
    return found


def check_runtimes() -> None:
    out = sh("php --version 2>&1")
    if "not found" in out.lower() or "not recognized" in out.lower() or not out.strip():
        print("[ok] php not present -> PHP CVE-2026-109618/109619 not applicable")
    else:
        flag("HIGH", "php-runtime-present",
             f"php --version -> {out.strip()[:80]} (PHP CVE-2026-109618/109619 "
             "9.2 criticals: upgrade to 8.5.12/8.4.27/8.3.36/8.2.35)")

    out = sh("node --version 2>&1")
    if "not found" in out.lower() or "not recognized" in out.lower() or not out.strip():
        print("[ok] node not on PATH -> Brotli CVE-2026-111497 not reachable via PATH")
    else:
        patched, note = _node_is_patched(out.strip())
        if patched:
            print(f"[ok] {note}")
        else:
            flag("CRITICAL", "node-brotli",
                 f"PATH node: {note}{_brotli_evidence('node')}")

    for exe in _bundled_node():
        out = sh(f'"{exe}" --version 2>&1')
        if not out.strip() or "not recognized" in out.lower():
            flag("MEDIUM", "bundled-node-unreadable",
                 f"{exe} present but version unreadable in sandbox - check manually "
                 "against Brotli decoder CVE-2026-111497 fixed versions "
                 "1.0.10/1.1.1/1.2.1 (Node patched lines: 22.23.4/24.22.1/26.12.1)")
            continue
        patched, note = _node_is_patched(out.strip())
        if patched:
            print(f"[ok] {exe}: {note}")
        else:
            flag("HIGH", "bundled-node-brotli",
                 f"{exe}: {note}{_brotli_evidence(exe)}")


def check_python_brotli() -> None:
    """Brotli CVE-2026-111497, Python leg.

    requests/urllib3/aiohttp decode Brotli automatically when a brotli module is
    importable, so a vulnerable Python Brotli is reachable from ordinary HTTP.

    Fixed: Python Brotli 1.2.1, brotlicffi 1.2.1.0.
    """
    out = sh("python -m pip list 2>&1")
    lines = [
        ln
        for ln in out.splitlines()
        if ln.lower().startswith(("brotli", "brotlicffi"))
    ]
    if not lines:
        print(
            "[ok] no python brotli/brotlicffi installed "
            "-> Brotli CVE-2026-111497 Python leg not applicable"
        )
        return
    for ln in lines:
        parts = ln.split()
        name, ver = parts[0], parts[1] if len(parts) > 1 else "?"
        flag("HIGH", "python-brotli-present",
             f"{name} {ver} - Brotli CVE-2026-111497: upgrade to "
             "Brotli>=1.2.1 / brotlicffi>=1.2.1.0, then restart all Python services")


def _brotli_cli_paths() -> list[pathlib.Path]:
    """Return standalone ``brotli.exe`` copies bundled by other apps.

    Node's bundled decoder is covered by ``check_runtimes`` via
    ``process.versions.brotli``.  Other apps (Git for Windows, OCR/AI bundles)
    ship a separate CLI whose decoder version is the decisive fact - and the
    mingw ``libbrotlidec.dll`` copies carry no version resource, so the CLI is
    the only reliable probe.

    The globs are bounded to one directory level on purpose: a recursive walk of
    ``Program Files`` is slow and would trip over permission-denied trees.
    """
    found: list[pathlib.Path] = []
    for root, pattern in (
        ("C:/Program Files", "*/mingw64/bin/brotli.exe"),
        ("C:/Program Files", "*/bin/brotli.exe"),
        ("C:/Program Files (x86)", "*/mingw64/bin/brotli.exe"),
    ):
        base = pathlib.Path(root)
        if not base.is_dir():
            continue
        try:
            found.extend(sorted(base.glob(pattern)))
        except OSError:
            continue  # unreadable tree - not a finding, just skip it
    extra = [
        pathlib.Path.home()
        / "AppData/Local/AMD/AI_Bundle/ComfyUI/Git/mingw64/bin/brotli.exe",
    ]
    found.extend(p for p in extra if p.exists())
    seen: set[str] = set()
    uniq: list[pathlib.Path] = []
    for p in found:
        key = str(p).lower()
        if key not in seen and p.exists():
            seen.add(key)
            uniq.append(p)
    return uniq


def check_native_brotli() -> None:
    """Brotli CVE-2026-111497 in native apps that ship their own brotli.exe.

    Affected range is Brotli 1.0.3-1.2.0; a 1.1.x copy needs 1.1.1, a 1.2.x copy
    needs 1.2.1.  Ranked below the Node finding because nothing here decodes
    attacker-supplied Brotli on its own - it takes someone running ``brotli -d``
    on hostile input.
    """
    paths = _brotli_cli_paths()
    if not paths:
        print("[ok] no standalone brotli.exe found -> native Brotli leg not applicable")
        return
    for exe in paths:
        out = sh(f'"{exe}" --version 2>&1')
        m = re.search(r"brotli\s+v?(\d+\.\d+(?:\.\d+)?)", out or "", re.I)
        if not m:
            flag("MEDIUM", "brotli-cli-unreadable",
                 f"{exe} present but version unreadable in sandbox - check manually "
                 "against the affected range 1.0.3-1.2.0 (CVE-2026-111497)")
            continue
        _patched, note = _brotli_is_patched(m.group(1))
        if _patched:
            print(f"[ok] {exe}: {note}")
        else:
            flag("MEDIUM", "brotli-cli-vulnerable",
                 f"{exe}: {note} - update the owning app so it stops shipping it")


def check_wallets() -> None:
    """Black October Electrum supply-chain theft: desktop crypto wallets are targets."""
    roots = [
        pathlib.Path.home() / "AppData/Roaming/Electrum",
        pathlib.Path.home() / "AppData/Local/Electrum",
        pathlib.Path.home() / "AppData/Roaming/Bitcoin",
        pathlib.Path.home() / "AppData/Roaming/Ethereum",
        pathlib.Path("C:/Program Files/Daedalus Mainnet"),
    ]
    found = [str(r) for r in roots if r.exists()]
    if found:
        flag("HIGH", "crypto-wallet-present",
             f"desktop wallet(s): {found} - Electrum supply-chain theft in "
             "the wave: verify installer provenance/checksums, move long-term "
             "funds to hardware wallet")
    else:
        print("[ok] no Electrum/Daedalus wallet dirs -> wallet-leg exposure limited")


def check_ca_trust() -> None:
    """Black October eMudhra certificate/key compromise: check Windows trust store."""
    out = sh('certutil -store -user Root 2>&1')
    hits = [ln.strip() for ln in out.splitlines()
            if "emudhra" in ln.lower() or "e-mudhra" in ln.lower()]
    if hits:
        flag("HIGH", "emudhra-ca-in-user-store",
             f"{len(hits)} hit(s): {hits[:3]} - eMudhra CA key compromise in the wave: "
             "remove/distrust, rely on OS store")
    else:
        print("[ok] no eMudhra CA in user Root store")


def check_ai_runtimes() -> None:
    """AI-agent attack leg: local model servers + auto-updaters (LM Studio, Ollama)."""
    lm = pathlib.Path.home() / "AppData/Local/LM Studio"
    if lm.exists():
        flag("MEDIUM", "lmstudio-service",
             f"{lm} present; LM Studio auto-starts --run-as-service "
             "(local LLM server = prompt-injection->tool-exec target: bind 127.0.0.1, "
             "update app so bundled Node >= 26.12.1)")
    oll = pathlib.Path.home() / "AppData/Local/Programs/Ollama"
    if oll.exists():
        flag("MEDIUM", "ollama-present",
             f"{oll} present - verify version, bind 127.0.0.1 only")
    java = sh('java -version 2>&1')
    if "not recognized" not in java.lower() and java.strip():
        # NOTE: the CVE previously cited here (CVE-2026-22020 "Java Flight
        # Recorder") returns zero records from the NVD API, i.e. it is not a
        # published identifier.  No advisory backs it, so it is not cited.
        # The old runtime is still worth upgrading on age alone -> LOW.
        flag("LOW", "java-present",
             f"java --version -> {java.strip().splitlines()[0][:60]} "
             "(old runtime - upgrade on general age/unsupported-CPU grounds; "
             "no specific advisory is attached to this finding)")


LOOPBACK = {"127.0.0.1", "::1", "localhost", ""}
WILDCARD = {"0.0.0.0", "::"}
# Process names that serve local LLM endpoints (Black October AI-server leg targets).
AI_PROCESS_HINTS = ("lmstudio", "lm studio", "lmlink", "ollama", "llama", "kobold",
                    "text-generation", "vllm", "localai", "lemonade")


def _parse_listener_entries(text: str) -> list[tuple[str, str, str]]:
    """Parse listener rows into (addr, port, process) tuples.

    Accepts two shapes so the scan survives a degraded environment:
      * netstat:  'TCP  0.0.0.0:135  0.0.0.0:0  LISTENING  2700'
      * custom:   '0.0.0.0:1234|lmstudio'
    Junk lines (headers, errors) are skipped.
    """
    entries: list[tuple[str, str, str]] = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or ":" not in ln:
            continue
        fields = ln.split()
        if len(fields) >= 5 and fields[0].upper() in ("TCP", "TCPV6"):
            if fields[3].upper() != "LISTENING":
                continue
            head, proc = fields[1], ""
        else:
            head, _, proc = ln.partition("|")
            head = head.strip()
        addr, _, port = head.rpartition(":")
        addr = addr.strip().strip("[]")
        if not port.strip().isdigit():
            continue
        entries.append((addr, port.strip(), proc.strip().lower()))
    return entries


def _pid_name_map() -> dict[str, str]:
    """Map PID -> image name (lowercase) via tasklist; empty when unavailable."""
    names: dict[str, str] = {}
    out = sh("tasklist /fo csv /nh")
    for line in out.splitlines():
        cols = [c.strip().strip('"') for c in line.split('","')]
        if len(cols) >= 2 and cols[1].isdigit():
            names[cols[1]] = cols[0].lower()
    return names


def _classify_listeners(entries: list[tuple[str, str, str]]) -> dict[str, list[str]]:
    """Bucket listeners into wildcard / non-loopback / exposed-AI-process.

    Only the exposed AI processes matter for the Black October AI-server leg, so
    they are kept separate from ordinary LAN-bound services.
    """
    buckets: dict[str, list[str]] = {
        "wildcard": [],
        "nonloopback": [],
        "ai_exposed": [],
    }
    for addr, port, proc in entries:
        if addr in LOOPBACK:
            continue
        label = f"{addr}:{port}" + (f" ({proc})" if proc else "")
        if addr in WILDCARD:
            buckets["wildcard"].append(label)
        else:
            buckets["nonloopback"].append(label)
        if proc and any(h in proc for h in AI_PROCESS_HINTS):
            buckets["ai_exposed"].append(label)
    return buckets


def _listener_scan() -> str:
    """Best-effort listener listing with owning process name (sandbox-safe).

    netstat -ano is the primary source: Get-NetTCPConnection returns data here but
    its -ExpandProperty process lookup fails, so netstat + tasklist is more robust.
    """
    net = sh("netstat -ano")
    if "LISTENING" in net.upper():
        names = _pid_name_map()
        rows = []
        for line in net.splitlines():
            fields = line.split()
            if len(fields) >= 5 and fields[0].upper() in ("TCP", "TCPV6") \
                    and fields[3].upper() == "LISTENING":
                rows.append(f"{fields[1]}|{names.get(fields[4], '')}")
        if rows:
            return "\n".join(rows)
    return sh("powershell -NoProfile -Command "
              "\"Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | "
              "ForEach-Object { \\\"{0}:{1}\\\" -f $_.LocalAddress, $_.LocalPort }\"")


def check_listening_sockets() -> None:
    """AI-server leg: local model endpoints bound to LAN/WAN are actively abused."""
    entries = _parse_listener_entries(_listener_scan())
    if not entries:
        print("[ok] listener scan unavailable - check manually "
              "(powershell: Get-NetTCPConnection -State Listen)")
        return
    buckets = _classify_listeners(entries)
    if buckets["ai_exposed"]:
        flag("HIGH", "ai-server-exposed",
             "local LLM endpoint(s) reachable off-host: "
             + ", ".join(sorted(set(buckets["ai_exposed"]))[:5])
             + " - bind to 127.0.0.1 (Black October AI-server leg: exposed LLM "
               "endpoints are used for prompt-injection -> tool execution)")
    # Wildcard ports owned by built-in Windows services are the OS default, not
    # a finding: only unnamed or third-party wildcard listeners are worth review.
    os_owned = {"system", "svchost.exe", "lsass.exe", "wininit.exe",
                "services.exe", "spoolsv.exe", "sshd.exe", "smss.exe"}
    wildcard = [w for w in buckets["wildcard"]
                if w.rsplit("(", 1)[-1].rstrip(")") not in os_owned]
    if wildcard:
        flag("MEDIUM", "listener-wildcard",
             f"{len(wildcard)} third-party listener(s) on 0.0.0.0/:: - e.g. "
             + ", ".join(sorted(set(wildcard))[:5]))
    if buckets["nonloopback"]:
        flag("MEDIUM", "listener-nonloopback",
             "non-loopback listen address(es): "
             + ", ".join(sorted(set(buckets["nonloopback"]))[:5]))
    if not buckets["ai_exposed"] and not wildcard and not buckets["nonloopback"]:
        print(f"[ok] listeners loopback-only ({len(entries)} sockets)")


def main() -> int:
    check_startup_hooks()
    check_litellm()
    check_run_keys()
    check_startup_folders()
    check_scheduled_tasks()
    check_runtimes()
    check_python_brotli()
    check_native_brotli()
    check_wallets()
    check_ca_trust()
    check_ai_runtimes()
    check_listening_sockets()
    print(f"\nTOTAL FINDINGS: {len(FINDINGS)}")
    return len(FINDINGS)


if __name__ == "__main__":
    sys.exit(min(main(), 125))
