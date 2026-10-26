"""Black October triage: local persistence + supply-chain IoC checks (read-only).

Checks the IoCs disclosed for the Black October wave (Oct 19-22 2026):
 - LiteLLM 1.107.0 supply chain: sitecustomize/usercustomize startup hooks,
   litellm presence, hourly RCE loader persistence.
 - Generic payload-drop persistence (Run keys, Startup folders, scheduled tasks).
 - Runtime exposure: PHP/Node presence for PHP2Shell / Node2Shell.

Run: python tools/black_october_triage.py   (exit code = number of findings)
"""

import pathlib
import subprocess
import sys

FINDINGS: list[tuple[str, str, str]] = []


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
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


def check_scheduled_tasks() -> None:
    out = sh("schtasks /query /fo csv")
    sus = [ln for ln in out.splitlines()
           if any(k in ln.lower() for k in ("updater", "telemetry", "helper", "sync",
                                            "host", "adobupdtmr", "winupdate",
                                            "appdata", "temp"))]
    if sus:
        flag("MEDIUM", "scheduled-tasks-suspect",
             f"{len(sus)} suspicious task(s): " + " | ".join(sus[:6]))
    print(f"[ok] scheduled tasks scanned ({len(out.splitlines())} lines)")


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
            f"node {core}: major {major} has NO official fix (EOL) - isolate or remove"
        )
    fx = tuple(int(x) for x in fixed.split("."))
    if (major, minor, patch) >= fx:
        return True, f"node {core} >= fixed {fixed}"
    return False, f"node {core} < fixed {fixed} - VULNERABLE to CVE-2026-111497"


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
        print("[ok] php not present -> PHP2Shell CVE-2026-109618/19 not applicable")
    else:
        flag("HIGH", "php-runtime-present",
             f"php --version -> {out.strip()[:80]} (PHP2Shell CVE-2026-109618/19: "
             "upgrade to 8.5.12/8.4.27/8.3.36/8.2.35)")

    out = sh("node --version 2>&1")
    if "not found" in out.lower() or "not recognized" in out.lower() or not out.strip():
        print("[ok] node not on PATH -> Node2Shell not applicable via PATH")
    else:
        patched, note = _node_is_patched(out.strip())
        if patched:
            print(f"[ok] {note}")
        else:
            flag("CRITICAL", "node2shell", f"PATH node: {note}")

    for exe in _bundled_node():
        out = sh(f'"{exe}" --version 2>&1')
        if not out.strip() or "not recognized" in out.lower():
            flag("MEDIUM", "bundled-node-unreadable",
                 f"{exe} present but version unreadable in sandbox - check manually "
                 "against Node2Shell fixed versions 22.23.4/24.22.1/26.12.1")
            continue
        patched, note = _node_is_patched(out.strip())
        if patched:
            print(f"[ok] {exe}: {note}")
        else:
            flag("HIGH", "bundled-node2shell", f"{exe}: {note}")


def check_python_brotli() -> None:
    """HTTP2Shell/Python2Shell: requests/urllib3/aiohttp auto-use brotli if installed.

    Fixed: Python Brotli 1.2.1, brotlicffi 1.2.1.0 (per Black October advisories).
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
            "-> HTTP2Shell Python leg not applicable"
        )
        return
    for ln in lines:
        parts = ln.split()
        name, ver = parts[0], parts[1] if len(parts) > 1 else "?"
        flag("HIGH", "python-brotli-present",
             f"{name} {ver} - HTTP2Shell CVE-2026-111497: upgrade to "
             "Brotli>=1.2.1 / brotlicffi>=1.2.1.0, then restart all Python services")


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
        flag("MEDIUM", "java-present",
             f"java --version -> {java.strip().splitlines()[0][:60]} "
             "(CVE-2026-22020 Java Flight Recorder: patch to latest CPU)")


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
    check_wallets()
    check_ca_trust()
    check_ai_runtimes()
    check_listening_sockets()
    print(f"\nTOTAL FINDINGS: {len(FINDINGS)}")
    return len(FINDINGS)


if __name__ == "__main__":
    sys.exit(min(main(), 125))
