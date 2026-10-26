# Black October — Local Exposure Triage & Fix Plan

Scope: this workstation (`DESKTOP-K3R3JAS`, `192.168.1.173`) as scanned by
`python tools/black_october_triage.py`. Exit code = number of findings (currently **15**).

Evidence basis: every item below comes from a live scan on this machine, not from
assumption. The CVE IDs and "fixed version" floors are the ones carried in the triage
harness from the earlier research phase — re-confirm them against vendor advisories
before you broadcast them, but the *versions installed here* are directly observed.

## Priority stack

| # | Sev | Finding | Evidence (observed) | Why it ranks here |
|---|-----|---------|---------------------|-------------------|
| 1 | CRITICAL | Node 2 Shell — Node **24.15.0** below fixed 24.22.1 | `node --version` → `v24.15.0`; also `C:\Program Files\nodejs\node.exe` | `node` is on PATH, so any tool/edge-runtime that shells to node hits the vulnerable binary. Trivial remote trigger class, trivial fix. |
| 2 | HIGH | LM Studio **LM Link** connector listening on **0.0.0.0:61236** | `netstat -ano` → PID 20444 → `lmlink-connector.exe` at `~/.lmstudio/extensions/frameworks/lmlink-connector-win-x86_64-avx2-0.1.1/` | LLM endpoint reachable off-host on the LAN = prompt-injection → tool-execution surface. Auto-starts at login. |
| 3 | HIGH | Daedalus Mainnet desktop wallet installed | `C:\Program Files\Daedalus Mainnet` present | Irreversible-loss class. Nothing else on this list can cost you money directly. |
| 4 | HIGH | Bundled Node **25.5.0** — EOL major, **no official fix** | `Bionic\...\node.exe` and `LM Studio\...\node.exe` both v25.5.0 | Cannot be patched by version bump; must be isolated or removed. No fix path = persistent. |
| 5 | HIGH | Auto-start surface broad and unaudited | HKCU Run keys: `OneDrive`, `LM Studio … --run-as-service`, `Teams`; Startup folder: `Lemonade Server.lnk`, `Wispr Flow.lnk` | Each is a persistence+wake-up path that survives patching of #1/#4. LM Studio's is what exposes #2. |
| 6 | MEDIUM | Java **1.8.0_491** (JFR CVE class) | `java -version` → `1.8.0_491` | Ancient runtime, still on PATH. Only matters if something loads untrusted JARs. |
| 7 | MEDIUM | 37 scheduled tasks matching the suspect heuristic | `schtasks` scan, incl. `\GoogleUserPEH\*` | Mostly benign Google/OS updaters; needs one manual pass, not panic. |
| 8 | MEDIUM | NetBIOS/139 listening on LAN address | `192.168.1.173:139 (system)` | Windows default, but it is legacy SMB-era surface. |
| 9 | MEDIUM | Third-party wildcard listener | `0.0.0.0:61236 (lmlink-connector.exe)` | Same socket as #2; counted separately because other apps inherit the pattern. |
| — | ok | PHP absent | `php` not found | PHP 2 Shell leg not applicable. |
| — | ok | No `brotli` / `brotlicffi` | not in `pyproject.toml` deps, not in pip list | HTTP 2 Shell Python leg not applicable. |
| — | ok | No eMudhra CA in user Root store | `certutil -store -user Root` | Cert-compromise leg clear. |
| — | ok | `.env` never committed | `git log --all -- .env` empty; `.env` is gitignored | No secret leak via this repo. |

## Fix plan

Do 1 → 4 first; they are the only items that are both reachable and fixable today.

**Step 1 — Patch Node (kills #1).**
Update the PATH install and the `C:\Program Files\nodejs` copy to >= 24.22.1
(or move to 22.23.4 / 26.12.1, the other fixed lines). Then verify:
`python tools/black_october_triage.py` — the `[CRITICAL] node2shell` line must be gone.

**Step 2 — Take LM Link off the LAN (kills #2 and #9).**
In LM Studio: disable LM Link / "local service" exposure
(`~/.lmstudio/settings.json` has `enableLocalService: true`), and remove
`LM Studio.exe --run-as-service` from the HKCU Run key so it stops binding at login.
Verification: `netstat -ano | Select-String 61236` returns nothing, or shows `127.0.0.1` only.

**Step 3 — Move wallet funds (kills #3).**
No technical fix makes a hot wallet safe. Move long-term balances to hardware;
keep only spending money on the desktop wallet. If Daedalus is unused, uninstall it.

**Step 4 — Contain the EOL Node 25.5.0 (kills #4).**
There is no patch. Decide per host: if Bionic is unused, uninstall it; if needed,
run it in a VM/container. Update LM Studio to a build bundling a fixed Node, because
you cannot edit its bundled runtime safely.

**Step 5 — Trim auto-start (kills #5).**
Keep only what you can name a reason for. Drop `--run-as-service` and any
Startup-folder `.lnk` you do not use (`Lemonade Server`, `Wispr Flow`).

**Step 6 — Low-priority hardening (#6, #7, #8).**
Upgrade or remove Java 8 if nothing needs it; manually review the 37 scheduled
tasks (the tool deliberately over-reports); leave 139 alone unless you have a
reason to disable NetBIOS over TCP/IP.

## Harness notes

- `tools/black_october_triage.py` — the scanner. Listener attribution uses
  `netstat -ano` + `tasklist` because `Get-NetTCPConnection`'s process lookup fails
  on this box (`Get-Process` inside the pipeline errors out).
- `tests/test_black_october_triage.py` — 24 regression tests covering the parser,
  the `lmlink` AI-process match, the OS-owned-wildcard filter and the Node version gate.
- Both files are **untracked** in git at the moment.
