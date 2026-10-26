# Black October — Local Exposure Triage & Fix Plan

Scope: this workstation (`DESKTOP-K3R3JAS`, `192.168.1.173`) as scanned by
`python tools/black_october_triage.py`. Exit code = number of findings (currently **17**).

Evidence basis: every item below comes from a live scan on this machine, not from
assumption. The *versions installed here* are directly observed.

Verification status of the external claims (checked 2026-10-26):

- **CVE-2026-111497 (Node2Shell) — confirmed.** Reported as fixed on 21 Oct 2026 in
  22.23.4, 24.22.1 and 26.12.1, matching the floors this scanner uses. Not in CISA's
  Known Exploited Vulnerabilities catalog, but a public PoC exists.
- **Node 25 — claim narrowed.** The advisory says only that EOL lines get no official
  fix. It does not name major 25. The scanner therefore reports "not a fixed release
  line" rather than "EOL"; treat Node 25 as unpatched, not as formally end-of-lifed.
- **CVE-2026-22020 (Java JFR) — UNVERIFIED.** No advisory was found for this ID. The
  installed Java 8 *is* observed and is worth upgrading on its own merits, but the CVE
  number attached to it in the scanner should be treated as unconfirmed.
- **Electrum supply-chain theft — partially confirmed.** Corroborated only at the
  "an Electrum theft happened in the wave" level; the installers/provenance angle was
  not independently verified.

## Priority stack

| # | Sev | Finding | Evidence (observed) | Why it ranks here |
|---|-----|---------|---------------------|-------------------|
| 1 | CRITICAL | Node 2 Shell — Node **24.15.0** below fixed 24.22.1 | `node --version` → `v24.15.0`; also `C:\Program Files\nodejs\node.exe` | `node` is on PATH, so any tool/edge-runtime that shells to node hits the vulnerable binary. Trivial remote trigger class, trivial fix. |
| 2 | HIGH | LM Studio **LM Link** connector listening on **0.0.0.0:61236** | `netstat -ano` → PID 20444 → `lmlink-connector.exe` at `~/.lmstudio/extensions/frameworks/lmlink-connector-win-x86_64-avx2-0.1.1/` | LLM endpoint reachable off-host on the LAN = prompt-injection → tool-execution surface. Auto-starts at login. |
| 3 | HIGH | Daedalus Mainnet desktop wallet installed | `C:\Program Files\Daedalus Mainnet` present | Irreversible-loss class. Nothing else on this list can cost you money directly. |
| 4 | HIGH | Bundled Node **25.5.0** — not a patched release line, **no fix available** | `Bionic\...\node.exe` and `LM Studio\...\node.exe` both v25.5.0 | The advisory patches only 22.23.4 / 24.22.1 / 26.12.1, so nothing on the 25 line is fixed. Cannot be patched by version bump; must be isolated or removed. |
| 5 | HIGH | Auto-start surface broad and unaudited | HKCU Run keys: `OneDrive`, `LM Studio … --run-as-service`, `Teams`; Startup folder: `Lemonade Server.lnk`, `Wispr Flow.lnk` | Each is a persistence+wake-up path that survives patching of #1/#4. LM Studio's is what exposes #2. |
| 6 | MEDIUM | Java **1.8.0_491** (JFR CVE class) | `java -version` → `1.8.0_491` | Ancient runtime, still on PATH. Only matters if something loads untrusted JARs. |
| 7 | MEDIUM | 3 scheduled tasks launch from a user-writable dir | `\OneDrive Reporting Task-…`, `\OneDrive Standalone Update Task-…`, `\OneDrive Startup Task-…`: actions in `%localappdata%` / `AppData\Local`, author `Microsoft Corporation` | All three are vendor-signed and benign in substance, so they rank last. They are listed only because the launch directory is the one a dropper uses; the 37 earlier "suspect task" hits were vendor/OS noise and are no longer reported. |
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

**Step 4 — Contain the Node 25.5.0 (kills #4).**
No fixed build exists on the 25 line. Decide per host: if Bionic is unused, uninstall it;
if needed, run it in a VM/container. Update LM Studio to a build bundling a fixed Node
(22.23.4 / 24.22.1 / 26.12.1), because you cannot edit its bundled runtime safely.

**Step 5 — Trim auto-start (kills #5).**
Keep only what you can name a reason for. Drop `--run-as-service` and any
Startup-folder `.lnk` you do not use (`Lemonade Server`, `Wispr Flow`).

**Step 6 — Low-priority hardening (#6, #7, #8).**
Upgrade or remove Java 8 if nothing needs it; confirm the three `user-writable` tasks
above are the vendor tasks they claim to be (the scanner now filters vendor/OS noise, so a
`CRITICAL scheduled-task-persistence` line means the dropper shape — user-writable path
plus LOLBin or download — and is not expected on this box); leave 139 alone unless you
have a reason to disable NetBIOS over TCP/IP.

## Harness notes

- `tools/black_october_triage.py` — the scanner. Listener attribution uses
  `netstat -ano` + `tasklist` because `Get-NetTCPConnection`'s process lookup fails
  on this box (`Get-Process` inside the pipeline errors out).
- `tests/test_black_october_triage.py` — 32 regression tests covering the parser,
  the `lmlink` AI-process match, the OS-owned-wildcard filter, the Node version gate,
  and the scheduled-task persistence classifier (`_parse_schtasks_verbose` /
  `_task_risk`) with explicit vendor-noise and OS-baseline regression cases.
- The scheduled-task check parses `schtasks /query /fo csv /nh /v` and inspects
  columns 1 (name), 7 (author) and 8 (action). It reports the persistence shape
  (action in a user-writable dir, plus a LOLBin or a second-stage fetch) and treats
  `%windir%` actions and `ProgramData\Microsoft` as OS baseline, so Defender's own
  rundll32 maintenance tasks are not findings.
- Both files are tracked in git. The scanner, its tests and this doc currently carry
  uncommitted changes (the scheduled-task classifier rewrite and the UTF-8 encoding
  pin); commit them before relying on this doc as the record.
- Known environmental failure, unrelated to these files:
  `tests/test_run_timeout.py::test_timeout_kills_process_tree` fails on this box
  because the strict run sandbox rejects `python -m x` before the timeout path is
  reached. It needs `AGENT_RUN_SANDBOX=dev` to exercise what it asserts.
