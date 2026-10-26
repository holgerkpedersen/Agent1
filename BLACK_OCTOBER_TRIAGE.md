# Black October — Local Exposure Triage & Fix Plan

Scope: this workstation (`DESKTOP-K3R3JAS`, `192.168.1.173`) as scanned by
`python tools/black_october_triage.py`. Exit code = number of findings (currently **17**).

Evidence basis: every item below comes from a live scan on this machine, not from
assumption. The *versions installed here* are directly observed.

Verification status of the external claims.

**Read this first — the CVE identifiers are NOT trustworthy.** A second
verification pass on 2026-10-26 could not reach the network at all: `web_search`
returned zero results for every query, including `CVE-2021-44228 log4shell` and
`Node.js release schedule LTS versions` — controls that must return hits. That is a
tool/network failure, not evidence about the event. No CVE below was independently
re-confirmed in that pass, and the scanner prints these IDs as though they were
established fact. **Treat every CVE number here as unconfirmed.** What *is* observed
first-hand on this box is listed in the "Evidence" column of the priority stack.

- **CVE-2026-111497 (Node2Shell) — unconfirmed.** Recorded as fixed on 21 Oct 2026 in
  22.23.4, 24.22.1 and 26.12.1, matching the floors this scanner uses. Not in CISA's
  Known Exploited Vulnerabilities catalog, but a public PoC is claimed. The *version
  gates* are correct regardless; only the advisory provenance is unverified.
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
| 1 | CRITICAL | Node 2 Shell — Node **24.15.0** below fixed 24.22.1 | `where.exe node` → `C:\Program Files\nodejs\node.exe` → `v24.15.0` (re-verified 2026-10-26 16:16). `winget list --name Node.js`: installed 24.15.0, available **24.22.1** | One binary, not two: the PATH copy *is* the `Program Files` copy, so a single upgrade closes both. `node` is on PATH, so any tool/edge-runtime that shells to node hits the vulnerable binary. Trivial remote trigger class, trivial fix. |
| 2 | HIGH | LM Studio **LM Link** connector listening on **0.0.0.0:61236** | `netstat -ano` → `0.0.0.0:61236` LISTENING, PID 20444 (re-verified 2026-10-26 16:16). Running build `lmlink-connector-win-x86_64-avx2-0.1.1`; `settings.json` → `"enableLocalService": true` | LLM endpoint reachable off-host on the LAN = prompt-injection → tool-execution surface. Auto-starts at login via the `--run-as-service` Run key. Fixable in two minutes; see Step 2. |
| 3 | HIGH | Daedalus Mainnet desktop wallet installed | `C:\Program Files\Daedalus Mainnet` present | Irreversible-loss class. Nothing else on this list can cost you money directly. |
| 4 | HIGH | Bundled Node **25.5.0** — not a patched release line, **no fix available** | `Bionic\...\bin\node.exe` → `v25.5.0` and `LM Studio\...\bin\node.exe` → `v25.5.0` (both run, re-verified 16:16). `winget`: LM Studio `0.4.26+4`, **no newer build offered** | The advisory patches only 22.23.4 / 24.22.1 / 26.12.1, so nothing on the 25 line is fixed. There is no patch available *and* no app update that would deliver one — containment only (Step 4). |
| 5 | HIGH | Auto-start surface broad and unaudited | HKCU Run keys: `OneDrive`, `LM Studio … --run-as-service`, `Teams`; Startup folder: `Lemonade Server.lnk`, `Wispr Flow.lnk` | Each is a persistence+wake-up path that survives patching of #1/#4. LM Studio's is what exposes #2. |
| 6 | MEDIUM | Java **1.8.0_491** (JFR CVE class) | `java -version` → `1.8.0_491`. `winget`: installed 8.0.4910.10, available **8.0.5110.32** | Ancient runtime, still on PATH. Only matters if something loads untrusted JARs. The CVE number attached to it is unconfirmed (see above); the upgrade is justified on age alone. |
| 7 | MEDIUM | 3 scheduled tasks launch from a user-writable dir | `\OneDrive Reporting Task-…`, `\OneDrive Standalone Update Task-…`, `\OneDrive Startup Task-…`: actions in `%localappdata%` / `AppData\Local`, author `Microsoft Corporation` | All three are vendor-signed and benign in substance, so they rank last. They are listed only because the launch directory is the one a dropper uses; the 37 earlier "suspect task" hits were vendor/OS noise and are no longer reported. |
| 8 | MEDIUM | NetBIOS/139 listening on LAN address | `192.168.1.173:139 (system)` | Windows default, but it is legacy SMB-era surface. |
| 9 | MEDIUM | Third-party wildcard listener | `0.0.0.0:61236 (lmlink-connector.exe)` | Same socket as #2; counted separately because other apps inherit the pattern. |
| — | ok | PHP absent | `php` not found | PHP 2 Shell leg not applicable. |
| — | ok | No `brotli` / `brotlicffi` | not in `pyproject.toml` deps, not in pip list | HTTP 2 Shell Python leg not applicable. |
| — | ok | No eMudhra CA in user Root store | `certutil -store -user Root` | Cert-compromise leg clear. |
| — | ok | `.env` never committed | `git log --all -- .env` empty; `.env` is gitignored | No secret leak via this repo. |

## Fix plan

Do 1 → 4 first; they are the only items that are both reachable and fixable today.

**Step 1 — Patch Node (kills #1). One command, ~2 minutes.**
`winget show` already lists the fix as available: installed `24.15.0` → available
`24.22.1`. Run in an **elevated** shell (the target is under `C:\Program Files`):

```
winget upgrade --id OpenJS.NodeJS.LTS --accept-source-agreements
```

Then open a *new* shell (PATH cache) and verify all three ways:
1. `node --version` → `v24.22.1` or higher.
2. `python tools/black_october_triage.py` — the `[CRITICAL] node2shell` line is gone
   and the total drops from 17 to 16.
3. If a global npm install breaks, reinstall it; don't roll Node back.

This is the single highest-value action on this list: cheapest fix, reachable binary.

**Step 2 — Take LM Link off the LAN (kills #2 and #9). Highest severity after #1.**
Confirmed on this box right now: `netstat -ano` shows `0.0.0.0:61236` LISTENING under
PID 20444, and `~/.lmstudio/settings.json` contains `"enableLocalService": true`.
That setting is the cause; the `--run-as-service` Run key is what makes it start at login.

1. In LM Studio: Settings → disable **LM Link** (or "local service"), which flips
   `enableLocalService` to `false`. Quit LM Studio fully afterwards.
2. Kill the current listener: `Stop-Process -Id 20444` (verify the PID first — it changes
   on restart; the process is `lmlink-connector.exe`).
3. Remove the Run key that re-binds at login:
   `Remove-ItemProperty -Path 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Run' -Name 'electron.app.LM Studio'`

Verification (must show nothing, or `127.0.0.1` only, not `0.0.0.0`):
`netstat -ano | Select-String 61236`

Note: the connector directory also holds `lmlink-connector-win-x86_64-avx2-0.1.0` and
`harmony-win-x86_64-avx2-0.3.6` beside the `0.1.1` build that is running. Stale extension
dirs are not a vulnerability by themselves, but delete them so it is unambiguous which
build is live.

**Step 3 — Move wallet funds (kills #3).**
No technical fix makes a hot wallet safe. Move long-term balances to hardware;
keep only spending money on the desktop wallet. If Daedalus is unused, uninstall it.

**Step 4 — Contain the Node 25.5.0 (kills #4). No update path exists.**
`winget list` shows **LM Studio 0.4.26+4 with no "available" column** — you are already
on the newest build, and it still bundles Node 25.5.0. So "just update LM Studio" is not
an option; my earlier plan overstated it. Bionic has `1.1.9+6` available (from `1.0.7+2`),
but its bundled runtime is also 25.5.0, so upgrading is unlikely to change the Node line —
verify after upgrading with `Bionic\...\bin\node.exe --version`.

Because there is no patch, the only real options are containment:
- Uninstall **Bionic** if you do not use it (it is one of two 25.5.0 copies).
- For LM Studio, accept the risk deliberately: the bundled Node only matters if a model
  or plugin can reach it. Keep `developer.allowDevelopmentPlugins` and
  `autoUpdateExtensionPacks` in mind — you currently have **both enabled**.
  Turning off `allowDevelopmentPlugins` narrows that path at zero cost if you don't
  write LM Studio plugins.
- Do **not** hand-replace the bundled `node.exe`; you break app signing and the next
  auto-update reverts it.

**Step 5 — Trim auto-start (kills #5).**
Keep only what you can name a reason for. Drop `--run-as-service` and any
Startup-folder `.lnk` you do not use (`Lemonade Server`, `Wispr Flow`).

**Step 6 — Low-priority hardening (#6, #7, #8).**
Java: `winget upgrade --id Oracle.JavaRuntimeEnvironment` takes `8.0.4910.10` →
`8.0.5110.32`. Only do this if something on the box actually needs Java — if not, remove
it instead, which is strictly better than patching an 8.x runtime.
Tasks: confirm the three `user-writable` tasks are the vendor tasks they claim to be (the
scanner filters vendor/OS noise, so a `CRITICAL scheduled-task-persistence` line means the
dropper shape — user-writable path plus LOLBin or download — and is not expected here).
139: leave NetBIOS alone unless you have a reason to disable NetBIOS over TCP/IP.

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
- Both files are tracked in git and committed: `00068cd` (scanner + this report) and
  `ead8012` (CVE-claim verification). The scheduled-task classifier rewrite and the
  UTF-8 encoding pin from earlier notes landed in `ead8012`, so this doc is the record.
  These commits are local only — push them or CI never sees them.
- Known environmental failure, unrelated to these files:
  `tests/test_run_timeout.py::test_timeout_kills_process_tree` fails on this box
  because the strict run sandbox rejects `python -m x` before the timeout path is
  reached. It needs `AGENT_RUN_SANDBOX=dev` to exercise what it asserts.
