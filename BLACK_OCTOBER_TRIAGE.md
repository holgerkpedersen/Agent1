# Black October — Local Exposure Triage & Fix Plan

Scope: this workstation (`DESKTOP-K3R3JAS`, `192.168.1.173`) as scanned by
`python tools/black_october_triage.py`. Exit code = number of findings (currently **18**,
re-counted 2026-10-26 after the OneDrive autostart removal and the disabled-task demotion).

Evidence basis: every item below comes from a live scan on this machine, not from
assumption. The *versions installed here* are directly observed.

Verification status of the external claims.

**Superseded: the CVE identifiers are now confirmed against authoritative sources.**
An earlier pass could not reach the network (`web_search` returned zero results even
for `CVE-2021-44228 log4shell`, a control that must return hits). The cause was the
search tool, not the network: the NVD REST API and `nodejs.org` are both reachable
directly. Method: query each ID against
`services.nvd.nist.gov/rest/json/cves/2.0?cveId=…` with `CVE-2021-44228` as a
positive control. The control returned `totalResults=1`, so a `0` result below is a
real absence, not a failed fetch. CISA's KEV feed was downloaded as well
(catalog size 1756).

**The headline correction: CVE-2026-111497 is not "Node2Shell". It is a Brotli
decoder bug that Node inherits by bundling Brotli.** The scanner misnamed it.

- **CVE-2026-111497 — CONFIRMED, but re-scoped.** NVD record exists, published
  `2026-10-22T14:17`, CVSS v4.0 **9.2 CRITICAL**. Description: buffer overflow in the
  **Brotli decoder** (`c/dec/decode.c`), affected range **Brotli 1.0.3 → 1.2.0**,
  fixed in **1.0.10 / 1.1.1 / 1.2.1**. Impact: memory disclosure, DoS, potential RCE
  from a crafted Brotli stream. Node merely bundles the library, so Node's own page
  rates the issue **high** while NVD rates it **9.2 critical** — the scanner adopts
  NVD's score. Not in KEV.
- **CVE-2026-109618 / CVE-2026-109619 — CONFIRMED (PHP).** Both `9.2` CRITICAL, both
  `Analyzed`: an out-of-memory `longjmp()` path that leaves request memory
  uninitialized, and stale global state pointing into freed request memory; each yields
  RCE. Fixed floors: `8.2.35`, `8.3.36`, `8.4.27`, `8.5.12`. **PHP is not installed
  here, so this leg does not apply.** Not in KEV.
- **CVE-2026-22020 (Java "Flight Recorder") — DOES NOT EXIST.** `totalResults=0`.
  No advisory, no record. The scanner cited it as fact; that citation has been removed
  and the Java finding is now `LOW` on age grounds alone.
- **None of the four IDs is in CISA KEV.** So the "ongoing, actively exploited" framing
  does not hold for these specific CVEs; treat them as patching debt, not as an active
  intrusion.
- **Node 25 — claim narrowed.** The advisory patches only 22.23.4 / 24.22.1 / 26.12.1.
  Major 25 is not among them, so no fix exists on that line. That is not the same as
  saying 25 is formally end-of-lifed.
- **Electrum supply-chain theft — still only partially confirmed.** The installer
  provenance angle was not independently verified; the wallet-presence finding stands
  on observation alone.

## Out-of-band finding — higher priority than everything below

This document covers **device-local** exposure. It is not the top of the overall
priority stack: a personal-data finding outranks all of it, because every item below
still requires an attacker to reach this machine, whereas a public home address needs
no exploit at all. See **`PII_EXPOSURE_TRIAGE.md`** — the Danish CVR filing that
published a residential address, plus the CPR § 28 / CVR suppression plan.

The numbering below is unchanged and remains relative to device-local risk only.

## Priority stack

| # | Sev | Finding | Evidence (observed) | Why it ranks here |
|---|-----|---------|---------------------|-------------------|
| 1 | CRITICAL | **Brotli decoder 1.2.0** (CVE-2026-111497) inside Node **24.15.0** on PATH | `node -p process.versions.brotli` → **`1.2.0`** on all three node binaries (re-verified this pass). `where.exe node` → `C:\Program Files\nodejs\node.exe` → `v24.15.0`. `winget`: installed 24.15.0, available **24.22.1** | The defect is the **bundled Brotli**, not Node: affected range is Brotli 1.0.3–1.2.0 and this box runs 1.2.0 in every binary. Path to fix is still a Node upgrade (24.22.1 ships 1.2.1). One binary, not two: the PATH copy *is* the `Program Files` copy. `node` is on PATH, so any tool that shells to node loads the vulnerable decoder. |
| 2 | HIGH | LM Studio **LM Link** connector listening on **0.0.0.0:61236** | `netstat -ano` → `0.0.0.0:61236` LISTENING, PID 20444 (re-verified 2026-10-26 16:16). Running build `lmlink-connector-win-x86_64-avx2-0.1.1`; `settings.json` → `"enableLocalService": true` | LLM endpoint reachable off-host on the LAN = prompt-injection → tool-execution surface. Auto-starts at login via the `--run-as-service` Run key. Fixable in two minutes; see Step 2. |
| 3 | HIGH | Daedalus Mainnet desktop wallet installed | `C:\Program Files\Daedalus Mainnet` present | Irreversible-loss class. Nothing else on this list can cost you money directly. |
| 4 | HIGH | Bundled Node **25.5.0** also carries Brotli **1.2.0** — no patched 25 line | `Bionic\...\bin\node.exe` → `v25.5.0`, brotli **1.2.0**; `LM Studio\...\bin\node.exe` → `v25.5.0`, brotli **1.2.0**. `winget`: LM Studio `0.4.26+4`, **no newer build offered** | Same affected Brotli 1.2.0 as #1, but the advisory patches only 22.23.4 / 24.22.1 / 26.12.1, so nothing on the 25 line is fixed and no app update delivers one — containment only (Step 4). |
| 5 | HIGH | Auto-start surface broad and unaudited | HKCU Run keys: `LM Studio … --run-as-service`, `Teams`; Startup folder: `Lemonade Server.lnk`, `Wispr Flow.lnk`. `OneDrive` **removed 2026-10-26** (§ Applied fixes) | Each is a persistence+wake-up path that survives patching of #1/#4. LM Studio's is what exposes #2. |
| 6 | LOW | Java **1.8.0_491** — old, not linked to any published CVE | `java -version` → `1.8.0_491`. `winget`: installed 8.0.4910.10, available **8.0.5110.32** | Downgraded from MEDIUM: the CVE previously cited for it (CVE-2026-22020) returns **zero NVD records**, so there is no advisory. Still on PATH and ancient, so upgrading (or removing) is justified on age alone. |
| 7 | LOW | 3 scheduled tasks launch from a user-writable dir | `\OneDrive Reporting Task-…`, `\OneDrive Standalone Update Task-…`, `\OneDrive Startup Task-…`: actions in `%localappdata%` / `AppData\Local`, author `Microsoft Corporation`. **`OneDrive Startup Task` disabled 2026-10-26** (§ Applied fixes); the two updaters remain Enabled | Now split by state, re-verified live: the **disabled** Startup Task reports `[LOW] scheduled-task-persistence: … task is Disabled - not currently active`, while the **two enabled** updaters stay `[MEDIUM]`. Before that change all three printed at MEDIUM, so the applied fix read as an open finding. All three are vendor-signed and benign in substance, so they rank last; the 37 earlier "suspect task" hits were vendor/OS noise and are no longer reported. The Startup Task mattered only because it would relaunch OneDrive at logon. |
| 8 | MEDIUM | NetBIOS/139 listening on LAN address | `192.168.1.173:139 (system)` | Windows default, but it is legacy SMB-era surface. |
| 9 | MEDIUM | Third-party wildcard listener | `0.0.0.0:61236 (lmlink-connector.exe)` | Same socket as #2; counted separately because other apps inherit the pattern. |
| 10 | MEDIUM | **Git for Windows** ships Brotli **1.2.0**; a second Git under `AMD\AI_Bundle\ComfyUI` ships **1.1.0** | `Git\mingw64\bin\brotli.exe --version` → **`brotli 1.2.0`**; `ComfyUI\Git\mingw64\bin\brotli.exe --version` → **`brotli 1.1.0`** (both re-verified this pass). `Tesseract-OCR\libbrotlidec.dll` is also present but has **no version resource**, so its version is unknown | Both are inside the affected range 1.0.3–1.2.0 — 1.1.0 needs **1.1.1**, not 1.2.1. Ranked below Node because nothing here decodes attacker-supplied Brotli automatically: it needs a user or script to run `brotli -d` on hostile input. Fix is a Git for Windows update (covers both copies). |
| — | ok | PHP absent | `php` not found | CVE-2026-109618/109619 (PHP 9.2 criticals) not applicable — no PHP on this box. |
| — | ok | No `brotli` / `brotlicffi` in Python | `importlib.util.find_spec` → all absent | Python-side Brotli decode leg not applicable. |
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

Then open a *new* shell (PATH cache) and verify three ways:
1. `node --version` → `v24.22.1` or higher.
2. `node -p process.versions.brotli` → **`1.2.1`** or higher. This is the check that
   actually proves the fix: the vulnerability is in the Brotli decoder, so a 24.22.1
   label with a still-1.2.0 Brotli would mean the fix did not land.
3. `python tools/black_october_triage.py` — both `node 24.15.0` lines are gone (the
   `[CRITICAL] node-brotli` PATH line and the `[HIGH] bundled-node-brotli` line for
   `C:\Program Files\nodejs\node.exe`, which is the same binary counted twice) and the
   total drops from **18 to 16**.

If a global npm install breaks, reinstall it; don't roll Node back.

This is the single highest-value action on this list: cheapest fix, reachable binary.

Naming note: this CVE is a **Brotli** decoder bug that Node inherits. The NVD
reference list includes `broccoli2shell.com`, so the campaign name circulating for it
is **Broccoli2Shell** — not the "Node2Shell" the earlier scan printed. The scan now
says `node-brotli` and names the component, so a wrong product name cannot send you
looking for the wrong patch.

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

**Step 6 — Low-priority hardening (#6, #7, #8, #10).**
Java: `winget upgrade --id Oracle.JavaRuntimeEnvironment` takes `8.0.4910.10` →
`8.0.5110.32`. Only do this if something on the box actually needs Java — if not, remove
it instead, which is strictly better than patching an 8.x runtime.
Tasks: confirm the three `user-writable` tasks are the vendor tasks they claim to be (the
scanner filters vendor/OS noise, so a `CRITICAL scheduled-task-persistence` line means the
dropper shape — user-writable path plus LOLBin or download — and is not expected here).
139: leave NetBIOS alone unless you have a reason to disable NetBIOS over TCP/IP.
Brotli CLIs (#10): `winget upgrade --id Git.Git` replaces the `1.2.0` copy with whatever
the current Git for Windows bundles — re-check afterwards with `brotli.exe --version`, and
do not assume the update lands on 1.2.1 or later. The `ComfyUI` copy is `1.1.0` and only
moves if you update that AMD AI bundle; if you no longer use it, deleting the bundle is
cheaper. `Tesseract-OCR\libbrotlidec.dll` has no version resource, so treat it as unknown
until you probe it (`python -c` cannot help here — run a Brotli decode through Tesseract or
check the installer's changelog).

## Applied fixes

### 2026-10-26 — OneDrive removed from startup

Requested by the user. OneDrive registered its autostart in **three** places, not one;
removing only the Run key would have been a fake fix.

| What changed | From | To | How |
|---|---|---|---|
| `HKCU\...\Run\OneDrive` | `"...\OneDrive.exe" /background` | **deleted** | `winreg.DeleteValue` |
| `HKCU\Software\Microsoft\OneDrive\AutoStartEnabled` | `1` | **`0`** | `winreg.SetValueEx` (REG_DWORD) |
| Scheduled task `\OneDrive Startup Task-S-1-5-…-1001` | Enabled (logon trigger → `OneDriveLauncher.exe /startInstances`) | **Disabled** | `schtasks /change /tn <name> /disable` |

Why each was needed:

- **`AutoStartEnabled`** is OneDrive's own "start at sign-in" switch. Left at `1`, the
  client re-creates the Run value on its next launch, so the deletion silently reverts.
- **The Startup Task** has a `<LogonTrigger>` and an action of
  `C:\Users\Holger\AppData\Local\Microsoft\OneDrive\26.189.0928.0009\OneDriveLauncher.exe`
  with `/startInstances` — it starts OneDrive at logon regardless of the Run key. This
  was the gap that would have undone the fix, and it was invisible while I only read
  `HKCU` Run keys.

Verified after the change (re-read from the registry / Task Scheduler, not from the writes):

- no `OneDrive` value in `HKCU` Run; none in `HKLM` Run or `HKLM` WOW64 Run;
- `AutoStartEnabled = 0`;
- `OneDrive Startup Task` → `Scheduled Task State = Disabled`;
- `schtasks /query` → **0** OneDrive tasks in `Ready` state.

Not touched, by design: **`OneDrive Reporting Task`** and **`OneDrive Standalone Update
Task`** remain Enabled. They are update/reporting jobs, not logon triggers, so they do not
start OneDrive; disabling them would only stop OneDrive patching itself.

Rollback: backups hold the original values and the full task XML —
`%LOCALAPPDATA%\Agent1\backups\onedrive-startup-<stamp>.json` (Run value + original
`AutoStartEnabled`) and `onedrive-startup-task-<stamp>.json` (task XML + prior state
`Enabled`, restorable with `schtasks /change /tn <name> /enable`).

Two caveats worth recording:

- OneDrive (PID 16944) was still running when this was applied; the change takes effect at
  the **next sign-in**. Until then the process keeps syncing — that is expected, not a
  failure.
- The client is user-mode sync, so this does **not** narrow the #5 persistence surface in
  the way disabling a network service would. It removes an autostart, not a capability.

Scanner note — both gaps found during this change were **fixed** in
`tools/black_october_triage.py`:

- **Blind scan reported as clean.** An `os.walk` over `C:\Windows\System32\Tasks` fails
  with `Access denied` and silently returns **0 hits**, which is how the first pass
  wrongly concluded the OneDrive tasks were gone — `schtasks /query` is the reliable
  probe on this box. `check_scheduled_tasks` now reports a zero-task parse as
  `MEDIUM scheduled-task-scan-empty` ("the task scan is BLIND, not clean") instead of
  printing `[ok]`.
- **The restarting task was invisible.** Nothing classified a task as "relaunches the
  app", so removing a Run key looked complete while a logon task undid it.
  `_autostart_restart_risk` now flags an *enabled* task with an `At logon` /
  `At startup` / `On workstation unlock` trigger whose action mentions a known
  autostart app, as `MEDIUM`. The parser reads columns 11 (`Scheduled Task State`) and
  18 (`Schedule Type`) for this — no per-task XML call needed.

Verified on this host's live task data (259 tasks parsed, 28 with logon triggers): the
now-**Disabled** OneDrive Startup Task is correctly **silent**, and the same real action
string with the state flipped to `Enabled` produces the MEDIUM finding. It stays MEDIUM
rather than CRITICAL because a vendor shipping both a Run key and a logon task is not
malicious — this is an autostart-*completeness* finding, not a persistence finding.


## Harness notes


- `tools/black_october_triage.py` — the scanner. Listener attribution uses
  `netstat -ano` + `tasklist` because `Get-NetTCPConnection`'s process lookup fails
  on this box (`Get-Process` inside the pipeline errors out).
- `tests/test_black_october_triage.py` — 65 regression tests covering the parser,
  the `lmlink` AI-process match, the OS-owned-wildcard filter, the Node version gate,
  the scheduled-task persistence classifier (`_parse_schtasks_verbose` /
  `_task_risk`) with explicit vendor-noise and OS-baseline regression cases, the
  autostart/blind-scan cases added with the OneDrive fix (`_autostart_restart_risk`,
  `scheduled-task-scan-empty`), and the disabled-task demotion cases added when an
  applied fix was still printing as an open MEDIUM finding.
- The scheduled-task check parses `schtasks /query /fo csv /nh /v` and inspects
  columns 1 (name), 7 (author), 8 (action), 11 (state) and 18 (trigger). It reports the
  persistence shape (action in a user-writable dir, plus a LOLBin or a second-stage
  fetch), treats `%windir%` actions and `ProgramData\Microsoft` as OS baseline (so
  Defender's own rundll32 maintenance tasks are not findings), and separately flags an
  enabled logon/startup task that relaunches a known autostart app.
- Both files are tracked in git and committed: `00068cd` (scanner + this report),
  `ead8012` (CVE-claim verification), `4b97e53`/`b289ad9` (triage report, CVR exposure),
  and `6bd306e` (OneDrive autostart removal). All of these are pushed to
  `origin/master`. The scheduled-task classifier rewrite and the UTF-8 encoding pin
  from earlier notes landed in `ead8012`, so this doc is the record.
- **Fixed 2026-10-26 (was: known environmental failure).**
  `tests/test_run_timeout.py::test_timeout_kills_process_tree` failed on a clean
  environment because the strict run sandbox rejected `python -m x` before the
  timeout/Popen path was ever reached — the test asserted the timeout transport
  while being stopped at the policy gate. The test now invokes
  `python -m py_compile setup.py`: `py_compile` is in
  `RUN_SHELL_MODULE_ALLOWLIST`, so the security gate stays **enabled in strict
  mode** (no env override, no wider sandbox) while the command still reaches the
  code under test, and because it is not a test run the full-run budget gate does
  not engage. `Popen` is faked, so nothing is executed. An earlier attempt set
  `AGENT_RUN_SANDBOX=dev` to force the command through; that was discarded
  because disabling the sandbox to satisfy a test is the wrong direction — the
  gate is real behaviour this repo wants tested, not something to switch off.
  Both tests in that file pass; the assertion was not weakened.
- **Fixed 2026-10-26.** `tests/test_packaging_metadata.py::test_wheel_contains_every_subpackage`
  hard-coded `--no-build-isolation`, so on a runner without setuptools pip failed
  with `BackendUnavailable: Cannot import 'setuptools.build_meta'` and the job went
  red for an environment reason rather than a packaging regression. The flag is now
  chosen by `_build_isolation_flags()`, which probes the backend named in
  `pyproject.toml` via `importlib.util.find_spec` (spec resolution only — the
  backend is never imported); when it is importable the fast `--no-build-isolation`
  build is used, otherwise pip is left to provision the backend itself. The
  no-setuptools branch is pinned by
  `test_build_isolation_flags_follow_backend_availability`, which monkeypatches the
  probe, plus `test_build_backend_is_read_from_pyproject` and
  `test_module_importable_probes_without_importing`. Note: the setuptools-absent
  branch is verified by that unit test, not by building a wheel in a stripped
  interpreter.
- **Root cause of the red CI run that prompted both fixes: neither change was
  committed.** `git status` showed `tests/test_packaging_metadata.py`,
  `tests/test_run_timeout.py` and this doc as modified-but-unstaged, so CI ran the
  pre-fix code and failed on exactly those two tests, which is the correct result
  for the old code. Local runs were green only because they used the working tree.
