# Security Priority Stack — one ranked list

This is the single place where the whole exposure stack is ordered. The detail lives in
two companion documents and is not duplicated here:

- `PII_EXPOSURE_TRIAGE.md` — the Danish CVR filing (priority 1).
- `BLACK_OCTOBER_TRIAGE.md` — device-local findings, 18 open, scanned with
  `python tools/black_october_triage.py`.

Read this file for **what to fix first**. Read those files for **how**.

**This file contains no personal data.** The owner's name, email, street address and
CVR number are never written here. Fill them in mentally, not in git.

## Ranking rule

Findings are ordered by **who can reach them**, then by **what losing it costs**.

1. Reachable by anyone, with no exploit → first. A published home address is in this
   group. It is the only finding that is already disclosed.
2. Reachable from the local network right now → second. This needs no user action by
   the owner and no knowledge of the machine.
3. Reachable only after someone gets onto the machine, but the loss cannot be undone →
   third. A hot wallet sits here.
4. Reachable only after someone gets onto the machine, and a patch exists → next.
5. Reachable only with a local action (running a CLI on hostile input) → later.
6. Age and default-configuration debt with no published advisory → last.

Severity labels alone do not set the order. CVE-2026-111497 carries CVSS 9.2 and still
ranks below a business-register record, because the CVE needs a crafted stream to
arrive, and the address is already published.

## The stack

| # | Priority | Finding | Status |
|---|---|---|---|
| 1 | **P0** | CVR filing publishes a residential address, phone and email. `protected = False`. | **Open. No fix applied.** |
| 2 | P1 | LM Studio LM Link listens on `0.0.0.0:61236` (`lmlink-connector.exe`), and re-binds at logon. | **Open.** |
| 3 | P1 | Daedalus Mainnet desktop wallet installed. | **Open.** Non-technical fix. |
| 4 | P2 | Brotli 1.2.0 inside Node 24.15.0 on `PATH` — CVE-2026-111497. | **Open. Patch available.** |
| 5 | P2 | Brotli 1.2.0 inside bundled Node 25.5.0 (Bionic, LM Studio). | **Open. No patch exists on the 25 line.** |
| 6 | P3 | Wide, unaudited auto-start surface (LM Studio, Teams, Lemonade Server, Wispr Flow). | **Open.** |
| 7 | P3 | Brotli 1.2.0 in Git for Windows; 1.1.0 in the ComfyUI Git bundle. | **Open.** |
| 8 | P4 | NetBIOS 139 on the LAN address; two vendor update tasks in user-writable directories. | **Open. Accepted risk.** |
| 9 | P5 | Java 1.8.0_491 on `PATH`. Age only — **no advisory is attached**. | **Open.** |

Already closed on this machine: OneDrive autostart removed (Run key deleted,
`AutoStartEnabled = 0`, Startup Task disabled). PHP is not installed. No Python
`brotli` package is installed. No eMudhra CA is in the user Root store. `.env` was
never committed.

## Plan

The order below is the order to work in. Do not skip ahead to the CVEs because they
carry the highest numbers.

### Step 1 — priority 1, the business register (needs no exploit)

1. Open `datacvr.virk.dk` in a browser and confirm that the record shows no address
   protection. The earlier machine query went through a third-party mirror
   (`cvrapi.dk`), and the official endpoint rejects automated requests. A single
   mirror is not enough evidence to act on.
2. Apply for address protection under **CPR-loven § 28** through the bopælskommune.
3. Assume the historical 1993–2005 record stays visible even after protection is
   granted. Protection reads as *current state*, so plan around the old row.
4. Request removal from the CVR republishers: `virio.dk`, `companydata.dk`,
   `selskabsinfo.dk`, `proff.dk`.

Honest limit: this reduces discoverability. It does not erase. Data already scraped
cannot be recalled, and the original filing is a legal historical record that will not
be deleted. Anyone who offers to erase it should be treated with suspicion.

### Step 2 — priority 2, take the LLM endpoint off the network (2 minutes)

1. In LM Studio, disable **LM Link** / local service. This sets `enableLocalService`
   to `false`.
2. Quit LM Studio fully.
3. Stop the current listener. Confirm the PID first, because it changes on restart.
4. Delete the Run key `electron.app.LM Studio` so it does not re-bind at logon.

Verify: port `61236` must show nothing, or `127.0.0.1` only. It must not show
`0.0.0.0`.

Why this ranks so high: an exposed LLM endpoint is a prompt-injection to
tool-execution path, it is reachable from the local network today, and the fix costs
two minutes.

### Step 3 — priority 3, the wallet (no technical fix exists)

Move long-term balances to a hardware wallet. Keep only spending money on the desktop
wallet. If Daedalus is unused, uninstall it. Verify installer checksums against the
vendor.

Honest limit: the Electrum supply-chain angle that motivated this finding was never
independently confirmed. The finding stands on observed wallet presence alone.

### Step 4 — priority 4, patch Node (one command, elevated shell)

```
winget upgrade --id OpenJS.NodeJS.LTS --accept-source-agreements
```

Installed `24.15.0`, available `24.22.1`. That build ships Brotli `1.2.1`.

Verify in a **new** shell:

1. `node --version` reports `v24.22.1` or higher.
2. `node -p process.versions.brotli` reports `1.2.1` or higher. **This is the check
   that proves the fix.** The defect is in the bundled Brotli, so a `24.22.1` label
   with a still-`1.2.0` Brotli means the fix did not land.
3. Re-run the scanner. The two Node 24.15.0 lines must be gone. The total drops from
   **18 to 16**.

Do not roll Node back to make a global npm install work. Reinstall that package.

### Step 5 — priority 5, contain the Node 25 copies (no patch exists)

The advisory fixes `22.23.4`, `24.22.1` and `26.12.1`. Major 25 is not among them, so
no update delivers a fix. LM Studio is already on its newest build and still bundles
25.5.0, so "just update the app" is not a path.

1. Uninstall Bionic if it is unused.
2. For LM Studio, accept the risk deliberately and narrow the path: turn off
   development plugins and automatic extension-pack updates if you do not write
   plugins.
3. Do **not** hand-replace the bundled `node.exe`. That breaks app signing and the
   next auto-update reverts it.

### Step 6 — priorities 6 and 7, trim auto-start and the Brotli CLIs

Delete every Run value and Startup shortcut you cannot name a reason for. Update Git
for Windows to move the bundled `brotli.exe` off `1.2.0`; re-check the version
afterwards and do not assume the update reaches `1.2.1`. The ComfyUI copy is `1.1.0`
and moves only with that AMD bundle — deleting the bundle is cheaper if unused.

### Step 7 — priorities 8 and 9, accept or defer

Leave NetBIOS alone unless there is a reason to disable NetBIOS over TCP/IP. Leave the
two vendor-signed update tasks alone. Upgrade or remove Java 8; removal is strictly
better than patching an 8.x runtime. There is no advisory for it, so this is age
hygiene, not an incident.

## What is not a vulnerability here

State this plainly, so the register stops carrying false entries:

- **CVE-2026-22020 does not exist.** The NVD API returns zero records for it, and a
  `CVE-2021-44228` positive control returned one, so the zero is a real absence. Any
  register that cites a Java "Flight Recorder" CVE is wrong. Remove it.
- **CVE-2026-109618 and CVE-2026-109619 do not apply.** They are PHP defects and PHP
  is not installed.
- **None of the cited CVEs is in CISA KEV.** As of the last check there was no
  confirmed in-the-wild exploitation of CVE-2026-111497. Treat all of this as patching
  debt, not as an active intrusion.
- **The impact claim for CVE-2026-111497 is narrower than its 9.2 score suggests.**
  Independent research argues the vulnerable fast path is not reachable from any
  Brotli stream, because the specification caps `MLEN`. The public proof of concept
  reached code execution only on a deliberately weakened, non-PIE build; on a standard
  build its author reported a crash and a limited memory read. The defensible
  expectation on this machine is **denial of service and memory disclosure**, with
  remote code execution **unproven**. The patch is still worth one command.
- **Both campaign names are in circulation.** "Node2Shell" and "Broccoli2Shell" both
  refer to CVE-2026-111497. The mechanism — a Brotli decoder defect that Node inherits
  by bundling Brotli — is what matters.

## Limits of this assessment

- Priority 1 was **not** re-verified against the official register during the final
  pass. Automated requests to `datacvr.virk.dk` are rejected, and the CVR number is
  deliberately redacted from this repository. `protected = False` is the last verified
  state.
- "No link between the personal data and the Black October campaign" rests on
  dark-web and paste-site sources that were unreachable at the time. That is **absence
  of evidence**, not a clean result.
- The narrowed exploitability claim comes from independent research and second-hand
  reporting of the proof of concept. It was not reproduced here.
