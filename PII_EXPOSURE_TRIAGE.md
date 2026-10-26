# Personal-Data Exposure Triage — Priority #1

Scope: the owner's personal-data footprint online, specifically the Danish business
register (CVR) exposure. This finding outranks every item in
`BLACK_OCTOBER_TRIAGE.md`, because those are *device-local* and need an attacker to
reach the machine first. A published home address needs no exploit at all.

**This file deliberately contains no personal data.** The owner's email, name, street
address and CVR number are redacted as `<OWNER-EMAIL>`, `<CVR-NR>`, `<STREET-ADDRESS>`.
Writing the real values here would republish the exact exposure this document exists to
contain, and would trip `tests/test_no_published_secrets.py`. Fill them in mentally, not
in git.

## Verification status — read this first

Two different evidence classes appear below and they are not equally strong.

- **VERIFIED (statute/authority).** Confirmed against lovguiden.dk, danskelove.dk and
  datacvr.virk.dk. Quoted text in "The statute" section.
- **VERIFIED (live register query).** The facts of the owner's own filing were
  re-confirmed by querying the public CVR API with the owner's CVR number and reading
  the returned record (HTTP 200). This replaces an earlier "carried over, unverified"
  status — those facts came from a compacted transcript and had never been persisted.

### Live query result

Field **presence** and flags only; literal values are deliberately not reproduced here.

| Field | Observed | Implication |
|---|---|---|
| `protected` | **`False`** | **No address protection is currently active.** The fix below is NOT in place. |
| `startdate` | `01/01 - 1993` | Confirms the 1993 start — previously unverified. |
| `enddate` | `11/11 - 2005` | Confirms the 2005 end — previously unverified. |
| `address` | **present** (17 chars) | A residential address is published *right now*. |
| `zipcode`, `city` | **present** | Enough to locate the address geographically. |
| `phone` | **present** | Exposed; not previously accounted for. |
| `email` | **present** | Exposed; a ready-made phishing target. |
| `owners`, `productionunits` | present | Additional linked records to check. |
| `creditbankrupt` | `False` | No insolvency flag to correct. |

Two corrections to the earlier analysis:

1. The dates and the exposure are now **fact, not a lead**. The record is live and the
   address is currently retrievable by anyone who knows the CVR number.
2. **`phone` and `email` are also published.** The earlier finding covered the address
   only. Published contact details turn an address leak into a directly contactable
   attack surface — this raises the finding's severity, it does not lower it.

The `web_search` tool was intermittently returning zero results during this session
(short Danish queries succeeded; several longer ones and every English query returned
empty). Absence of a search hit below is therefore *not* evidence of absence. The live
API query above was used precisely because search was unreliable.

## Why this is the top finding

An *enkeltmandsvirksomhed* (sole proprietorship) has no legal separation from its
owner: the registered business address **is** the owner's residential address. CVR
holds current *and historical* records and publishes them openly. So a 1990s
registration made a home address permanently queryable by name or CVR number.

Ranked above the Brotli RCE (CVE-2026-111497) not because it is more severe in the
abstract, but because it is **already disclosed**. The CVE needs a crafted stream to
arrive; the address is already in an attacker's hands and is the ingredient that makes
a phishing call or letter credible.

## The statute (VERIFIED)

CPR-loven § 28, quoted from danskelove.dk / lovguiden.dk:

> "Enhver har ret til ved henvendelse til sin bopælskommune at få navne- og
> adressebeskyttelse, således at vedkommendes navn og adresse i CPR ikke må
> videregives til private … Beskyttelsen bortfalder efter et år, medmindre
> vedkommende ønsker et tidligere bortfald …"

Four consequences that drive the plan:

1. **It is a right, not a discretionary grant.** "Enhver har ret" — everyone has it.
   An earlier draft of this analysis said the protection is tested against a specific
   threat to the applicant. **That was wrong**, and it would have set a false
   expectation. There is no threat test to satisfy for the standard protection.
2. **It expires after one year** and must be renewed. A one-off application is not a
   permanent fix — this is the single most likely way the fix silently lapses.
3. **Duration is extendable.** Per stk. 2 the protection period's length is set by
   regulation, and borgerhaandbog.dk states the municipality normally approves up to
   one year "medmindre helt særlige omstændigheder taler for en længere periode".
4. **The CVR transfer is automatic.** From datacvr.virk.dk:

   > "Hvis du har navne- og adressebeskyttelse i Det Centrale Personregister (CPR),
   > får du dog automatisk beskyttelsen af din privatadresse overført til Det Centrale
   > Virksomhedsregister. Det betyder, at din privatadresse ikke bliver offentliggjort
   > i CVR, så længe beskyttelsen i CPR er gældende."

   So there is **no separate CVR application**. The Folketinget evaluation paper
   (ft.dk, 2020-21) confirms the same mechanism and adds the failure mode: you *may*
   ask Erhvervsstyrelsen for the protection **not** to apply in CVR. Do not tick that.

## Fix plan

### Step 1 — Apply for navne- og adressebeskyttelse (the actual fix)

Apply digitally via borger.dk (MitID), or at the bopælskommune if you cannot self-serve.
You may cover yourself and children at the same address. Nothing else on this list
matters until this is in place.

### Step 2 — Confirm the transfer, don't assume it

Automatic does not mean infallible. Look up your own `<CVR-NR>` on `datacvr.virk.dk`
and check whether `<STREET-ADDRESS>` still appears. Also verify the CPR record itself.
This is the check that proves the fix landed — the equivalent of re-reading
`process.versions.brotli` rather than trusting a version label.

### Step 2b — Remove the published phone and email (new this pass)

The live query showed `phone` and `email` are published alongside the address. § 28 /
CVR suppression covers the **address**; it is not established that it clears contact
fields. Handle them explicitly:

1. In the CVR self-service flow (Virk / Erhvervsstyrelsen), blank the business phone and
   email, or replace them with role addresses (`kontakt@…`) that are not personal.
2. Replace the personal email with an alias you do not use for banking or MitID. CVR
   email is a ready-made, *verified* phishing target — the attacker gets a real address
   attached to a real company name.
3. If a phone number must stay for the business, use a separate number, not your mobile.

### Step 3 — Set a renewal reminder

Because § 28 protection lapses after a year, put a calendar reminder at ~11 months.
Without this the whole exercise quietly reverses itself.

### Step 4 — Broker removal (residual risk, partial by nature)

NOT VERIFIED this session — searches on Danish broker opt-out returned zero results.
The categories to work through are:

- **CVR-derived aggregators.** Several sites republish CVR data daily and are not bound
  by the CPR protection. Examples surfaced in search: virio.dk, companydata.dk,
  selskabsinfo.dk, proff.dk.
- **People-search / address sites.** Standard removal requests; expect partial success.
- **Search engines.** Request cache/page removal for pages still showing the address
  after the register itself is suppressed.

The honest framing: this step reduces discoverability, it does not achieve erasure.
Data already scraped is outside CVR's control.

### Step 5 — What cannot be fixed

The underlying business filing is a **legal historical record** and will not be deleted.
The objective is suppression of publication going forward, not deletion of history.
Anyone asserting they can erase the original filing should be treated with suspicion.

## Honest gaps

- Danish broker opt-out procedures were not confirmed — the § 28/CPR mechanism was, but
  the broker landscape changes constantly and needs a fresh check.
- The earlier "no Black October link" conclusion rested on dark-web and paste sources
  that were unreachable at the time. That remains **absence of evidence**, not a clean
  result.
- The live query came from a third-party mirror (`cvrapi.dk`), not the authoritative
  register: `datacvr.virk.dk` returned HTTP 403 to an automated request and
  `distribution.virk.dk` timed out. The mirror's data matched expectations exactly, but
  treat `protected = False` as *strongly indicated* and confirm it in the official UI at
  `datacvr.virk.dk` before acting — do not treat a single third-party mirror as final.
- `phone` and `email` being published was observed on this pass and has not been
  separately cross-checked against the official register.
- Not verified: whether an *active* CPR § 28 protection retroactively suppresses the
  historical (`enddate` 2005) address record, or only the current one. The law says the
  private address "ikke bliver offentliggjort … så længe beskyttelsen er gældende",
  which reads as *current-state*, so the historical row may remain visible even after
  protection is granted. **Assume it stays visible** and plan broker removal on that
  basis. This is the single most important unknown in this document.

## Related

- `BLACK_OCTOBER_TRIAGE.md` — the device-local findings (Brotli/Node, LM Link, wallets).
- `tests/test_no_published_secrets.py` — guards this repo against committed secrets and
  PII. Set `AGENT1_PII_SENTINELS` to enable the PII scans, which otherwise skip loudly.
