---
name: playwright
description: Drive a real browser with Playwright for scraping, E2E checks and page inspection — Python sync API (the style used in this repo) plus the playwright-cli reference. Covers selectors, waiting, persistent profiles, login walls and cleanup. (use when: automating a browser, scraping a site, writing an E2E test, inspecting DOM/network, screenshots, or debugging a flaky Playwright script)
---

# Playwright Browser Automation

Playwright drives Chromium, Firefox and WebKit with real browsers. This repo uses the
**Python sync API** (`from playwright.sync_api import sync_playwright`) in ad-hoc probe
scripts; `playwright-cli` is the official agent CLI for one-off interactive work.
Installed here: Playwright **1.62.0** (verify with `python -m pip show playwright`).

## 0. Preflight (do this first)

```powershell
python -m pip show playwright                 # version present?
python -m playwright install chromium          # browser binary present? run once per machine
```

If `install` has never run, every launch fails with "Executable doesn't exist".
Playwright is **not** in `requirements.txt` / `pyproject.toml`, so it is a local-only
dependency — do not assume CI has it.

## 1. The one rule that kills flakiness: auto-wait, never sleep

Every action (`click`, `fill`, `screenshot`) waits for the element to be **actionable**.
Never add `time.sleep()`; never poll in a loop. Use locators, not raw selectors.

```python
page.get_by_role("button", name="Sign in").click()      # role/name = user-facing
page.get_by_label("Email").fill("a@b.com")              # label text
page.get_by_placeholder("Search").fill("devops")
page.get_by_text("Jobindex", exact=True).click()        # visible text
page.locator("#id").filter(has_text="Open")             # CSS + filter when needed
```

Prefer this order: role/label/text > `data-testid` > CSS/XPath. Position-based and
nth-child selectors break on layout changes — treat them as a last resort.

## 2. Minimal working script (this repo's style)

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)      # headless=False to watch it work
    context = browser.new_context(
        user_agent="Mozilla/5.0 ...",               # some sites block the default UA
        viewport={"width": 1366, "height": 900},
        locale="da-DK",
    )
    page = context.new_page()
    page.goto("https://example.com", wait_until="domcontentloaded")
    page.get_by_role("link", name="Jobs").click()
    print(page.title())
    print(page.locator("article").count())
    page.screenshot(path="page.png", full_page=True)
    browser.close()          # close() releases the browser process — always reach it
```

`with sync_playwright()` is mandatory: it starts/stops the driver process. Forgetting
`browser.close()` leaks processes (a common cause of "address in use" and hung CI).

## 3. Waiting, scrolling, extraction

```python
page.wait_for_selector(".job-card", state="visible", timeout=15000)
page.get_by_text("Results").wait_for()
page.mouse.wheel(0, 2000)                      # scroll for lazy-loaded lists
page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

rows = page.locator(".job-card")
for i in range(rows.count()):
    rows.nth(i).inner_text()                   # text; .get_attribute("href") for links
page.eval_on_selector_all(".job-card", "els => els.map(e => e.innerText)")  # bulk, fast
```

Set timeouts per call (`timeout=`) or globally: `page.set_default_timeout(30000)`.
`wait_until="networkidle"` is convenient but often never settles on ad-heavy sites —
prefer `domcontentloaded` + an explicit locator wait.

## 4. Persistent profile (login, cookies, real fingerprint)

Use a **throwaway directory in the repo**, never the user's real Chrome profile:

```python
ctx = p.chromium.launch_persistent_context(
    user_data_dir="_pw_profile",     # git-ignored scratch; delete when done
    headless=False,                  # logins usually need headless=False
)
page = ctx.pages[0] if ctx.pages else ctx.new_page()
```

Why this matters here: previous runs created `_ji_profile*`, `_tmp_chrome_*`,
`_tmp_userdata_copy` and a `Default/` junction into
`C:\Users\Holger\AppData\Local\Chrome\User Data`. Those are 20+ GB of copies, and
reading/copying the real profile's `Cookies` DB exposes browser credentials. Rules:

- Never copy, decrypt or read the user's Chrome profile (`Cookies`, `Login Data`).
- Never create a junction/symlink to it. `rmdir <junction>` removes the **link** only;
  `Remove-Item -Recurse` on it walks into the real profile — do not do that.
- Give each probe its own `user_data_dir`, and delete it when the task ends.

For a login the user must do themselves, launch headless=False, print an instruction,
and poll for the post-login URL:

```python
page.wait_for_url("**/dashboard**", timeout=300_000)   # human-in-the-loop login
ctx.storage_state(path="auth.json")                    # reuse later, no re-login
```

Reuse later with `browser.new_context(storage_state="auth.json")`.

## 5. Network: capture data instead of scraping DOM

Intercepting the site's own JSON API is more reliable than parsing HTML.

```python
with page.expect_response(lambda r: "/api/jobs" in r.url and r.status == 200) as info:
    page.get_by_role("button", name="Next").click()
data = info.value.json()

page.on("response", lambda r: print(r.status, r.url) if r.request.resource_type == "xhr" else None)
page.route("**/*.png", lambda route: route.abort())     # speed: block images/fonts
```

## 6. Debugging a failing script

- `headless=False` and watch it; add `p.chromium.launch(headless=False, slow_mo=200)`.
- Trace: `ctx.tracing.start(screenshots=True, snapshots=True)` … `ctx.tracing.stop(path="trace.zip")`,
  then `python -m playwright show-trace trace.zip`.
- Screenshot + HTML on failure: `page.screenshot(path="fail.png", full_page=True)`.
- Console/page errors: `page.on("console", lambda m: print(m.type, m.text))`,
  `page.on("pageerror", print)`.
- Element not found → check for an iframe: `page.frame_locator("#widget").locator(...)`.
- "Element is not visible" → scroll into view first: `loc.scroll_into_view_if_needed()`.
- Strict-mode violation ("resolves to 2 elements") → narrow with `.filter()` or `.first`.

## 7. Cleanup rules (repo hygiene)

Scratch output belongs in one directory and must be ignorable — `.gitignore` covers
`_tmp_*`, `*.png`, `*.db` and `.scratch-quarantine/`.

- Name probes `_tmp_<topic>.py`; write their output to `_tmp_*` files, not the root.
- Do not commit probe scripts, scraped JSON, screenshots, profile dirs or cookie DBs.
- Delete the `user_data_dir`, trace zip and screenshots when finished.
- The harness blocks recursive deletes; move retired scratch to `.scratch-quarantine/`
  (reversible) instead of trying to delete it.

## 8. playwright-cli (official agent CLI, for interactive work)

Structured snapshot refs beat selectors when exploring a page by hand:

```bash
playwright-cli open https://example.com
playwright-cli snapshot                 # accessibility tree with refs e1, e2, ...
playwright-cli find --regex "/sign (in|up)/i"
playwright-cli click e15
playwright-cli fill e5 "user@example.com" --submit
playwright-cli eval "el => el.textContent" e5
playwright-cli screenshot --filename=page.png
playwright-cli requests ; playwright-cli console warning
playwright-cli state-save auth.json     # cookies + storage for reuse
playwright-cli close
```

Install with `npm i -g @playwright/cli` (not installed in this workspace). Docs:
https://playwright.dev/agent-cli/skills — Microsoft's own skill file is the reference
model for this runbook.

## 9. Safety and ethics

- Respect `robots.txt` and the site's ToS; rate-limit (`page.wait_for_timeout(1000)`
  between pages) and keep request volume low.
- Never automate a login you are not allowed to perform. If the user logs in with a
  Google account, hand them the browser (headless=False) and wait for them.
- Store credentials in `.env` (git-ignored), never hardcode them in a probe script.
- Scraped personal data: keep it local, delete it after use, do not commit it.
