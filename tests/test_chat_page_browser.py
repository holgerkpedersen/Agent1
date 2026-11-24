"""Browser regression tests for the /chat page (dashboard chat surface).

These are a permanent, executable version of the checks that were run by hand
when the page was reported broken in a live browser:

* the keyboard-shortcuts <dialog> painted over the page on every load, and a
  hard refresh (Ctrl-F5) re-exposed it;
* the ENTIRE inline script rendered as visible text under the Send button -
  the opening ``<script>`` tag had been dropped, so no JS ran at all and the
  dialog's Close button was dead.

The source-level tests in ``test_chat_shortcuts_dialog.py`` pin the same two
invariants for CI; this module drives the real page in a real browser: it
serves ``static/chat.html`` through the live dashboard server, stubs every API
call with ``page.route`` (no LLM/API key needed) and asserts that the buttons
have working JS behind them.

Playwright is a local dev extra (not in requirements.txt), so the whole module
skips cleanly when it - or its browser - is missing.
"""

from __future__ import annotations

import importlib.util
import json
import threading

import pytest

pytestmark = pytest.mark.browser


def _browser_available() -> bool:
    """True only if playwright AND a launchable chromium are present."""
    if importlib.util.find_spec("playwright") is None:
        return False
    try:  # pragma: no cover - environment dependent
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            browser.close()
            return True
    except Exception:
        return False


requires_browser = pytest.mark.skipif(
    not _browser_available(),
    reason="needs playwright (python) + chromium to drive the real page",
)


# --------------------------------------------------------------------------
# Live server: serve static/chat.html verbatim, like production does.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def chat_url() -> "str":
    from agent_core.monitoring import DashboardAPIServer

    holder = DashboardAPIServer(None, port=0)  # port 0 -> OS-assigned
    httpd = holder.start()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/chat"
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()
        thread.join(timeout=5)


# --------------------------------------------------------------------------
# API stubs - the page must work with no LLM behind it.
# --------------------------------------------------------------------------
STUB_COMMANDS = [
    {
        "name": "help",
        "summary": "List commands",
        "needs_input": False,
        "usage": "/help",
    },
    {
        "name": "model",
        "summary": "Switch model",
        "needs_input": True,
        "usage": "/model <m>",
    },
]


def _stub_apis(page) -> None:
    """Answer every /api/chat/* call the page makes, deterministically."""

    def _json(route, payload, status=200):
        route.fulfill(
            status=status,
            content_type="application/json",
            body=json.dumps(payload),
        )

    page.route(
        "**/api/chat/history",
        lambda r: _json(r, {"messages": [], "model": "stub/model"}),
    )
    page.route("**/api/chat/commands", lambda r: _json(r, {"commands": STUB_COMMANDS}))
    page.route(
        "**/api/chat/send",
        lambda r: _json(
            r,
            {
                "ok": True,
                "reply": "pong from stub",
                "model": "stub/model",
            },
        ),
    )
    # The page now drives turns through the SSE stream endpoint, so stub it
    # with the same event shape the server emits (answer + done).
    def _stream(route):
        body = (
            'event: answer\\ndata: "pong from stub"\\n\\n'
            'event: done\\ndata: "stub/model"\\n\\n'
        )
        route.fulfill(
            status=200,
            content_type="text/event-stream",
            body=body,
        )

    page.route("**/api/chat/stream", _stream)
    page.route("**/api/chat/reset", lambda r: _json(r, {"ok": True}))


@pytest.fixture()
def chat_page(chat_url: str):
    """A fresh tab with the API stubbed and no console errors tolerated."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        _stub_apis(page)
        errors: list[str] = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on(
            "console",
            lambda m: errors.append(m.text) if m.type == "error" else None,
        )
        page.goto(chat_url, wait_until="domcontentloaded")
        page.wait_for_selector("#chatInput")
        page._js_errors = errors  # type: ignore[attr-defined]
        yield page
        browser.close()


# --------------------------------------------------------------------------
# The reported symptoms.
# --------------------------------------------------------------------------
@requires_browser
def test_inline_script_is_not_visible_text(chat_page) -> None:
    """The inline script must be code, not body text under the Send button.

    A dropped ``<script>`` tag made ~46 KB of JS render as page copy and
    disabled every handler on the page.
    """
    assert chat_page.locator("script").count() >= 2, "page scripts must be in <script>"
    body = chat_page.inner_text("body")
    for marker in ('"use strict"', "addEventListener", "</script>"):
        assert marker not in body, f"JS source leaked into the page text: {marker!r}"
    # The composer (the thing the user needs) is visible.
    assert chat_page.locator("#chatInput").is_visible()
    assert chat_page.locator("#sendBtn").is_visible()


@requires_browser
def test_shortcuts_dialog_hidden_on_load_and_after_reload(chat_url: str) -> None:
    """Ctrl-F5 (hard reload) must not show the shortcuts overlay.

    Regression: an explicit ``display`` on ``.shortcuts`` outranked the
    ``[hidden]`` attribute, so the modal painted over the page on every load.
    """
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        _stub_apis(page)
        page.goto(chat_url, wait_until="domcontentloaded")

        def dialog_open() -> bool:
            return page.evaluate("document.getElementById('shortcutsDialog').open")

        assert not dialog_open(), "shortcuts dialog must be closed on load"
        assert page.locator(".shortcuts-card").is_hidden(), (
            "shortcuts overlay must not paint before it is opened"
        )

        # Ctrl-F5: reload WITHOUT the browser cache, like a hard refresh.
        page.reload(wait_until="domcontentloaded")
        assert not dialog_open(), "shortcuts dialog reopened after reload"
        browser.close()


@requires_browser
def test_shortcut_dialog_close_button_works(chat_page) -> None:
    """The Close button is dead if no JS ran; prove the handler is bound."""
    chat_page.click("#shortcutsBtn")
    assert chat_page.locator(".shortcuts-card").is_visible(), "open button must show it"

    chat_page.click("#shortcutsClose")
    assert not chat_page.evaluate("document.getElementById('shortcutsDialog').open"), (
        "Close button did not close the dialog"
    )


@requires_browser
def test_shortcut_dialog_escape_and_backdrop(chat_page) -> None:
    """Escape and a backdrop click both dismiss the overlay."""
    chat_page.click("#shortcutsBtn")
    chat_page.keyboard.press("Escape")
    assert not chat_page.evaluate("document.getElementById('shortcutsDialog').open"), (
        "Escape did not close the dialog"
    )

    # Open again, then click outside the card (the backdrop). The page closes
    # on a click whose target is the <dialog> itself.
    chat_page.click("#shortcutsBtn")
    box = chat_page.locator(".shortcuts-card").bounding_box()
    assert box
    chat_page.mouse.click(box["x"] - 10, box["y"] + (box["height"] / 2))
    assert not chat_page.evaluate("document.getElementById('shortcutsDialog').open"), (
        "backdrop click did not close the dialog"
    )


@requires_browser
def test_ctrl_slash_toggles_the_dialog(chat_page) -> None:
    """Ctrl+/ toggles the overlay, Ctrl+K focuses the composer."""
    chat_page.keyboard.press("Control+KeyS")  # placeholder guard, see below
    assert not chat_page.evaluate("document.getElementById('shortcutsDialog').open")

    chat_page.keyboard.press("Control+Slash")
    assert chat_page.evaluate("document.getElementById('shortcutsDialog').open"), (
        "Ctrl+/ must open the shortcuts dialog"
    )
    chat_page.keyboard.press("Escape")  # close it again before the next shortcut
    chat_page.click("#chatInput")
    chat_page.keyboard.press("Control+KeyK")
    assert chat_page.evaluate("document.activeElement.id") == "chatInput", (
        "Ctrl+K must focus the composer"
    )


# --------------------------------------------------------------------------
# Do the buttons have JS behind them?
# --------------------------------------------------------------------------
@requires_browser
def test_send_button_sends_and_renders(chat_page) -> None:
    """Clicking Send posts to /api/chat/send and renders both messages."""
    chat_page.fill("#chatInput", "hello world")
    chat_page.click("#sendBtn")

    chat_page.wait_for_selector(".msg.assistant .bubble:has-text('pong from stub')")
    assert chat_page.locator(".msg.user").count() == 1, "user turn missing"
    assert chat_page.locator(".msg.assistant").count() == 1, "reply turn missing"
    # The composer is ready for the next message.
    assert chat_page.input_value("#chatInput") == ""


@requires_browser
def test_enter_sends_and_shift_enter_newlines(chat_page) -> None:
    """Enter sends; Shift+Enter inserts a newline instead of sending."""
    box = chat_page.locator("#chatInput")

    box.click()
    chat_page.keyboard.press("Shift+Enter")
    assert "\n" in box.input_value(), "Shift+Enter must insert a newline"

    chat_page.fill("#chatInput", "second message")
    chat_page.keyboard.press("Enter")
    chat_page.wait_for_selector(".msg.assistant .bubble:has-text('pong from stub')")
    assert chat_page.locator(".msg.user").count() == 1, (
        "Shift+Enter must not have sent the draft"
    )


@requires_browser
def test_reset_button_clears_transcript(chat_page) -> None:
    """Reset asks for confirmation, then empties the transcript.

    Headless chromium auto-dismisses ``window.confirm`` as a *dismiss*, which
    is exactly what my first browser probe tripped over: the page's JS is
    fine, the test has to accept the dialog.
    """
    chat_page.on("dialog", lambda d: d.accept())
    chat_page.fill("#chatInput", "reset me")
    chat_page.click("#sendBtn")
    chat_page.wait_for_selector(".msg.assistant .bubble:has-text('pong from stub')")

    chat_page.click("#resetBtn")
    chat_page.wait_for_selector("#emptyState")
    assert chat_page.locator(".msg").count() == 0, "transcript must be empty"


@requires_browser
def test_theme_toggle_switches_theme(chat_page) -> None:
    """The theme button flips data-theme and the aria-label glyph state."""
    before = chat_page.evaluate("document.documentElement.dataset.theme")
    chat_page.click("#themeToggle")
    after = chat_page.evaluate("document.documentElement.dataset.theme")
    assert before != after, "theme toggle did not change [data-theme]"


@requires_browser
def test_slash_button_opens_command_menu(chat_page) -> None:
    """/ Commands opens a filtered menu built from the catalog."""
    chat_page.type("#chatInput", "/")
    chat_page.wait_for_selector(".slash-menu.open .slash-item")

    items = chat_page.locator(".slash-menu.open .slash-item")
    assert items.count() == len(STUB_COMMANDS), (
        f"menu must list the catalog, got {items.count()}"
    )
    # Typing a filter narrows it to `model`.
    chat_page.type("#chatInput", "mod")
    chat_page.wait_for_function(
        "document.querySelectorAll('.slash-menu.open .slash-item').length === 1"
    )


@requires_browser
def test_no_js_errors_on_load(chat_page) -> None:
    """A page that is broken at load time raises JS errors; none are allowed."""
    chat_page.wait_for_timeout(300)  # let late handlers settle
    assert chat_page._js_errors == [], f"console/page errors: {chat_page._js_errors}"
