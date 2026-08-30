"""Regression tests: TTTHEME navbar button glyphs were displaced from their
round buttons (darkmode / monochrome toggles at /index.html).

Root causes fixed:
1. index.html referenced static/ttheme/css/lib/bootstrap.min.css and
   js/lib/{jquery,bootstrap.bundle,iconify-icon}.min.js that did NOT exist
   in the repo -> every Bootstrap layout class used by the round buttons
   (w-40-px/h-40-px rely on d-flex/rounded-circle/justify-content-center)
   was dead, so the glyph sat off-centre in an unstyled square.
2. ttheme/js/app.js wrote the words "dark"/"light" INTO the theme button,
   pushing the CSS ::after glyph out of the circle. It now only sets
   aria-label (the visible glyph comes from style.css).
3. The monochrome toggle's <iconify-icon> never rendered offline because no
   icon source was registered; static/assets/agent1-icon-bundle.js now
   registers every icon used by the dashboard via addIcon().
"""
from __future__ import annotations

import json
import os
import re
import threading
import typing
import urllib.error
import urllib.request
from pathlib import Path

import pytest

#: Repo root derived from this file's location — a hardcoded C:\Dev\Agent1
#: broke CI the moment the checkout landed anywhere else (D:\a\Agent1\...).
REPO = str(Path(__file__).resolve().parent.parent)
LIBS = [
    os.path.join("static", "ttheme", "css", "lib", "bootstrap.min.css"),
    os.path.join("static", "ttheme", "js", "lib", "jquery-3.7.1.min.js"),
    os.path.join("static", "ttheme", "js", "lib", "bootstrap.bundle.min.js"),
    os.path.join("static", "ttheme", "js", "lib", "iconify-icon.min.js"),
]


@pytest.fixture(scope="module")
def base_url() -> typing.Iterator[str]:
    from agent_core.monitoring import DashboardAPIServer

    holder = DashboardAPIServer(None, port=0)  # port 0 -> OS-assigned
    httpd = holder.start()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        shutdown = threading.Thread(target=httpd.shutdown, daemon=True)
        shutdown.start()
        shutdown.join(timeout=5)
        httpd.server_close()
        thread.join(timeout=5)


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""


def test_no_missing_static_refs_on_rendered_page(base_url: str) -> None:
    status, body = _get(base_url + "/index.html")
    assert status == 200
    html = body.decode("utf-8")
    # All @@HEAD@@/@SIDEBAR@@/@@HEADER@@/@@FOOTER@@ markers must be replaced.
    assert "@@" not in html
    refs = sorted(set(re.findall(r'(?:src|href)="(static/[^"]+)"', html)))
    assert refs, "page must reference static assets"
    broken = []
    for ref in refs:
        st, data = _get(f"{base_url}/{ref}")
        if st != 200 or not data:
            broken.append(ref)
    assert broken == [], f"404/broken static refs served to the browser: {broken}"


def test_frontend_libraries_are_vendored() -> None:
    import os

    for rel in LIBS:
        path = os.path.join(REPO, rel)
        assert os.path.isfile(path), f"missing vendored library: {rel}"
        assert os.path.getsize(path) > 10_000, f"suspiciously small library: {rel}"

    css = open(os.path.join(REPO, LIBS[0]), encoding="utf-8",
               errors="replace").read()
    # Layout classes the round navbar buttons depend on:
    for cls in (".rounded-circle", ".d-flex", ".justify-content-center",
                ".align-items-center", ".flex-wrap"):
        assert cls in css, f"bootstrap.min.css lacks {cls} -> buttons misalign"


def test_icon_bundle_registers_all_dashboard_icons_offline() -> None:
    import os

    lib_path = os.path.join(REPO, LIBS[3])
    lib = open(lib_path, encoding="utf-8", errors="replace").read()
    assert "addIcon" in lib, "iconify runtime without addIcon API"

    bundle_path = os.path.join(REPO, "static", "assets", "agent1-icon-bundle.js")
    raw = open(bundle_path, encoding="utf-8").read()
    match = re.search(r"var ICONS = (\{.*?\});", raw, flags=re.S)
    assert match, "agent1-icon-bundle.js has no ICONS registry"
    icons = json.loads(match.group(1))
    # Icons used by partials/_header.html and _sidebar.html:
    for name in ("tabler:palette", "heroicons:bars-3-solid", "mdi:github",
                 "solar:home-smile-angle-outline", "line-md:gauge"):
        assert name in icons, f"offline icon bundle lacks {name}"
    assert len(icons) >= 14
    # Regression guard (cropped-icons bug): every record MUST carry explicit
    # width/height. addIcon() without dimensions falls back to a bogus
    # 16x16 viewBox and crops any icon drawn on a larger grid (24/1024/...).
    for name, rec in icons.items():
        assert isinstance(rec.get("width"), (int, float)) and \
            rec["width"] > 0, f"{name}: missing width -> icon renders cropped"
        assert isinstance(rec.get("height"), (int, float)) and \
            rec["height"] > 0, f"{name}: missing height -> icon renders cropped"
        assert "<svg" not in str(rec.get("body", "")), \
            f"{name}: body must be inner SVG content only"


def test_theme_button_script_does_not_write_text_into_button(base_url: str) -> None:
    """The old bug: innerText 'dark'/'light' pushed the glyph out of the circle."""
    status, body = _get(base_url + "/static/ttheme/js/app.js")
    assert status == 200
    code_only = re.sub(rb"//[^\n]*|/\*.*?\*/", b"", body, flags=re.S)
    assert not re.search(rb"\.innerText\s*=", code_only), \
        "app.js assigns innerText into the theme button again"


def test_header_buttons_keep_round_shape_classes(base_url: str) -> None:
    status, body = _get(base_url + "/index.html")
    assert status == 200
    html = body.decode("utf-8")
    theme_btn = re.search(
        r'<button type="button" data-theme-toggle(.*?)</button>', html, re.S)
    mono_btn = re.search(
        r'<button type="button" onclick="toggleMonochrome\(\)"(.*?)</button>',
        html, re.S)
    assert theme_btn and mono_btn, "navbar toggle buttons missing from header"
    for btn in (theme_btn, mono_btn):
        classes = btn.group(1)
        assert "w-40-px" in classes and "h-40-px" in classes
        assert "rounded-circle" in classes
        assert "justify-content-center" in classes


def _extract_function(source: str, name: str) -> str:
    """Return the top-level `function <name>(...) { ... }` source (brace-matched).

    The parameter list must be skipped by balanced parens first — it may itself
    contain braces (e.g. destructuring like ``({ localStorageTheme })``), which
    would otherwise terminate the scan after the signature only.
    """
    start = source.index(f"function {name}(")
    i = source.index("(", start)
    depth = 0
    for j in range(i, len(source)):
        if source[j] == "(":
            depth += 1
        elif source[j] == ")":
            depth -= 1
            if depth == 0:
                break
    k = source.index("{", j)  # body start, after the closing paren of params
    depth = 0
    for m in range(k, len(source)):
        if source[m] == "{":
            depth += 1
        elif source[m] == "}":
            depth -= 1
            if depth == 0:
                return source[start : m + 1]
    raise AssertionError(f"unbalanced braces while extracting {name}")


def test_dark_mode_is_default_when_no_preference_stored(
    base_url: str, tmp_path: Path
) -> None:
    """Regression: app.js used to force the theme to "light" whenever nothing
    was stored in localStorage. On a fresh visit that overwrote the declared
    <html data-theme="dark"> in index.html and rendered the whole dashboard
    bright, even though the static markup is dark-first.

    The resolver must now honour the attribute declared on <html> (or fall
    back to "dark"), while an explicitly stored user preference still wins.
    """
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node not available to execute the theme resolver")

    status, body = _get(base_url + "/static/ttheme/js/app.js")
    assert status == 200
    fn_src = _extract_function(body.decode("utf-8"), "calculateSettingAsThemeString")

    harness = tmp_path / "theme_resolver_harness.js"
    harness.write_text(
        f"{fn_src}\n"
        "global.document = { documentElement: { getAttribute: () => global.__declared } };\n"
        "const run = (stored, declared) => {\n"
        "  global.__declared = declared;\n"
        "  return calculateSettingAsThemeString({ localStorageTheme: stored });\n"
        "};\n"
        "console.log(JSON.stringify({\n"
        "  fresh_dark_declared: run(null, 'dark'),\n"
        "  fresh_light_declared: run(null, 'light'),\n"
        "  fresh_no_declaration: run(null, null),\n"
        "  stored_light_wins: run('light', 'dark'),\n"
        "  stored_dark_wins: run('dark', null)\n"
        "}));\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        ["node", str(harness)], capture_output=True, text=True, timeout=30
    )
    assert proc.returncode == 0, f"node harness failed: {proc.stderr}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    # Fresh visit with <html data-theme="dark"> (what index.html declares):
    assert result["fresh_dark_declared"] == "dark", \
        "first load must stay dark to match the declared markup"
    # No stored preference and no declaration -> dark-first fallback:
    assert result["fresh_no_declaration"] == "dark"
    # An explicit light declaration is still honoured:
    assert result["fresh_light_declared"] == "light"
    # A stored user preference always wins over the declared attribute:
    assert result["stored_light_wins"] == "light"
    assert result["stored_dark_wins"] == "dark"
