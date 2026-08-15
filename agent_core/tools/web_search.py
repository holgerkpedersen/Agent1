"""DuckDuckGo web search for the NLP tool loop.

Dependency-free client (urllib + DuckDuckGo HTML endpoint), adapted from the
ReactAgent repo (commands/websearch/websearch_command.py).  Query-only: no
arbitrary URL fetching (decision #006 — protects local services from SSRF).
Results are treated as UNTRUSTED input (decision #004) and formatted with an
explicit marker before they are fed back into the conversation.
"""
from __future__ import annotations

import html
import os
import re
import urllib.parse
import urllib.request
from typing import Any

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

#: Search/ad infrastructure domains — skipped in results.
_SKIP_DOMAINS = frozenset({
    "duckduckgo.com", "google.com", "bing.com", "yahoo.com",
    "facebook.com", "twitter.com", "instagram.com",
    "amazon.com", "amazon.co.uk", "amazon.de",
})

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS_LIMIT = 10
MAX_QUERY_LENGTH = 400
MAX_OUTPUT_CHARS = 5000

#: Marker prepended to every search result block (decision #005): web content
#: is attacker-controlled and must not be trusted like local file content.
UNTRUSTED_MARKER = "[UNTRUSTED WEB CONTENT]"

_REQUEST_TIMEOUT = float(os.environ.get("WEB_SEARCH_TIMEOUT", "10"))


def sanitize_query(query: str) -> str:
    """Trim, collapse whitespace, and truncate the search query."""
    value = re.sub(r"\s+", " ", query.strip())
    return value[:MAX_QUERY_LENGTH]


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(value))
    text = text.replace("\xa0", " ")  # &nbsp; decodes to a non-breaking space
    return re.sub(r"\s{2,}", " ", text).strip()


def _domain(url: str) -> str:
    m = re.match(r"https?://([^/]+)", url)
    return m.group(1).lstrip("www.") if m else ""


def search_ddg(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """Query DuckDuckGo HTML and return ``{title, url, snippet}`` dicts.

    Returns an empty list on network errors; callers surface the failure.
    """
    params = urllib.parse.urlencode({"q": query, "kl": "en-us"})
    url = f"https://html.duckduckgo.com/html/?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            page = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    blocks = re.findall(
        r'<a[^>]+class=["\']result__a["\'][^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>'
        r'.*?<a[^>]+class=["\']result__snippet["\'][^>]*>(.*?)</a>',
        page, re.DOTALL,
    )
    for raw_url, raw_title, raw_snippet in blocks:
        if raw_url.startswith("/"):
            uddg = re.search(r"uddg=([^&]+)", raw_url)
            if not uddg:
                continue
            raw_url = urllib.parse.unquote(uddg.group(1))
        if not raw_url.startswith("http"):
            continue
        if _domain(raw_url) in _SKIP_DOMAINS:
            continue
        results.append({
            "title": _strip_tags(raw_title),
            "url": raw_url,
            "snippet": _strip_tags(raw_snippet),
        })
        if len(results) >= max(1, min(max_results, MAX_RESULTS_LIMIT)):
            break
    return results


def format_results(query: str, results: list[dict[str, Any]]) -> str:
    """Numbered, untrusted-marked result block for the tool response."""
    if not results:
        return f"No web results found for: {query}"
    lines = [f"Web search results for: {query}", UNTRUSTED_MARKER]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}")
        lines.append(f"   {r.get('url', '')}")
        snippet = r.get("snippet", "").strip()
        if snippet:
            lines.append(f"   {snippet}")
    output = "\n".join(lines)
    return output[:MAX_OUTPUT_CHARS]
