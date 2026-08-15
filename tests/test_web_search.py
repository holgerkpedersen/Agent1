"""Tests for the web_search NLP tool (parsing, formatting, handler, schema)."""
from unittest.mock import patch

from agent_core.tool_schemas import NLP_TOOL_NAMES, NLP_TOOL_SCHEMAS
from agent_core.tools.web_search import (
    MAX_RESULTS_LIMIT,
    UNTRUSTED_MARKER,
    format_results,
    sanitize_query,
    search_ddg,
)

_FIXTURE_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc">Example &amp; Co</a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpage&amp;rut=abc">The <b>example</b> snippet&nbsp;text.</a>
</div>
<div class="result">
  <a class="result__a" href="https://docs.python.org/3/">Python Docs</a>
  <a class="result__snippet" href="https://docs.python.org/3/">Official Python documentation.</a>
</div>
<div class="result">
  <a class="result__a" href="https://duckduckgo.com/">Search Home</a>
  <a class="result__snippet" href="https://duckduckgo.com/">Should be skipped.</a>
</div>
</body></html>
"""


class TestSearchDdgParsing:
    def test_parses_results_with_redirect_unwrap(self):
        with patch("urllib.request.urlopen") as mock:
            mock.return_value.__enter__.return_value.read.return_value = _FIXTURE_HTML.encode()
            results = search_ddg("example", max_results=5)
        assert len(results) == 2
        assert results[0]["url"] == "https://example.com/page"
        assert results[0]["title"] == "Example & Co"
        assert results[0]["snippet"] == "The example snippet text."

    def test_skips_search_infrastructure_domains(self):
        with patch("urllib.request.urlopen") as mock:
            mock.return_value.__enter__.return_value.read.return_value = _FIXTURE_HTML.encode()
            results = search_ddg("example", max_results=5)
        urls = [r["url"] for r in results]
        assert all("duckduckgo.com" not in u for u in urls)

    def test_max_results_capped(self):
        html = "".join(
            f'<a class="result__a" href="https://s{i}.example/">t</a>'
            f'<a class="result__snippet" href="https://s{i}.example/">s</a>'
            for i in range(20)
        )
        with patch("urllib.request.urlopen") as mock:
            mock.return_value.__enter__.return_value.read.return_value = html.encode()
            results = search_ddg("x", max_results=100)
        assert len(results) <= MAX_RESULTS_LIMIT

    def test_network_error_returns_empty(self):
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            assert search_ddg("x") == []


class TestSanitizeQuery:
    def test_trims_and_collapses(self):
        assert sanitize_query("   python   async   ") == "python async"

    def test_truncates_long_queries(self):
        assert len(sanitize_query("x" * 500)) <= 400


class TestFormatResults:
    def test_marks_untrusted(self):
        out = format_results("python", [{"title": "T", "url": "https://e.com", "snippet": "S"}])
        assert UNTRUSTED_MARKER in out
        assert "1. T" in out
        assert "https://e.com" in out

    def test_empty_results(self):
        assert "No web results found" in format_results("x", [])


class TestHandlerAndSchema:
    def test_web_search_schema_registered(self):
        names = [s["function"]["name"] for s in NLP_TOOL_SCHEMAS]
        assert "web_search" in names
        assert "web_search" in NLP_TOOL_NAMES

    def test_handler_formats_output(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))
        results = [{"title": "PEP 8", "url": "https://peps.python.org/pep-0008/", "snippet": "Style guide"}]

        async def run():
            with patch("agent_core.tools.web_search.search_ddg", return_value=results):
                return await agent._execute_tool_call(
                    "web_search", {"query": "python pep8"},
                )

        out = asyncio.run(run())
        assert "PEP 8" in out
        assert UNTRUSTED_MARKER in out

    def test_handler_requires_query(self, tmp_path):
        import asyncio
        from agent import Agent
        agent = Agent(workspace=str(tmp_path))

        async def run():
            return await agent._execute_tool_call("web_search", {})

        assert "requires a query" in asyncio.run(run())
