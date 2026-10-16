"""Regression tests for benchmark.py provider endpoint resolution (P2).

These pin the --provider/--url behaviour of resolve_base_url(): which
chat-completions URL a given provider (and an explicit --url override)
resolves to.  Written BEFORE the lint-defect fixes in benchmark.py; they
must stay green across any refactor of that code.
"""

import benchmark


class TestResolveBaseUrl:
    def test_default_provider_is_lmstudio(self):
        assert benchmark.DEFAULT_PROVIDER == "lmstudio"
        assert benchmark.PROVIDER_BASE_URLS["llama"] == "http://127.0.0.1:8080/v1"

    def test_llama_provider_appends_chat_completions(self):
        url = benchmark.resolve_base_url("llama")
        assert url == "http://127.0.0.1:8080/v1/chat/completions"

    def test_lmstudio_default_when_provider_unknown(self):
        # Unknown providers must not crash the run by producing a broken URL;
        # they fall back to the default provider's endpoint.
        url = benchmark.resolve_base_url("ollama")
        assert url == benchmark.PROVIDER_BASE_URLS[benchmark.DEFAULT_PROVIDER] + "/chat/completions"

    def test_explicit_url_wins_over_provider(self):
        # Explicit --url always beats the --provider default, for any provider.
        url = benchmark.resolve_base_url("llama", "http://192.168.0.5:1234/v1")
        assert url == "http://192.168.0.5:1234/v1/chat/completions"

    def test_bare_host_port_gets_v1_suffix(self):
        url = benchmark.resolve_base_url("llama", "http://192.168.0.71:8080")
        assert url == "http://192.168.0.71:8080/v1/chat/completions"

    def test_full_chat_completions_url_is_not_double_suffixed(self):
        url = benchmark.resolve_base_url("llama", "http://192.168.0.71:8080/v1/chat/completions/")
        assert url == "http://192.168.0.71:8080/v1/chat/completions"

    def test_trailing_slash_on_base_url_is_normalized(self):
        url = benchmark.resolve_base_url("llama", "http://host:8080/v1/")
        assert url == "http://host:8080/v1/chat/completions"
