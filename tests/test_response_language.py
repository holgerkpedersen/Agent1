"""Contract tests for reply-language handling (Kanban card "Agent1 sometimes
change to another language.").

Root cause: the whole harness is English — system prompt, tool schemas, tool
results, logs.  The only language instruction was one clause buried at the end
of a long RULES list ("Be concise. Answer in the user's language."), so after
long English tool loops the model drifted to English and answered a German
user in English.

The contract pinned here:

* ONE module owns language policy: ``agent_core.language``.  Detection,
  override handling and the prompt text all come from it.
* The system prompt carries a dedicated ``LANGUAGE`` section near the top,
  before ``WORKING METHOD`` — prominence, not a buried clause.
* A non-English user message earns a per-turn reminder that NAMES the
  detected language ("Reply in German.").  An English/undecidable message
  adds nothing: the prompt stays byte-identical to the pre-feature baseline.
* ``AGENT1_LANGUAGE`` overrides detection (codes and names); the raw aliases
  turn the directive off.
* The dynamic block is stripped every rebuild, so a long-lived session never
  accumulates stale LANGUAGE blocks.
"""

from __future__ import annotations

import pytest

import agent as agent_mod
import agent_core.language as language


# ── detection: best-effort heuristic, documented and deterministic ─────────


@pytest.mark.parametrize(
    "text,name",
    [
        (
            "Warum ist die Testdatei fehlgeschlagen? Ich habe alles ausgeführt.",
            "German",
        ),
        ("Pourquoi le test a échoué ? Je suis très content, merci beaucoup.", "French"),
        (
            "¿Por qué ha fallado la prueba? Estoy muy contento, muchas gracias.",
            "Spanish",
        ),
        ("Perché il test è fallito? Sono molto contento, grazie mille.", "Italian"),
        ("Waarom is de test mislukt? Ik ben erg blij, dank je wel.", "Dutch"),
        ("Por que o teste falhou? Estou muito feliz, obrigado por tudo.", "Portuguese"),
    ],
)
def test_detect_language_recognises_common_languages(text: str, name: str) -> None:
    assert language.detect_language(text) == name


@pytest.mark.parametrize(
    "text",
    [
        "",
        "run the tests",
        "commit changes",
        "hi",  # greeting: too little evidence to claim a language
    ],
)
def test_detect_language_is_inconclusive_for_english(text: str) -> None:
    """English (the harness's own language) yields no directive.

    English is the documented fallback; naming it per turn would only repeat
    what the base prompt already says, so ``None`` means "add nothing".
    """
    assert language.detect_language(text) is None


def test_detect_language_is_pure_and_deterministic() -> None:
    text = "Warum ist die Testdatei fehlgeschlagen? Ich habe alles ausgeführt."
    assert language.detect_language(text) == language.detect_language(text)


# ── AGENT1_SHELL-style explicit override wins over detection ───────────────


@pytest.mark.parametrize(
    "value,name",
    [
        ("de", "German"),
        ("de-DE", "German"),
        ("german", "German"),
        ("deutsch", "German"),
        ("fr", "French"),
        ("es", "Spanish"),
        ("pt-BR", "Portuguese"),
        ("EN", "English"),
    ],
)
def test_override_language_normalises_codes(
    monkeypatch: pytest.MonkeyPatch, value: str, name: str
) -> None:
    monkeypatch.setenv("AGENT1_LANGUAGE", value)
    assert language.override_language() == name
    # The override beats detection even when the text says otherwise.
    assert language.language_directive("Hello, what is this?").count(name) >= 1


@pytest.mark.parametrize(
    "value,name", [("off", None), ("none", None), ("raw", None), ("auto", None)]
)
def test_override_language_off_disables_the_directive(
    monkeypatch: pytest.MonkeyPatch, value: str, name: None
) -> None:
    monkeypatch.setenv("AGENT1_LANGUAGE", value)
    assert language.override_language() is None
    assert language.language_block("Warum ist die Testdatei fehlgeschlagen?") == ""


def test_override_absent_reads_no_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT1_LANGUAGE", raising=False)
    assert language.override_language() is None


# ── the per-turn reminder names the language ───────────────────────────────


def test_language_block_names_the_detected_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT1_LANGUAGE", raising=False)
    block = language.language_block("Warum ist die Testdatei fehlgeschlagen?")
    assert block.startswith(language.LANGUAGE_MARKER)
    assert "German" in block
    assert "verbatim" in block  # code/paths/logs must never be translated


def test_language_block_is_empty_for_english(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENT1_LANGUAGE", raising=False)
    assert language.language_block("run the tests") == ""
    assert language.language_block(None) == ""


# ── system prompt: one prominent LANGUAGE section, no buried clause ────────


def test_system_prompt_has_a_language_section_before_the_working_method() -> None:
    from agent_core.shell_utils import SYSTEM_PROMPT

    lowered = SYSTEM_PROMPT.lower()
    assert "language" in lowered
    # Prominence: the dedicated section comes before the working method, it
    # is not buried at the end of the rules list.
    assert lowered.index("language (hard rule)") < lowered.index("working method")


def test_system_prompt_language_section_is_declared_once() -> None:
    from agent_core.shell_utils import SYSTEM_PROMPT

    assert SYSTEM_PROMPT.lower().count("language (hard rule)") == 1


def test_system_prompt_no_longer_carries_the_buried_clause() -> None:
    """The old one-clause afterthought is gone — one wording, one meaning."""
    from agent_core.shell_utils import SYSTEM_PROMPT

    assert "answer in the user's language" not in SYSTEM_PROMPT.lower()


def test_system_prompt_language_rules_are_concise() -> None:
    """ASD-STE100 style: short imperative sentences, no hedging words."""
    from agent_core.shell_utils import SYSTEM_PROMPT

    section = SYSTEM_PROMPT.split("LANGUAGE (hard rule):", 1)[1]
    section = section.split("\n\nWORKING METHOD", 1)[0]
    for line in section.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        words = [w for w in line.replace(",", " ").split() if len(w) > 2]
        assert len(words) <= 25, f"clause too long: {line!r}"


# ── dynamic-block plumbing: rebuild every turn, never accumulate ────────────


def test_stripping_removes_the_language_block() -> None:
    from agent_core.run_utils import strip_dynamic_system_blocks

    text = language.LANGUAGE_MARKER + " Reply in German.\n"
    assert strip_dynamic_system_blocks("BASE" + text) == "BASE"


def test_language_block_marker_is_a_marker_constant() -> None:
    # The marker string the stripper must know is the one the builder emits.
    assert language.LANGUAGE_MARKER.startswith("\n\n")
    assert language.LANGUAGE_MARKER.endswith("LANGUAGE:")


# ── agent wiring: reminder lands last, exactly once per rebuild ────────────


def _make_agent(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Build an Agent isolated in *tmp_path*.

    Uses ``monkeypatch.chdir`` (repo convention), NOT a bare ``os.chdir``: a
    leaked cwd makes later test modules that open repo files by relative path
    (e.g. tests/test_agent_module_surface.py reading "agent.py") fail with
    FileNotFoundError for reasons unrelated to the code under test.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch_state = {
        "CHAT_HISTORY_JSON_PATH": str(tmp_path / "chat_history.json"),
        "AGENT_MEMORY_JSON_PATH": str(tmp_path / "agent_memory.json"),
    }
    for key, value in monkeypatch_state.items():
        setattr(agent_mod, key, value)
    monkeypatch.chdir(tmp_path)
    agent = agent_mod.Agent(workspace=str(workspace))
    return agent


def test_refresh_appends_the_language_block_last_exactly_once(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT1_LANGUAGE", raising=False)
    agent = _make_agent(tmp_path, monkeypatch)
    agent._chat_history = [
        {"role": "system", "content": agent_mod._SYSTEM_PROMPT},
        {"role": "user", "content": "Warum ist die Testdatei fehlgeschlagen?"},
    ]

    agent._refresh_system_message("Warum ist die Testdatei fehlgeschlagen?")
    first = str(agent._chat_history[0]["content"])
    assert first.count(language.LANGUAGE_MARKER.strip()) == 1
    # Last: the reminder sits directly before the conversation, closest to it.
    assert first.rstrip().endswith("verbatim.")

    # A second rebuild replaces it — no accumulation of stale language blocks.
    agent._refresh_system_message("Warum ist die Testdatei fehlgeschlagen?")
    second = str(agent._chat_history[0]["content"])
    assert second.count(language.LANGUAGE_MARKER.strip()) == 1


def test_refresh_adds_nothing_for_english_users(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """English (or undecidable) turns keep the prompt byte-identical."""
    monkeypatch.delenv("AGENT1_LANGUAGE", raising=False)
    agent = _make_agent(tmp_path, monkeypatch)
    agent._chat_history = [{"role": "system", "content": agent_mod._SYSTEM_PROMPT}]
    agent._refresh_system_message("run the tests")
    content = str(agent._chat_history[0]["content"])
    assert language.LANGUAGE_MARKER.strip() not in content
