"""Reply-language policy — the ONE owner of how Agent1 picks its answer language.

Kanban card: "Agent1 sometimes change to another language."  The harness is
English end to end (system prompt, tool schemas, tool results, logs), and the
only language instruction was one clause buried at the end of a long rules
list.  After long English tool loops the models drifted to English and
answered non-English users in English.

This module makes the policy explicit and centralised:

* :data:`LANGUAGE_MARKER` — the marker that marks the per-turn dynamic block.
  ``agent_core.run_utils.strip_dynamic_system_blocks`` cuts on it, so a
  long-lived session rebuilds one fresh language block each turn instead of
  accumulating stale copies.
* :func:`detect_language` — pure, deterministic best-effort heuristic over
  the user's own text (script evidence first, then function-word evidence).
* :func:`override_language` — the explicit ``AGENT1_LANGUAGE`` override
  (same pattern as ``AGENT1_SHELL``): codes and names map to a canonical
  English language name, the raw aliases turn the directive off.
* :func:`language_directive` / :func:`language_block` — the prompt text.

Best-effort by design: the heuristic names a language only when the evidence
is clear (two distinct native markers and a unique winner).  English is the
documented fallback of this English-language harness, so an inconclusive
message yields ``None`` and adds NOTHING to the prompt — the prompt stays
byte-identical to the pre-feature baseline for English users.

The system-prompt section text lives in :data:`LANGUAGE_SECTION`;
``agent_core.shell_utils.SYSTEM_PROMPT`` embeds it verbatim, so the rule the
model reads and the rule this module documents never diverge.
"""

from __future__ import annotations

import os
import re
import unicodedata

__all__ = [
    "ENV_LANGUAGE",
    "LANGUAGE_MARKER",
    "LANGUAGE_SECTION",
    "detect_language",
    "language_block",
    "language_directive",
    "override_language",
]

#: Environment variable that pins the reply language explicitly (user intent
#: wins over detection, same pattern as ``AGENT1_SHELL``).
ENV_LANGUAGE = "AGENT1_LANGUAGE"

#: Marker that introduces the per-turn language block in the system prompt.
#: The dynamic-block stripper cuts here so blocks never accumulate.
LANGUAGE_MARKER = "\n\nLANGUAGE:"

#: Aliases that switch the directive off: no per-turn language reminder.
_OFF_ALIASES = frozenset({"", "off", "none", "raw", "default", "auto"})

#: Code/name aliases mapped to canonical English display names.  Keys are
#: lower-case; both ISO codes (with optional region) and English/German names
#: resolve.  One meaning per name: the value is what the prompt shows.
_LANGUAGE_ALIASES: dict[str, str] = {
    "de": "German",
    "de-de": "German",
    "de-at": "German",
    "de-ch": "German",
    "deu": "German",
    "german": "German",
    "deutsch": "German",
    "fr": "French",
    "fr-fr": "French",
    "fra": "French",
    "french": "French",
    "français": "French",
    "es": "Spanish",
    "es-es": "Spanish",
    "es-mx": "Spanish",
    "spa": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "it-it": "Italian",
    "ita": "Italian",
    "italian": "Italian",
    "nl": "Dutch",
    "nl-nl": "Dutch",
    "nl-be": "Dutch",
    "nld": "Dutch",
    "dutch": "Dutch",
    "pt": "Portuguese",
    "pt-pt": "Portuguese",
    "pt-br": "Portuguese",
    "por": "Portuguese",
    "portuguese": "Portuguese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "eng": "English",
    "english": "English",
    "ru": "Russian",
    "ru-ru": "Russian",
    "rus": "Russian",
    "russian": "Russian",
    "ar": "Arabic",
    "arabic": "Arabic",
    "he": "Hebrew",
    "hebrew": "Hebrew",
    "el": "Greek",
    "greek": "Greek",
    "tr": "Turkish",
    "tr-tr": "Turkish",
    "turkish": "Turkish",
    "pl": "Polish",
    "polish": "Polish",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "chinese": "Chinese",
    "ja": "Japanese",
    "ja-jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "ko-kr": "Korean",
    "korean": "Korean",
}

# Language evidence: words that carry a language's function-word fingerprint.
# Matching is accent-insensitive and whole-word only.  A language is named
# when it has at least _MIN_DISTINCT_MARKERS distinct hits AND wins outright;
# ties mean "inconclusive" — better silence than a wrong directive.
_WORD_EVIDENCE: dict[str, frozenset[str]] = {
    # Diacritic-only markers are counted as one extra marker each (see
    # _markers); ß/ł/ñ and friends are unambiguous for these languages.
    "German": frozenset(
        {
            "und",
            "ist",
            "nicht",
            "die",
            "der",
            "das",
            "den",
            "dem",
            "des",
            "ein",
            "eine",
            "einen",
            "einem",
            "einer",
            "mit",
            "von",
            "fuer",
            "auf",
            "aus",
            "bei",
            "um",
            "aber",
            "auch",
            "noch",
            "schon",
            "sehr",
            "mehr",
            "kann",
            "muss",
            "soll",
            "will",
            "ich",
            "du",
            "wir",
            "mir",
            "mich",
            "dich",
            "mein",
            "meine",
            "dein",
            "wie",
            "was",
            "wer",
            "wo",
            "warum",
            "weil",
            "dass",
            "ueber",
            "unter",
            "zwischen",
            "ja",
            "nein",
            "bitte",
            "danke",
        }
    ),
    "French": frozenset(
        {
            "le",
            "la",
            "les",
            "des",
            "une",
            "un",
            "est",
            "dans",
            "pour",
            "avec",
            "pas",
            "plus",
            "tres",
            "oui",
            "non",
            "merci",
            "bonjour",
            "salut",
            "comment",
            "pourquoi",
            "quand",
            "quoi",
            "faire",
            "vous",
            "nous",
            "ils",
            "elles",
            "cette",
            "ce",
            "cet",
            "sur",
            "sous",
            "entre",
            "chez",
            "sans",
            "parce",
            "que",
            "qui",
            "aux",
            "au",
            "du",
            "mon",
            "ma",
            "mes",
            "ton",
            "ta",
            "tes",
        }
    ),
    "Spanish": frozenset(
        {
            "el",
            "los",
            "las",
            "una",
            "uno",
            "que",
            "del",
            "por",
            "para",
            "con",
            "sin",
            "muy",
            "mas",
            "si",
            "hola",
            "adios",
            "gracias",
            "como",
            "cuando",
            "donde",
            "hacer",
            "porque",
            "estoy",
            "esta",
            "este",
            "esto",
            "son",
            "ser",
            "estar",
            "tengo",
            "tener",
            "ese",
            "esa",
        }
    ),
    "Italian": frozenset(
        {
            "che",
            "e",
            "per",
            "con",
            "non",
            "sono",
            "come",
            "perche",
            "grazie",
            "ciao",
            "molto",
            "piu",
            "questo",
            "questa",
            "il",
            "la",
            "le",
            "gli",
            "una",
            "uno",
            "del",
            "della",
            "nel",
            "sul",
            "mio",
            "mia",
            "tuo",
            "tua",
        }
    ),
    "Dutch": frozenset(
        {
            "en",
            "de",
            "het",
            "een",
            "is",
            "niet",
            "maar",
            "ook",
            "nog",
            "zeer",
            "meer",
            "ja",
            "nee",
            "dank",
            "hallo",
            "hoe",
            "waarom",
            "wanneer",
            "maken",
            "ik",
            "jij",
            "hij",
            "wij",
            "mijn",
            "jouw",
        }
    ),
    "Portuguese": frozenset(
        {
            "o",
            "a",
            "os",
            "as",
            "um",
            "uma",
            "que",
            "e",
            "de",
            "do",
            "da",
            "em",
            "por",
            "para",
            "com",
            "sem",
            "muito",
            "mais",
            "sim",
            "nao",
            "ola",
            "obrigado",
            "obrigada",
            "como",
            "quando",
            "porque",
            "eu",
            "voce",
            "nos",
            "meu",
            "minha",
        }
    ),
}

#: Characters whose presence (any count) is a strong per-language hint inside
#: Latin-script text.  Accent-insensitive matching strips them, so they are
#: counted separately on the raw text before stripping.
_DISTINCTIVE_CHARS: dict[str, frozenset[str]] = {
    "German": frozenset("äöüß"),
    "French": frozenset("œêîûëïùàâç"),
    "Spanish": frozenset("ñ¿¡"),
    "Italian": frozenset("ìò"),
    "Portuguese": frozenset("ãõ"),
    "Polish": frozenset("łśżźćń"),
    "Turkish": frozenset("ğışİ"),
}

#: Script evidence: a whole script is decisive, one character suffices.
#: Checked in order; the first hit wins (Japanese kana before Chinese Han in
#: mixed Japanese text).
_SCRIPT_EVIDENCE: tuple[tuple[str, str], ...] = (
    ("Japanese", "\u3040-\u30ff"),  # Hiragana + Katakana
    ("Korean", "\u1100-\u11ff\uac00-\ud7af"),  # Hangul jamo + syllables
    ("Chinese", "\u4e00-\u9fff\u3400-\u4dbf"),  # CJK ideographs
    ("Russian", "\u0400-\u04ff"),  # Cyrillic
    ("Arabic", "\u0600-\u06ff"),
    ("Hebrew", "\u0590-\u05ff"),
    ("Greek", "\u0370-\u03ff"),
    ("Hindi", "\u0900-\u097f"),  # Devanagari
    ("Thai", "\u0e00-\u0e7f"),
)

# One word token: letters (any script), digits, apostrophes and hyphens.
_TOKEN_RE = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)


def _fold(text: str) -> str:
    """Lower-case *text* and strip accents (evidence matching is accent-blind).

    ``Warum`` and ``warum``, ``très`` and ``tres`` must score the same; the
    word lists are stored in this folded, accent-stripped form.
    """
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.lower()


def override_language() -> str | None:
    """Return the canonical name pinned by ``AGENT1_LANGUAGE``, else ``None``.

    Same precedence pattern as ``AGENT1_SHELL``: an explicit user setting wins
    over detection.  Empty/absent, and the raw aliases (off, none, raw,
    default, auto), mean "no directive" — :func:`detect_language` decides.
    """
    value = (os.environ.get(ENV_LANGUAGE) or "").strip().lower()
    if not value or value in _OFF_ALIASES:
        return None
    return _LANGUAGE_ALIASES.get(value)


def detect_language(text: str | None) -> str | None:
    """Best-effort language of the user's own text, ``None`` when unclear.

    Deterministic and pure: same input, same answer.  Script evidence wins
    outright (a Cyrillic sentence is Russian); Latin-script text needs at
    least two distinct native markers and a unique winner among the
    candidates.  Ties, one marker only, or plain English yield ``None`` —
    English is this harness's own language and the documented fallback, so
    nothing extra gets injected for it.

    A message that mixes languages (a German user pasting an English log)
    resolves by volume: the language of the user's own prose outvotes a short
    quoted snippet because function words dominate the token stream.
    """
    if not text or not text.strip():
        return None
    # 1. Script evidence: decisive, no threshold games.
    for name, rng in _SCRIPT_EVIDENCE:
        if re.search(f"[{rng}]", text):
            return name
    # 2. Latin-script languages: count distinct function-word markers.
    tokens = {_fold(m) for m in _TOKEN_RE.findall(text)}
    distinctive = {ch for ch in text.lower() if unicodedata.combining(ch)} or set(
        ch for ch in text.lower() if ch in "ßłñğışİ"
    )
    scores: dict[str, int] = {}
    for name, words in _WORD_EVIDENCE.items():
        markers = len(words & tokens)
        extra = len(_DISTINCTIVE_CHARS.get(name, frozenset()) & distinctive)
        scores[name] = markers + extra
    best = max(scores.values(), default=0)
    if best < 2:
        return None
    winners = [name for name, score in scores.items() if score == best]
    if len(winners) != 1:
        return None  # inconclusive: refuse to guess between candidates
    return winners[0]


def language_directive(user_text: str | None = None) -> str:
    """Return the one-line directive naming the reply language, else ``""``.

    Precedence: an explicit off alias (off, none, raw, default, auto)
    disables the per-turn reminder outright — no detection fallback.  An
    explicit language wins over detection; a plain setting falls through to
    :func:`detect_language` on *user_text*.  No language decided means an
    empty string — the prompt stays byte-identical to the baseline.
    """
    value = (os.environ.get(ENV_LANGUAGE) or "").strip().lower()
    if value and value in _OFF_ALIASES:
        return ""  # explicit opt-out wins over detection outright
    name = _LANGUAGE_ALIASES.get(value) if value else None
    if name is None:
        name = detect_language(user_text)
    if not name:
        return ""
    return (
        f"Reply in {name}, the language of the user's latest message.\n"
        "Do not switch languages because tool output, logs, file contents\n"
        "or code are in another language. Keep code, commands, paths and\n"
        "identifiers verbatim."
    )


def language_block(user_text: str | None = None) -> str:
    """Render the per-turn dynamic language block (empty when nothing applies).

    Appended LAST in the system message, directly before the conversation, so
    the reminder sits closer to the model's attention than any tool output.
    Rebuilt every turn; :func:`agent_core.run_utils.strip_dynamic_system_blocks`
    cuts it at :data:`LANGUAGE_MARKER` so one copy exists per rebuild.
    """
    directive = language_directive(user_text)
    if not directive:
        return ""
    return f"{LANGUAGE_MARKER} {directive}\n"


#: The dedicated system-prompt section, embedded verbatim by
#: ``shell_utils.SYSTEM_PROMPT`` before the working method (prominence).
#: Style follows the workspace ASD-STE100 runbook: one instruction per line,
#: active voice, no hedging.
LANGUAGE_SECTION = (
    "LANGUAGE (hard rule):\n"
    "Answer in the language of the user's latest message.\n"
    "If their message mixes languages, answer in the language they wrote\n"
    "themselves.\n"
    "Do not switch because tool output, logs, file contents or code are in\n"
    "another language. Keep code, commands, paths, identifiers and tool\n"
    "output verbatim. Never translate them.\n"
    "Never add a translation of your answer into another language.\n\n"
)
