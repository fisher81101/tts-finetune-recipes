"""Shared text normalization + G2P helper.

Used at both dataset prep AND inference so phonemes always match.
Splits text on lexicon words (longest-match-first, word-boundary) and runs
misaki espeak on everything else. Splices in lexicon IPA verbatim.

Add your own domain vocabulary to lexicon.json.
Extend normalize_text() for your own language/domain.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

# ── Text normalization ────────────────────────────────────────────────────────
# Shared with 01_prepare_dataset.py — extend these for your own language/domain.

_RUPEES_RE       = re.compile(r"Rs\.?\s*(\d[\d,]*)", flags=re.IGNORECASE)
_INDIAN_GROUP_RE = re.compile(r"\b\d{1,2}(?:,\d{2})+,\d{3}\b")
_LONG_DIGITS_RE  = re.compile(r"\b\d{7,}\b")
_WORD_RE         = re.compile(r"\b[A-Za-z]+\b")
_LC_RUN_RE       = re.compile(r"[a-z]{2,}")

_ONES = [
    "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _two(n: int) -> str:
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + (" " + _ONES[o] if o else "")


def _three(n: int) -> str:
    h, rest = divmod(n, 100)
    parts = []
    if h:
        parts.append(_ONES[h] + " hundred")
    if rest:
        if parts:
            parts.append("and")
        parts.append(_two(rest))
    return " ".join(parts)


def _indian_number_words(n: int) -> str:
    if n == 0:
        return "zero"
    parts = []
    crore, rest = divmod(n, 10_000_000)
    if crore:
        parts.append(_indian_number_words(crore) + " crore")
    lakh, rest = divmod(rest, 100_000)
    if lakh:
        parts.append(_two(lakh) + " lakh")
    thousand, rest = divmod(rest, 1000)
    if thousand:
        parts.append(_two(thousand) + " thousand")
    if rest:
        parts.append(_three(rest))
    return " ".join(parts)


def _expand_rupees(m):
    try:
        return _indian_number_words(int(m.group(1).replace(",", ""))) + " rupees"
    except ValueError:
        return m.group(0)


def _expand_indian_group(m):
    try:
        return _indian_number_words(int(m.group(0).replace(",", "")))
    except ValueError:
        return m.group(0)


def _maybe_spell_acronym(m):
    word = m.group(0)
    if sum(1 for c in word if c.isupper()) < 2:
        return word
    if _LC_RUN_RE.search(word):
        return word
    return " ".join(c.upper() for c in word)


def normalize_text(text: str) -> str:
    """Normalize a transcript string before G2P.

    Extend this function for your own language or domain:
    - Add regex patterns for dates, units, abbreviations
    - Replace rupee expander with your currency
    - Swap acronym spellout for your language conventions
    """
    text = text.replace("@", " at ").replace("&", " and ")
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))
    text = re.sub(r"(?<=[A-Za-z])/(?=[A-Za-z])", " ", text)
    text = _RUPEES_RE.sub(_expand_rupees, text)
    text = _LONG_DIGITS_RE.sub(lambda m: " ".join(m.group(0)), text)
    text = _INDIAN_GROUP_RE.sub(_expand_indian_group, text)
    text = _WORD_RE.sub(_maybe_spell_acronym, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

_LEX_PATH = Path(__file__).with_name("lexicon.json")

_lexicon = None
_g2p = None
_pattern = None
_language = None


def _load(language: str = "en-gb"):
    global _lexicon, _pattern, _g2p, _language
    if _lexicon is None:
        raw = json.loads(_LEX_PATH.read_text(encoding="utf-8")) if _LEX_PATH.exists() else {}
        _lexicon = {k: v for k, v in raw.items() if not k.startswith("_")}
        if _lexicon:
            keys = sorted(_lexicon.keys(), key=len, reverse=True)
            _pattern = re.compile(
                r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b",
                re.IGNORECASE,
            )
    if _g2p is None or _language != language:
        from misaki import espeak
        _g2p = espeak.EspeakG2P(language=language)
        _language = language


def _clean_ipa(ipa: str) -> str:
    return ipa.replace("ː", "").replace("̃", "")


def g2p_with_lexicon(text: str, language: str = "en-gb") -> str:
    """Convert text to IPA phonemes using lexicon overrides + espeak fallback."""
    _load(language)
    lex = _lexicon

    if not lex or not _pattern:
        ipa, _ = _g2p(text)
        return _clean_ipa(ipa)

    parts: list[str] = []
    last = 0
    for m in _pattern.finditer(text):
        if m.start() > last:
            ipa, _ = _g2p(text[last:m.start()])
            parts.append(_clean_ipa(ipa))
        canon = next(k for k in lex if k.lower() == m.group(0).lower())
        parts.append(" " + lex[canon] + " ")
        last = m.end()
    if last < len(text):
        ipa, _ = _g2p(text[last:])
        parts.append(_clean_ipa(ipa))

    return "".join(parts).strip()
