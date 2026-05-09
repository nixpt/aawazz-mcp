"""Language detection for the v1.4 ``respond`` lang_mismatch policy.

Phase 3 of v1.4 (SPEC §5). Wraps ``lingua-language-detector`` and exposes
:func:`detect_language(text) -> ISO 639-1 code | None`. Gracefully no-ops
when the ``[langdetect]`` extra isn't installed — callers see ``None``
and the dispatcher's lang_mismatch policy falls back to ``"off"``
behaviour (synthesize whatever the caller asked for).

The lingua → ISO 639-1 map is hand-curated for v1.4.0; lingua's
``Language`` enum has ~75 entries but we only need the languages our TTS
providers cover (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh, ja,
hu, ko, hi, ne, vi, uk plus their writing-system kin).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("aawazz_mcp.audio.lang_detect")

_LINGUA_AVAILABLE: bool | None = None
_DETECTOR: Any | None = None


# lingua's Language enum names → ISO 639-1 codes covered by aawazz providers.
# Add more here as we add provider language coverage.
_LINGUA_TO_ISO: dict[str, str] = {
    "ENGLISH": "en",
    "SPANISH": "es",
    "FRENCH": "fr",
    "GERMAN": "de",
    "ITALIAN": "it",
    "PORTUGUESE": "pt",
    "POLISH": "pl",
    "TURKISH": "tr",
    "RUSSIAN": "ru",
    "DUTCH": "nl",
    "CZECH": "cs",
    "ARABIC": "ar",
    "CHINESE": "zh",
    "JAPANESE": "ja",
    "HUNGARIAN": "hu",
    "KOREAN": "ko",
    "HINDI": "hi",
    "NEPALI": "ne",
    "VIETNAMESE": "vi",
    "UKRAINIAN": "uk",
    "BENGALI": "bn",
    "MARATHI": "mr",
    "TAMIL": "ta",
    "TELUGU": "te",
    "URDU": "ur",
    "PUNJABI": "pa",
    "GUJARATI": "gu",
    "MALAYALAM": "ml",
    "KANNADA": "kn",
    "PERSIAN": "fa",
    "INDONESIAN": "id",
    "MALAY": "ms",
    "THAI": "th",
    "TURKISH": "tr",
    "GREEK": "el",
    "HEBREW": "he",
    "ROMANIAN": "ro",
    "SWEDISH": "sv",
    "DANISH": "da",
    "NORWEGIAN": "no",
    "FINNISH": "fi",
    "BULGARIAN": "bg",
    "SERBIAN": "sr",
    "CROATIAN": "hr",
    "SLOVAK": "sk",
    "SLOVENE": "sl",
    "CATALAN": "ca",
    "BASQUE": "eu",
    "WELSH": "cy",
    "IRISH": "ga",
    "ICELANDIC": "is",
    "LATVIAN": "lv",
    "LITHUANIAN": "lt",
    "ESTONIAN": "et",
    "AFRIKAANS": "af",
    "SWAHILI": "sw",
    "TAGALOG": "tl",
    "ALBANIAN": "sq",
}


def _ensure_detector() -> Any | None:
    """Build lingua's all-languages detector once, cache. Returns None if
    lingua isn't installed (the [langdetect] extra is opt-in)."""
    global _LINGUA_AVAILABLE, _DETECTOR

    if _LINGUA_AVAILABLE is False:
        return None
    if _DETECTOR is not None:
        return _DETECTOR

    try:
        from lingua import LanguageDetectorBuilder
    except Exception:
        log.debug("lingua-language-detector not installed; lang detection no-ops")
        _LINGUA_AVAILABLE = False
        return None

    try:
        _DETECTOR = LanguageDetectorBuilder.from_all_languages().build()
        _LINGUA_AVAILABLE = True
        return _DETECTOR
    except Exception:
        log.exception("lingua detector init failed")
        _LINGUA_AVAILABLE = False
        return None


def is_available() -> bool:
    """True iff lingua is installed and its detector built successfully.

    Cheap (cached) — safe to call on every respond() invocation.
    """
    return _ensure_detector() is not None


def detect_language(text: str, *, min_chars: int = 4) -> str | None:
    """Detect ``text``'s language and return an ISO 639-1 code, or ``None``
    if undetectable, lingua isn't installed, or the text is too short.

    Args:
        text: The text to classify. Whitespace stripped before analysis.
        min_chars: Skip detection on text below this length (lingua is
            unreliable on very short fragments).

    Returns:
        ISO 639-1 code (``"en"``, ``"ru"``, ``"hi"``, …) or ``None``.
    """
    if not text:
        return None
    text = text.strip()
    if len(text) < min_chars:
        return None

    detector = _ensure_detector()
    if detector is None:
        return None

    try:
        lang = detector.detect_language_of(text)
    except Exception:  # noqa: BLE001 — never break a respond() call over detection
        log.exception("lingua detect_language_of failed")
        return None

    if lang is None:
        return None

    name = getattr(lang, "name", None) or getattr(lang, "iso_code_639_1", None)
    if not isinstance(name, str):
        return None
    iso = _LINGUA_TO_ISO.get(name.upper())
    if iso is None:
        log.debug("lingua language %r not in _LINGUA_TO_ISO map", name)
    return iso
