"""Coverage for :mod:`aawazz_mcp.providers.kokoro` — non-network unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from aawazz_mcp import providers  # noqa: F401  - register
from aawazz_mcp import registry
from aawazz_mcp.provider_base import ProviderError, TtsRequest
from aawazz_mcp.providers.kokoro import (
    _LOCALE_LANG,
    _VOICE_CATALOG,
    _VOICE_RX,
    _lang_from_voice_id,
    _lang_tag_from_voice_id,
)


def test_kokoro_registered() -> None:
    p = registry.get_tts("kokoro")
    assert p.name == "kokoro"


def test_voice_id_regex_matches_canonical_form() -> None:
    cases = [
        ("af_bella", "a", "f", "bella"),
        ("am_adam", "a", "m", "adam"),
        ("bf_emma", "b", "f", "emma"),
        ("jm_kumo", "j", "m", "kumo"),
        ("zf_xiaoxiao", "z", "f", "xiaoxiao"),
    ]
    for voice_id, locale, gender, name in cases:
        m = _VOICE_RX.match(voice_id)
        assert m, voice_id
        assert m.group("locale") == locale
        assert m.group("gender") == gender
        assert m.group("name") == name


def test_voice_id_regex_rejects_malformed() -> None:
    for v in ["", "f_bella", "afbella", "Af_Bella", "a_bella"]:
        assert _VOICE_RX.match(v) is None, f"unexpectedly matched {v!r}"


def test_lang_from_voice_id_covers_all_locales() -> None:
    """Every locale prefix in _LOCALE_LANG resolves to a non-empty lang."""
    for prefix, lang in _LOCALE_LANG.items():
        sample_voice = f"{prefix}f_test"
        assert _lang_from_voice_id(sample_voice) == lang


def test_lang_tag_from_voice_id_returns_kokoro_tag() -> None:
    """Kokoro's ``lang`` argument is an espeak-ng tag, not ISO 639-1.
    Notable divergence: Mandarin is ``cmn``, not ``zh``."""
    assert _lang_tag_from_voice_id("af_bella") == "en-us"
    assert _lang_tag_from_voice_id("bf_emma") == "en-gb"
    assert _lang_tag_from_voice_id("jf_alpha") == "ja"
    assert _lang_tag_from_voice_id("zf_xiaoxiao") == "cmn"
    assert _lang_tag_from_voice_id("ff_siwis") == "fr-fr"


def test_voice_catalog_all_resolve_to_known_languages() -> None:
    """Every voice in the static catalog has a recognized locale prefix."""
    for voice_id in _VOICE_CATALOG:
        assert _VOICE_RX.match(voice_id), voice_id
        assert _lang_from_voice_id(voice_id) in _LOCALE_LANG.values(), voice_id


def test_capabilities_empty_when_autodownload_off(
    monkeypatch, tmp_path: Path
) -> None:
    from aawazz_mcp.providers.kokoro import KokoroTtsProvider

    monkeypatch.setenv("AAWAZZ_KOKORO_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_KOKORO_AUTO_DOWNLOAD", "0")
    p = KokoroTtsProvider()
    caps = p.capabilities()
    assert caps.languages == frozenset()


def test_capabilities_advertises_languages_when_autodownload_on(
    monkeypatch, tmp_path: Path
) -> None:
    """Auto-download lets the router send language=en/ja/zh/etc. our way;
    synthesize() does the model fetch lazily."""
    from aawazz_mcp.providers.kokoro import KokoroTtsProvider

    monkeypatch.setenv("AAWAZZ_KOKORO_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_KOKORO_AUTO_DOWNLOAD", "1")
    p = KokoroTtsProvider()
    caps = p.capabilities()
    # Catalog spans 8 distinct languages (en, ja, zh, es, fr, hi, it, pt).
    assert caps.languages == frozenset(_LOCALE_LANG.values())
    assert len(caps.voices) == len(_VOICE_CATALOG)


@pytest.mark.asyncio
async def test_synthesize_unknown_voice_raises(
    monkeypatch, tmp_path: Path
) -> None:
    from aawazz_mcp.providers.kokoro import KokoroTtsProvider

    monkeypatch.setenv("AAWAZZ_KOKORO_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_KOKORO_AUTO_DOWNLOAD", "0")
    p = KokoroTtsProvider()

    with pytest.raises(ProviderError, match="unknown Kokoro voice"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="en",
                voice="kokoro:not-a-real-voice",
                output_path=str(tmp_path / "out.wav"),
            )
        )


@pytest.mark.asyncio
async def test_synthesize_unsupported_default_lang_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """No default voice for an unmapped language → clean error."""
    from aawazz_mcp.providers.kokoro import KokoroTtsProvider

    monkeypatch.setenv("AAWAZZ_KOKORO_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_KOKORO_AUTO_DOWNLOAD", "0")
    p = KokoroTtsProvider()

    with pytest.raises(ProviderError, match="no default Kokoro voice"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="ar",  # not in our defaults map
                voice=None,
                output_path=str(tmp_path / "out.wav"),
            )
        )
