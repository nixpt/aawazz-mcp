"""Coverage for :mod:`aawazz_mcp.providers.xtts` — non-network unit tests.

These tests don't actually load the ~2 GB XTTS-v2 model. We exercise
the registration / capability surface and the speaker-wav resolution
logic, which is where most of XTTS's provider-side complexity lives.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aawazz_mcp import providers  # noqa: F401  - register
from aawazz_mcp import registry
from aawazz_mcp.provider_base import ProviderError, TtsRequest
from aawazz_mcp.providers.xtts import (
    _XTTS_LANGUAGES,
    XttsTtsProvider,
)


def test_xtts_registered() -> None:
    p = registry.get_tts("xtts")
    assert p.name == "xtts"


def test_xtts_languages_match_model_card() -> None:
    """17 languages per the XTTS-v2 model card."""
    assert len(_XTTS_LANGUAGES) == 17
    for code in ("en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
                 "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"):
        assert code in _XTTS_LANGUAGES


def test_capabilities_empty_when_not_installed(monkeypatch) -> None:
    """Phase-3 lazy-import pattern: provider exposes empty capabilities
    when the heavy ``TTS`` package is absent so the router skips us."""
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: False
    )
    p = XttsTtsProvider()
    caps = p.capabilities()
    assert caps.languages == frozenset()


def test_capabilities_lists_languages_when_installed(monkeypatch) -> None:
    """When TTS is installed, capabilities advertises the full XTTS lang
    set so the router can route any of the 17."""
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: True
    )
    p = XttsTtsProvider()
    caps = p.capabilities()
    assert caps.languages == _XTTS_LANGUAGES
    # XTTS opts out of DSP — its own expressivity control is the right
    # axis to vary, not numpy post-processing.
    assert caps.accepts_dsp_profiles is False
    assert caps.speed_range == (1.0, 1.0)


def test_speaker_wav_from_voice_clone_prefix(monkeypatch) -> None:
    """voice='xtts:cloned-from-/path/to/ref.wav' → speaker_wav='/path/...'."""
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: True
    )
    p = XttsTtsProvider()

    req = TtsRequest(
        text="hi",
        language="en",
        voice="xtts:cloned-from-/abs/voices/me.wav",
    )
    assert p._resolve_speaker_wav(req) == "/abs/voices/me.wav"


def test_speaker_wav_from_extra_overrides_voice(monkeypatch) -> None:
    """extra={'speaker_wav': ...} wins over the voice= prefix."""
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: True
    )
    p = XttsTtsProvider()

    req = TtsRequest(
        text="hi",
        language="en",
        voice="xtts:cloned-from-/loser.wav",
        extra={"speaker_wav": "/winner.wav"},
    )
    assert p._resolve_speaker_wav(req) == "/winner.wav"


def test_speaker_wav_none_when_neither_provided(monkeypatch) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: True
    )
    p = XttsTtsProvider()
    req = TtsRequest(text="hi", language="en")
    assert p._resolve_speaker_wav(req) is None


@pytest.mark.asyncio
async def test_synthesize_unsupported_language_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """vi (Vietnamese) isn't in XTTS-v2; must hard-fail before model load."""
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: True
    )
    p = XttsTtsProvider()
    with pytest.raises(ProviderError, match="does not support language"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="vi",
                voice="xtts:cloned-from-/x.wav",
                output_path=str(tmp_path / "out.wav"),
            )
        )


@pytest.mark.asyncio
async def test_synthesize_no_speaker_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """XTTS must hard-fail without a speaker_wav or built-in speaker name —
    we do NOT silently load a default that may surprise the caller."""
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: True
    )
    p = XttsTtsProvider()
    with pytest.raises(ProviderError, match="reference WAV or a built-in speaker"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="en",
                voice=None,
                output_path=str(tmp_path / "out.wav"),
            )
        )


@pytest.mark.asyncio
async def test_synthesize_not_installed_raises(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "aawazz_mcp.providers.xtts._probe_xtts", lambda: False
    )
    p = XttsTtsProvider()
    with pytest.raises(ProviderError, match="not installed"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="en",
                voice="xtts:cloned-from-/x.wav",
                output_path=str(tmp_path / "out.wav"),
            )
        )
