"""Capability + import-shape coverage for the four built-in providers.

We don't load actual TTS / STT models here — that's covered by
``test_smoke_local.py`` (slow). These tests validate that each provider
declares sensible capabilities and answers basic Protocol surface.
"""

from __future__ import annotations

import pytest

from aawazz_mcp import providers  # noqa: F401  - triggers registration
from aawazz_mcp import registry
from aawazz_mcp.provider_base import (
    SttCapabilities,
    SttProvider,
    TtsCapabilities,
    TtsProvider,
)


# ── tiny-tts ────────────────────────────────────────────────────────────────


def test_tiny_tts_registered() -> None:
    p = registry.get_tts("tiny-tts")
    assert p.name == "tiny-tts"
    assert isinstance(p, TtsProvider)


def test_tiny_tts_capabilities() -> None:
    p = registry.get_tts("tiny-tts")
    caps = p.capabilities()
    assert isinstance(caps, TtsCapabilities)
    assert caps.languages == frozenset({"en"})
    assert caps.requires_network is False
    assert caps.accepts_dsp_profiles is True
    assert any(v.id == "tiny-tts:MALE" for v in caps.voices)
    assert caps.speed_range == (0.5, 2.0)


# ── gtts ────────────────────────────────────────────────────────────────────


def test_gtts_registered() -> None:
    p = registry.get_tts("gtts")
    assert p.name == "gtts"
    assert isinstance(p, TtsProvider)


def test_gtts_capabilities_when_installed() -> None:
    pytest.importorskip("gtts")
    p = registry.get_tts("gtts")
    caps = p.capabilities()
    assert caps.requires_network is True
    assert "es" in caps.languages
    assert "ja" in caps.languages
    assert "ne" in caps.languages
    assert caps.sample_rate == 24000


# ── moonshine ───────────────────────────────────────────────────────────────


def test_moonshine_registered() -> None:
    p = registry.get_stt("moonshine")
    assert p.name == "moonshine"
    assert isinstance(p, SttProvider)


def test_moonshine_capabilities() -> None:
    p = registry.get_stt("moonshine")
    caps = p.capabilities()
    assert isinstance(caps, SttCapabilities)
    assert "en" in caps.languages
    # All v1.2 dispatcher languages still covered.
    for expected in ("es", "zh", "ja", "ko", "ar", "vi", "uk"):
        assert expected in caps.languages, f"missing {expected}"
    # Per-language arch table populated.
    assert "tiny_streaming" in caps.model_archs["en"]
    assert "base" in caps.model_archs["es"]


# ── whisper ─────────────────────────────────────────────────────────────────


def test_whisper_registered() -> None:
    p = registry.get_stt("whisper")
    assert p.name == "whisper"
    assert isinstance(p, SttProvider)


def test_whisper_capabilities_when_installed() -> None:
    pytest.importorskip("transformers")
    p = registry.get_stt("whisper")
    caps = p.capabilities()
    assert "ne" in caps.languages
    assert caps.model_archs.get("ne") == ("whisper-small",)


# ── routing assumptions LocalBackend depends on ─────────────────────────────


def test_built_in_set_unchanged() -> None:
    """LocalBackend currently hardcodes ``"tiny-tts"`` / ``"gtts"`` /
    ``"moonshine"`` / ``"whisper"`` as provider names. If any rename, the
    backend wiring breaks — surface that here."""
    tts_names = {p.name for p in registry.list_tts()}
    stt_names = {p.name for p in registry.list_stt()}
    assert "tiny-tts" in tts_names
    assert "gtts" in tts_names
    assert "moonshine" in stt_names
    assert "whisper" in stt_names
