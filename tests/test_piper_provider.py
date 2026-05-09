"""Coverage for :mod:`aawazz_mcp.providers.piper`.

These tests are non-network — they don't download voices. End-to-end
synthesis is exercised via the smoke harness once a voice is on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aawazz_mcp import providers  # noqa: F401  - register built-ins
from aawazz_mcp import registry
from aawazz_mcp.provider_base import ProviderError, TtsRequest
from aawazz_mcp.providers.piper import (
    _VOICE_RX,
    _lang_from_voice_id,
    _scan_installed_voices,
)


# ── Registration + module surface ──────────────────────────────────────────


def test_piper_registered() -> None:
    p = registry.get_tts("piper")
    assert p.name == "piper"


def test_voice_id_regex_matches_canonical_form() -> None:
    """Match the rhasspy/piper-voices canonical naming."""
    cases = [
        ("en_US-amy-medium", "en", "US", "amy", "medium"),
        ("en_GB-jenny-medium", "en", "GB", "jenny", "medium"),
        ("es_ES-davefx-medium", "es", "ES", "davefx", "medium"),
        ("de_DE-thorsten-low", "de", "DE", "thorsten", "low"),
        # Quality labels can have hyphens (rare but valid).
        ("en_US-libritts_r-medium", "en", "US", "libritts_r", "medium"),
    ]
    for voice_id, fam, region, name, quality in cases:
        m = _VOICE_RX.match(voice_id)
        assert m, f"failed: {voice_id!r}"
        assert m.group("lang_family") == fam
        assert m.group("lang_region") == region
        assert m.group("voice_name") == name
        assert m.group("voice_quality") == quality


def test_voice_id_regex_rejects_malformed() -> None:
    invalid = ["", "en", "en_US", "en_US-amy", "no-region-here", "_US-amy-med"]
    for v in invalid:
        assert _VOICE_RX.match(v) is None, f"unexpectedly matched {v!r}"


def test_lang_from_voice_id() -> None:
    assert _lang_from_voice_id("en_US-amy-medium") == "en"
    assert _lang_from_voice_id("ja_JP-rinne-medium") == "ja"
    assert _lang_from_voice_id("invalid") == ""


# ── Voice scan ─────────────────────────────────────────────────────────────


def test_scan_empty_dir_returns_empty(tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    assert _scan_installed_voices(voices_dir) == {}


def test_scan_missing_dir_returns_empty(tmp_path: Path) -> None:
    voices_dir = tmp_path / "does-not-exist"
    assert _scan_installed_voices(voices_dir) == {}


def test_scan_picks_up_paired_files(tmp_path: Path) -> None:
    """A voice is only counted if both .onnx and .onnx.json exist."""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    paired = voices_dir / "en_US-amy-medium.onnx"
    paired.write_bytes(b"")
    (voices_dir / "en_US-amy-medium.onnx.json").write_text("{}")

    orphan = voices_dir / "en_GB-orphan-medium.onnx"
    orphan.write_bytes(b"")
    # No matching .onnx.json — should be skipped.

    found = _scan_installed_voices(voices_dir)
    assert "en_US-amy-medium" in found
    assert "en_GB-orphan-medium" not in found


def test_scan_skips_invalid_voice_id_filenames(tmp_path: Path) -> None:
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    (voices_dir / "not-a-voice-id.onnx").write_bytes(b"")
    (voices_dir / "not-a-voice-id.onnx.json").write_text("{}")
    assert _scan_installed_voices(voices_dir) == {}


# ── Provider behavior with no voices on disk ───────────────────────────────


def test_capabilities_empty_when_autodownload_off(
    monkeypatch, tmp_path: Path
) -> None:
    """No voices on disk + auto-download off → empty languages so the
    routing chain skips this provider."""
    from aawazz_mcp.providers.piper import PiperTtsProvider

    monkeypatch.setenv("AAWAZZ_PIPER_VOICES_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_PIPER_AUTO_DOWNLOAD", "0")
    p = PiperTtsProvider()
    caps = p.capabilities()
    assert caps.languages == frozenset()
    assert caps.voices == ()


def test_capabilities_advertises_downloadable_when_autodownload_on(
    monkeypatch, tmp_path: Path
) -> None:
    """No voices installed + auto-download on → advertise the rhasspy/piper-voices
    language catalog so the router can route to us; synthesize() does the
    download lazily."""
    from aawazz_mcp.providers.piper import PiperTtsProvider, _DOWNLOADABLE_LANGS

    monkeypatch.setenv("AAWAZZ_PIPER_VOICES_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_PIPER_AUTO_DOWNLOAD", "1")
    p = PiperTtsProvider()
    caps = p.capabilities()
    assert caps.languages == _DOWNLOADABLE_LANGS
    # Voices block stays installed-only — we don't advertise voice IDs we
    # don't have on disk.
    assert caps.voices == ()


@pytest.mark.asyncio
async def test_synthesize_no_voice_no_autodownload_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """With auto-download disabled, requesting an absent voice fails clean."""
    from aawazz_mcp.providers.piper import PiperTtsProvider

    monkeypatch.setenv("AAWAZZ_PIPER_VOICES_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_PIPER_AUTO_DOWNLOAD", "0")
    p = PiperTtsProvider()

    with pytest.raises(ProviderError, match="not installed"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="en",
                voice="en_US-amy-medium",
                output_path=str(tmp_path / "out.wav"),
            )
        )


@pytest.mark.asyncio
async def test_synthesize_invalid_voice_id_raises(
    monkeypatch, tmp_path: Path
) -> None:
    from aawazz_mcp.providers.piper import PiperTtsProvider

    monkeypatch.setenv("AAWAZZ_PIPER_VOICES_DIR", str(tmp_path / "empty"))
    monkeypatch.setenv("AAWAZZ_PIPER_AUTO_DOWNLOAD", "0")
    p = PiperTtsProvider()

    with pytest.raises(ProviderError, match="invalid Piper voice"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="en",
                voice="bogus",
                output_path=str(tmp_path / "out.wav"),
            )
        )


@pytest.mark.asyncio
async def test_synthesize_no_voice_for_lang_raises(
    monkeypatch, tmp_path: Path
) -> None:
    """Caller didn't pick a voice and none is installed for the language."""
    from aawazz_mcp.providers.piper import PiperTtsProvider

    monkeypatch.setenv("AAWAZZ_PIPER_VOICES_DIR", str(tmp_path / "empty"))
    p = PiperTtsProvider()

    with pytest.raises(ProviderError, match="no Piper voice"):
        await p.synthesize(
            TtsRequest(
                text="hi",
                language="en",
                voice=None,
                output_path=str(tmp_path / "out.wav"),
            )
        )
