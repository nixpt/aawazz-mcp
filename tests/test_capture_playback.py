"""Coverage for phase-6 capture + playback providers."""

from __future__ import annotations

import pytest

from aawazz_mcp import providers  # noqa: F401  - register
from aawazz_mcp import registry
from aawazz_mcp.provider_base import (
    CaptureRequest,
    ProviderError,
)


# ── Registration ────────────────────────────────────────────────────────────


def test_sounddevice_capture_registered() -> None:
    p = registry.get_capture("sounddevice")
    assert p.name == "sounddevice"


def test_shell_playback_registered() -> None:
    p = registry.get_playback("shell")
    assert p.name == "shell"


def test_capture_unknown_raises() -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get_capture("does-not-exist")


def test_playback_unknown_raises() -> None:
    with pytest.raises(KeyError, match="not registered"):
        registry.get_playback("does-not-exist")


# ── Capability probes plumb through monkeypatch ─────────────────────────────


def test_sounddevice_has_input_device_observes_monkeypatch(monkeypatch) -> None:
    """The provider does a module-level lookup so tests can patch the canonical
    function without rebinding the provider's reference."""
    p = registry.get_capture("sounddevice")

    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.has_input_device", lambda: False
    )
    assert p.has_input_device() is False

    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.has_input_device", lambda: True
    )
    assert p.has_input_device() is True


def test_shell_playback_has_player_observes_monkeypatch(monkeypatch) -> None:
    p = registry.get_playback("shell")

    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.has_player", lambda: False
    )
    assert p.has_player() is False

    monkeypatch.setattr(
        "aawazz_mcp.audio.playback.has_player", lambda: True
    )
    assert p.has_player() is True


# ── Capture record contract ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_capture_requires_save_path() -> None:
    p = registry.get_capture("sounddevice")
    with pytest.raises(ProviderError, match="requires save_path"):
        await p.record(CaptureRequest(duration_s=1.0, save_path=None))


@pytest.mark.asyncio
async def test_capture_propagates_hint_from_record_helper(
    monkeypatch, tmp_path
) -> None:
    """A dict-shaped error from the underlying capture helper becomes a
    ProviderError; the helper's ``hint`` survives as ``ProviderError.hint``."""
    p = registry.get_capture("sounddevice")

    def fake(*args, **kwargs):
        return {
            "audio_path": None,
            "error": "mic capture timed out (no samples arrived)",
            "hint": "OS mute / UEFI mute / PipeWire routing.",
        }

    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.record_to_wav_hard_timeout", fake
    )

    with pytest.raises(ProviderError) as excinfo:
        await p.record(
            CaptureRequest(
                duration_s=1.0, save_path=str(tmp_path / "out.wav")
            )
        )
    assert "timed out" in str(excinfo.value)
    assert excinfo.value.hint == "OS mute / UEFI mute / PipeWire routing."


@pytest.mark.asyncio
async def test_capture_pid_file_only_passed_when_set(monkeypatch, tmp_path) -> None:
    """The provider must NOT forward pid_file=None to the underlying helper —
    a v1.0 three-arg test mock would explode on the unexpected positional.
    Only forward pid_file when the caller actually passed it via extra=."""
    p = registry.get_capture("sounddevice")
    seen: dict = {}

    def fake_three_arg(duration_s, output_path, timeout_s):
        seen["argc"] = 3
        return {
            "audio_path": output_path,
            "duration_s": duration_s,
            "sample_rate": 16000,
        }

    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.record_to_wav_hard_timeout",
        fake_three_arg,
    )
    await p.record(
        CaptureRequest(duration_s=0.5, save_path=str(tmp_path / "out.wav"))
    )
    assert seen["argc"] == 3
