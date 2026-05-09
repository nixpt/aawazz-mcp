"""Wave 1C audio I/O — capability probes + bounded mic capture.

Most tests don't need a real mic — they just check that the capability probes
return booleans and degrade cleanly. The ``@pytest.mark.mic`` test does a
1-second smoke recording and is skipped when no input device is available.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from aawazz_mcp.audio.capture import (
    clamp_duration,
    has_input_device,
    record_to_wav,
)
from aawazz_mcp.audio.playback import has_player, play


# ---------------------------------------------------------------------------
# Capability probes — must return bool, never raise. Safe on any host.
# ---------------------------------------------------------------------------


def test_has_input_device_returns_bool() -> None:
    """has_input_device() returns a bool regardless of host audio config."""
    result = has_input_device()
    assert isinstance(result, bool)


def test_has_player_returns_bool() -> None:
    """has_player() returns a bool regardless of which players are installed."""
    result = has_player()
    assert isinstance(result, bool)


@pytest.mark.skipif(not shutil.which("paplay"), reason="no paplay on PATH")
def test_has_player_finds_paplay() -> None:
    """When paplay is on PATH (captain's box), has_player() must return True."""
    assert has_player() is True


def test_play_no_audio_path_returns_false_when_no_player(monkeypatch: pytest.MonkeyPatch) -> None:
    """play() must return False (not raise) when no player resolves."""
    # Simulate an empty PATH for player resolution.
    monkeypatch.setattr("aawazz_mcp.audio.playback._resolve_player", lambda: None)
    assert play("/nonexistent/path.wav") is False


# ---------------------------------------------------------------------------
# Duration clamp — assert bounds without hitting the mic.
# ---------------------------------------------------------------------------


def test_clamp_duration_caps_at_30() -> None:
    """clamp_duration clamps oversized requests to 30.0s."""
    assert clamp_duration(60.0) == 30.0
    assert clamp_duration(31.0) == 30.0


def test_clamp_duration_floors_at_half_second() -> None:
    """clamp_duration floors tiny requests at 0.5s."""
    assert clamp_duration(0.0) == 0.5
    assert clamp_duration(0.1) == 0.5


def test_clamp_duration_passes_through_in_range() -> None:
    """In-range values pass through unchanged."""
    assert clamp_duration(1.0) == 1.0
    assert clamp_duration(5.0) == 5.0
    assert clamp_duration(30.0) == 30.0
    assert clamp_duration(0.5) == 0.5


async def test_record_to_wav_applies_clamp_before_capture(
    tmp_aawazz_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``record_to_wav`` must clamp its duration BEFORE invoking the capturer.

    Stubs out the underlying sync recorder so we can verify the clamped
    duration without a real mic and without a 30-second wait. The clamp itself
    is unit-tested via ``test_clamp_duration_caps_at_30``; this test confirms
    ``record_to_wav`` actually wires the clamp through to the worker.
    """
    captured: dict[str, float] = {}

    def fake_record_sync(duration_s: float, output_path: str, sample_rate: int) -> dict:
        captured["duration_s"] = duration_s
        captured["sample_rate"] = sample_rate
        return {
            "audio_path": output_path,
            "audio_duration_s": duration_s,
            "sample_rate": sample_rate,
        }

    monkeypatch.setattr("aawazz_mcp.audio.capture._record_sync", fake_record_sync)

    out = tmp_aawazz_home / "clamp.wav"
    res = await record_to_wav(60.0, str(out))
    # Over-long: clamped down to 30.0
    assert captured["duration_s"] == 30.0
    assert res["audio_duration_s"] == 30.0

    res2 = await record_to_wav(0.0, str(out))
    # Sub-minimum: clamped up to 0.5
    assert captured["duration_s"] == 0.5
    assert res2["audio_duration_s"] == 0.5


# ---------------------------------------------------------------------------
# Real-mic smoke — only when an input device exists.
# ---------------------------------------------------------------------------


@pytest.mark.mic
@pytest.mark.skipif(not has_input_device(), reason="no input device")
async def test_record_smoke(tmp_aawazz_home: Path) -> None:
    """Record 1 second; assert WAV exists and round-trips through soundfile."""
    import soundfile as sf

    out = tmp_aawazz_home / "smoke.wav"
    res = await record_to_wav(1.0, str(out))
    assert res.get("audio_path") == str(out), res
    assert res["audio_duration_s"] == 1.0
    assert res["sample_rate"] == 16000
    assert out.exists()

    audio, sr = sf.read(str(out))
    assert sr == 16000
    # ~1 second of audio at 16 kHz — allow a small tolerance for rounding.
    assert 15000 <= len(audio) <= 17000
