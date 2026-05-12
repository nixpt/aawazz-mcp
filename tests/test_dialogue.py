"""Coverage for the dialogue composer + the ``dialogue`` MCP tool."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from aawazz_mcp.audio import dialogue as dlg


def _write_tone(
    path, duration_s: float = 0.2, sample_rate: int = 22050, freq: float = 440.0,
    channels: int = 1,
) -> None:
    """Write a short sine tone for tests that need a real WAV on disk."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    mono = (np.sin(2 * np.pi * freq * t) * 0.3).astype(np.float32)
    if channels == 2:
        audio = np.stack([mono, mono], axis=1)
    else:
        audio = mono
    sf.write(str(path), audio, sample_rate, subtype="PCM_16")


# ── compose() — input validation ────────────────────────────────────────────


def test_compose_rejects_empty_turns() -> None:
    with pytest.raises(ValueError, match="no turns"):
        dlg.compose([], [], pause_ms=300, stereo=False)


def test_compose_rejects_mismatched_lengths(tmp_path) -> None:
    p = tmp_path / "a.wav"
    _write_tone(p)
    with pytest.raises(ValueError, match="len"):
        dlg.compose([str(p)], ["voice-a", "voice-b"])


# ── compose() — concat + pause ──────────────────────────────────────────────


def test_compose_mono_concat_adds_silence_between_turns(tmp_path) -> None:
    sr = 22050
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_tone(a, duration_s=0.2, sample_rate=sr)
    _write_tone(b, duration_s=0.2, sample_rate=sr)

    audio, out_sr = dlg.compose(
        [str(a), str(b)], ["v1", "v2"], pause_ms=500, stereo=False
    )
    assert out_sr == sr
    # Expected length: 0.2 + 0.5 + 0.2 = 0.9 s
    assert abs(len(audio) / sr - 0.9) < 0.01
    # Mono shape.
    assert audio.ndim == 1


def test_compose_pause_clamped(tmp_path) -> None:
    """pause_ms is clamped to 0..2000."""
    sr = 22050
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_tone(a, sample_rate=sr)
    _write_tone(b, sample_rate=sr)

    # 5000ms request → clamped to 2000ms.
    audio, _ = dlg.compose([str(a), str(b)], ["v1", "v2"], pause_ms=5000)
    # 0.2 + 2.0 + 0.2 = 2.4 s
    assert abs(len(audio) / sr - 2.4) < 0.01


def test_compose_zero_pause_no_silence(tmp_path) -> None:
    sr = 22050
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_tone(a, duration_s=0.2, sample_rate=sr)
    _write_tone(b, duration_s=0.2, sample_rate=sr)
    audio, _ = dlg.compose([str(a), str(b)], ["v1", "v2"], pause_ms=0)
    assert abs(len(audio) / sr - 0.4) < 0.01


# ── compose() — resampling ──────────────────────────────────────────────────


def test_compose_resamples_to_majority_rate(tmp_path) -> None:
    """Mixed sample rates: target is the most-common rate; others resampled."""
    a = tmp_path / "a-22k.wav"
    b = tmp_path / "b-22k.wav"
    c = tmp_path / "c-48k.wav"
    _write_tone(a, duration_s=0.2, sample_rate=22050)
    _write_tone(b, duration_s=0.2, sample_rate=22050)
    _write_tone(c, duration_s=0.2, sample_rate=48000)

    audio, sr = dlg.compose(
        [str(a), str(b), str(c)], ["v1", "v2", "v3"], pause_ms=0
    )
    assert sr == 22050
    # 3 × 0.2 = 0.6 s
    assert abs(len(audio) / sr - 0.6) < 0.02


def test_compose_downmixes_stereo_source_to_mono(tmp_path) -> None:
    a = tmp_path / "stereo.wav"
    _write_tone(a, duration_s=0.2, channels=2)
    audio, _ = dlg.compose([str(a)], ["v1"], pause_ms=0)
    assert audio.ndim == 1


# ── compose() — stereo two-speaker pan ──────────────────────────────────────


def test_compose_stereo_with_two_voices_pans_per_speaker(tmp_path) -> None:
    sr = 22050
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write_tone(a, duration_s=0.2, sample_rate=sr, freq=440.0)
    _write_tone(b, duration_s=0.2, sample_rate=sr, freq=880.0)

    audio, _ = dlg.compose(
        [str(a), str(b)], ["amy", "ryan"], pause_ms=0, stereo=True
    )
    assert audio.ndim == 2
    assert audio.shape[1] == 2

    # Inspect the two halves: first half should be left-channel only,
    # second half should be right-channel only.
    half = int(0.2 * sr)
    left_first = np.max(np.abs(audio[:half, 0]))
    right_first = np.max(np.abs(audio[:half, 1]))
    left_second = np.max(np.abs(audio[half:, 0]))
    right_second = np.max(np.abs(audio[half:, 1]))
    # Amy is left → first half has left signal, second half left ~ 0.
    assert left_first > 0.1
    assert right_first < 1e-6
    # Ryan is right → second half has right signal, second half left ~ 0.
    assert left_second < 1e-6
    assert right_second > 0.1


def test_compose_stereo_with_one_voice_falls_back_to_mono(tmp_path) -> None:
    """stereo=True is a no-op when there's only one unique voice."""
    sr = 22050
    a = tmp_path / "a.wav"
    _write_tone(a, sample_rate=sr)
    audio, _ = dlg.compose([str(a)], ["solo"], stereo=True)
    assert audio.ndim == 1


def test_compose_stereo_with_three_voices_falls_back_to_mono(tmp_path) -> None:
    """stereo=True is a no-op with 3+ unique voices (no clean L/R mapping)."""
    sr = 22050
    a, b, c = tmp_path / "a.wav", tmp_path / "b.wav", tmp_path / "c.wav"
    for p in (a, b, c):
        _write_tone(p, sample_rate=sr)
    audio, _ = dlg.compose([str(a), str(b), str(c)], ["v1", "v2", "v3"], stereo=True)
    assert audio.ndim == 1


def test_compose_stereo_assignment_by_first_appearance(tmp_path) -> None:
    """First unique voice → left, regardless of which appears first by index."""
    sr = 22050
    paths = [tmp_path / f"{i}.wav" for i in range(4)]
    for p in paths:
        _write_tone(p, duration_s=0.1, sample_rate=sr)
    # Sequence: ryan, amy, amy, ryan — first unique = ryan → left.
    voices = ["ryan", "amy", "amy", "ryan"]
    audio, _ = dlg.compose([str(p) for p in paths], voices, pause_ms=0, stereo=True)
    # Turn 0 = ryan = left.
    chunk = int(0.1 * sr)
    assert np.max(np.abs(audio[:chunk, 0])) > 0.1
    assert np.max(np.abs(audio[:chunk, 1])) < 1e-6
