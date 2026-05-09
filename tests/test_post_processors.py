"""Coverage for the phase-5 post-processor pipeline.

Three angles:
1. Each built-in registers and exposes a sensible direction.
2. Each transforms audio in a meaningful way (shape preserved or altered
   per profile semantics).
3. The chain runner enforces direction and surfaces clean ProviderErrors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aawazz_mcp import providers  # noqa: F401  - register
from aawazz_mcp import post_processors  # noqa: F401  - register
from aawazz_mcp import registry
from aawazz_mcp.backends.local import _apply_audio_chain
from aawazz_mcp.provider_base import ProviderError


def _sine_wav(path: Path, duration_s: float = 1.0, sr: int = 16000,
              freq: float = 220.0, leading_silence_s: float = 0.0,
              trailing_silence_s: float = 0.0) -> None:
    """Write a synthetic sine WAV with optional silence padding."""
    t_voiced = np.arange(int(duration_s * sr)) / sr
    voiced = 0.5 * np.sin(2 * np.pi * freq * t_voiced).astype(np.float32)

    parts = []
    if leading_silence_s > 0:
        parts.append(np.zeros(int(leading_silence_s * sr), dtype=np.float32))
    parts.append(voiced)
    if trailing_silence_s > 0:
        parts.append(np.zeros(int(trailing_silence_s * sr), dtype=np.float32))

    audio = np.concatenate(parts)
    sf.write(str(path), audio, sr, subtype="PCM_16")


# ── Registration ─────────────────────────────────────────────────────────────


def test_dsp_profiles_all_register() -> None:
    """Each of the 7 DSP effects registers under its dsp:<NAME> key."""
    expected = {"dsp:DEEP", "dsp:BRIGHT", "dsp:SOFT", "dsp:GRAVEL",
                "dsp:ROBOT", "dsp:ECHO", "dsp:WIDE"}
    names = {p.name for p in registry.list_post()}
    assert expected.issubset(names)


def test_gain_auto_registered() -> None:
    p = registry.get_post("gain:auto")
    assert p.name == "gain:auto"
    assert p.direction == "both"


def test_vad_webrtc_registered() -> None:
    p = registry.get_post("vad:webrtc")
    assert p.name == "vad:webrtc"
    assert p.direction == "both"


def test_dsp_profiles_are_tts_direction() -> None:
    """DSP profiles only make sense applied to synthesized speech."""
    for p in registry.list_post():
        if p.name.startswith("dsp:"):
            assert p.direction == "tts"


# ── Behavior ────────────────────────────────────────────────────────────────


def test_gain_auto_normalizes_peak() -> None:
    p = registry.get_post("gain:auto")
    audio = np.array([0.1, -0.2, 0.05, -0.15], dtype=np.float32)
    out = p.process(audio, 16000)
    assert np.isclose(np.max(np.abs(out)), 0.95, atol=0.01)


def test_gain_auto_handles_silence() -> None:
    """Silence → no division by zero, no scaling."""
    p = registry.get_post("gain:auto")
    audio = np.zeros(1000, dtype=np.float32)
    out = p.process(audio, 16000)
    assert np.allclose(out, 0.0)


def test_dsp_deep_alters_audio() -> None:
    """DEEP resamples + lowpasses — output audibly different from input."""
    p = registry.get_post("dsp:DEEP")
    audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000)).astype(np.float32)
    out = p.process(audio, 16000)
    # Length may differ (resample); just check it ran and is finite.
    assert out.size > 0
    assert np.all(np.isfinite(out))


def test_vad_webrtc_trims_silence_from_speechlike_audio(tmp_path: Path) -> None:
    """Silence + speech-like band-limited noise + silence → trimmed audio
    is shorter than the original and non-empty.

    Pure sine tones don't trip webrtcvad reliably (it's trained on speech
    spectral features); we use band-limited white noise as a proxy.
    """
    pytest.importorskip("webrtcvad")
    p = registry.get_post("vad:webrtc")

    sr = 16000
    rng = np.random.default_rng(0)
    silence = np.zeros(sr, dtype=np.float32)  # 1s silence

    # 1.5s of band-limited noise (300-3400 Hz, telephony band) — webrtcvad
    # was tuned for this. Loud enough to clear the VAD threshold.
    n_voiced = int(1.5 * sr)
    noise = rng.standard_normal(n_voiced).astype(np.float32) * 0.5
    # Crude lowpass via cumulative average
    voiced = np.convolve(noise, np.ones(8) / 8, mode="same").astype(np.float32)

    audio = np.concatenate([silence, voiced, silence])
    out = p.process(audio, sr)

    # Output is non-empty and shorter than the 3s input.
    assert out.size > 0
    assert len(out) < len(audio)


# ── Chain runner ────────────────────────────────────────────────────────────


def test_apply_chain_unknown_processor_raises(tmp_path: Path) -> None:
    wav = tmp_path / "in.wav"
    _sine_wav(wav)
    with pytest.raises(ProviderError, match="unknown post-processor"):
        _apply_audio_chain(str(wav), ["does:not:exist"], direction="tts")


def test_apply_chain_direction_mismatch_raises(tmp_path: Path) -> None:
    """A tts-only processor (dsp:DEEP) in a stt chain must raise."""
    wav = tmp_path / "in.wav"
    _sine_wav(wav)
    with pytest.raises(ProviderError, match="direction='tts'"):
        _apply_audio_chain(str(wav), ["dsp:DEEP"], direction="stt")


def test_apply_chain_direction_both_works_either_way(tmp_path: Path) -> None:
    wav = tmp_path / "in.wav"
    _sine_wav(wav)
    # gain:auto is direction="both" — runnable in both contexts.
    _apply_audio_chain(str(wav), ["gain:auto"], direction="tts")
    _apply_audio_chain(str(wav), ["gain:auto"], direction="stt")


def test_apply_chain_runs_steps_in_order(tmp_path: Path) -> None:
    """Sanity: chaining DSP+gain leaves a finite, non-zero WAV. We're
    not asserting exact byte-equality — that depends on numpy precision."""
    wav = tmp_path / "in.wav"
    _sine_wav(wav, duration_s=0.5)
    _apply_audio_chain(
        str(wav), ["dsp:DEEP", "gain:auto"], direction="tts"
    )
    audio, sr = sf.read(str(wav))
    assert sr == 16000
    assert audio.size > 0
    assert np.any(audio != 0.0)


def test_apply_chain_empty_is_noop(tmp_path: Path) -> None:
    """Empty chain returns immediately without touching the file."""
    wav = tmp_path / "in.wav"
    _sine_wav(wav)
    mtime_before = wav.stat().st_mtime_ns
    _apply_audio_chain(str(wav), None, direction="tts")
    _apply_audio_chain(str(wav), [], direction="tts")
    assert wav.stat().st_mtime_ns == mtime_before
