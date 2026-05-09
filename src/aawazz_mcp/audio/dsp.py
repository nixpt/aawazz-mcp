"""Lightweight DSP voice profiles — post-process tiny-tts output with numpy.

Every profile is a pure function ``(audio: NDArray, sr: int) -> NDArray``
so they compose trivially. Zero additional dependencies beyond numpy
(already required by aawazz-mcp).
"""

from __future__ import annotations

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────

def _resample_pitch(audio: np.ndarray, sr: int, ratio: float) -> np.ndarray:
    """Change pitch by resampling + playing at original rate.

    ``ratio > 1`` → higher pitch (squeeze), ``ratio < 1`` → lower pitch (stretch).
    Uses linear interpolation for speed.
    """
    n = len(audio)
    if ratio <= 0:
        return audio
    indices = np.arange(0, n, ratio)
    indices = indices[indices < n].astype(np.int32)
    return audio[indices].copy()


def _lowpass(audio: np.ndarray, sr: int, cutoff_hz: float = 2000, order: int = 2) -> np.ndarray:
    """Simple Butterworth-like lowpass via repeated moving-average windows."""
    window = int(sr / cutoff_hz)
    if window < 2:
        return audio
    window = min(window, len(audio) // 4)
    kernel = np.ones(window) / window
    for _ in range(order):
        audio = np.convolve(audio, kernel, mode="same")
    return audio


def _highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 2000) -> np.ndarray:
    """Subtract a lowpass to get a highpass (simple shelf)."""
    lp = _lowpass(audio, sr, cutoff_hz, order=1)
    return audio - lp


def _add_echo(audio: np.ndarray, sr: int, delay_s: float = 0.3, decay: float = 0.4) -> np.ndarray:
    """Add a single echo tap."""
    delay_samples = int(sr * delay_s)
    if delay_samples <= 0 or delay_samples >= len(audio):
        return audio
    out = audio.copy()
    out[delay_samples:] += audio[:-delay_samples] * decay
    # Normalise peak to avoid clipping
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out / peak * 0.95
    return out


def _add_reverb(audio: np.ndarray, sr: int, decay: float = 0.3, tail_s: float = 0.5) -> np.ndarray:
    """Simple Schroeder-like comb-filter reverb."""
    delay_samples = int(sr * tail_s)
    if delay_samples <= 0:
        return audio
    out = np.zeros(len(audio) + delay_samples)
    out[:len(audio)] = audio
    for _ in range(8):
        d = int(delay_samples * (0.5 + np.random.random() * 0.5))
        for i in range(len(audio)):
            if i + d < len(out):
                out[i + d] += audio[i] * decay
    # Trim, normalise
    out = out[:len(audio)]
    peak = np.max(np.abs(out))
    if peak > 0.99:
        out = out / peak * 0.95
    return out


def _waveshape_distortion(audio: np.ndarray, drive: float = 2.0) -> np.ndarray:
    """Soft-clip waveshaping for warm distortion."""
    return np.tanh(audio * drive)


def _robotize(audio: np.ndarray, sr: int) -> np.ndarray:
    """Rectify + bandpass for classic robot voice."""
    rectified = np.abs(audio)
    bp = rectified - _lowpass(rectified, sr, 300, order=2)
    bp = _lowpass(bp, sr, 4000, order=1)
    peak = np.max(np.abs(bp))
    if peak > 0:
        bp = bp / peak * 0.9
    return bp


def _tremolo(audio: np.ndarray, sr: int, rate_hz: float = 5.0, depth: float = 0.5) -> np.ndarray:
    """Amplitude modulation for tremolo effect."""
    t = np.arange(len(audio)) / sr
    mod = 1.0 - depth * (1.0 + np.sin(2 * np.pi * rate_hz * t))
    return audio * mod


# ── profile registry ─────────────────────────────────────────────────────

VOICE_PROFILES: dict[str, str] = {
    "MALE": "Default tiny-tts voice, no post-processing",
    "DEEP": "Lower pitch (resample 0.75x), subtle lowpass for warmth",
    "BRIGHT": "Higher pitch (resample 1.2x), gentle highpass",
    "SOFT": "Warm lowpass filter at 3000 Hz, smoothed transients",
    "GRAVEL": "Subtle saturation + slight pitch-down (0.88x)",
    "ROBOT": "Full-wave rectification + bandpass — classic robot",
    "ECHO": "Single echo tap at 300ms, 40% decay",
    "WIDE": "Pitch-up (1.1x) + reverb tail for spaciousness",
}


def apply_profile(audio: np.ndarray, sr: int, profile: str) -> np.ndarray:
    """Apply a named voice profile to the audio buffer.

    Args:
        audio: 1-D float array in [-1, 1] range.
        sr: Sample rate (Hz).
        profile: Profile name (case-insensitive, like ``"DEEP"``).

    Returns:
        Processed audio (may be shorter/longer due to resampling).
    """
    name = (profile or "MALE").upper()

    if name == "MALE":
        return audio

    if name == "DEEP":
        x = _resample_pitch(audio, sr, 0.75)
        x = _lowpass(x, sr, 3000, 1)
        return x

    if name == "BRIGHT":
        x = _resample_pitch(audio, sr, 1.2)
        x = x + 0.3 * _highpass(x, sr, 3000)
        peak = np.max(np.abs(x))
        if peak > 0.99:
            x = x / peak * 0.95
        return x

    if name == "SOFT":
        return _lowpass(audio, sr, 3000, order=3)

    if name == "GRAVEL":
        x = _resample_pitch(audio, sr, 0.88)
        x = _waveshape_distortion(x, drive=1.4)
        peak = np.max(np.abs(x))
        if peak > 0.99:
            x = x / peak * 0.95
        return x

    if name == "ROBOT":
        return _robotize(audio, sr)

    if name == "ECHO":
        return _add_echo(audio, sr, delay_s=0.3, decay=0.4)

    if name == "WIDE":
        x = _resample_pitch(audio, sr, 1.1)
        x = _add_reverb(x, sr, decay=0.25, tail_s=0.4)
        return x

    msg = f"unknown voice profile {profile!r}; valid: {', '.join(sorted(VOICE_PROFILES))}"
    raise ValueError(msg)
