"""Smoke coverage for :mod:`aawazz_mcp.audio.dsp`."""

from __future__ import annotations

import numpy as np
import pytest


def _sine(duration_s: float = 0.5, sr: int = 16000, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return 0.5 * np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.mark.parametrize(
    "profile",
    ["MALE", "DEEP", "BRIGHT", "SOFT", "GRAVEL", "ROBOT", "ECHO", "WIDE"],
)
def test_apply_profile_returns_finite_audio(profile: str) -> None:
    from aawazz_mcp.audio.dsp import apply_profile

    audio = _sine()
    out = apply_profile(audio, sr=16000, profile=profile)

    assert isinstance(out, np.ndarray)
    assert out.size > 0
    assert np.all(np.isfinite(out))
    assert np.max(np.abs(out)) <= 1.0


def test_male_profile_is_passthrough() -> None:
    from aawazz_mcp.audio.dsp import apply_profile

    audio = _sine()
    out = apply_profile(audio, sr=16000, profile="MALE")

    assert np.array_equal(out, audio)


def test_apply_profile_is_case_insensitive() -> None:
    from aawazz_mcp.audio.dsp import apply_profile

    audio = _sine()
    a = apply_profile(audio, sr=16000, profile="deep")
    b = apply_profile(audio, sr=16000, profile="DEEP")

    assert np.array_equal(a, b)


def test_apply_profile_unknown_raises() -> None:
    from aawazz_mcp.audio.dsp import apply_profile

    with pytest.raises(ValueError, match="unknown voice profile"):
        apply_profile(_sine(), sr=16000, profile="NOPE")


def test_voice_profiles_registry_complete() -> None:
    from aawazz_mcp.audio.dsp import VOICE_PROFILES

    expected = {"MALE", "DEEP", "BRIGHT", "SOFT", "GRAVEL", "ROBOT", "ECHO", "WIDE"}
    assert set(VOICE_PROFILES) == expected
