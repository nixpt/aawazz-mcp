"""DSP voice profiles graduate to PostProcessor instances.

The 7 effects from :mod:`aawazz_mcp.audio.dsp` (DEEP, BRIGHT, SOFT,
GRAVEL, ROBOT, ECHO, WIDE) become ``dsp:<NAME>`` post-processors,
registered under the new abstraction. The numeric implementations stay
in ``audio/dsp.py`` (still imported by the v1.0 ``LocalBackend`` legacy
path) — this module is the bridge into the registry.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from aawazz_mcp.audio.dsp import VOICE_PROFILES, apply_profile
from aawazz_mcp.registry import register_post


# Skip MALE — it's the passthrough identity, not a post-processing effect.
_DSP_PROFILES: tuple[str, ...] = tuple(
    p for p in VOICE_PROFILES if p != "MALE"
)


def _make_dsp_class(profile: str) -> type:
    """Build a fresh class per profile so each can register under its own
    name. The registry instantiates the class and stores the instance."""

    class _DspProcessor:
        name: str = f"dsp:{profile}"
        direction: Literal["tts", "stt", "both"] = "tts"
        _profile = profile

        def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
            return apply_profile(audio, int(sample_rate), self._profile)

    _DspProcessor.__name__ = f"DspProfile_{profile}"
    _DspProcessor.__qualname__ = _DspProcessor.__name__
    return _DspProcessor


for _profile in _DSP_PROFILES:
    register_post(f"dsp:{_profile}")(_make_dsp_class(_profile))
