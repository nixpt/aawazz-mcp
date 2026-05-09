"""``gain:auto`` post-processor — peak-normalize audio so the loudest
sample reaches a target dBFS. Pure numpy, useful both for TTS output
(consistent loudness across providers) and STT input (fixed gain helps
ASR accuracy).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from aawazz_mcp.registry import register_post


_TARGET_PEAK = 0.95  # leave 0.5 dBFS of headroom


@register_post("gain:auto")
class AutoGainProcessor:
    name = "gain:auto"
    direction: Literal["tts", "stt", "both"] = "both"

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if audio.size == 0:
            return audio
        peak = float(np.max(np.abs(audio)))
        if peak <= 1e-6:
            return audio  # silence; don't divide by zero
        scale = _TARGET_PEAK / peak
        return (audio.astype(np.float32) * scale).astype(audio.dtype, copy=False)
