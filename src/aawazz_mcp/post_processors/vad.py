"""``vad:webrtc`` post-processor — silence-trim front and back of audio.

WebRTC VAD operates on 10/20/30 ms PCM-16 frames at 8/16/32/48 kHz. The
trim algorithm scans for the first and last frame flagged as voiced;
leading and trailing silence is removed. Useful as an STT preprocessor
(strip dead air before transcription) and a TTS post-processor (trim
the synthesizer's tail-pad).

Requires the ``[vad]`` extra (``webrtcvad-wheels``). Without it, the
processor registers but :meth:`process` raises a clear
:class:`ProviderError`.

Frame size and aggressiveness are fixed for v1.3.0 (30 ms,
aggressiveness 2). Future configurability lands as kwargs on
``post_process=`` once the pipeline supports per-step config.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from aawazz_mcp.provider_base import ProviderError
from aawazz_mcp.registry import register_post

log = logging.getLogger("aawazz_mcp.post_processors.vad")


_FRAME_MS = 30
_AGGRESSIVENESS = 2  # 0=least, 3=most aggressive
_SUPPORTED_RATES: frozenset[int] = frozenset({8000, 16000, 32000, 48000})


def _probe_webrtcvad() -> bool:
    try:
        import webrtcvad  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


def _resample_to_supported(
    audio: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, int]:
    """If sample_rate is supported by webrtcvad, return as-is; otherwise
    resample to the nearest supported rate via numpy linear interp."""
    if sample_rate in _SUPPORTED_RATES:
        return audio, sample_rate
    target = min(_SUPPORTED_RATES, key=lambda r: abs(r - sample_rate))
    n = int(len(audio) * target / sample_rate)
    if n <= 0:
        return audio, sample_rate
    src = np.linspace(0, len(audio) - 1, num=n)
    return np.interp(src, np.arange(len(audio)), audio), target


def _to_int16(audio: np.ndarray) -> bytes:
    """Convert float32 in [-1, 1] (or int16 already) to int16 bytes."""
    if audio.dtype != np.int16:
        clipped = np.clip(audio, -1.0, 1.0)
        scaled = (clipped * 32767.0).astype(np.int16)
    else:
        scaled = audio
    return scaled.tobytes()


@register_post("vad:webrtc")
class WebRtcVadProcessor:
    name = "vad:webrtc"
    direction: Literal["tts", "stt", "both"] = "both"

    def __init__(self) -> None:
        self._available = _probe_webrtcvad()

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if not self._available:
            msg = (
                "vad:webrtc requires webrtcvad-wheels; install via "
                "``pip install aawazz-mcp[vad]``"
            )
            raise ProviderError(msg)
        if audio.size == 0:
            return audio

        # Flatten to mono if needed.
        if audio.ndim > 1:
            audio = audio.mean(axis=tuple(range(1, audio.ndim)))

        proc_audio, proc_rate = _resample_to_supported(audio, sample_rate)

        import webrtcvad  # noqa: PLC0415

        vad = webrtcvad.Vad(_AGGRESSIVENESS)
        frame_samples = (proc_rate * _FRAME_MS) // 1000
        # Pre-encode as int16 once; webrtcvad needs raw bytes.
        pcm_int16 = (
            np.clip(proc_audio, -1.0, 1.0) * 32767.0
        ).astype(np.int16)

        n_frames = len(pcm_int16) // frame_samples
        if n_frames == 0:
            return audio

        # Find first / last voiced frame in resampled space.
        first_voiced: int | None = None
        last_voiced: int | None = None
        for i in range(n_frames):
            chunk = pcm_int16[i * frame_samples : (i + 1) * frame_samples]
            try:
                voiced = vad.is_speech(chunk.tobytes(), proc_rate)
            except Exception:  # noqa: BLE001
                # Bad frame size or other webrtcvad complaint — treat as silence.
                voiced = False
            if voiced:
                if first_voiced is None:
                    first_voiced = i
                last_voiced = i

        if first_voiced is None:
            log.debug("vad:webrtc — no voiced frames found; returning original")
            return audio

        # Map resampled-space frame indices back to original-rate samples.
        ratio = sample_rate / proc_rate
        start = int(first_voiced * frame_samples * ratio)
        end = int((last_voiced + 1) * frame_samples * ratio)
        # Add small guard padding so we don't clip an attack/release transient.
        guard_samples = int(0.05 * sample_rate)  # 50 ms
        start = max(0, start - guard_samples)
        end = min(len(audio), end + guard_samples)
        return audio[start:end]
