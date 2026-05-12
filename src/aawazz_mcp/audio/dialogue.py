"""Dialogue composition — concatenate per-turn WAVs into one stream.

Given a list of per-turn WAVs (produced by repeated ``speak()`` calls
with different voices), stitch them into a single dialogue file with:

- optional inter-turn silence (default 300 ms — natural pause cadence),
- resampling to a common sample rate (the most-common turn's rate wins),
- optional stereo-pan: when exactly two distinct voices appear, the
  first is placed on the left channel and the second on the right, for
  a two-speaker conversation feel.

Pure-numpy / soundfile / scipy.signal — no extra deps.
"""

from __future__ import annotations

import logging
from collections import Counter
from math import gcd

import numpy as np
import soundfile as sf

_LOG = logging.getLogger("aawazz.audio.dialogue")

_MIN_PAUSE_MS: int = 0
_MAX_PAUSE_MS: int = 2000


def _resample(audio: np.ndarray, src_sr: int, tgt_sr: int) -> np.ndarray:
    """Polyphase resample. Anti-aliased — voice-quality preserving."""
    if src_sr == tgt_sr:
        return audio
    from scipy.signal import resample_poly  # noqa: PLC0415

    g = gcd(src_sr, tgt_sr)
    return resample_poly(audio, tgt_sr // g, src_sr // g, axis=0)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def compose(
    turn_paths: list[str],
    turn_voices: list[str],
    pause_ms: int = 300,
    stereo: bool = False,
) -> tuple[np.ndarray, int]:
    """Read per-turn WAVs, concat with silence pads, optionally stereo-pan.

    Returns ``(audio: np.ndarray, sample_rate: int)``. Audio is shape
    ``(N,)`` in mono mode or ``(N, 2)`` in stereo mode. Caller writes
    via :func:`soundfile.write`.
    """
    if not turn_paths:
        msg = "dialogue: no turns to compose"
        raise ValueError(msg)
    if len(turn_paths) != len(turn_voices):
        msg = (
            f"dialogue: turn_paths len ({len(turn_paths)}) != "
            f"turn_voices len ({len(turn_voices)})"
        )
        raise ValueError(msg)

    pause_ms = max(_MIN_PAUSE_MS, min(_MAX_PAUSE_MS, int(pause_ms)))

    # Read each turn. Down-mix stereo source to mono (we'll pan later if
    # stereo=True; treating every input as a single per-turn voice
    # stream is the simplest invariant).
    per_turn: list[tuple[np.ndarray, int]] = []
    for path in turn_paths:
        audio, sr = sf.read(path, dtype="float32")
        per_turn.append((_to_mono(audio), int(sr)))

    # Target sample rate: pick the most common across turns so we resample
    # the fewest samples. Ties broken by first occurrence.
    sr_counts = Counter(sr for _, sr in per_turn)
    target_sr = sr_counts.most_common(1)[0][0]

    # Resample everything to target_sr.
    per_turn = [(_resample(a, sr, target_sr), target_sr) for a, sr in per_turn]

    pause_samples = int(target_sr * pause_ms / 1000)
    silence = np.zeros(pause_samples, dtype=np.float32) if pause_samples else None

    if not stereo or len(set(turn_voices)) != 2:
        # Mono concat. Stereo with 1 or 3+ voices falls back to mono
        # because the "two-speaker pan" semantics only work cleanly
        # for exactly two distinct voices.
        chunks: list[np.ndarray] = []
        for i, (audio, _) in enumerate(per_turn):
            chunks.append(audio)
            if silence is not None and i < len(per_turn) - 1:
                chunks.append(silence)
        return np.concatenate(chunks).astype(np.float32), target_sr

    # Stereo two-speaker mode. First unique voice → left, second → right.
    voice_order: list[str] = []
    for v in turn_voices:
        if v not in voice_order:
            voice_order.append(v)
    left_voice, right_voice = voice_order[0], voice_order[1]

    chunks_stereo: list[np.ndarray] = []
    for i, ((audio, _), voice) in enumerate(zip(per_turn, turn_voices)):
        n = len(audio)
        stereo_chunk = np.zeros((n, 2), dtype=np.float32)
        if voice == left_voice:
            stereo_chunk[:, 0] = audio
        else:  # voice == right_voice
            stereo_chunk[:, 1] = audio
        chunks_stereo.append(stereo_chunk)
        if silence is not None and i < len(per_turn) - 1:
            chunks_stereo.append(np.zeros((pause_samples, 2), dtype=np.float32))

    return np.concatenate(chunks_stereo, axis=0).astype(np.float32), target_sr
