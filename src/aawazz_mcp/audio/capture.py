"""Bounded mic capture via sounddevice → WAV.

Wave 1C owns this module.

Contract:
    record_to_wav(duration_s, output_path, sample_rate=16000) -> dict

Returns: ``{audio_path, audio_duration_s, sample_rate}``.

Sample rate: Moonshine wants 16 kHz mono. Match.

Failure mode: if no input device exists or sounddevice raises, surface a
structured error rather than crashing. Caller (LocalBackend.listen) wraps the
error into the tool response.

Hard cap: clamp ``duration_s`` to ``[0.5, 30.0]``. An LLM agent could otherwise
spin a 5-minute recording and hang the runtime.
"""

from __future__ import annotations


def has_input_device() -> bool:
    """True iff sounddevice can enumerate at least one input device.

    Used by ``voices_list().capabilities.listen``. Must NOT raise — return
    False on any sounddevice failure.

    Wave 1C: wrap ``sounddevice.query_devices(kind='input')`` in try/except.
    """
    raise NotImplementedError("Wave 1C")


async def record_to_wav(
    duration_s: float,
    output_path: str,
    sample_rate: int = 16000,
) -> dict:
    """Record `duration_s` seconds of mono mic audio at `sample_rate`, write WAV.

    Args:
        duration_s: Clamped to [0.5, 30.0].
        output_path: Absolute path. Parent dir must exist.
        sample_rate: Hz. Default 16000 (Moonshine native rate).

    Returns:
        ``{audio_path: str, audio_duration_s: float, sample_rate: int}``.

    Wave 1C:
        sounddevice.rec → wait/sleep → soundfile.write to output_path.
        Run sounddevice in a thread (it blocks); use asyncio.to_thread.
    """
    raise NotImplementedError("Wave 1C")
