"""Bounded mic capture via sounddevice → WAV.

Wave 1C owns this module.

Contract:
    has_input_device() -> bool
    record_to_wav(duration_s, output_path, sample_rate=16000) -> dict

Returns: ``{audio_path, audio_duration_s, sample_rate}`` on success, or
``{audio_path: None, error: str, hint: str}`` on failure.

Sample rate: Moonshine wants 16 kHz mono. Match.

Failure mode: if no input device exists or sounddevice raises, surface a
structured error rather than crashing. Caller (LocalBackend.listen) wraps the
error into the tool response.

Hard cap: clamp ``duration_s`` to ``[0.5, 30.0]``. An LLM agent could otherwise
spin a 5-minute recording and hang the runtime.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

_LOG = logging.getLogger("aawazz.audio")

# Hard caps — keep in sync with docstring and tool docs in Wave 2.
_MIN_DURATION_S = 0.5
_MAX_DURATION_S = 30.0


def clamp_duration(duration_s: float) -> float:
    """Clamp recording duration to ``[0.5, 30.0]`` seconds.

    Exposed so tests can assert clamping behaviour without a real mic, and so
    Wave 2 can reuse the exact same bounds in the tool docstring.
    """
    if duration_s < _MIN_DURATION_S:
        return _MIN_DURATION_S
    if duration_s > _MAX_DURATION_S:
        return _MAX_DURATION_S
    return float(duration_s)


def has_input_device() -> bool:
    """True iff sounddevice can enumerate at least one input device.

    Used by ``voices_list().capabilities.listen``. Must NOT raise — returns
    False on any sounddevice failure (missing PortAudio, no device, sounddevice
    itself failing to import).
    """
    try:
        import sounddevice as sd  # noqa: PLC0415 — defer import; module may be missing
    except Exception as exc:  # pragma: no cover — covered by capability tests on host
        _LOG.warning("sounddevice import failed: %s", exc)
        return False

    try:
        info = sd.query_devices(kind="input")
    except Exception as exc:
        _LOG.warning("sounddevice.query_devices failed: %s", exc)
        return False

    # query_devices(kind='input') returns a dict for the default input or a
    # list when no default is set. Treat any non-empty result as "has input".
    if info is None:
        return False
    if isinstance(info, dict):
        # A dict result is the default input device — present by definition.
        return bool(info)
    if isinstance(info, (list, tuple)):
        return len(info) > 0
    # Unknown shape — be conservative and say no.
    return False


def _record_sync(duration_s: float, output_path: str, sample_rate: int) -> dict[str, Any]:
    """Synchronous mic capture + WAV write. Runs in a worker thread."""
    try:
        import sounddevice as sd  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
    except Exception as exc:
        return {
            "audio_path": None,
            "error": f"audio dependency missing: {exc}",
            "hint": "install sounddevice + soundfile (PortAudio system lib required)",
        }

    try:
        # int16 mono — matches soundfile's PCM_16 default and Moonshine's input.
        audio = sd.rec(
            int(duration_s * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
        )
        sd.wait()
    except Exception as exc:
        return {
            "audio_path": None,
            "error": f"mic capture failed: {exc}",
            "hint": "is an input device connected? check `voices_list().capabilities.listen`",
        }

    try:
        sf.write(output_path, audio, sample_rate)
    except Exception as exc:
        return {
            "audio_path": None,
            "error": f"WAV write failed: {exc}",
            "hint": f"is the parent directory writable? path={output_path!r}",
        }

    return {
        "audio_path": output_path,
        "audio_duration_s": float(duration_s),
        "sample_rate": int(sample_rate),
    }


async def record_to_wav(
    duration_s: float,
    output_path: str,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    """Record `duration_s` seconds of mono mic audio at `sample_rate`, write WAV.

    Args:
        duration_s: Clamped to ``[0.5, 30.0]``. An LLM agent could otherwise
            request a 5-minute recording and hang the MCP runtime.
        output_path: Absolute path. Parent dir must exist.
        sample_rate: Hz. Default 16000 (Moonshine native rate).

    Returns:
        On success: ``{audio_path: str, audio_duration_s: float, sample_rate: int}``.
        On failure: ``{audio_path: None, error: str, hint: str}``.

    The recording itself is dispatched to a worker thread via
    ``asyncio.to_thread`` so the FastMCP event loop stays responsive during the
    blocking ``sounddevice.rec`` + ``sd.wait`` call.
    """
    clamped = clamp_duration(duration_s)
    return await asyncio.to_thread(_record_sync, clamped, output_path, sample_rate)
