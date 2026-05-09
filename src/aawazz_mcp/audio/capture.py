"""Bounded mic capture via sounddevice → WAV.

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

# Hard caps — keep in sync with docstring and the `listen` tool docstring.
_MIN_DURATION_S = 0.5
_MAX_DURATION_S = 30.0


def clamp_duration(duration_s: float) -> float:
    """Clamp recording duration to ``[0.5, 30.0]`` seconds.

    Exposed so tests can assert clamping behaviour without a real mic, and so
    the `listen` tool docstring can document the same bounds.
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
        import numpy as np  # noqa: PLC0415
        import sounddevice as sd  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415
    except Exception as exc:
        return {
            "audio_path": None,
            "error": f"audio dependency missing: {exc}",
            "hint": "install sounddevice + soundfile (PortAudio system lib required)",
        }

    try:
        # Pre-allocate the buffer with zeros so early-stop (sd.stop() via
        # SIGUSR1 in _capture_worker) leaves the post-cut tail as silence
        # rather than uninitialized memory — Moonshine handles trailing
        # silence cleanly but would hallucinate on noise.
        frames = int(duration_s * sample_rate)
        audio = np.zeros((frames, 1), dtype=np.int16)
        sd.rec(out=audio, samplerate=sample_rate)
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

    NOTE: ``sd.wait()`` blocks forever when the device enumerates but produces
    no samples (OS-mute, UEFI-mute, PulseAudio/PipeWire routing wrong source).
    Callers that must avoid hangs should use :func:`record_to_wav_hard_timeout`
    instead — same contract, but subprocess-isolated so the parent can hard-kill
    a wedged capture.
    """
    clamped = clamp_duration(duration_s)
    return await asyncio.to_thread(_record_sync, clamped, output_path, sample_rate)


def _capture_worker(
    duration_s: float,
    output_path: str,
    queue: "Any",
) -> None:
    """Child-process mic capture entry point.

    Runs the async :func:`record_to_wav` to completion via ``asyncio.run`` and
    pushes the result onto ``queue``. The parent process can hard-kill this
    worker (terminate / kill) when ``sd.wait`` wedges, which is impossible if
    the recording runs in the parent's thread pool.

    Installs a ``SIGUSR1`` handler that calls ``sd.stop()`` so an external
    toggle script can gracefully cut the recording — ``sd.wait`` returns
    immediately, the partial buffer (zero-padded by ``_record_sync``) is
    written, and the worker exits cleanly. Wrapper scripts find the worker
    PID via the ``pid_file`` argument to :func:`record_to_wav_hard_timeout`.
    """
    import signal as _signal

    def _stop_handler(_signum, _frame):
        try:
            import sounddevice as _sd  # already imported below; cheap on cache
            _sd.stop()
        except Exception:
            pass

    try:
        _signal.signal(_signal.SIGUSR1, _stop_handler)
    except Exception:
        pass  # platform without SIGUSR1 (Windows) — fall through, no toggle support

    try:
        result = asyncio.run(record_to_wav(duration_s=duration_s, output_path=output_path))
    except Exception as exc:  # noqa: BLE001
        result = {
            "audio_path": None,
            "error": f"mic capture failed: {exc}",
            "hint": "check sounddevice / PortAudio input device configuration",
        }
    queue.put(result)


def record_to_wav_hard_timeout(
    duration_s: float,
    output_path: str,
    timeout_s: float,
    pid_file: str | None = None,
) -> dict[str, Any]:
    """Subprocess-isolated :func:`record_to_wav` with a real hard timeout.

    Spawns a child process via ``multiprocessing.get_context("spawn")``, joins
    with ``timeout_s``, and ``terminate`` + ``kill`` the worker if it's still
    alive. Returns the same payload shape as :func:`record_to_wav` plus two new
    failure modes:

    - ``error: "mic capture timed out (no samples arrived)"`` — wait elapsed,
      worker was force-killed. Most often: mic enumerable but muted at OS /
      UEFI / routing layer.
    - ``error: "mic capture process exited with code N"`` — child died (segv,
      OOM, sandbox kill).

    Sync function. Async callers should wrap with ``asyncio.to_thread`` so the
    event loop stays responsive during the join.

    Used by both the agent-facing MCP ``listen`` tool and the operator-side
    ``aawazz-dictate`` CLI. Plain :func:`record_to_wav` remains the primitive
    for callers that explicitly opt out of subprocess isolation.

    Args:
        pid_file: Optional path to write the capture worker's PID. External
            toggle scripts ``kill -SIGUSR1 <pid>`` to gracefully early-stop
            the recording (subprocess catches the signal, writes the partial
            WAV, exits 0). The file is unlinked on return regardless of
            outcome — best-effort, never raises.
    """
    # Local import keeps multiprocessing out of the hot path for callers that
    # only need the basic record_to_wav primitive.
    import multiprocessing as mp
    import os

    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_capture_worker,
        args=(duration_s, output_path, queue),
        daemon=True,
    )
    proc.start()

    # Best-effort PID file for graceful-cut signaling. If the write fails,
    # we still record (just without toggle support).
    pid_file_written = False
    if pid_file is not None and proc.pid is not None:
        try:
            with open(pid_file, "w") as fh:
                fh.write(str(proc.pid))
            pid_file_written = True
        except Exception:
            pass

    try:
        proc.join(timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join(2.0)
            if proc.is_alive():
                proc.kill()
                proc.join(2.0)
            return {
                "audio_path": None,
                "error": "mic capture timed out (no samples arrived)",
                "hint": (
                    f"device enumerated but produced no samples in {timeout_s:.1f}s. "
                    "Check: OS mute, UEFI mute, PulseAudio/PipeWire source routing, "
                    "container audio passthrough."
                ),
            }

        if proc.exitcode not in (0, None):
            return {
                "audio_path": None,
                "error": f"mic capture process exited with code {proc.exitcode}",
                "hint": "check sounddevice / PortAudio input device configuration",
            }

        try:
            return queue.get_nowait()
        except Exception:
            return {
                "audio_path": None,
                "error": "mic capture process returned no result",
                "hint": "check sounddevice / PortAudio input device configuration",
            }
    finally:
        if pid_file_written:
            try:
                os.unlink(pid_file)
            except Exception:
                pass
