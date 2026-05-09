"""Built-in mic capture provider — sounddevice via ``audio.capture``.

Phase 6 of v1.3 (SPEC §1.4). Wraps the existing v1.0
:func:`aawazz_mcp.audio.capture.record_to_wav_hard_timeout` behind the
:class:`CaptureProvider` Protocol so future plugins (PortAudio direct,
ffmpeg subprocess, PipeWire native, etc.) can swap in without touching
LocalBackend.

The hard-timeout subprocess strategy stays the canonical default —
mic enumeration that succeeds but produces no samples (UEFI / OS
mute, routing) returns a structured error in ``duration_s + 5s``
rather than wedging the MCP runtime.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from aawazz_mcp.audio import capture as _capture_module
from aawazz_mcp.provider_base import (
    CaptureRequest,
    CaptureResult,
    ProviderError,
)
from aawazz_mcp.registry import register_capture


@register_capture("sounddevice")
class SoundDeviceCaptureProvider:
    name = "sounddevice"
    version = "1.0"

    def has_input_device(self) -> bool:
        # Look up via the module so ``monkeypatch.setattr`` on the
        # canonical path is observed by callers.
        return bool(_capture_module.has_input_device())

    async def record(self, request: CaptureRequest) -> CaptureResult:
        if request.save_path is None:
            msg = (
                "sounddevice capture requires save_path; the caller "
                "(LocalBackend) resolves a tempfile before dispatching"
            )
            raise ProviderError(msg)

        timeout_s = request.duration_s + 5.0
        # Optional graceful-stop hook used by aawazz-dictate (SIGUSR1).
        pid_file: str | None = None
        if request.extra:
            pid_file = request.extra.get("pid_file")

        t0 = time.time()
        # Only forward pid_file when set so callers (incl. test mocks) that
        # use the v1.0 three-arg signature aren't broken by an unexpected
        # extra positional.
        args: tuple = (
            request.duration_s,
            request.save_path,
            timeout_s,
        )
        if pid_file is not None:
            args = args + (pid_file,)
        meta: dict[str, Any] = await asyncio.to_thread(
            _capture_module.record_to_wav_hard_timeout, *args
        )
        latency_ms = int((time.time() - t0) * 1000)

        if meta.get("error") or meta.get("audio_path") is None:
            err = meta.get("error", "mic capture failed")
            raise ProviderError(err, hint=meta.get("hint"))

        return CaptureResult(
            audio_path=meta["audio_path"],
            sample_rate=int(meta.get("sample_rate", request.sample_rate)),
            duration_s=float(meta.get("duration_s", request.duration_s)),
            latency_ms=latency_ms,
        )

    async def aclose(self) -> None:
        pass
