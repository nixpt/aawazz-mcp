"""Built-in capture provider — shell-out to ``termux-microphone-record``.

Mirrors :mod:`aawazz_mcp.providers.sounddevice_capture` but targets the
Termux:API binary instead of PortAudio. The win on Termux/Android is that
``sounddevice`` can't enumerate devices through proot-distro — the mic is
reachable only via Android's MediaRecorder service.

Auto-selection: see :func:`aawazz_mcp.audio.capture.default_provider_name`
— on a Termux host with this binary present, this provider becomes the
implicit default for :func:`listen` instead of ``sounddevice``.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from aawazz_mcp.audio import termux_mic as _termux_mic_module
from aawazz_mcp.provider_base import (
    CaptureRequest,
    CaptureResult,
    ProviderError,
)
from aawazz_mcp.registry import register_capture


@register_capture("termux-mic")
class TermuxMicCaptureProvider:
    name = "termux-mic"
    version = "1.0"

    def has_input_device(self) -> bool:
        # Module-level lookup so tests can monkeypatch the probe.
        return bool(_termux_mic_module.has_microphone())

    async def record(self, request: CaptureRequest) -> CaptureResult:
        if request.save_path is None:
            msg = (
                "termux-mic capture requires save_path; the caller "
                "(LocalBackend) resolves a tempfile before dispatching"
            )
            raise ProviderError(msg)

        t0 = time.time()
        meta: dict[str, Any] = await asyncio.to_thread(
            _termux_mic_module.record_to_wav,
            float(request.duration_s),
            request.save_path,
        )

        if meta.get("error") or meta.get("audio_path") is None:
            raise ProviderError(
                meta.get("error", "termux-mic capture failed"),
                hint=meta.get("hint"),
            )

        return CaptureResult(
            audio_path=meta["audio_path"],
            sample_rate=int(meta.get("sample_rate", request.sample_rate)),
            duration_s=float(meta.get("duration_s", request.duration_s)),
            latency_ms=int(meta.get("latency_ms", (time.time() - t0) * 1000)),
        )

    async def aclose(self) -> None:
        pass
