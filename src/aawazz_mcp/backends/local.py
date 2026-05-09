"""Local backend — bundles tiny-tts + Moonshine + sounddevice.

Wave 1A owns this module. Reference: ~/.local/aawazz/{mouth,ears}/server.py
captures the proven contract; lift the model-load + invocation logic, drop the
FastAPI scaffolding, route through the loaders in :mod:`aawazz_mcp.models`.

CRITICAL — stdout safety:
    `tiny_tts.TinyTTS.speak()` does ``print(f"Synthesizing: ...")``. Under MCP
    stdio transport this corrupts the JSON-RPC frame stream. Wrap every call
    site in ``contextlib.redirect_stdout(sys.stderr)``. The contextmanager
    helper lives in :mod:`aawazz_mcp.models.tts_loader`.

CRITICAL — concurrent calls:
    `TinyTTS` and `moonshine_voice.Transcriber` are not thread-safe. FastMCP
    serves tools concurrently. Use an ``asyncio.Lock`` per loader to serialize.
"""

from __future__ import annotations

from aawazz_mcp.backends.base import Backend
from aawazz_mcp.config import AawazzConfig


class LocalBackend(Backend):
    """In-process tiny-tts + Moonshine + sounddevice."""

    def __init__(self, cfg: AawazzConfig) -> None:
        self.cfg = cfg
        # Wave 1A:
        # - lazy-init self._tts_loader: TtsLoader | None = None
        # - lazy-init self._stt_loader: SttLoader | None = None
        # - asyncio.Lock per loader

    async def warm(self) -> None:
        raise NotImplementedError("Wave 1A: load tts + default stt arch")

    async def speak(self, **kwargs) -> dict:
        raise NotImplementedError("Wave 1A: stdout-redirect + tts.speak() + soundfile.info()")

    async def transcribe(self, **kwargs) -> dict:
        raise NotImplementedError("Wave 1A: load_wav_file + transcriber.transcribe_without_streaming()")

    async def listen(self, **kwargs) -> dict:
        raise NotImplementedError(
            "Wave 1A: aawazz_mcp.audio.capture.record_to_wav() + transcribe(self,...)"
        )
