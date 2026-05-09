"""Lazy Whisper-based STT loader — used for languages Moonshine doesn't cover.

Currently loaded models:
  - ``amitpant7/Nepali-Automatic-Speech-Recognition`` (Whisper Small, ne)

The pipeline is constructed on first call so importing this module doesn't
drag ``transformers`` in at module-load time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("aawazz_mcp.whisper_stt")

_WHISPER_MODELS: dict[str, str] = {
    "ne": "amitpant7/Nepali-Automatic-Speech-Recognition",
}


def supported_languages() -> set[str]:
    return set(_WHISPER_MODELS)


class WhisperSttLoader:
    """Lazy Whisper pipeline cached by language code."""

    def __init__(self) -> None:
        self._pipe: Any | None = None
        self._model_id: str | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    async def load(self, language: str) -> None:
        """Load the pipeline for *language*. Idempotent if already loaded."""
        model_id = _WHISPER_MODELS.get(language)
        if model_id is None:
            msg = f"no Whisper model registered for language {language!r}"
            raise ValueError(msg)
        async with self._lock:
            if self._pipe is not None and self._model_id == model_id:
                return
            await asyncio.to_thread(self._load_blocking, model_id)
            self._model_id = model_id

    def _load_blocking(self, model_id: str) -> None:
        from transformers import pipeline

        log.info("loading Whisper model %s", model_id)
        t0 = time.time()
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
        )
        log.info("Whisper model loaded in %.2fs", time.time() - t0)

    async def transcribe(self, audio_path: str, language: str) -> dict:
        """Transcribe a local WAV. Returns ``{text, audio_duration_s, latency_ms}``."""
        path = Path(audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {path}")

        # Lazy load
        if self._pipe is None:
            await self.load(language)

        import soundfile as sf

        async with self._lock:
            audio_data, sr = await asyncio.to_thread(sf.read, str(path))
            audio_duration_s = len(audio_data) / float(sr) if sr else 0.0

            t0 = time.time()
            result = await asyncio.to_thread(self._pipe, str(path))
            latency_ms = int((time.time() - t0) * 1000)

        text = (result.get("text") or "").strip()

        return {
            "text": text,
            "audio_duration_s": audio_duration_s,
            "sample_rate": int(sr),
            "latency_ms": latency_ms,
        }
