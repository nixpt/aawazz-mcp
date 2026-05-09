"""Lazy moonshine-voice loader keyed by ``(language, model_arch)``.

Reload semantics: switching ``model_arch`` (e.g. ``tiny_streaming`` →
``base_streaming``) tears down the old ``Transcriber`` and constructs a new
one. Same for ``language``. Most callers stay on the default ``en`` /
``tiny_streaming`` and never trigger a reload.

Cache: weights live in ``~/.cache/moonshine_voice/`` (override
``MOONSHINE_VOICE_CACHE``). Don't invent ``~/.cache/aawazz/``.

Concurrency: an ``asyncio.Lock`` per loader serializes both reload and
transcribe — Moonshine's ``Transcriber`` is not thread-safe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("aawazz_mcp.stt_loader")


def _arch_from_string(name: str):
    """Translate the SPEC's lowercase ``model_arch`` strings to ``ModelArch``.

    Accepts ``tiny``, ``tiny_streaming`` / ``tiny-streaming``, etc. Raises
    ``ValueError`` with the valid set on miss.
    """
    from moonshine_voice import ModelArch

    key = name.upper().replace("-", "_")
    try:
        return ModelArch[key]
    except KeyError as e:
        valid = ", ".join(m.name.lower() for m in ModelArch)
        raise ValueError(
            f"unknown model_arch {name!r}; valid: {valid}"
        ) from e


class SttLoader:
    """Lazy Moonshine ``Transcriber`` cached by ``(language, arch)``."""

    def __init__(self) -> None:
        self._transcriber: Any | None = None
        self._key: tuple[str, str] | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded_archs(self) -> list[str]:
        """Return ``["<lang>/<arch>"]`` if a model is loaded, else ``[]``.

        Used by :func:`aawazz_mcp.resources.health` so callers can see what's
        warm without forcing a load.
        """
        if self._key is None:
            return []
        return [f"{self._key[0]}/{self._key[1]}"]

    async def load(
        self, language: str = "en", model_arch: str = "tiny_streaming"
    ) -> None:
        """Eager-load a specific (language, model_arch). Used by ``--warm``."""
        async with self._lock:
            await self._load_blocking_async(language, model_arch)

    async def _load_blocking_async(self, language: str, model_arch: str) -> None:
        """Caller MUST hold ``self._lock`` before invoking."""
        key = (language, model_arch)
        if self._transcriber is not None and self._key == key:
            return

        await asyncio.to_thread(self._load_blocking, language, model_arch)
        self._key = key

    def _load_blocking(self, language: str, model_arch: str) -> None:
        """Heavy import + model init. Caller holds the lock; runs in a thread."""
        from moonshine_voice import Transcriber, get_model_for_language

        arch = _arch_from_string(model_arch)
        log.info("loading Moonshine model lang=%s arch=%s", language, arch.name)
        t0 = time.time()
        model_path, model_arch_obj = get_model_for_language(language, arch)
        # Drop the previous transcriber before replacing — frees the ONNX
        # session and helps callers who switch archs frequently.
        self._transcriber = None
        self._transcriber = Transcriber(
            model_path=str(model_path), model_arch=model_arch_obj
        )
        log.info(
            "Moonshine loaded in %.2fs (%s)", time.time() - t0, model_path
        )

    async def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        model_arch: str = "tiny_streaming",
    ) -> dict:
        """Transcribe a local WAV. Returns ``{text, audio_duration_s, sample_rate, latency_ms}``.

        Reloads the underlying ``Transcriber`` if ``(language, model_arch)``
        differs from the currently-cached one.
        """
        from moonshine_voice import load_wav_file

        path = Path(audio_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"audio file not found: {path}")
        if not path.is_file():
            raise IsADirectoryError(f"not a regular file: {path}")

        async with self._lock:
            await self._load_blocking_async(language, model_arch)

            audio_data, sr = await asyncio.to_thread(
                load_wav_file, str(path)
            )
            audio_duration_s = (
                len(audio_data) / float(sr) if sr else 0.0
            )

            t0 = time.time()
            transcript = await asyncio.to_thread(
                self._transcriber.transcribe_without_streaming,
                audio_data,
                sr,
            )
            latency_ms = int((time.time() - t0) * 1000)

        # Don't ``str(transcript)`` — Moonshine's __str__ includes a
        # ``[<start>s] `` prefix per line. Concatenate the clean text instead.
        lines = getattr(transcript, "lines", None) or []
        text = " ".join(
            line.text.strip()
            for line in lines
            if getattr(line, "text", "")
        ).strip()

        return {
            "text": text,
            "audio_duration_s": audio_duration_s,
            "sample_rate": int(sr),
            "latency_ms": latency_ms,
        }
