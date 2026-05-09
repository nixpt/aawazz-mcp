"""Built-in XTTS-v2 TTS provider — voice-cloning multilingual synthesis.

Phase 4 of v1.3 (SPEC §8). Wraps :class:`TTS.api.TTS` with the
``tts_models/multilingual/multi-dataset/xtts_v2`` checkpoint. Coqui's
XTTS-v2 covers 17 languages and clones a target voice from a 3–30 s
reference WAV.

Requires the ``[xtts]`` extra (~2 GB model downloaded on first use,
plus heavyweight deps including torch + librosa + numba). Without the
extra, the provider registers but reports empty capabilities.

Voice cloning
-------------
Every XTTS synthesize call needs a reference WAV. Two ways to specify::

    speak(tts_provider="xtts", language="en", voice="xtts:cloned-from-/abs/ref.wav")
    speak(tts_provider="xtts", language="en", extra={"speaker_wav": "/abs/ref.wav"})

The ``cloned-from-<path>`` voice ID is parsed back to ``speaker_wav``;
``extra["speaker_wav"]`` overrides if both are present. No speaker_wav
→ ProviderError with a hint pointing at the API.

XTTS-v2 also exposes a few built-in studio speakers via ``speaker=``;
expose them via ``extra={"speaker": "Daisy Studious"}``.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from aawazz_mcp.provider_base import (
    ProviderError,
    TtsCapabilities,
    TtsRequest,
    TtsResult,
    VoiceCatalogEntry,
)
from aawazz_mcp.registry import register_tts

log = logging.getLogger("aawazz_mcp.providers.xtts")


# XTTS-v2 supported languages per the model card.
_XTTS_LANGUAGES: frozenset[str] = frozenset({
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl",
    "cs", "ar", "zh", "ja", "hu", "ko", "hi",
})


_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
_VOICE_PREFIX = "xtts:"
_CLONE_PREFIX = "xtts:cloned-from-"


def _xtts_version() -> str:
    try:
        from importlib.metadata import version
        return version("coqui-tts")
    except Exception:
        return "unknown"


def _probe_xtts() -> bool:
    try:
        import TTS  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


@register_tts("xtts")
class XttsTtsProvider:
    name = "xtts"

    def __init__(self) -> None:
        self._available = _probe_xtts()
        self._version = _xtts_version() if self._available else "not-installed"
        self._tts: Any | None = None
        # Honor offline ops who pre-fetched the model + want to skip the HF
        # download attempt.
        self._auto_download = (
            os.environ.get("AAWAZZ_XTTS_AUTO_DOWNLOAD", "1") == "1"
        )

    @property
    def version(self) -> str:
        return self._version

    def capabilities(self) -> TtsCapabilities:
        if not self._available:
            return TtsCapabilities(
                languages=frozenset(),
                voices=(),
                requires_network=False,
                sample_rate=24000,
                accepts_dsp_profiles=False,
                speed_range=(1.0, 1.0),
                notes=(
                    "coqui-tts not installed; install via "
                    "``pip install aawazz-mcp[xtts]``."
                ),
            )

        # XTTS has no fixed voice catalog — voices are cloned from reference
        # WAVs. We surface a single placeholder entry that documents the API.
        voices = (
            VoiceCatalogEntry(
                id="xtts:cloned-from-<reference.wav>",
                language="",
                description=(
                    "Pass a reference WAV via voice='xtts:cloned-from-/abs/path' "
                    "or extra={'speaker_wav': '/abs/path'}. 3–30s of clean "
                    "target voice."
                ),
                default=False,
            ),
        )
        return TtsCapabilities(
            languages=_XTTS_LANGUAGES,
            voices=voices,
            requires_network=False,
            sample_rate=24000,
            # XTTS has its own style/expressivity control; layering DSP
            # profiles on top muddies that signal.
            accepts_dsp_profiles=False,
            # XTTS exposes ``length_penalty`` rather than direct speed control;
            # for v1.3 we lock to 1.0× and document.
            speed_range=(1.0, 1.0),
            notes=(
                "Coqui XTTS-v2 (~2 GB model on first use). Voice cloning "
                "via speaker_wav reference. 17 languages."
            ),
        )

    async def synthesize(self, request: TtsRequest) -> TtsResult:
        if not self._available:
            msg = (
                "coqui-tts not installed; install via "
                "``pip install aawazz-mcp[xtts]``"
            )
            raise ProviderError(msg)
        if request.output_path is None:
            msg = "XttsTtsProvider requires output_path"
            raise ProviderError(msg)
        if request.language not in _XTTS_LANGUAGES:
            msg = (
                f"XTTS-v2 does not support language {request.language!r}; "
                f"supported: {sorted(_XTTS_LANGUAGES)}"
            )
            raise ProviderError(msg)

        speaker_wav = self._resolve_speaker_wav(request)
        speaker = request.extra.get("speaker") if request.extra else None
        if not speaker_wav and not speaker:
            msg = (
                "XTTS-v2 requires either a reference WAV or a built-in "
                "speaker name. Pass voice='xtts:cloned-from-/abs/ref.wav', "
                "extra={'speaker_wav': '/abs/ref.wav'}, or "
                "extra={'speaker': '<built-in speaker name>'}"
            )
            raise ProviderError(msg)

        tts_obj = await self._ensure_loaded()

        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        import asyncio  # noqa: PLC0415

        kwargs: dict[str, Any] = {
            "text": request.text,
            "file_path": str(out),
            "language": request.language,
        }
        if speaker_wav:
            kwargs["speaker_wav"] = speaker_wav
        if speaker:
            kwargs["speaker"] = speaker

        t0 = time.time()
        try:
            await asyncio.to_thread(tts_obj.tts_to_file, **kwargs)
        except Exception as e:  # noqa: BLE001
            msg = f"XTTS synthesis failed: {e}"
            raise ProviderError(msg) from e

        latency_ms = int((time.time() - t0) * 1000)

        import soundfile as sf  # noqa: PLC0415

        info = sf.info(str(out))
        voice_used = (
            f"xtts:cloned-from-{speaker_wav}"
            if speaker_wav
            else f"xtts:speaker={speaker}"
        )
        return TtsResult(
            audio_path=str(out),
            sample_rate=int(info.samplerate),
            duration_s=float(info.duration),
            latency_ms=latency_ms,
            voice_used=voice_used,
        )

    def _resolve_speaker_wav(self, request: TtsRequest) -> str | None:
        # extra wins over voice= for explicit caller intent.
        if request.extra:
            wav = request.extra.get("speaker_wav")
            if wav:
                return str(wav)
        if request.voice and request.voice.startswith(_CLONE_PREFIX):
            return request.voice[len(_CLONE_PREFIX):]
        return None

    async def _ensure_loaded(self) -> Any:
        if self._tts is not None:
            return self._tts

        log.info("loading XTTS-v2 model (~2 GB; first call may take minutes)")
        import asyncio  # noqa: PLC0415

        from TTS.api import TTS  # noqa: PLC0415

        try:
            self._tts = await asyncio.to_thread(TTS, _MODEL_NAME)
        except Exception as e:
            msg = (
                f"XTTS-v2 model load failed: {e}; "
                "first-use downloads ~2 GB from Hugging Face"
            )
            raise ProviderError(msg) from e
        return self._tts

    async def warm(self) -> None:
        if not self._available:
            return
        await self._ensure_loaded()

    async def aclose(self) -> None:
        self._tts = None
