"""Built-in gTTS provider — non-English fallback via Google Translate TTS.

Lifts the v1.2.1 ``_speak_gtts`` code path (with the MP3→PCM-WAV transcode
fix) out of :class:`LocalBackend` into a :class:`TtsProvider`. Behavior is
identical to v1.2.x; the surface is just the new Protocol.

Requires the ``[multilingual]`` extra (`gtts` package). If gtts isn't
installed, the provider still registers but reports empty capabilities; any
``synthesize`` call returns a clean :class:`ProviderError` rather than a
runtime ImportError.
"""

from __future__ import annotations

import time
from pathlib import Path

from aawazz_mcp.provider_base import (
    ProviderError,
    TtsCapabilities,
    TtsRequest,
    TtsResult,
    VoiceCatalogEntry,
)
from aawazz_mcp.registry import register_tts


def _probe_gtts() -> tuple[bool, frozenset[str]]:
    try:
        from gtts.lang import tts_langs
        return True, frozenset(tts_langs().keys())
    except Exception:
        return False, frozenset()


def _gtts_version() -> str:
    try:
        from importlib.metadata import version
        return version("gtts")
    except Exception:
        return "unknown"


@register_tts("gtts")
class GttsTtsProvider:
    name = "gtts"

    def __init__(self) -> None:
        self._available, self._languages = _probe_gtts()
        self._version = _gtts_version() if self._available else "not-installed"

    @property
    def version(self) -> str:
        return self._version

    def capabilities(self) -> TtsCapabilities:
        # When gtts isn't installed: empty languages → routing layer skips us.
        # When installed: declare the full Google Translate TTS language set.
        notes = (
            "Requires internet access (Google Translate TTS). "
            "Install via ``pip install aawazz-mcp[multilingual]``."
            if self._available
            else "gtts package not installed; install via ``pip install aawazz-mcp[multilingual]``."
        )
        return TtsCapabilities(
            languages=self._languages,
            voices=(
                VoiceCatalogEntry(
                    id="gtts:default",
                    language="",
                    description="Single Google Translate TTS voice per language",
                    default=False,
                ),
            ),
            requires_network=True,
            sample_rate=24000,
            accepts_dsp_profiles=True,
            speed_range=(1.0, 1.0),
            notes=notes,
        )

    async def synthesize(self, request: TtsRequest) -> TtsResult:
        if not self._available:
            msg = (
                "gtts not installed; install via "
                "``pip install aawazz-mcp[multilingual]``"
            )
            raise ProviderError(msg)
        if request.output_path is None:
            msg = "GttsTtsProvider requires output_path"
            raise ProviderError(msg)
        if request.language not in self._languages:
            msg = (
                f"gTTS does not support language {request.language!r}; "
                f"valid: {sorted(self._languages)[:20]}..."
                if len(self._languages) > 20
                else f"gTTS does not support language {request.language!r}; "
                f"valid: {sorted(self._languages)}"
            )
            raise ProviderError(msg)

        out = Path(request.output_path)
        mp3_tmp = out.with_suffix(out.suffix + ".mp3")

        from gtts import gTTS  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415

        t0 = time.time()
        try:
            tts = gTTS(request.text, lang=request.language)
            tts.save(str(mp3_tmp))
        except Exception as e:
            msg = f"gTTS synthesis failed: {e}; gTTS requires internet access"
            raise ProviderError(msg) from e

        try:
            audio_data, sr = sf.read(str(mp3_tmp))
            sf.write(str(out), audio_data, int(sr), subtype="PCM_16")
        except Exception as e:
            msg = (
                f"gTTS MP3 → WAV transcode failed: {e}; "
                "libsndfile may lack MP3 support in this build"
            )
            raise ProviderError(msg) from e
        finally:
            mp3_tmp.unlink(missing_ok=True)

        latency_ms = int((time.time() - t0) * 1000)
        info = sf.info(str(out))

        return TtsResult(
            audio_path=str(out),
            sample_rate=int(info.samplerate),
            duration_s=float(info.duration),
            latency_ms=latency_ms,
            voice_used=f"gtts:{request.language}",
        )

    @property
    def supports_streaming(self) -> bool:
        # gTTS is HTTP-based and only returns a complete MP3 — no streaming.
        return False

    async def synthesize_stream(self, request: TtsRequest, text_stream):  # noqa: ARG002
        msg = "GttsTtsProvider does not support synthesize_stream"
        raise ProviderError(msg)
        yield  # noqa: B901  - unreachable; marks as async generator

    async def aclose(self) -> None:
        # gtts is stateless; nothing to release.
        pass
