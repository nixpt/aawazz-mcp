"""Built-in tiny-tts provider — single-voice English synthesis.

Wraps :class:`aawazz_mcp.models.tts_loader.TtsLoader` (existing v1.0 loader)
behind the v1.3 :class:`TtsProvider` Protocol. No behavior change vs v1.2.x —
the loader is the same instance, only the call surface is new.

DSP voice profiles (DEEP / BRIGHT / ...) are *not* this provider's job. They
graduate to ``PostProcessor`` instances in phase 5; until then,
:class:`LocalBackend` applies them inline after this provider returns.
"""

from __future__ import annotations

from aawazz_mcp.models.tts_loader import TtsLoader
from aawazz_mcp.provider_base import (
    ProviderError,
    TtsCapabilities,
    TtsRequest,
    TtsResult,
    VoiceCatalogEntry,
)
from aawazz_mcp.registry import register_tts


def _tiny_tts_version() -> str:
    try:
        from importlib.metadata import version
        return version("tiny-tts")
    except Exception:
        return "unknown"


@register_tts("tiny-tts")
class TinyTtsProvider:
    name = "tiny-tts"

    def __init__(self) -> None:
        self._loader: TtsLoader | None = None
        self._version = _tiny_tts_version()

    @property
    def version(self) -> str:
        return self._version

    def _get_loader(self) -> TtsLoader:
        if self._loader is None:
            self._loader = TtsLoader()
        return self._loader

    def capabilities(self) -> TtsCapabilities:
        return TtsCapabilities(
            languages=frozenset({"en"}),
            voices=(
                VoiceCatalogEntry(
                    id="tiny-tts:MALE",
                    language="en",
                    description="Tiny-TTS bundled MALE voice",
                    default=True,
                ),
            ),
            requires_network=False,
            sample_rate=44100,
            accepts_dsp_profiles=True,
            speed_range=(0.5, 2.0),
            notes=(
                "~1.6M params; bundled with the tiny-tts package; auto-downloads "
                "weights from Hugging Face on first use."
            ),
        )

    async def synthesize(self, request: TtsRequest) -> TtsResult:
        if request.output_path is None:
            msg = (
                "TinyTtsProvider requires output_path — caller (LocalBackend) "
                "resolves the default before dispatching to this provider"
            )
            raise ProviderError(msg)

        voice = request.voice or "MALE"
        meta = await self._get_loader().synthesize(
            text=request.text,
            output_path=request.output_path,
            voice=voice,
            speed=request.speed,
        )
        return TtsResult(
            audio_path=meta["audio_path"],
            sample_rate=int(meta["sample_rate"]),
            duration_s=float(meta["duration_s"]),
            latency_ms=int(meta["latency_ms"]),
            voice_used=f"tiny-tts:{voice}",
        )

    async def warm(self) -> None:
        """Eagerly load the tiny-tts model. Used by ``aawazz-mcp --warm``."""
        await self._get_loader().load()

    async def aclose(self) -> None:
        self._loader = None
