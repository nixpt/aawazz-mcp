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
    TtsChunk,
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

    @property
    def supports_streaming(self) -> bool:
        return True

    async def synthesize_stream(self, request: TtsRequest, text_stream):
        """Synth-per-chunk: each text chunk → tiny-tts ``synthesize`` →
        :class:`TtsChunk`. The cumulative audio is concatenated and written
        as a complete WAV at ``request.output_path`` on the final chunk so
        callers expecting a single file still get one.
        """
        if request.output_path is None:
            msg = "TinyTtsProvider streaming requires output_path"
            raise ProviderError(msg)

        import asyncio  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415

        loader = self._get_loader()
        voice = request.voice or "MALE"
        speed = float(request.speed)

        chunks: list[np.ndarray] = []
        sample_rate = 22050  # tiny-tts default; updated from real synth output

        async for sentence in text_stream:
            sentence = sentence.strip()
            if not sentence:
                continue

            tmp = tempfile.NamedTemporaryFile(
                prefix="aawazz-tinytts-stream-", suffix=".wav", delete=False
            )
            tmp.close()
            try:
                meta = await loader.synthesize(
                    text=sentence,
                    output_path=tmp.name,
                    voice=voice,
                    speed=speed,
                )
                audio, sr = await asyncio.to_thread(sf.read, tmp.name)
            finally:
                Path(tmp.name).unlink(missing_ok=True)

            sample_rate = int(sr or meta.get("sample_rate") or 22050)
            chunks.append(audio.astype(np.float32))
            yield TtsChunk(
                audio=audio.astype(np.float32),
                sample_rate=sample_rate,
                is_final=False,
            )

        # Write the cumulative WAV.
        if chunks:
            full = np.concatenate(chunks)
            sf.write(
                str(Path(request.output_path)),
                full,
                sample_rate,
                subtype="PCM_16",
            )
            yield TtsChunk(
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=sample_rate,
                is_final=True,
            )
        else:
            # No text produced — write a tiny silence so output_path exists.
            silence = (0.0, np.zeros(int(sample_rate * 0.05), dtype=np.float32))
            sf.write(
                str(Path(request.output_path)),
                silence[1],
                sample_rate,
                subtype="PCM_16",
            )
            yield TtsChunk(
                audio=silence[1], sample_rate=sample_rate, is_final=True,
            )

    async def aclose(self) -> None:
        self._loader = None
