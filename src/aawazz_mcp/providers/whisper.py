"""Built-in Whisper STT provider — Nepali support via HF transformers.

Wraps :class:`aawazz_mcp.models.whisper_stt.WhisperSttLoader` (existing v1.2
loader) behind the v1.3 :class:`SttProvider` Protocol. Currently registers
one model — Nepali Whisper-Small (`amitpant7/Nepali-Automatic-Speech-Recognition`)
— matching v1.2.x.

Requires the ``[multilingual]`` extra (`transformers` + `torch`). Without it,
the provider registers but reports empty languages.
"""

from __future__ import annotations

from aawazz_mcp.provider_base import (
    ProviderError,
    SttCapabilities,
    SttRequest,
    SttResult,
)
from aawazz_mcp.registry import register_stt


def _probe_transformers() -> bool:
    try:
        import transformers  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


def _transformers_version() -> str:
    try:
        from importlib.metadata import version
        return version("transformers")
    except Exception:
        return "unknown"


@register_stt("whisper")
class WhisperSttProvider:
    name = "whisper"

    def __init__(self) -> None:
        self._available = _probe_transformers()
        self._version = _transformers_version() if self._available else "not-installed"
        self._loader = None

    @property
    def version(self) -> str:
        return self._version

    def _get_loader(self):
        if self._loader is None:
            from aawazz_mcp.models.whisper_stt import WhisperSttLoader  # noqa: PLC0415
            self._loader = WhisperSttLoader()
        return self._loader

    def capabilities(self) -> SttCapabilities:
        if not self._available:
            return SttCapabilities(
                languages=frozenset(),
                model_archs={},
                accepts_url=False,
                cold_load_seconds_estimate=0.0,
                notes=(
                    "transformers not installed; install via "
                    "``pip install aawazz-mcp[multilingual]``."
                ),
            )

        from aawazz_mcp.models.whisper_stt import supported_languages  # noqa: PLC0415

        langs = supported_languages()
        return SttCapabilities(
            languages=frozenset(langs),
            model_archs={lang: ("whisper-small",) for lang in langs},
            accepts_url=False,
            cold_load_seconds_estimate=8.0,
            notes=(
                "HF transformers Whisper-class pipelines. Nepali (ne) uses "
                "amitpant7/Nepali-Automatic-Speech-Recognition (~927 MB). "
                "First call downloads weights from Hugging Face."
            ),
        )

    async def transcribe(self, request: SttRequest) -> SttResult:
        if not self._available:
            msg = (
                "transformers not installed; install via "
                "``pip install aawazz-mcp[multilingual]``"
            )
            raise ProviderError(msg)

        from aawazz_mcp.models.whisper_stt import supported_languages  # noqa: PLC0415

        if request.language not in supported_languages():
            msg = (
                f"Whisper provider has no model registered for language "
                f"{request.language!r}; supported: {sorted(supported_languages())}"
            )
            raise ProviderError(msg)

        meta = await self._get_loader().transcribe(
            audio_path=request.audio_path,
            language=request.language,
        )
        return SttResult(
            text=meta["text"],
            audio_duration_s=float(meta["audio_duration_s"]),
            sample_rate=int(meta["sample_rate"]),
            latency_ms=int(meta["latency_ms"]),
            model_arch="whisper-small",
        )

    async def aclose(self) -> None:
        self._loader = None
