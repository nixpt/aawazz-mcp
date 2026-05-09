"""Built-in Moonshine STT provider.

Wraps :class:`aawazz_mcp.models.stt_loader.SttLoader` (existing v1.0 loader)
behind the v1.3 :class:`SttProvider` Protocol. Per-language arch table
matches the v1.2 dispatcher catalog.
"""

from __future__ import annotations

from aawazz_mcp.provider_base import (
    ProviderError,
    SttCapabilities,
    SttRequest,
    SttResult,
)
from aawazz_mcp.registry import register_stt


def _moonshine_version() -> str:
    try:
        from importlib.metadata import version
        return version("moonshine-voice")
    except Exception:
        return "unknown"


# Per-language available archs; copied from v1.2 dispatcher.lang_models. Update
# alongside ``moonshine_voice`` upgrades or new language coverage.
_LANG_MODELS: dict[str, tuple[str, ...]] = {
    "en": ("tiny", "tiny_streaming", "base", "small_streaming", "medium_streaming"),
    "es": ("base",),
    "zh": ("base",),
    "ja": ("tiny", "base"),
    "ko": ("tiny",),
    "ar": ("base",),
    "vi": ("base",),
    "uk": ("base",),
}


@register_stt("moonshine")
class MoonshineSttProvider:
    name = "moonshine"

    def __init__(self) -> None:
        from aawazz_mcp.models.stt_loader import SttLoader  # noqa: PLC0415
        self._SttLoader = SttLoader
        self._loader = None
        self._version = _moonshine_version()

    @property
    def version(self) -> str:
        return self._version

    def _get_loader(self):
        if self._loader is None:
            self._loader = self._SttLoader()
        return self._loader

    def capabilities(self) -> SttCapabilities:
        return SttCapabilities(
            languages=frozenset(_LANG_MODELS.keys()),
            model_archs=dict(_LANG_MODELS),
            accepts_url=False,
            cold_load_seconds_estimate=3.0,
            notes=(
                "ONNX Moonshine ASR via moonshine_voice. "
                "Non-commercial Moonshine Community License — see "
                "https://www.moonshine.ai/license."
            ),
        )

    async def transcribe(self, request: SttRequest) -> SttResult:
        if request.language not in _LANG_MODELS:
            msg = (
                f"Moonshine does not support language {request.language!r}; "
                f"valid: {sorted(_LANG_MODELS)}"
            )
            raise ProviderError(msg)

        arch = request.model_arch or _LANG_MODELS[request.language][0]
        if arch not in _LANG_MODELS[request.language]:
            msg = (
                f"Moonshine has no {arch!r} arch for language "
                f"{request.language!r}; valid: {_LANG_MODELS[request.language]}"
            )
            raise ProviderError(msg)

        meta = await self._get_loader().transcribe(
            audio_path=request.audio_path,
            language=request.language,
            model_arch=arch,
        )
        return SttResult(
            text=meta["text"],
            audio_duration_s=float(meta["audio_duration_s"]),
            sample_rate=int(meta["sample_rate"]),
            latency_ms=int(meta["latency_ms"]),
            model_arch=arch,
        )

    async def warm(
        self, language: str = "en", model_arch: str = "tiny_streaming"
    ) -> None:
        """Eagerly load a specific (language, arch). Used by ``aawazz-mcp --warm``."""
        await self._get_loader().load(language=language, model_arch=model_arch)

    async def aclose(self) -> None:
        self._loader = None
