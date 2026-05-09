"""Built-in Piper TTS provider — multi-voice ONNX synthesis (Rhasspy).

Phase 3 of v1.3 (SPEC §8). Wraps :class:`piper.PiperVoice`. Voices are ONNX
files (~60-100 MB each) downloaded from
``https://huggingface.co/rhasspy/piper-voices``.

Voice management
----------------
Voices live under ``~/.local/share/aawazz/piper-voices/`` (or whatever
``$AAWAZZ_PIPER_VOICES_DIR`` points at). Each voice is a pair:

    en_US-amy-medium.onnx
    en_US-amy-medium.onnx.json

If a voice is requested but not on disk, the provider auto-downloads via
``piper.download_voices`` when ``$AAWAZZ_PIPER_AUTO_DOWNLOAD`` is unset or
``"1"``. Set it to ``"0"`` to require manual download via::

    python -m piper.download_voices --download-dir ~/.local/share/aawazz/piper-voices en_US-amy-medium

Capabilities
------------
:meth:`capabilities` reflects only what's *currently* on disk so the routing
chain skips this provider when there's no voice for a language. Auto-download
on first synthesize() expands the installed set; a subsequent
``voices_list()`` call sees the new voice.

Requires the ``[piper]`` extra. Without it, the provider registers but
reports empty capabilities and any synthesize call returns a ProviderError.
"""

from __future__ import annotations

import logging
import os
import re
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

log = logging.getLogger("aawazz_mcp.providers.piper")


_VOICE_RX = re.compile(
    r"^(?P<lang_family>[^-]+)_(?P<lang_region>[^-]+)"
    r"-(?P<voice_name>[^-]+)-(?P<voice_quality>.+)$"
)

# Snapshot of the language families covered by ``rhasspy/piper-voices`` on
# Hugging Face (158 voices across 44 languages, fetched 2026-05-09 from
# https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json).
# Used to populate capabilities.languages when auto-download is enabled —
# otherwise the router skips Piper for any language the user hasn't already
# downloaded a voice for, defeating auto-download. Update on Piper voice
# catalog growth.
_DOWNLOADABLE_LANGS: frozenset[str] = frozenset({
    "ar", "bg", "ca", "cs", "cy", "da", "de", "el", "en", "es", "eu", "fa",
    "fi", "fr", "hi", "hu", "id", "is", "it", "ka", "kk", "ku", "lb", "lv",
    "ml", "ne", "nl", "no", "pl", "pt", "ro", "ru", "sk", "sl", "sq", "sr",
    "sv", "sw", "te", "tr", "uk", "ur", "vi", "zh",
})


def _piper_version() -> str:
    try:
        from importlib.metadata import version
        return version("piper-tts")
    except Exception:
        return "unknown"


def _probe_piper() -> bool:
    try:
        import piper  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


def _voices_dir() -> Path:
    """Where Piper ONNX voices live. ``$AAWAZZ_PIPER_VOICES_DIR`` overrides."""
    env = os.environ.get("AAWAZZ_PIPER_VOICES_DIR")
    if env:
        return Path(env).expanduser()
    home = (
        os.environ.get("AAWAZZ_HOME")
        or str(Path.home() / ".local/share/aawazz")
    )
    return Path(home) / "piper-voices"


def _scan_installed_voices(voices_dir: Path) -> dict[str, Path]:
    """Return ``{voice_id: onnx_path}`` for any installed voices."""
    if not voices_dir.exists():
        return {}
    out: dict[str, Path] = {}
    for f in voices_dir.glob("*.onnx"):
        voice_id = f.stem
        if _VOICE_RX.match(voice_id) and (f.with_suffix(".onnx.json")).exists():
            out[voice_id] = f
    return out


def _lang_from_voice_id(voice_id: str) -> str:
    """``"en_US-amy-medium"`` → ``"en"`` (lang_family only)."""
    m = _VOICE_RX.match(voice_id)
    if not m:
        return ""
    return m.group("lang_family")


@register_tts("piper")
class PiperTtsProvider:
    name = "piper"

    def __init__(self) -> None:
        self._available = _probe_piper()
        self._version = _piper_version() if self._available else "not-installed"
        self._voices_dir = _voices_dir()
        self._installed: dict[str, Path] = (
            _scan_installed_voices(self._voices_dir)
            if self._available
            else {}
        )
        self._loaded: dict[str, Any] = {}  # voice_id → PiperVoice instance
        self._auto_download = (
            os.environ.get("AAWAZZ_PIPER_AUTO_DOWNLOAD", "1") == "1"
        )

    @property
    def version(self) -> str:
        return self._version

    def capabilities(self) -> TtsCapabilities:
        installed_langs: set[str] = set()
        voices: list[VoiceCatalogEntry] = []
        for voice_id in sorted(self._installed):
            lang = _lang_from_voice_id(voice_id)
            if lang:
                installed_langs.add(lang)
            voices.append(
                VoiceCatalogEntry(
                    id=f"piper:{voice_id}",
                    language=lang,
                    description=f"Piper voice {voice_id}",
                )
            )

        # When auto-download is on, we can serve any language in the rhasspy
        # piper-voices catalog — declare them all so the router lets the
        # request through (synthesize() then triggers the download). When
        # off, we can only serve what's already on disk.
        if self._available and self._auto_download:
            languages = frozenset(installed_langs | _DOWNLOADABLE_LANGS)
        else:
            languages = frozenset(installed_langs)

        if not self._available:
            notes = (
                "piper-tts not installed; install via "
                "``pip install aawazz-mcp[piper]``."
            )
        elif not self._installed:
            if self._auto_download:
                notes = (
                    f"piper-tts installed; no voices in {self._voices_dir}; "
                    "auto-download enabled — first synthesize() fetches "
                    "the requested voice from rhasspy/piper-voices."
                )
            else:
                notes = (
                    f"piper-tts installed; no voices in {self._voices_dir}; "
                    "auto-download disabled. Run ``python -m "
                    f"piper.download_voices --download-dir {self._voices_dir}"
                    " <voice_id>`` first."
                )
        else:
            modes = []
            if self._auto_download:
                modes.append("auto-download enabled")
            modes.append(f"voices dir: {self._voices_dir}")
            notes = " · ".join(modes)

        return TtsCapabilities(
            languages=languages,
            voices=tuple(voices),
            requires_network=False,
            sample_rate=22050,
            accepts_dsp_profiles=True,
            speed_range=(0.5, 2.0),
            notes=notes,
        )

    async def synthesize(self, request: TtsRequest) -> TtsResult:
        if not self._available:
            msg = (
                "piper-tts not installed; install via "
                "``pip install aawazz-mcp[piper]``"
            )
            raise ProviderError(msg)
        if request.output_path is None:
            msg = "PiperTtsProvider requires output_path"
            raise ProviderError(msg)

        voice_id = self._resolve_voice(request.voice, request.language)
        voice_obj = await self._ensure_voice_loaded(voice_id)

        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        from piper import SynthesisConfig  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415

        # Piper's length_scale is inverse of "speed": >1 = slower, <1 = faster.
        length_scale = 1.0 / max(0.1, float(request.speed))
        syn_config = SynthesisConfig(length_scale=length_scale)

        t0 = time.time()
        chunks: list[np.ndarray] = []
        sample_rate = 22050
        try:
            for chunk in voice_obj.synthesize(request.text, syn_config=syn_config):
                chunks.append(chunk.audio_float_array)
                sample_rate = chunk.sample_rate
        except Exception as e:  # noqa: BLE001
            msg = f"Piper synthesis failed for voice {voice_id!r}: {e}"
            raise ProviderError(msg) from e

        if not chunks:
            msg = f"Piper produced no audio for text {request.text[:40]!r}"
            raise ProviderError(msg)

        audio = np.concatenate(chunks).astype(np.float32)
        # Clip just in case length_scale produced an over-driven sample.
        np.clip(audio, -1.0, 1.0, out=audio)
        sf.write(str(out), audio, int(sample_rate), subtype="PCM_16")

        latency_ms = int((time.time() - t0) * 1000)
        info = sf.info(str(out))

        return TtsResult(
            audio_path=str(out),
            sample_rate=int(info.samplerate),
            duration_s=float(info.duration),
            latency_ms=latency_ms,
            voice_used=f"piper:{voice_id}",
        )

    def _resolve_voice(
        self, requested: str | None, language: str
    ) -> str:
        """Pick a voice ID. Caller-specified wins; else first installed for the
        language; else ProviderError."""
        if requested:
            voice_id = (
                requested[len("piper:"):]
                if requested.startswith("piper:")
                else requested
            )
            return voice_id

        for voice_id in sorted(self._installed):
            if _lang_from_voice_id(voice_id) == language:
                return voice_id

        msg = (
            f"no Piper voice installed for language {language!r}; "
            f"installed: {sorted(self._installed)}; "
            f"download with: python -m piper.download_voices "
            f"--download-dir {self._voices_dir} <voice_id>"
        )
        raise ProviderError(msg)

    async def _ensure_voice_loaded(self, voice_id: str) -> Any:
        if voice_id in self._loaded:
            return self._loaded[voice_id]

        if not _VOICE_RX.match(voice_id):
            msg = (
                f"invalid Piper voice ID {voice_id!r}; "
                f"expected '<lang>_<region>-<name>-<quality>' "
                f"(e.g. 'en_US-amy-medium')"
            )
            raise ProviderError(msg)

        onnx = self._voices_dir / f"{voice_id}.onnx"
        if not onnx.exists():
            if self._auto_download:
                await self._download_voice(voice_id)
                onnx = self._voices_dir / f"{voice_id}.onnx"
                if not onnx.exists():
                    msg = (
                        f"Piper voice {voice_id!r} download succeeded but "
                        f"file missing at {onnx}"
                    )
                    raise ProviderError(msg)
            else:
                msg = (
                    f"Piper voice {voice_id!r} not installed at "
                    f"{self._voices_dir}; download with: "
                    f"python -m piper.download_voices --download-dir "
                    f"{self._voices_dir} {voice_id} (or set "
                    f"AAWAZZ_PIPER_AUTO_DOWNLOAD=1 for managed downloads)"
                )
                raise ProviderError(msg)

        log.info("loading Piper voice %s from %s", voice_id, onnx)
        import asyncio  # noqa: PLC0415

        from piper import PiperVoice  # noqa: PLC0415

        voice_obj = await asyncio.to_thread(PiperVoice.load, str(onnx))
        self._loaded[voice_id] = voice_obj
        return voice_obj

    async def _download_voice(self, voice_id: str) -> None:
        log.info(
            "auto-downloading Piper voice %s to %s",
            voice_id,
            self._voices_dir,
        )
        self._voices_dir.mkdir(parents=True, exist_ok=True)
        import asyncio  # noqa: PLC0415

        from piper.download_voices import download_voice  # noqa: PLC0415

        try:
            await asyncio.to_thread(
                download_voice, voice_id, self._voices_dir
            )
        except Exception as e:
            msg = (
                f"Piper voice {voice_id!r} download failed: {e}; "
                f"manual: python -m piper.download_voices "
                f"--download-dir {self._voices_dir} {voice_id}"
            )
            raise ProviderError(msg) from e
        # Refresh the installed catalog.
        self._installed = _scan_installed_voices(self._voices_dir)

    async def warm(self, voice_id: str | None = None) -> None:
        """Eagerly load one voice. ``aawazz-mcp --warm`` calls this without
        an arg; we pick the first installed voice."""
        if not self._available or not self._installed:
            return
        if voice_id is None:
            voice_id = sorted(self._installed)[0]
        await self._ensure_voice_loaded(voice_id)

    @property
    def supports_streaming(self) -> bool:
        # Piper streams natively but the v1.4 phase-2 wiring only ships
        # tiny-tts streaming. Piper streaming follow-up is filed.
        return False

    async def synthesize_stream(self, request: TtsRequest, text_stream):  # noqa: ARG002
        msg = "PiperTtsProvider streaming arrives in a follow-up; use batch synthesize"
        raise ProviderError(msg)
        yield  # noqa: B901  - unreachable; marks as async generator

    async def aclose(self) -> None:
        self._loaded.clear()
