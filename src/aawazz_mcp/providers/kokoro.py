"""Built-in Kokoro-82M TTS provider — pure-ONNX multi-voice synthesis.

Phase 4 of v1.3 (SPEC §8). Wraps :class:`kokoro_onnx.Kokoro`. The model
is one ONNX file (``kokoro-v1.0.onnx``, ~330 MB) plus a voices.bin
catalog; together they cover 50+ voices across English (American +
British), Japanese, Mandarin, Spanish, French, Hindi, Italian,
Portuguese.

Voice management
----------------
Model files live under ``~/.local/share/aawazz/kokoro/`` (or
``$AAWAZZ_KOKORO_DIR``). Two files needed::

    kokoro-v1.0.onnx        # ~330 MB
    voices-v1.0.bin         # ~25 MB

When ``$AAWAZZ_KOKORO_AUTO_DOWNLOAD`` is unset or ``"1"``, missing files
are pulled from the upstream GitHub release on first synthesize().

Voice IDs
---------
Kokoro voice IDs are 3-character prefixes followed by a name::

    af_bella   → American Female, "bella"
    am_adam    → American Male, "adam"
    bf_emma    → British Female, "emma"
    jf_alpha   → Japanese Female, "alpha"
    zm_yunxi   → Mandarin (Chinese) Male, "yunxi"

The first letter is locale (a=American, b=British, j=Japanese,
z=Mandarin/Zhongwen, e=Spanish/Español, f=French, h=Hindi, i=Italian,
p=Portuguese). The second is gender (f=female, m=male).
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

log = logging.getLogger("aawazz_mcp.providers.kokoro")


_VOICE_RX = re.compile(r"^(?P<locale>[a-z])(?P<gender>[fm])_(?P<name>[a-z][a-z0-9_]*)$")

# Locale prefix → ISO 639-1 language code.
_LOCALE_LANG: dict[str, str] = {
    "a": "en",  # American English
    "b": "en",  # British English
    "j": "ja",  # Japanese
    "z": "zh",  # Mandarin (Zhongwen)
    "e": "es",  # Spanish (Español)
    "f": "fr",  # French
    "h": "hi",  # Hindi
    "i": "it",  # Italian
    "p": "pt",  # Portuguese
}

# Locale prefix → kokoro_onnx ``lang`` argument. These are espeak-ng tags
# (kokoro_onnx phonemizes via espeak), which DON'T match ISO 639-1 1:1.
# Notable divergence: Mandarin is ``cmn``, not ``zh``.
_LOCALE_LANG_TAG: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "j": "ja",
    "z": "cmn",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "p": "pt-br",
}

# Catalog snapshot for kokoro-v1.0 (hexgrad/Kokoro-82M, fetched 2026-05-09).
# Keep in sync with the voices.bin file the auto-downloader pulls.
_VOICE_CATALOG: tuple[str, ...] = (
    # American Female
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    # American Male
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck", "am_santa",
    # British Female
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    # British Male
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    # Japanese
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    # Mandarin
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    # Spanish
    "ef_dora", "em_alex", "em_santa",
    # French
    "ff_siwis",
    # Hindi
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    # Italian
    "if_sara", "im_nicola",
    # Portuguese
    "pf_dora", "pm_alex", "pm_santa",
)


_MODEL_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/kokoro-v1.0.onnx"
)
_VOICES_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/"
    "download/model-files-v1.0/voices-v1.0.bin"
)
_MODEL_FILENAME = "kokoro-v1.0.onnx"
_VOICES_FILENAME = "voices-v1.0.bin"


def _kokoro_version() -> str:
    try:
        from importlib.metadata import version
        return version("kokoro-onnx")
    except Exception:
        return "unknown"


def _probe_kokoro() -> bool:
    try:
        import kokoro_onnx  # noqa: F401, PLC0415
        return True
    except Exception:
        return False


def _kokoro_dir() -> Path:
    env = os.environ.get("AAWAZZ_KOKORO_DIR")
    if env:
        return Path(env).expanduser()
    home = (
        os.environ.get("AAWAZZ_HOME")
        or str(Path.home() / ".local/share/aawazz")
    )
    return Path(home) / "kokoro"


def _lang_from_voice_id(voice_id: str) -> str:
    m = _VOICE_RX.match(voice_id)
    if not m:
        return ""
    return _LOCALE_LANG.get(m.group("locale"), "")


def _lang_tag_from_voice_id(voice_id: str) -> str:
    """The ``lang`` arg expected by ``kokoro_onnx.Kokoro.create``."""
    m = _VOICE_RX.match(voice_id)
    if not m:
        return "en-us"
    return _LOCALE_LANG_TAG.get(m.group("locale"), "en-us")


@register_tts("kokoro")
class KokoroTtsProvider:
    name = "kokoro"

    def __init__(self) -> None:
        self._available = _probe_kokoro()
        self._version = _kokoro_version() if self._available else "not-installed"
        self._kokoro_dir = _kokoro_dir()
        self._kokoro: Any | None = None
        self._auto_download = (
            os.environ.get("AAWAZZ_KOKORO_AUTO_DOWNLOAD", "1") == "1"
        )

    @property
    def version(self) -> str:
        return self._version

    def _is_installed(self) -> bool:
        return (
            (self._kokoro_dir / _MODEL_FILENAME).exists()
            and (self._kokoro_dir / _VOICES_FILENAME).exists()
        )

    def capabilities(self) -> TtsCapabilities:
        installed = self._is_installed()
        # When installed (or auto-download will install on demand), we cover
        # the catalog languages. Otherwise empty so the router skips us.
        if not self._available:
            return TtsCapabilities(
                languages=frozenset(),
                voices=(),
                requires_network=False,
                sample_rate=24000,
                accepts_dsp_profiles=True,
                speed_range=(0.5, 2.0),
                notes=(
                    "kokoro-onnx not installed; install via "
                    "``pip install aawazz-mcp[kokoro]``."
                ),
            )

        if installed or self._auto_download:
            languages = frozenset(_LOCALE_LANG.values())
        else:
            languages = frozenset()

        voices = tuple(
            VoiceCatalogEntry(
                id=f"kokoro:{vid}",
                language=_lang_from_voice_id(vid),
                description=f"Kokoro voice {vid}",
            )
            for vid in _VOICE_CATALOG
        )

        if installed:
            notes = f"model dir: {self._kokoro_dir}"
        elif self._auto_download:
            notes = (
                f"model files not yet in {self._kokoro_dir}; "
                "auto-download enabled — first synthesize() pulls "
                f"~355 MB from GitHub release ({_MODEL_URL!s})."
            )
        else:
            notes = (
                f"model files missing in {self._kokoro_dir}; "
                "auto-download disabled. Drop the two release files "
                f"({_MODEL_FILENAME}, {_VOICES_FILENAME}) there or set "
                "AAWAZZ_KOKORO_AUTO_DOWNLOAD=1."
            )

        return TtsCapabilities(
            languages=languages,
            voices=voices,
            requires_network=False,
            sample_rate=24000,
            accepts_dsp_profiles=True,
            speed_range=(0.5, 2.0),
            notes=notes,
        )

    async def synthesize(self, request: TtsRequest) -> TtsResult:
        if not self._available:
            msg = (
                "kokoro-onnx not installed; install via "
                "``pip install aawazz-mcp[kokoro]``"
            )
            raise ProviderError(msg)
        if request.output_path is None:
            msg = "KokoroTtsProvider requires output_path"
            raise ProviderError(msg)

        voice_id = self._resolve_voice(request.voice, request.language)
        kokoro = await self._ensure_loaded()

        out = Path(request.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        import asyncio  # noqa: PLC0415
        import soundfile as sf  # noqa: PLC0415

        lang_tag = _lang_tag_from_voice_id(voice_id)
        speed = float(request.speed)

        t0 = time.time()
        try:
            audio, sr = await asyncio.to_thread(
                kokoro.create,
                request.text,
                voice_id,
                speed,
                lang_tag,
            )
        except Exception as e:  # noqa: BLE001
            msg = (
                f"Kokoro synthesis failed for voice {voice_id!r} "
                f"(lang={lang_tag}): {e}"
            )
            raise ProviderError(msg) from e

        sf.write(str(out), audio, int(sr), subtype="PCM_16")
        latency_ms = int((time.time() - t0) * 1000)
        info = sf.info(str(out))

        return TtsResult(
            audio_path=str(out),
            sample_rate=int(info.samplerate),
            duration_s=float(info.duration),
            latency_ms=latency_ms,
            voice_used=f"kokoro:{voice_id}",
        )

    def _resolve_voice(
        self, requested: str | None, language: str
    ) -> str:
        if requested:
            voice_id = (
                requested[len("kokoro:"):]
                if requested.startswith("kokoro:")
                else requested
            )
            if voice_id not in _VOICE_CATALOG:
                msg = (
                    f"unknown Kokoro voice {voice_id!r}; "
                    f"valid: {list(_VOICE_CATALOG)[:8]}... "
                    f"({len(_VOICE_CATALOG)} total)"
                )
                raise ProviderError(msg)
            return voice_id

        # Default voice per language. Pick a recognizable American Female /
        # equivalent for each lang.
        defaults = {
            "en": "af_bella",
            "ja": "jf_alpha",
            "zh": "zf_xiaoxiao",
            "es": "ef_dora",
            "fr": "ff_siwis",
            "hi": "hf_alpha",
            "it": "if_sara",
            "pt": "pf_dora",
        }
        if language in defaults:
            return defaults[language]
        msg = (
            f"no default Kokoro voice for language {language!r}; "
            f"supported: {sorted(defaults.keys())}"
        )
        raise ProviderError(msg)

    async def _ensure_loaded(self) -> Any:
        if self._kokoro is not None:
            return self._kokoro

        model_path = self._kokoro_dir / _MODEL_FILENAME
        voices_path = self._kokoro_dir / _VOICES_FILENAME

        if not (model_path.exists() and voices_path.exists()):
            if self._auto_download:
                await self._download_model()
            else:
                msg = (
                    f"Kokoro model files missing in {self._kokoro_dir}; "
                    f"download {_MODEL_FILENAME} and {_VOICES_FILENAME} from "
                    f"the kokoro-onnx GitHub release, or set "
                    "AAWAZZ_KOKORO_AUTO_DOWNLOAD=1 for managed downloads"
                )
                raise ProviderError(msg)

        log.info(
            "loading Kokoro model from %s + %s", model_path, voices_path
        )
        import asyncio  # noqa: PLC0415

        from kokoro_onnx import Kokoro  # noqa: PLC0415

        self._kokoro = await asyncio.to_thread(
            Kokoro, str(model_path), str(voices_path)
        )
        return self._kokoro

    async def _download_model(self) -> None:
        log.info("auto-downloading Kokoro model to %s", self._kokoro_dir)
        self._kokoro_dir.mkdir(parents=True, exist_ok=True)

        import asyncio  # noqa: PLC0415
        import urllib.request  # noqa: PLC0415

        def _fetch(url: str, dest: Path) -> None:
            log.info("fetching %s -> %s", url, dest)
            with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:
                while chunk := resp.read(1 << 20):  # 1 MiB chunks
                    fh.write(chunk)

        try:
            await asyncio.to_thread(
                _fetch, _MODEL_URL, self._kokoro_dir / _MODEL_FILENAME
            )
            await asyncio.to_thread(
                _fetch, _VOICES_URL, self._kokoro_dir / _VOICES_FILENAME
            )
        except Exception as e:
            msg = (
                f"Kokoro model download failed: {e}; manual download from "
                f"{_MODEL_URL} and {_VOICES_URL} into {self._kokoro_dir}"
            )
            raise ProviderError(msg) from e

    async def warm(self, voice_id: str | None = None) -> None:
        if not self._available:
            return
        await self._ensure_loaded()

    @property
    def supports_streaming(self) -> bool:
        # Kokoro has create_stream but the v1.4 phase-2 wiring only ships
        # tiny-tts streaming. Kokoro streaming follow-up is filed.
        return False

    async def synthesize_stream(self, request: TtsRequest, text_stream):  # noqa: ARG002
        msg = "KokoroTtsProvider streaming arrives in a follow-up; use batch synthesize"
        raise ProviderError(msg)
        yield  # noqa: B901  - unreachable; marks as async generator

    async def aclose(self) -> None:
        self._kokoro = None
