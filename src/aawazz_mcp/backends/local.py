"""Local backend — bundles tiny-tts + Moonshine + sounddevice.

Implements the :class:`Backend` ABC against in-process loaders. All four
methods (``speak`` / ``transcribe`` / ``listen`` / ``warm``) are async; tiny-tts
and Moonshine are CPU-bound so ``asyncio.to_thread`` keeps the event loop
responsive (loaders own that detail).

CRITICAL — stdout safety:
    See :func:`aawazz_mcp.models.tts_loader.stdout_to_stderr`. The wrap lives
    inside :class:`TtsLoader._speak_blocking`, which means LocalBackend never
    has to worry about it directly. If you add a new tts call site, route
    through the loader.

CRITICAL — voice validation:
    tiny-tts only ships ``MALE``. We reject unknown voices with a structured
    error rather than silently downgrading, so callers can detect misconfig.
    ``available_voices`` lifts ``SPK2ID.keys()`` from tiny-tts.

Errors as structured returns: tools that raise crash the FastMCP tool call,
losing the actionable hint. We catch known failure modes (invalid voice,
unknown arch, missing file, no input device) and return a payload shaped
``{error, hint?, backend, ...}``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from aawazz_mcp.audio.paths import (
    default_output_dir,
    hashed_wav_name,
    text_hash as _text_hash,
)
from aawazz_mcp.backends.base import Backend
from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.models.stt_loader import SttLoader, _arch_from_string
from aawazz_mcp.models.tts_loader import TtsLoader

log = logging.getLogger("aawazz_mcp.backends.local")

# tiny-tts voice catalog. tiny_tts.utils.config.SPK2ID == {"MALE": 0}; we
# resolve at import time (cheap) and bail with a structured error rather
# than letting tts.speak silently downgrade.
_VOICE_MALE = "MALE"
_AVAILABLE_VOICES = [_VOICE_MALE]


def _err(message: str, **extra: Any) -> dict:
    """Compose an error payload. Always tags ``backend: "local"``."""
    payload: dict[str, Any] = {"error": message, "backend": "local"}
    payload.update(extra)
    return payload


class LocalBackend(Backend):
    """In-process tiny-tts + Moonshine."""

    def __init__(self, cfg: AawazzConfig) -> None:
        self.cfg = cfg
        self._tts_loader: TtsLoader | None = None
        self._stt_loader: SttLoader | None = None

    # ---------------------------------------------------------------- helpers

    def _get_tts(self) -> TtsLoader:
        if self._tts_loader is None:
            self._tts_loader = TtsLoader()
        return self._tts_loader

    def _get_stt(self) -> SttLoader:
        if self._stt_loader is None:
            self._stt_loader = SttLoader()
        return self._stt_loader

    # ------------------------------------------------------------------ warm

    async def warm(self) -> None:
        """Eagerly load TTS + the configured default STT arch.

        Used by ``aawazz-mcp --warm`` and ``scripts/prefetch_models.py``. Lazy
        is the runtime default — eager warm at startup is opt-in because some
        MCP runtimes time out the ``initialize`` call after 5-10s.
        """
        await self._get_tts().load()
        await self._get_stt().load(
            language=self.cfg.default_language,
            model_arch=self.cfg.default_model_arch,
        )

    # ------------------------------------------------------------------ speak

    async def speak(
        self,
        text: str,
        voice: str = "MALE",
        speed: float = 1.0,
        output_path: str | None = None,
        play: bool = False,
    ) -> dict:
        """Synthesize ``text`` to a WAV. See SPEC §1.1 for response shape."""
        # Voice validation — reject anything that isn't MALE. tiny-tts itself
        # would silently downgrade, which masks misconfig; we tighten here.
        normalized_voice = (voice or "").upper()
        if normalized_voice != _VOICE_MALE:
            return _err(
                f"voice {voice!r} not supported by tiny-tts; "
                f"only {_VOICE_MALE!r} ships",
                available_voices=list(_AVAILABLE_VOICES),
                requested_voice=voice,
            )

        # Bound the speed range. tiny-tts itself accepts any positive float
        # but extreme values produce garbage.
        if not (0.5 <= speed <= 2.0):
            return _err(
                f"speed {speed} out of range [0.5, 2.0]",
                requested_speed=speed,
            )
        if not text or not text.strip():
            return _err("text is empty", requested_text=text)
        if len(text) > 4000:
            return _err(
                f"text length {len(text)} exceeds 4000-char cap",
                requested_text_length=len(text),
            )

        # Resolve output_path: explicit absolute path wins; else default dir
        # + hashed name. ``default_output_dir`` falls back to tempdir if
        # ``$AAWAZZ_HOME`` is unwritable.
        if output_path:
            out = Path(output_path).expanduser()
            if not out.is_absolute():
                return _err(
                    f"output_path must be absolute, got {output_path!r}",
                    requested_output_path=output_path,
                )
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out = default_output_dir() / hashed_wav_name(text)

        try:
            meta = await self._get_tts().synthesize(
                text=text,
                output_path=str(out),
                voice=normalized_voice,
                speed=float(speed),
            )
        except Exception as e:  # noqa: BLE001 — tiny-tts wraps a pile of errors
            log.exception("tts synthesize failed")
            return _err(
                f"synthesis failed: {e}",
                hint="check stderr for tiny-tts traceback",
            )

        played = False
        if play:
            try:
                from aawazz_mcp.audio.playback import play as _play

                played = bool(_play(meta["audio_path"]))
            except Exception as e:  # noqa: BLE001 — never crash speak() over playback
                log.warning("playback failed: %s", e)

        return {
            "audio_path": meta["audio_path"],
            "duration_s": meta["duration_s"],
            "sample_rate": meta["sample_rate"],
            "latency_ms": meta["latency_ms"],
            "voice": normalized_voice,
            "speed": float(speed),
            "text_hash": _text_hash(text),
            "played": played,
            "backend": "local",
        }

    # -------------------------------------------------------------- transcribe

    async def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        model_arch: str = "tiny_streaming",
    ) -> dict:
        """Transcribe a local WAV (or http(s) URL). See SPEC §1.2."""
        # Validate model_arch up front so the caller gets a clean error before
        # we hit a model load.
        try:
            _arch_from_string(model_arch)
        except ValueError as e:
            return _err(
                str(e),
                requested_model_arch=model_arch,
                hint="valid: tiny, tiny_streaming, base, base_streaming, "
                "small_streaming, medium_streaming",
            )

        # http(s) URL → download to tempfile, transcribe, unlink.
        downloaded_tmp: Path | None = None
        if audio_path.startswith(("http://", "https://")):
            try:
                downloaded_tmp = await self._download_audio_to_temp(audio_path)
                local_path = str(downloaded_tmp)
            except Exception as e:  # noqa: BLE001
                log.exception("download_audio_to_temp failed")
                return _err(
                    f"failed to download {audio_path}: {e}",
                    requested_audio_path=audio_path,
                )
        else:
            local_path = audio_path

        try:
            try:
                meta = await self._get_stt().transcribe(
                    audio_path=local_path,
                    language=language,
                    model_arch=model_arch,
                )
            except FileNotFoundError as e:
                return _err(
                    str(e),
                    requested_audio_path=audio_path,
                )
            except IsADirectoryError as e:
                return _err(str(e), requested_audio_path=audio_path)
            except Exception as e:  # noqa: BLE001
                log.exception("stt transcribe failed")
                return _err(
                    f"transcribe failed: {e}",
                    requested_audio_path=audio_path,
                )
        finally:
            if downloaded_tmp is not None:
                try:
                    downloaded_tmp.unlink(missing_ok=True)
                except OSError:
                    log.warning("could not unlink temp %s", downloaded_tmp)

        return {
            "text": meta["text"],
            "audio_duration_s": meta["audio_duration_s"],
            "sample_rate": meta["sample_rate"],
            "latency_ms": meta["latency_ms"],
            "model_arch": model_arch,
            "language": language,
            "audio_path": audio_path,  # echo the caller's input (URL or path)
            "backend": "local",
        }

    async def _download_audio_to_temp(self, url: str) -> Path:
        """Stream `url` to ``${TMPDIR}/aawazz-stt-<sha8>.wav``. Caller unlinks."""
        import hashlib
        import tempfile

        import httpx

        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        tmp = Path(tempfile.gettempdir()) / f"aawazz-stt-{digest}.wav"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(connect=5.0, read=120.0)
        ) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with tmp.open("wb") as fh:
                    async for chunk in resp.aiter_bytes():
                        fh.write(chunk)
        return tmp

    # ------------------------------------------------------------------ listen

    async def listen(
        self,
        duration_s: float = 5.0,
        language: str = "en",
        model_arch: str = "tiny_streaming",
        save_audio: bool = False,
    ) -> dict:
        """Capture mic, transcribe. See SPEC §1.3.

        Always local — remote-mode mic-tunneling is out of scope; the
        dispatcher routes ``listen`` straight here regardless of cfg.mode.
        """
        # Hard cap matches SPEC §1.3.
        if not (0.5 <= duration_s <= 30.0):
            return _err(
                f"duration_s {duration_s} out of range [0.5, 30.0]",
                requested_duration_s=duration_s,
            )

        # Validate arch eagerly so user sees the error before mic capture.
        try:
            _arch_from_string(model_arch)
        except ValueError as e:
            return _err(
                str(e),
                requested_model_arch=model_arch,
                hint="valid: tiny, tiny_streaming, base, base_streaming, "
                "small_streaming, medium_streaming",
            )

        # Resolve capture path. ``save_audio=False`` writes to tempdir + unlinks
        # post-transcribe. ``save_audio=True`` writes under AAWAZZ_HOME and is
        # returned to the caller.
        import tempfile

        if save_audio:
            ts_name = hashed_wav_name(f"listen-{time.time()}")
            capture_path = default_output_dir() / ts_name
        else:
            tmp = tempfile.NamedTemporaryFile(
                prefix="aawazz-listen-", suffix=".wav", delete=False
            )
            tmp.close()
            capture_path = Path(tmp.name)

        capture_path.parent.mkdir(parents=True, exist_ok=True)

        # Capture mic audio via the audio module. Use the hard-timeout variant
        # so a wedged sd.wait() (mic enumerates but produces no samples — OS
        # mute, UEFI mute, routing) returns a structured error in
        # duration_s + 5s rather than hanging the MCP runtime indefinitely.
        # Same helper backs aawazz-dictate; one fix surface for both legs.
        import asyncio as _asyncio

        try:
            from aawazz_mcp.audio.capture import record_to_wav_hard_timeout

            capture = await _asyncio.to_thread(
                record_to_wav_hard_timeout,
                duration_s,
                str(capture_path),
                duration_s + 5.0,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("record_to_wav_hard_timeout failed")
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(
                f"mic capture failed: {e}",
                hint="check that a default input device exists; "
                "voices_list().capabilities.listen reports availability",
                requested_duration_s=duration_s,
            )
        if capture.get("error") or capture.get("audio_path") is None:
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(
                f"mic capture failed: {capture.get('error', 'no audio path returned')}",
                hint=capture.get(
                    "hint",
                    "check that a default input device exists; "
                    "voices_list().capabilities.listen reports availability",
                ),
                requested_duration_s=duration_s,
                requested_audio_path=str(capture_path),
            )

        try:
            meta = await self._get_stt().transcribe(
                audio_path=str(capture_path),
                language=language,
                model_arch=model_arch,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("listen → transcribe failed")
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(
                f"transcribe failed: {e}",
                requested_audio_path=str(capture_path),
            )

        result = {
            "text": meta["text"],
            "audio_duration_s": meta["audio_duration_s"],
            "sample_rate": meta["sample_rate"],
            "latency_ms": meta["latency_ms"],
            "model_arch": model_arch,
            "language": language,
            "audio_path": str(capture_path) if save_audio else None,
            "backend": "local",
        }

        if not save_audio:
            try:
                capture_path.unlink(missing_ok=True)
            except OSError:
                log.warning("could not unlink temp capture %s", capture_path)

        return result
