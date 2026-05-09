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
from aawazz_mcp.models.stt_loader import _arch_from_string
from aawazz_mcp.provider_base import (
    CaptureRequest,
    ProviderError,
    SttRequest,
    TtsRequest,
)
from aawazz_mcp.routing import Router
from aawazz_mcp import registry as _registry

# Importing the providers + post_processors packages triggers built-in
# registration as a side effect.
import aawazz_mcp.post_processors  # noqa: F401, PLC0415
import aawazz_mcp.providers  # noqa: F401, PLC0415


def _apply_audio_chain(
    audio_path: str,
    chain: list[str] | None,
    direction: str,
) -> None:
    """Run a post-process / pre-process chain on the audio at ``audio_path``.

    Each step is looked up in the registry by name. Direction is enforced:
    a "tts" processor in a pre-process chain or vice versa raises
    ProviderError. Reads + writes the file once (chain runs in memory).
    """
    if not chain:
        return

    import soundfile as sf  # noqa: PLC0415

    audio, sr = sf.read(audio_path)
    for name in chain:
        try:
            proc = _registry.get_post(name)
        except KeyError as e:
            msg = (
                f"unknown post-processor {name!r}; registered: "
                f"{sorted(p.name for p in _registry.list_post())}"
            )
            raise ProviderError(msg) from e
        if proc.direction != direction and proc.direction != "both":
            msg = (
                f"post-processor {name!r} has direction={proc.direction!r}; "
                f"can't run in a {direction!r} chain"
            )
            raise ProviderError(msg)
        try:
            audio = proc.process(audio, int(sr))
        except ProviderError:
            raise
        except Exception as e:  # noqa: BLE001
            msg = f"post-processor {name!r} failed: {e}"
            raise ProviderError(msg) from e

    sf.write(audio_path, audio, int(sr), subtype="PCM_16")

log = logging.getLogger("aawazz_mcp.backends.local")

# tiny-tts voice catalog. tiny_tts.utils.config.SPK2ID == {"MALE": 0}; we
# resolve at import time (cheap) and bail with a structured error rather
# than letting tts.speak silently downgrade.
# Additional profiles are DSP post-processing effects (see audio/dsp.py).
_VOICE_MALE = "MALE"
_AVAILABLE_VOICES = [
    "MALE",
    "DEEP",
    "BRIGHT",
    "SOFT",
    "GRAVEL",
    "ROBOT",
    "ECHO",
    "WIDE",
]
_DSP_VOICES = frozenset(_AVAILABLE_VOICES) - {"MALE"}


def _err(message: str, **extra: Any) -> dict:
    """Compose an error payload. Always tags ``backend: "local"``."""
    payload: dict[str, Any] = {"error": message, "backend": "local"}
    payload.update(extra)
    return payload


class LocalBackend(Backend):
    """In-process tiny-tts + Moonshine."""

    def __init__(self, cfg: AawazzConfig) -> None:
        self.cfg = cfg
        self._router = Router(cfg.routing)

    # ------------------------------------------------------------------ warm

    async def warm(self) -> None:
        """Eagerly load the default TTS + STT providers.

        Resolves the default chains via the router so this honors per-language
        config too. Lazy is the runtime default — eager warm at startup is
        opt-in because some MCP runtimes time out ``initialize`` after 5-10s.
        Providers without a ``warm`` method are skipped silently.
        """
        try:
            tts = self._router.resolve_tts(self.cfg.default_language)
            warm = getattr(tts, "warm", None)
            if warm is not None:
                await warm()
        except ProviderError:
            log.warning("warm: no tts provider available for default language")

        try:
            stt = self._router.resolve_stt(self.cfg.default_language)
            warm = getattr(stt, "warm", None)
            if warm is not None:
                # Moonshine warm takes lang+arch; whisper takes nothing.
                try:
                    await warm(
                        language=self.cfg.default_language,
                        model_arch=self.cfg.default_model_arch,
                    )
                except TypeError:
                    await warm()
        except ProviderError:
            log.warning("warm: no stt provider available for default language")

    # ------------------------------------------------------------------ speak

    async def speak(
        self,
        text: str,
        voice: str = "MALE",
        speed: float = 1.0,
        output_path: str | None = None,
        play: bool = False,
        language: str = "en",
        tts_provider: str | None = None,
        post_process: list[str] | None = None,
        playback_provider: str | None = None,
    ) -> dict:
        """Synthesize ``text`` via the routing chain. See SPEC §3 for routing
        semantics; SPEC §1.1 for the response shape.

        ``tts_provider`` overrides the chain (hard-fail if unavailable);
        otherwise the per-language preference list from ``cfg.routing`` is
        consulted, then the ``default`` chain.

        ``post_process`` is an ordered list of post-processor names from the
        registry (e.g. ``["dsp:DEEP", "gain:auto"]``). Each step runs in turn
        on the synthesized WAV. ``voice`` set to a DSP profile name
        (``"DEEP"``/``"BRIGHT"``/...) auto-prepends the corresponding
        ``dsp:<NAME>`` step for v1.2 back-compat.
        """
        normalized_voice = (voice or "").upper()

        # Common input validation (independent of provider choice).
        if not text or not text.strip():
            return _err("text is empty", requested_text=text)
        if len(text) > 4000:
            return _err(
                f"text length {len(text)} exceeds 4000-char cap",
                requested_text_length=len(text),
            )
        if not (0.5 <= speed <= 2.0):
            return _err(
                f"speed {speed} out of range [0.5, 2.0]",
                requested_speed=speed,
            )

        # Route to a provider via the routing chain (or per-call override).
        try:
            provider = self._router.resolve_tts(language, override=tts_provider)
        except ProviderError as e:
            return _err(
                str(e),
                requested_language=language,
                requested_tts_provider=tts_provider,
            )

        # Voice validation + DSP detection. tiny-tts has the strict
        # MALE/DSP allow-list inherited from v1.2; other providers validate
        # internally. DSP voices are a v1.2 hack — phase 5 moves them to
        # PostProcessor objects driven by the ``post_process=`` param.
        is_dsp_voice = normalized_voice in _DSP_VOICES
        if provider.name == "tiny-tts" and normalized_voice not in _AVAILABLE_VOICES:
            return _err(
                f"voice {voice!r} not supported; "
                f"available: {', '.join(_AVAILABLE_VOICES)}",
                available_voices=list(_AVAILABLE_VOICES),
                requested_voice=voice,
            )

        # Translate voice for synthesis. tiny-tts uses MALE; gtts ignores
        # voice; other providers receive ``voice`` as-is.
        if provider.name == "tiny-tts":
            req_voice = "MALE" if is_dsp_voice else normalized_voice
        elif provider.name == "gtts":
            req_voice = None
        else:
            req_voice = voice

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
            tts_result = await provider.synthesize(
                TtsRequest(
                    text=text,
                    language=language,
                    voice=req_voice,
                    speed=float(speed),
                    output_path=str(out),
                )
            )
        except ProviderError as e:
            return _err(
                str(e),
                requested_language=language,
                requested_tts_provider=tts_provider or provider.name,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("tts synthesize failed via %s", provider.name)
            return _err(
                f"synthesis failed: {e}",
                hint="check stderr for provider traceback",
                requested_tts_provider=tts_provider or provider.name,
            )

        audio_path = tts_result.audio_path
        duration_s = tts_result.duration_s
        sample_rate = tts_result.sample_rate
        latency_ms = tts_result.latency_ms

        # Phase-5 post-process pipeline. Legacy ``voice="DEEP"`` (and other
        # DSP profile names) auto-prepends ``dsp:DEEP`` to the chain when the
        # provider accepts DSP profiles. Caller's explicit chain still runs
        # after the legacy step.
        chain: list[str] = []
        if is_dsp_voice and provider.capabilities().accepts_dsp_profiles:
            chain.append(f"dsp:{normalized_voice}")
        if post_process:
            chain.extend(post_process)

        if chain:
            try:
                _apply_audio_chain(str(out), chain, direction="tts")
            except ProviderError as e:
                return _err(str(e), requested_post_process=chain)
            # Re-read metadata since the chain may have changed length.
            import soundfile as sf  # noqa: PLC0415

            info = sf.info(str(out))
            duration_s = float(info.duration)
            sample_rate = int(info.samplerate)

        played = False
        if play:
            pb_name = playback_provider or "shell"
            try:
                player = _registry.get_playback(pb_name)
                played = bool(await player.play(audio_path))
            except KeyError:
                log.warning(
                    "playback provider %r not registered; skipping play", pb_name
                )
            except Exception as e:  # noqa: BLE001 — never crash speak() over playback
                log.warning("playback failed: %s", e)

        # Response shape preserves v1.2 back-compat: gtts emits voice="gtts"
        # and backend="local-gtts"; everything else stays "local".
        if provider.name == "gtts":
            return {
                "audio_path": audio_path,
                "duration_s": duration_s,
                "sample_rate": sample_rate,
                "latency_ms": latency_ms,
                "voice": "gtts",
                "speed": 1.0,
                "text_hash": _text_hash(text),
                "played": played,
                "backend": "local-gtts",
                "provider": provider.name,
                "post_process_chain": chain,
            }
        return {
            "audio_path": audio_path,
            "duration_s": duration_s,
            "sample_rate": sample_rate,
            "latency_ms": latency_ms,
            "voice": normalized_voice,
            "speed": float(speed),
            "text_hash": _text_hash(text),
            "played": played,
            "backend": "local",
            "provider": provider.name,
            "post_process_chain": chain,
        }

    # -------------------------------------------------------------- transcribe

    async def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        model_arch: str = "tiny_streaming",
        stt_provider: str | None = None,
        pre_process: list[str] | None = None,
    ) -> dict:
        """Transcribe a local WAV (or http(s) URL) via the routing chain.

        See SPEC §3 for routing semantics; SPEC §1.2 for response shape.
        ``stt_provider`` overrides the chain (hard-fail on missing or
        language-incompatible).

        ``pre_process`` is an ordered list of post-processor names with
        direction ``"stt"`` or ``"both"`` (e.g. ``["vad:webrtc"]`` to trim
        leading/trailing silence, ``["gain:auto"]`` to peak-normalize).
        Each step runs on the audio buffer before transcription. The
        original audio file at ``audio_path`` is NOT modified — a tempfile
        copy is processed and unlinked after.
        """
        # Resolve the STT provider via the routing chain.
        try:
            provider = self._router.resolve_stt(language, override=stt_provider)
        except ProviderError as e:
            return _err(
                str(e),
                requested_language=language,
                requested_stt_provider=stt_provider,
            )

        # Validate model_arch up front for moonshine-class providers (skip
        # for whisper which doesn't expose a Moonshine arch). Whisper / future
        # providers validate their own arch.
        if provider.name == "moonshine":
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

        # Phase-5 pre_process pipeline. Run on a tempfile copy so the
        # caller's original audio file is never mutated.
        preprocessed_tmp: Path | None = None
        if pre_process:
            import shutil  # noqa: PLC0415
            import tempfile as _tempfile  # noqa: PLC0415

            tmp = _tempfile.NamedTemporaryFile(
                prefix="aawazz-pre-", suffix=".wav", delete=False
            )
            tmp.close()
            preprocessed_tmp = Path(tmp.name)
            shutil.copy(local_path, preprocessed_tmp)
            try:
                _apply_audio_chain(
                    str(preprocessed_tmp), pre_process, direction="stt"
                )
                local_path = str(preprocessed_tmp)
            except ProviderError as e:
                preprocessed_tmp.unlink(missing_ok=True)
                if downloaded_tmp is not None:
                    downloaded_tmp.unlink(missing_ok=True)
                return _err(
                    str(e),
                    requested_audio_path=audio_path,
                    requested_pre_process=pre_process,
                )

        # Whisper has no caller-facing arch; pass None so the provider
        # uses its registered model.
        req_arch: str | None = (
            None if provider.name == "whisper" else model_arch
        )

        try:
            stt_result = await provider.transcribe(
                SttRequest(
                    audio_path=local_path,
                    language=language,
                    model_arch=req_arch,
                )
            )
        except ProviderError as e:
            return _err(str(e), requested_audio_path=audio_path)
        except FileNotFoundError as e:
            return _err(str(e), requested_audio_path=audio_path)
        except IsADirectoryError as e:
            return _err(str(e), requested_audio_path=audio_path)
        except Exception as e:
            log.exception("%s transcribe failed", provider.name)
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
            if preprocessed_tmp is not None:
                try:
                    preprocessed_tmp.unlink(missing_ok=True)
                except OSError:
                    log.warning(
                        "could not unlink pre-processed temp %s",
                        preprocessed_tmp,
                    )

        return {
            "text": stt_result.text,
            "audio_duration_s": stt_result.audio_duration_s,
            "sample_rate": stt_result.sample_rate,
            "latency_ms": stt_result.latency_ms,
            "model_arch": stt_result.model_arch or model_arch,
            "language": language,
            "audio_path": audio_path,  # echo the caller's input (URL or path)
            "backend": "local",
            "provider": provider.name,
            "pre_process_chain": list(pre_process) if pre_process else [],
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
        stt_provider: str | None = None,
        pre_process: list[str] | None = None,
        capture_provider: str | None = None,
    ) -> dict:
        """Capture mic, transcribe via the routing chain. See SPEC §1.3.

        Always local — remote-mode mic-tunneling is out of scope; the
        dispatcher routes ``listen`` straight here regardless of cfg.mode.
        ``stt_provider`` overrides the chain. ``pre_process`` runs on the
        captured audio before STT (e.g. ``["vad:webrtc"]`` to trim
        leading/trailing silence). When ``save_audio=True`` the saved file
        reflects the pre-processed audio, not the raw capture.
        """
        # Hard cap matches SPEC §1.3.
        if not (0.5 <= duration_s <= 30.0):
            return _err(
                f"duration_s {duration_s} out of range [0.5, 30.0]",
                requested_duration_s=duration_s,
            )

        # Resolve provider before mic capture so the arch-validation error
        # surfaces fast.
        try:
            provider = self._router.resolve_stt(language, override=stt_provider)
        except ProviderError as e:
            return _err(
                str(e),
                requested_language=language,
                requested_stt_provider=stt_provider,
            )

        if provider.name == "moonshine":
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

        # Resolve the capture provider (default: sounddevice). The
        # ``record()`` contract: hard-timeout subprocess isolation, raises
        # ProviderError with hint on mic-mute / sandbox / no-device.
        cap_name = capture_provider or "sounddevice"
        try:
            capturer = _registry.get_capture(cap_name)
        except KeyError:
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(
                f"capture provider {cap_name!r} not registered",
                requested_capture_provider=cap_name,
            )

        try:
            await capturer.record(
                CaptureRequest(
                    duration_s=duration_s,
                    sample_rate=16000,
                    save_path=str(capture_path),
                )
            )
        except ProviderError as e:
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(
                str(e),
                hint=e.hint or (
                    "check that a default input device exists; "
                    "voices_list().capabilities.listen reports availability"
                ),
                requested_duration_s=duration_s,
                requested_audio_path=str(capture_path),
            )
        except Exception as e:  # noqa: BLE001
            log.exception("capture provider %r failed", cap_name)
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(
                f"mic capture failed: {e}",
                hint="check that a default input device exists; "
                "voices_list().capabilities.listen reports availability",
                requested_duration_s=duration_s,
            )

        # Phase-5 pre_process pipeline. Mutates the capture file in place;
        # if save_audio=True the saved WAV reflects the post-pre-process
        # state (cleaner output for downstream consumers).
        if pre_process:
            try:
                _apply_audio_chain(
                    str(capture_path), pre_process, direction="stt"
                )
            except ProviderError as e:
                if not save_audio:
                    capture_path.unlink(missing_ok=True)
                return _err(
                    str(e),
                    requested_audio_path=str(capture_path),
                    requested_pre_process=pre_process,
                )

        req_arch: str | None = (
            None if provider.name == "whisper" else model_arch
        )
        try:
            stt_result = await provider.transcribe(
                SttRequest(
                    audio_path=str(capture_path),
                    language=language,
                    model_arch=req_arch,
                )
            )
        except ProviderError as e:
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(str(e), requested_audio_path=str(capture_path))
        except Exception as e:  # noqa: BLE001
            log.exception("listen → transcribe failed")
            if not save_audio:
                capture_path.unlink(missing_ok=True)
            return _err(
                f"transcribe failed: {e}",
                requested_audio_path=str(capture_path),
            )

        result = {
            "text": stt_result.text,
            "audio_duration_s": stt_result.audio_duration_s,
            "sample_rate": stt_result.sample_rate,
            "latency_ms": stt_result.latency_ms,
            "model_arch": stt_result.model_arch or model_arch,
            "language": language,
            "audio_path": str(capture_path) if save_audio else None,
            "backend": "local",
            "provider": provider.name,
            "pre_process_chain": list(pre_process) if pre_process else [],
        }

        if not save_audio:
            try:
                capture_path.unlink(missing_ok=True)
            except OSError:
                log.warning("could not unlink temp capture %s", capture_path)

        return result
