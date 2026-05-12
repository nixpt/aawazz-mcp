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

from aawazz_mcp.audio import playback as _playback_audio
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
    LlmRequest,
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
        # voice; other providers treat the legacy "MALE" default as "pick
        # whatever default voice you have" rather than a literal voice id.
        if provider.name == "tiny-tts":
            req_voice = "MALE" if is_dsp_voice else normalized_voice
        elif provider.name == "gtts":
            req_voice = None
        else:
            req_voice = None if voice == "MALE" else voice

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
            pb_name = playback_provider or _playback_audio.default_provider_name()
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

    # ------------------------------------------------------------------ respond

    async def respond(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict] | None = None,
        system_prompt: str | None = None,
        llm_provider: str | None = None,
        llm_model: str | None = None,
        tts_provider: str | None = None,
        language: str = "en",
        voice: str = "MALE",
        speed: float = 1.0,
        play: bool = False,
        stream: bool = False,
        post_process: list[str] | None = None,
        output_path: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        timeout_s: float = 30.0,
        lang_mismatch: str = "route",
    ) -> dict:
        """v1.4 LLM-bridge: generate text via the routed LLM provider, then
        synthesize via the existing speak path. See SPEC_v1.4 §6.

        Either ``prompt`` (one-shot user message) or ``messages``
        (multi-turn caller-managed state) must be provided. ``system_prompt``
        is prepended automatically if not present in ``messages``.
        """
        # Input validation.
        if (prompt is None or not str(prompt).strip()) and not messages:
            return _err(
                "respond requires either prompt or messages",
                hint="pass prompt=<one-shot text> or messages=[{role,content},...]",
            )
        if prompt and messages:
            return _err(
                "pass prompt OR messages, not both",
            )

        if prompt:
            msgs: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        else:
            # mypy/ruff: messages was just confirmed truthy
            msgs = [dict(m) for m in (messages or [])]

        # Resolve LLM provider via the routing chain.
        try:
            llm = self._router.resolve_llm(override=llm_provider)
        except ProviderError as e:
            return _err(
                str(e),
                hint=e.hint,
                requested_llm_provider=llm_provider,
            )

        llm_req = LlmRequest(
            messages=tuple(msgs),
            system_prompt=system_prompt,
            model=llm_model,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            timeout_s=float(timeout_s),
        )

        # Stream path branches early — output_path resolution happens inside
        # _respond_stream since the streaming TTS provider needs it up front.
        if stream:
            return await self._respond_stream(
                llm=llm,
                llm_req=llm_req,
                language=language,
                voice=voice,
                speed=speed,
                play=play,
                post_process=post_process,
                output_path=output_path,
                tts_provider=tts_provider,
                llm_model=llm_model,
            )

        # Batch path (v1.4 phase 1 behavior).
        try:
            llm_result = await llm.complete(llm_req)
        except ProviderError as e:
            return _err(
                str(e),
                hint=e.hint,
                requested_llm_provider=llm.name,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("llm complete failed via %s", llm.name)
            return _err(
                f"llm complete failed: {e}",
                requested_llm_provider=llm.name,
            )

        if not llm_result.text:
            return _err(
                "llm returned empty text",
                requested_llm_provider=llm.name,
                requested_llm_model=llm_model or llm_result.model,
                finish_reason=llm_result.finish_reason,
            )

        # ── Phase 3: language-mismatch detection ────────────────────────────
        # Detect language of the LLM output BEFORE TTS. If detected != request
        # language, apply the lang_mismatch policy. Default ``"route"`` finds
        # a TTS provider that DOES speak the detected language; ``"warn"``
        # synthesizes anyway but tags the response; ``"error"`` hard-fails;
        # ``"off"`` skips detection entirely.
        from aawazz_mcp.audio.lang_detect import detect_language  # noqa: PLC0415

        language_mismatch: dict | None = None
        effective_language = language
        effective_tts_provider = tts_provider

        if lang_mismatch != "off":
            detected = detect_language(llm_result.text)
            if detected and detected != language:
                language_mismatch = {
                    "requested": language,
                    "detected": detected,
                }
                if lang_mismatch == "error":
                    return _err(
                        f"language mismatch: requested {language!r} but LLM "
                        f"emitted {detected!r}",
                        hint=(
                            "set lang_mismatch='route' to auto-route the TTS, "
                            "'warn' to synthesize anyway, or 'off' to skip detection"
                        ),
                        language_mismatch=language_mismatch,
                        text=llm_result.text,
                    )
                if lang_mismatch == "route" and not tts_provider:
                    # Try to find a TTS provider that supports detected lang.
                    # If none does, fall through to warn-style behavior.
                    try:
                        rerouted = self._router.resolve_tts(detected)
                    except ProviderError:
                        log.info(
                            "lang_mismatch=route: no TTS provider supports "
                            "detected %r; falling through to warn",
                            detected,
                        )
                    else:
                        effective_tts_provider = rerouted.name
                        effective_language = detected
                        log.info(
                            "lang_mismatch=route: detected %r → "
                            "rerouting TTS to %r",
                            detected,
                            rerouted.name,
                        )

        # TTS via existing speak path. Carries DSP / post_process / playback
        # behavior verbatim — no special-casing for respond.
        speak_result = await self.speak(
            text=llm_result.text,
            voice=voice,
            speed=float(speed),
            output_path=output_path,
            play=play,
            language=effective_language,
            tts_provider=effective_tts_provider,
            post_process=post_process,
        )

        if "error" in speak_result:
            return {
                "text": llm_result.text,
                "error": speak_result["error"],
                "model": llm_result.model,
                "llm_provider": llm.name,
                "llm_latency_ms": llm_result.latency_ms,
                "prompt_tokens": llm_result.prompt_tokens,
                "completion_tokens": llm_result.completion_tokens,
                "finish_reason": llm_result.finish_reason,
                "audio_path": None,
                "tts_provider": speak_result.get("provider"),
                "backend": "local",
            }

        return {
            "text": llm_result.text,
            "audio_path": speak_result["audio_path"],
            "duration_s": speak_result["duration_s"],
            "sample_rate": speak_result["sample_rate"],
            "played": speak_result["played"],
            "voice": speak_result.get("voice"),
            "model": llm_result.model,
            "llm_provider": llm.name,
            "tts_provider": speak_result.get("provider"),
            "llm_latency_ms": llm_result.latency_ms,
            "tts_latency_ms": speak_result["latency_ms"],
            "total_latency_ms": llm_result.latency_ms + speak_result["latency_ms"],
            "prompt_tokens": llm_result.prompt_tokens,
            "completion_tokens": llm_result.completion_tokens,
            "finish_reason": llm_result.finish_reason,
            "post_process_chain": speak_result.get("post_process_chain", []),
            "language_detected": (
                language_mismatch["detected"]
                if language_mismatch
                else llm_result.language_detected
            ),
            "language_mismatch": language_mismatch,
            "backend": "local",
        }

    # ----------------------------------------------------- respond (streaming)

    async def _respond_stream(
        self,
        *,
        llm,
        llm_req,
        language: str,
        voice: str,
        speed: float,
        play: bool,
        post_process: list[str] | None,
        output_path: str | None,
        tts_provider: str | None,
        llm_model: str | None,
    ) -> dict:
        """Stream-orchestrate respond. Pipes ``llm.stream()`` deltas through
        the sentence chunker, drives ``tts.synthesize_stream``, plays each
        TtsChunk as it arrives. Returns a final dict with concatenated text,
        full WAV path, and first-audio / total latency metrics.

        When the TTS provider doesn't support streaming, falls back to:
        accumulate full LLM text, then call provider.synthesize batch — same
        latency as ``respond(stream=False)``.
        """
        from aawazz_mcp.audio.sentence_chunker import chunk_stream  # noqa: PLC0415

        # Resolve TTS provider (en-path uses tiny-tts by default; routing
        # respects per-language config and tts_provider override).
        try:
            provider = self._router.resolve_tts(
                language, override=tts_provider
            )
        except ProviderError as e:
            return _err(
                str(e), hint=e.hint, requested_language=language,
                requested_tts_provider=tts_provider,
            )

        # Resolve output_path early — both streaming and fallback paths need it.
        if output_path:
            out = Path(output_path).expanduser()
            if not out.is_absolute():
                return _err(
                    f"output_path must be absolute, got {output_path!r}",
                    requested_output_path=output_path,
                )
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out = default_output_dir() / hashed_wav_name(f"respond-stream-{time.time()}")

        # Fallback: TTS provider without streaming → behave like batch
        # respond(stream=False) under the hood. Keeps the contract uniform.
        if not provider.supports_streaming:
            log.info(
                "tts provider %r does not support streaming; falling back to batch",
                provider.name,
            )
            return await self._respond_stream_fallback_batch(
                llm=llm, llm_req=llm_req, language=language,
                voice=voice, speed=speed, play=play,
                post_process=post_process, output_path=str(out),
                tts_provider=tts_provider,
            )

        # Real streaming: pipe LLM deltas → sentence chunker → TTS chunks → play.
        accumulated_text: list[str] = []
        first_audio_at: float | None = None
        first_token_at: float | None = None
        played_any = False
        chunks_played = 0
        playback_provider = _registry.get_playback(
            _playback_audio.default_provider_name()
        )

        t0 = time.time()

        async def _delta_stream():
            """Translate llm.stream() into a string async-iterator,
            recording first-token-arrival time on the way through."""
            nonlocal first_token_at
            async for c in llm.stream(llm_req):
                if c.text and first_token_at is None:
                    first_token_at = time.time() - t0
                if c.text:
                    accumulated_text.append(c.text)
                    yield c.text

        sentences = chunk_stream(_delta_stream(), language=language)

        tts_request = TtsRequest(
            text="",
            language=language,
            voice=voice,
            speed=speed,
            output_path=str(out),
        )

        try:
            async for tts_chunk in provider.synthesize_stream(
                tts_request, sentences
            ):
                if tts_chunk.is_final:
                    continue
                if first_audio_at is None:
                    first_audio_at = time.time() - t0
                # Write the chunk to a tempfile and play synchronously
                # before pulling the next chunk. Synthesis of the next
                # sentence runs in parallel via asyncio.to_thread inside
                # provider.synthesize_stream.
                if play:
                    import soundfile as sf  # noqa: PLC0415
                    import tempfile as _tempfile  # noqa: PLC0415

                    tmp = _tempfile.NamedTemporaryFile(
                        prefix="aawazz-stream-", suffix=".wav", delete=False
                    )
                    tmp.close()
                    try:
                        sf.write(
                            tmp.name,
                            tts_chunk.audio,
                            int(tts_chunk.sample_rate),
                            subtype="PCM_16",
                        )
                        try:
                            ok = await playback_provider.play(tmp.name)
                            played_any = played_any or ok
                            chunks_played += 1
                        except Exception as e:  # noqa: BLE001
                            log.warning("chunk playback failed: %s", e)
                    finally:
                        Path(tmp.name).unlink(missing_ok=True)
        except ProviderError as e:
            return _err(
                str(e), hint=e.hint,
                requested_language=language,
                requested_tts_provider=provider.name,
            )
        except Exception as e:  # noqa: BLE001
            log.exception(
                "stream synthesize failed via %s", provider.name
            )
            return _err(
                f"stream synthesis failed: {e}",
                requested_tts_provider=provider.name,
            )

        total_text = "".join(accumulated_text).strip()
        total_ms = int((time.time() - t0) * 1000)

        # Final WAV is at out (provider wrote it on is_final). Apply
        # post_process chain on the cumulative file (same semantics as
        # batch respond → speak's post_process pipeline).
        post_chain = list(post_process) if post_process else []
        if post_chain:
            try:
                _apply_audio_chain(str(out), post_chain, direction="tts")
            except ProviderError as e:
                log.warning("post_process on streamed WAV failed: %s", e)
        return {
            "text": total_text,
            "audio_path": str(out),
            "played": played_any,
            "model": llm_model or "",
            "llm_provider": llm.name,
            "tts_provider": provider.name,
            "first_token_ms": int(first_token_at * 1000) if first_token_at else None,
            "first_audio_ms": int(first_audio_at * 1000) if first_audio_at else None,
            "total_latency_ms": total_ms,
            "chunks_played": chunks_played,
            "post_process_chain": post_chain,
            "stream": True,
            "backend": "local",
        }

    async def _respond_stream_fallback_batch(
        self,
        *,
        llm,
        llm_req,
        language: str,
        voice: str,
        speed: float,
        play: bool,
        post_process: list[str] | None,
        output_path: str,
        tts_provider: str | None,
    ) -> dict:
        """Stream LLM, then run a single batch TTS. Used when the resolved
        TTS provider doesn't support synthesize_stream. Same end-to-end
        latency as ``respond(stream=False)`` but the LLM half streams so
        callers can observe progress (debug logging only in v1.4.0)."""
        accumulated_text: list[str] = []
        async for chunk in llm.stream(llm_req):
            if chunk.text:
                accumulated_text.append(chunk.text)
        full_text = "".join(accumulated_text).strip()
        if not full_text:
            return _err(
                "llm returned empty text (streaming)",
                requested_llm_provider=llm.name,
            )

        speak_result = await self.speak(
            text=full_text,
            voice=voice,
            speed=float(speed),
            output_path=output_path,
            play=play,
            language=language,
            tts_provider=tts_provider,
            post_process=post_process,
        )
        if "error" in speak_result:
            return {
                "text": full_text,
                "error": speak_result["error"],
                "audio_path": None,
                "llm_provider": llm.name,
                "stream": True,
                "fallback": "batch",
                "backend": "local",
            }
        return {
            "text": full_text,
            "audio_path": speak_result["audio_path"],
            "duration_s": speak_result["duration_s"],
            "sample_rate": speak_result["sample_rate"],
            "played": speak_result["played"],
            "llm_provider": llm.name,
            "tts_provider": speak_result.get("provider"),
            "tts_latency_ms": speak_result["latency_ms"],
            "post_process_chain": speak_result.get("post_process_chain", []),
            "stream": True,
            "fallback": "batch",
            "backend": "local",
        }
