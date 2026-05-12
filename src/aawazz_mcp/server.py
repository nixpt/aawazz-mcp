"""FastMCP server — tool registrations for speak / transcribe / listen / voices_list.

:func:`build_server` returns a FastMCP instance with the four tools registered,
the ``aawazz://health`` resource exposed, and a lifespan async context manager
that warms models on startup (when ``cfg.warm``) and ``aclose``s the dispatcher
on shutdown.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from aawazz_mcp import registry as _registry
from aawazz_mcp.audio import dialogue as _dialogue
from aawazz_mcp.audio import termux_tts as _termux_tts
from aawazz_mcp.audio.paths import default_output_dir
from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.dispatcher import Dispatcher
from aawazz_mcp.vision import termux_camera as _termux_camera

INSTRUCTIONS_MD = """\
# aawazz — voice for any agent

Local-CPU TTS + STT MCP server. Tools:

- `speak(text, voice="MALE", speed=1.0)` — synthesize speech (tiny-tts).
- `say(text, engine=None, pitch=1.0, rate=1.0, stream="MUSIC")` — Termux/Android only:
  speech-only via Android TextToSpeech, no WAV output, low latency.
- `transcribe(audio_path, language="en", model_arch="tiny_streaming")` — STT on a WAV file.
- `listen(duration_s=5.0, language="en")` — capture mic for `duration_s` and transcribe.
- `voices_list()` — voice/language/model catalog + capability probe (no model load).
- `capture_photo(camera_id=0)` — Termux/Android only: snap a JPEG via the
  device camera and return its path (no LLM, no vision model — just the
  capture stage; pair with `respond` / `describe` for scene understanding).
- `dialogue(turns, stereo=False, pause_ms=300, play=False)` — synthesize a
  multi-turn conversation; each turn picks its own voice. Returns one WAV.

By default this server bundles its own copies of tiny-tts and Moonshine. If you
have an `aawazz-mouth` / `aawazz-ears` FastAPI pair running on the same host,
pass `--remote http://127.0.0.1:7861,http://127.0.0.1:7862` to delegate.
"""


def build_server(cfg: AawazzConfig) -> FastMCP:
    """Construct the FastMCP server with Dispatcher-backed tool implementations.

    Args:
        cfg: Resolved configuration (mode=local|remote, transport, warm, etc.).

    Returns:
        A FastMCP instance with `speak`, `transcribe`, `listen`, `voices_list`
        tools registered, an `aawazz://health` resource, and a lifespan that
        warms models on startup (when ``cfg.warm``) and ``aclose``s the
        dispatcher on shutdown.
    """
    dispatcher = Dispatcher(cfg)

    @asynccontextmanager
    async def lifespan(_mcp: FastMCP):
        # Warm BEFORE yielding so the first request after startup hits warm
        # models. aclose AFTER yield in finally so it runs even on shutdown error.
        if cfg.warm:
            await dispatcher.warm()
        try:
            yield {"dispatcher": dispatcher, "cfg": cfg}
        finally:
            await dispatcher.aclose()

    mcp = FastMCP("aawazz", instructions=INSTRUCTIONS_MD, lifespan=lifespan)

    @mcp.tool()
    async def speak(  # noqa: D401
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
        """Synthesize speech from text.

        Args:
            text: 1..4000 characters.
            voice: Voice id or DSP profile. tiny-tts ships "MALE"; DSP profiles:
                DEEP, BRIGHT, SOFT, GRAVEL, ROBOT, ECHO, WIDE.
            speed: Playback speed, 0.5..2.0.
            output_path: Absolute path to write the WAV. Default: under
                ``$AAWAZZ_HOME/mouth/<utc-ts>-<sha8>.wav``.
            play: If true and a desktop audio player is available
                (paplay/aplay/afplay), autoplay the output WAV.
            language: ISO 639-1 language code. Default routing: "en" uses
                tiny-tts + DSP profiles; other languages use gTTS.
            tts_provider: Override the routing chain for this call. Hard-fails
                if the provider is missing or doesn't support the language.
                Use ``voices_list().providers.tts[*].name`` to see what's
                registered. Default ``None`` follows ``cfg.routing.tts``.
            post_process: Ordered list of TTS-direction post-processor names
                applied to the synthesized WAV (e.g. ``["dsp:DEEP", "gain:auto"]``).
                See ``voices_list().providers.post_processors`` for what's
                available. Legacy DSP voice names (``"DEEP"``, ``"BRIGHT"``, …)
                auto-prepend the matching ``dsp:<NAME>`` step.

        Returns:
            ``{audio_path, duration_s, sample_rate, latency_ms, voice, speed,
            text_hash, played, backend, provider, post_process_chain}``.
        """
        return await dispatcher.speak(
            text=text,
            voice=voice,
            speed=speed,
            output_path=output_path,
            play=play,
            language=language,
            tts_provider=tts_provider,
            post_process=post_process,
            playback_provider=playback_provider,
        )

    @mcp.tool()
    async def transcribe(
        audio_path: str,
        language: str = "en",
        model_arch: str = "tiny_streaming",
        stt_provider: str | None = None,
        pre_process: list[str] | None = None,
    ) -> dict:
        """Transcribe a WAV file (or http(s) URL).

        Args:
            audio_path: Absolute path to a WAV (16/24/32-bit PCM) or http(s) URL.
                URLs are downloaded to ``$TMPDIR/aawazz-stt-<sha8>.wav`` and
                unlinked after transcription.
            language: ISO 639-1 code, e.g. "en".
            model_arch: Moonshine arch (tiny / tiny_streaming / base /
                base_streaming / small_streaming / medium_streaming).
                Ignored when the resolved provider isn't Moonshine.
            stt_provider: Override the routing chain for this call. Hard-fails
                if the provider is missing or doesn't support the language.
            pre_process: Ordered list of STT-direction post-processor names
                applied to the audio before transcription (e.g.
                ``["vad:webrtc"]`` to trim silence, ``["gain:auto"]`` to
                peak-normalize). The original ``audio_path`` is not modified.

        Returns:
            ``{text, audio_duration_s, sample_rate, latency_ms, model_arch,
            language, audio_path, backend, provider, pre_process_chain}``.
        """
        return await dispatcher.transcribe(
            audio_path=audio_path,
            language=language,
            model_arch=model_arch,
            stt_provider=stt_provider,
            pre_process=pre_process,
        )

    @mcp.tool()
    async def listen(
        duration_s: float = 5.0,
        language: str = "en",
        model_arch: str = "tiny_streaming",
        save_audio: bool = False,
        stt_provider: str | None = None,
        pre_process: list[str] | None = None,
        capture_provider: str | None = None,
    ) -> dict:
        """Capture `duration_s` of microphone audio and transcribe.

        Args:
            duration_s: 0.5..30.0 — hard cap so an agent can't spin forever.
            language: ISO 639-1 code.
            model_arch: See :func:`transcribe`.
            save_audio: If true, return the captured WAV path; otherwise discard.
            stt_provider: Override the routing chain. Same semantics as
                :func:`transcribe`'s ``stt_provider``.

        Returns:
            ``{text, audio_duration_s, sample_rate, latency_ms, model_arch,
            language, audio_path, backend, provider}`` (backend always "local").

        Notes:
            Always runs locally — the mic is on the host running this MCP server.
            If no input device is available (headless / sandboxed), returns a
            structured error rather than crashing the server.
        """
        return await dispatcher.listen(
            duration_s=duration_s,
            language=language,
            model_arch=model_arch,
            save_audio=save_audio,
            stt_provider=stt_provider,
            pre_process=pre_process,
            capture_provider=capture_provider,
        )

    @mcp.tool()
    async def respond(
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
        """Generate text via the routed LLM provider, then synthesize and
        optionally play it. v1.4 phase 2 — ``stream=True`` pipes LLM tokens
        through the sentence chunker into ``synthesize_stream`` so audio
        starts playing while the LLM is still generating. v1.4 phase 3 —
        ``lang_mismatch`` policy detects the LLM output language and
        re-routes TTS when it differs from ``language``.

        Args:
            prompt: One-shot user message. Mutually exclusive with ``messages``.
            messages: OpenAI-compat ``[{role, content}, ...]`` for multi-turn.
                Caller manages history.
            system_prompt: Prepended to messages if not already present.
                Load-bearing for adapter-flavoured models — bodhi reverts to
                its base identity (Bonsai, sometimes in Russian) without one.
            llm_provider: Override the LLM routing chain. Hard-fails if
                unregistered or unavailable. Default chain: ``["pipefish"]``.
            llm_model: Optional model name passed to the LLM provider.
                For pipefish: any model from ``voices_list().providers.llm[*]
                .backend_models``.
            tts_provider / language / voice / speed / play / post_process /
            output_path: forwarded to :func:`speak` after the LLM produces text.

        Returns:
            ``{text, audio_path, duration_s, sample_rate, played,
            llm_provider, tts_provider, llm_latency_ms, tts_latency_ms,
            total_latency_ms, prompt_tokens, completion_tokens, model,
            finish_reason, language_detected, backend}``.
        """
        return await dispatcher.respond(
            prompt=prompt,
            messages=messages,
            system_prompt=system_prompt,
            llm_provider=llm_provider,
            llm_model=llm_model,
            tts_provider=tts_provider,
            language=language,
            voice=voice,
            speed=speed,
            play=play,
            stream=stream,
            post_process=post_process,
            output_path=output_path,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
            lang_mismatch=lang_mismatch,
        )

    @mcp.tool()
    async def say(
        text: str,
        engine: str | None = None,
        language: str = "en",
        region: str | None = None,
        variant: str | None = None,
        pitch: float = 1.0,
        rate: float = 1.0,
        stream: str = "MUSIC",
    ) -> dict:
        """Speak text via Android TextToSpeech — no WAV, no file output.

        Distinct from :func:`speak`: routes through the host's system TTS
        engine (Termux:API ``termux-tts-speak``) and plays directly to an
        Android audio stream. Returns after the engine finishes
        synthesizing + playing, with no persisted ``audio_path``.

        Useful for low-latency one-shot feedback where the caller doesn't
        need a WAV file. Termux/Android only — falls back to a structured
        error on hosts where ``termux-tts-speak`` isn't on PATH.

        Args:
            text: 1..4000 characters.
            engine: TTS engine package name (e.g. ``"com.samsung.SMT"``,
                ``"com.google.android.tts"``). Default: Android system
                default. Enumerate installed engines via
                :func:`aawazz_mcp.audio.termux_tts.available_engines`.
            language: ISO 639-1 code, e.g. ``"en"``.
            region: Region of language, e.g. ``"US"`` for ``en_US``.
            variant: Language variant (engine-defined).
            pitch: 0.5..2.0. 1.0 is normal pitch; higher = brighter,
                lower = deeper.
            rate: 0.5..2.0. 1.0 is normal rate; 0.5 is half speed,
                2.0 is double speed.
            stream: Android audio stream — one of ``ALARM``, ``MUSIC``,
                ``NOTIFICATION``, ``RING``, ``SYSTEM``, ``VOICE_CALL``.
                Default ``"MUSIC"``: same channel as media playback,
                so audio is reliably audible at the user's media volume.
                ``NOTIFICATION`` auto-ducks music and respects ringer
                mode but is often muted on Android devices (notification
                volume = 0); use it explicitly when you actually want
                notification-style behaviour.

        Returns:
            On success: ``{engine, language, region, variant, pitch,
            rate, stream, text_length, latency_ms, played: true}``.
            On failure: ``{error, hint?, stderr?, latency_ms?}``.
        """
        return await asyncio.to_thread(
            _termux_tts.speak,
            text=text,
            engine=engine,
            language=language,
            region=region,
            variant=variant,
            pitch=pitch,
            rate=rate,
            stream=stream,
        )

    @mcp.tool()
    async def capture_photo(
        camera_id: int = 0,
        output_path: str | None = None,
    ) -> dict:
        """Capture a JPEG from the device camera. Termux/Android only.

        Wraps Termux:API's ``termux-camera-photo``. Returns the path to
        the saved file plus dimensions and size; does NOT do any vision
        analysis or LLM call. Pair with ``respond()`` (or an upcoming
        ``describe()`` tool) to feed the image to a multimodal LLM.

        Args:
            camera_id: Camera index from the device's ``Camera2`` list.
                ``0`` is the rear camera on most devices. Enumerate via
                :func:`aawazz_mcp.vision.termux_camera.available_cameras`.
            output_path: Absolute path to save the JPEG. Default: under
                ``/sdcard/aawazz-eyes/<ts>-cam<id>-<rand>.jpg`` —
                ``/sdcard`` is the only directory reliably reachable by
                both proot-distro and the Termux:API service. Override
                the dir via ``$AAWAZZ_TERMUX_CAMERA_DIR``.

        Returns:
            On success: ``{image_path, width, height, size_bytes,
            camera_id, latency_ms}``.
            On failure: ``{error, hint?, stderr?, latency_ms?}``.
        """
        return await asyncio.to_thread(
            _termux_camera.capture,
            camera_id=camera_id,
            output_path=output_path,
        )

    @mcp.tool()
    async def dialogue(
        turns: list[dict],
        output_path: str | None = None,
        play: bool = False,
        pause_ms: int = 300,
        stereo: bool = False,
        speed: float = 1.0,
        language: str = "en",
        tts_provider: str | None = None,
        post_process: list[str] | None = None,
        playback_provider: str | None = None,
    ) -> dict:
        """Synthesize a multi-turn dialogue and concatenate into one WAV.

        Each turn is a ``{"voice": str, "text": str}`` dict. Different
        voices per turn give the conversational feel — pick e.g. one
        Piper voice for each speaker (``piper:en_US-amy-medium`` and
        ``piper:en_US-ryan-medium``).

        Stereo mode: when exactly two distinct voices appear in ``turns``,
        ``stereo=True`` places the first on the left channel and the
        second on the right — natural two-speaker spatial separation.
        For 1 or 3+ unique voices, ``stereo=True`` falls back to mono
        because the L/R-per-voice mapping doesn't generalise cleanly.

        Args:
            turns: 1..50 turns. Each ``{"voice": str, "text": str}``.
                Text is 1..4000 chars; voice is a provider-specific ID
                (e.g. ``piper:en_US-amy-medium``) — same shape as
                :func:`speak`'s ``voice`` arg.
            output_path: Absolute path to write the concatenated WAV.
                Default: under ``$AAWAZZ_HOME/mouth/dialogue-<ts>-<hash>.wav``.
            play: Autoplay the final WAV via the routed playback provider.
            pause_ms: Inter-turn silence in milliseconds, 0..2000.
                Default 300 — natural conversation cadence.
            stereo: See above.
            speed, language, tts_provider, post_process, playback_provider:
                Defaults applied to every turn's :func:`speak` call.

        Returns:
            On success: ``{audio_path, duration_s, sample_rate, channels,
            latency_ms, turn_count, played, turn_timings:
            [{voice, text_length, audio_path?, duration_s, latency_ms},
            ...]}``.
            On failure: ``{error, hint?, turn_index?, ...}``.
        """
        if not turns:
            return {"error": "turns is empty; provide 1..50 turns"}
        if len(turns) > 50:
            return {
                "error": f"too many turns ({len(turns)}); max 50",
                "turn_count": len(turns),
            }

        # Pre-validate every turn so we don't synthesize a partial dialogue
        # only to discover the 7th turn is malformed.
        for i, turn in enumerate(turns):
            if not isinstance(turn, dict):
                return {"error": f"turn {i} is not a dict", "turn_index": i}
            if "voice" not in turn or "text" not in turn:
                return {
                    "error": f"turn {i} missing voice or text",
                    "turn_index": i,
                }
            text = turn["text"]
            if not isinstance(text, str) or not text.strip():
                return {
                    "error": f"turn {i} text is empty",
                    "turn_index": i,
                }
            if len(text) > 4000:
                return {
                    "error": f"turn {i} text length {len(text)} > 4000",
                    "turn_index": i,
                }

        t0 = time.time()
        turn_timings: list[dict] = []
        turn_paths: list[str] = []
        turn_voices: list[str] = [t["voice"] for t in turns]

        # Per-turn synthesis. Each goes to its own tempfile so the
        # default-output-dir doesn't fill with intermediates.
        tmp_files: list[Path] = []
        try:
            for i, turn in enumerate(turns):
                tmp = tempfile.NamedTemporaryFile(
                    prefix=f"aawazz-dialogue-{i:02d}-",
                    suffix=".wav",
                    delete=False,
                )
                tmp.close()
                tmp_files.append(Path(tmp.name))
                result = await dispatcher.speak(
                    text=turn["text"],
                    voice=turn["voice"],
                    speed=speed,
                    output_path=tmp.name,
                    play=False,
                    language=language,
                    tts_provider=tts_provider,
                    post_process=post_process,
                    playback_provider=None,
                )
                if result.get("error"):
                    return {
                        "error": f"turn {i} synthesis failed: {result['error']}",
                        "turn_index": i,
                        "turn_voice": turn["voice"],
                        **{
                            k: v
                            for k, v in result.items()
                            if k in ("hint", "requested_tts_provider")
                        },
                    }
                turn_paths.append(result["audio_path"])
                turn_timings.append(
                    {
                        "voice": turn["voice"],
                        "text_length": len(turn["text"]),
                        "duration_s": float(result.get("duration_s", 0.0)),
                        "latency_ms": int(result.get("latency_ms", 0)),
                    }
                )

            # Compose. Pure-numpy; runs in this coroutine since CPU work
            # for typical dialogues (<1 min total) is well under a second.
            import soundfile as sf  # noqa: PLC0415

            audio, sample_rate = _dialogue.compose(
                turn_paths=turn_paths,
                turn_voices=turn_voices,
                pause_ms=pause_ms,
                stereo=stereo,
            )

            # Resolve output path.
            if output_path is None:
                out_dir = default_output_dir()
                stamp = int(time.time())
                output_path = str(out_dir / f"dialogue-{stamp}.wav")
            else:
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            sf.write(output_path, audio, sample_rate, subtype="PCM_16")
            info = sf.info(output_path)

            played = False
            if play:
                # Mirror the speak()-side default of "shell" so this PR
                # doesn't depend on the playback default-resolution helper
                # introduced in PR #12 (termux-media). Hosts that need
                # a different provider can pass ``playback_provider=``.
                pb_name = playback_provider or "shell"
                try:
                    player = _registry.get_playback(pb_name)
                    played = bool(await player.play(output_path))
                except KeyError:
                    pass  # silent — same shape as speak()

            return {
                "audio_path": output_path,
                "duration_s": float(info.duration),
                "sample_rate": int(info.samplerate),
                "channels": int(info.channels),
                "latency_ms": int((time.time() - t0) * 1000),
                "turn_count": len(turns),
                "played": played,
                "turn_timings": turn_timings,
            }
        finally:
            for f in tmp_files:
                f.unlink(missing_ok=True)

    @mcp.tool()
    async def voices_list() -> dict:
        """Return the voice / language / model-arch catalog plus capability probe.

        Does NOT load any models — pure metadata, cheap to call as a health probe.

        Returns:
            ``{tts: {backend, voices}, stt: {backend, languages, model_archs},
            capabilities: {listen, play, backend_mode, remote_url}}``.
        """
        return await dispatcher.voices_list()

    @mcp.resource(
        "aawazz://health",
        name="health",
        description="Backend mode, remote URLs, models-loaded flags, and capability probe.",
        mime_type="application/json",
    )
    async def health() -> str:
        """Return the health JSON (SPEC §1.5)."""
        return json.dumps(await dispatcher.health(), indent=2)

    return mcp
