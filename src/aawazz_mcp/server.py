"""FastMCP server — tool registrations for speak / transcribe / listen / voices_list.

:func:`build_server` returns a FastMCP instance with the four tools registered,
the ``aawazz://health`` resource exposed, and a lifespan async context manager
that warms models on startup (when ``cfg.warm``) and ``aclose``s the dispatcher
on shutdown.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.dispatcher import Dispatcher

INSTRUCTIONS_MD = """\
# aawazz — voice for any agent

Local-CPU TTS + STT MCP server. Tools:

- `speak(text, voice="MALE", speed=1.0)` — synthesize speech (tiny-tts).
- `transcribe(audio_path, language="en", model_arch="tiny_streaming")` — STT on a WAV file.
- `listen(duration_s=5.0, language="en")` — capture mic for `duration_s` and transcribe.
- `voices_list()` — voice/language/model catalog + capability probe (no model load).

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
