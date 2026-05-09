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
            language: ISO 639-1 language code. "en" uses tiny-tts + DSP profiles;
                other languages use gTTS (Google TTS, requires internet).

        Returns:
            ``{audio_path, duration_s, sample_rate, latency_ms, voice, speed,
            text_hash, played, backend}``.
        """
        return await dispatcher.speak(
            text=text,
            voice=voice,
            speed=speed,
            output_path=output_path,
            play=play,
            language=language,
        )

    @mcp.tool()
    async def transcribe(
        audio_path: str,
        language: str = "en",
        model_arch: str = "tiny_streaming",
    ) -> dict:
        """Transcribe a WAV file (or http(s) URL) via Moonshine.

        Args:
            audio_path: Absolute path to a WAV (16/24/32-bit PCM) or http(s) URL.
                URLs are downloaded to ``$TMPDIR/aawazz-stt-<sha8>.wav`` and
                unlinked after transcription.
            language: ISO 639-1 code, e.g. "en".
            model_arch: One of tiny / tiny_streaming / base / base_streaming /
                small_streaming / medium_streaming.

        Returns:
            ``{text, audio_duration_s, sample_rate, latency_ms, model_arch,
            language, audio_path, backend}``.
        """
        return await dispatcher.transcribe(
            audio_path=audio_path,
            language=language,
            model_arch=model_arch,
        )

    @mcp.tool()
    async def listen(
        duration_s: float = 5.0,
        language: str = "en",
        model_arch: str = "tiny_streaming",
        save_audio: bool = False,
    ) -> dict:
        """Capture `duration_s` of microphone audio and transcribe.

        Args:
            duration_s: 0.5..30.0 — hard cap so an agent can't spin forever.
            language: ISO 639-1 code.
            model_arch: See :func:`transcribe`.
            save_audio: If true, return the captured WAV path; otherwise discard.

        Returns:
            ``{text, audio_duration_s, sample_rate, latency_ms, model_arch,
            language, audio_path, backend}`` (backend always "local").

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
