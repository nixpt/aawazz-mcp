"""Remote backend — httpx async client to existing aawazz-mouth/ears FastAPI services.

Wave 1B owns this module. Wire format MUST match the existing Rust arm in
``joker-mcp::modalities``::

    POST {AAWAZZ_MOUTH_URL or http://127.0.0.1:7861}/tts
        body: {"text": str, "voice": str, "speed": float}
        resp: {"audio_path", "duration_s", "sample_rate", "latency_ms",
               "voice", "speed", "text_hash"}

    POST {AAWAZZ_EARS_URL or http://127.0.0.1:7862}/transcribe
        body: {"audio_path": str, "language": str, "model_arch": str}
        resp: {"text", "audio_duration_s", "sample_rate", "latency_ms",
               "model_arch", "language", "audio_path"}

Failure policy: **fail loud, no silent fallback.** When the server is
unreachable / 5xx, return a structured error::

    {
        "error": "remote aawazz-mouth at <url> unreachable: <reason>",
        "hint": "is aawazz-mouth running? `systemctl --user status aawazz-mouth`. Or pass --no-remote.",
        "backend": "remote",
        "url": <url>,
    }

Use ``httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=120.0))`` —
short connect, long read (matches the Rust arm).
"""

from __future__ import annotations

from aawazz_mcp.backends.base import Backend


class RemoteBackend(Backend):
    """Thin httpx client over the existing FastAPI servers."""

    def __init__(self, mouth_url: str | None, ears_url: str | None) -> None:
        self.mouth_url = mouth_url  # full path including /tts, e.g. "http://127.0.0.1:7861/tts"
        self.ears_url = ears_url    # full path including /transcribe
        # Wave 1B: lazy-init self._client: httpx.AsyncClient | None = None

    async def warm(self) -> None:
        # Remote servers warm themselves; no-op.
        return None

    async def speak(self, **kwargs) -> dict:
        raise NotImplementedError("Wave 1B: POST mouth_url with {text, voice, speed}")

    async def transcribe(self, **kwargs) -> dict:
        raise NotImplementedError("Wave 1B: POST ears_url with {audio_path, language, model_arch}")

    async def listen(self, **kwargs) -> dict:
        # NEVER routed here — Dispatcher always picks local for listen.
        raise RuntimeError(
            "RemoteBackend.listen called — dispatcher should route listen to LocalBackend regardless of mode"
        )
