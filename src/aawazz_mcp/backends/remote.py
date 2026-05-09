"""Remote backend — httpx async client to existing aawazz-mouth/ears FastAPI services.

Wire format mirrors the existing Rust arm in
``joker-mcp::modalities`` (TTS @ ~lines 625-705, STT @ ~lines 361-440)::

    POST {AAWAZZ_MOUTH_URL or http://127.0.0.1:7861}/tts
        body: {"text": str, "voice": str, "speed": float}
        resp: {"audio_path", "duration_s", "sample_rate", "latency_ms",
               "voice", "speed", "text_hash"}

    POST {AAWAZZ_EARS_URL or http://127.0.0.1:7862}/transcribe
        body: {"audio_path": str, "language": str, "model_arch": str}
        resp: {"text", "audio_duration_s", "sample_rate", "latency_ms",
               "model_arch", "language", "audio_path"}

**Failure policy: fail loud, no silent fallback.** When the server is
unreachable / 5xx / returns malformed JSON, return a structured error::

    {
        "error": "remote aawazz-mouth at <url> unreachable: <reason>",
        "hint": "is aawazz-mouth running? `systemctl --user status aawazz-mouth`. ...",
        "backend": "remote",
        "url": <url>,
    }

Timeouts: ``httpx.AsyncClient(timeout=httpx.Timeout(connect=2.0, read=120.0))`` —
short connect = fast fail, long read = matches the s144 server's worst case.

The client is lazy-instantiated on first call so importing the module is free
(Wave 2's ``voices_list`` may construct a Dispatcher without ever calling speak).
"""

from __future__ import annotations

import asyncio
import time

import httpx

from aawazz_mcp.audio.paths import text_hash as _sha8
from aawazz_mcp.backends.base import Backend


_DEFAULT_TIMEOUT = httpx.Timeout(connect=2.0, read=120.0, write=10.0, pool=2.0)


def _mouth_error(url: str, reason: str) -> dict:
    return {
        "error": f"remote aawazz-mouth at {url} unreachable: {reason}",
        "hint": (
            "is aawazz-mouth running? `systemctl --user status aawazz-mouth`. "
            "Or unset AAWAZZ_MOUTH_URL to use the bundled backend."
        ),
        "backend": "remote",
        "url": url,
    }


def _ears_error(url: str, reason: str) -> dict:
    return {
        "error": f"remote aawazz-ears at {url} unreachable: {reason}",
        "hint": (
            "is aawazz-ears running? `systemctl --user status aawazz-ears`. "
            "Or unset AAWAZZ_EARS_URL to use the bundled backend."
        ),
        "backend": "remote",
        "url": url,
    }


def _server_error(service: str, url: str, status: int, body: str) -> dict:
    truncated = body if len(body) <= 400 else body[:400] + "…(truncated)"
    return {
        "error": f"remote aawazz-{service} at {url} returned HTTP {status}: {truncated}",
        "hint": "check the FastAPI server logs (`journalctl --user -u aawazz-{service}`).".format(service=service),
        "backend": "remote",
        "url": url,
        "status": status,
    }


class RemoteBackend(Backend):
    """Thin httpx client over the existing FastAPI servers.

    Holds a single shared ``AsyncClient`` constructed on first call. The client
    is **never** closed in normal operation — FastMCP servers are long-lived and
    httpx clients are cheap to keep open. Tests may call :meth:`aclose`.
    """

    def __init__(self, mouth_url: str | None, ears_url: str | None) -> None:
        # Already-normalised by AawazzConfig; full path including /tts or /transcribe.
        self.mouth_url = mouth_url
        self.ears_url = ears_url
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    # ----- Client lifecycle -------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ----- Backend ABC ------------------------------------------------------

    async def warm(self) -> None:
        """Remote servers warm themselves on FastAPI startup; nothing to do."""
        return None

    async def speak(
        self,
        text: str,
        voice: str = "MALE",
        speed: float = 1.0,
        output_path: str | None = None,  # noqa: ARG002 — remote ignores: server picks the path
        play: bool = False,  # noqa: ARG002 — remote ignores: playback is local-only (Wave 2 layer)
    ) -> dict:
        if self.mouth_url is None:
            return _mouth_error("<unset>", "AAWAZZ_MOUTH_URL not configured")

        url = self.mouth_url
        client = await self._get_client()
        body = {"text": text, "voice": voice, "speed": float(speed)}

        t0 = time.monotonic()
        try:
            resp = await client.post(url, json=body)
        except httpx.ConnectError as e:
            return _mouth_error(url, f"connection refused ({e})")
        except httpx.ConnectTimeout as e:
            return _mouth_error(url, f"connect timeout ({e})")
        except httpx.ReadTimeout as e:
            return _mouth_error(url, f"read timeout after 120s ({e})")
        except httpx.HTTPError as e:
            return _mouth_error(url, f"http error ({type(e).__name__}: {e})")
        latency_ms = int((time.monotonic() - t0) * 1000)

        if not resp.is_success:
            return _server_error("mouth", url, resp.status_code, resp.text)

        try:
            parsed = resp.json()
        except ValueError as e:
            return {
                "error": f"remote aawazz-mouth at {url} returned invalid JSON: {e}",
                "hint": "the server's response was not valid JSON; check its logs.",
                "backend": "remote",
                "url": url,
                "raw_body": resp.text[:400],
            }

        # Normalise the response into the v1.0 SPEC shape. The s144 server already
        # returns text_hash, but compute defensively from the input if absent.
        return {
            "audio_path": parsed.get("audio_path"),
            "duration_s": parsed.get("duration_s"),
            "sample_rate": parsed.get("sample_rate"),
            "latency_ms": parsed.get("latency_ms", latency_ms),
            "voice": parsed.get("voice", voice),
            "speed": parsed.get("speed", float(speed)),
            "text_hash": parsed.get("text_hash") or _sha8(text),
            "played": False,  # remote can't play on the MCP host; Wave 2's tool layer may post-play.
            "backend": "remote",
        }

    async def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        model_arch: str = "tiny_streaming",
    ) -> dict:
        if self.ears_url is None:
            return _ears_error("<unset>", "AAWAZZ_EARS_URL not configured")

        url = self.ears_url
        client = await self._get_client()
        body = {
            "audio_path": audio_path,
            "language": language,
            "model_arch": model_arch,
        }

        t0 = time.monotonic()
        try:
            resp = await client.post(url, json=body)
        except httpx.ConnectError as e:
            return _ears_error(url, f"connection refused ({e})")
        except httpx.ConnectTimeout as e:
            return _ears_error(url, f"connect timeout ({e})")
        except httpx.ReadTimeout as e:
            return _ears_error(url, f"read timeout after 120s ({e})")
        except httpx.HTTPError as e:
            return _ears_error(url, f"http error ({type(e).__name__}: {e})")
        latency_ms = int((time.monotonic() - t0) * 1000)

        if not resp.is_success:
            return _server_error("ears", url, resp.status_code, resp.text)

        try:
            parsed = resp.json()
        except ValueError as e:
            return {
                "error": f"remote aawazz-ears at {url} returned invalid JSON: {e}",
                "hint": "the server's response was not valid JSON; check its logs.",
                "backend": "remote",
                "url": url,
                "raw_body": resp.text[:400],
            }

        return {
            "text": parsed.get("text", ""),
            "audio_duration_s": parsed.get("audio_duration_s"),
            "sample_rate": parsed.get("sample_rate"),
            "latency_ms": parsed.get("latency_ms", latency_ms),
            "model_arch": parsed.get("model_arch", model_arch),
            "language": parsed.get("language", language),
            "audio_path": parsed.get("audio_path", audio_path),
            "backend": "remote",
        }

    async def listen(
        self,
        duration_s: float = 5.0,  # noqa: ARG002
        language: str = "en",  # noqa: ARG002
        model_arch: str = "tiny_streaming",  # noqa: ARG002
        save_audio: bool = False,  # noqa: ARG002
    ) -> dict:
        # NEVER routed here — Dispatcher always picks local for listen. The mic
        # lives on the host running this MCP server; remote servers can't reach
        # it. If this fires, something in Dispatcher routing is broken.
        raise RuntimeError(
            "RemoteBackend.listen called — dispatcher must route listen to LocalBackend "
            "regardless of cfg.mode (mic is on the MCP server's host)."
        )
