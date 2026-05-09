"""Real-network smoke against running aawazz-mouth (7861) + aawazz-ears (7862).

Marked ``@pytest.mark.remote`` — opt-in, skipped by default. Run with::

    pytest -m remote tests/test_smoke_remote.py

Auto-skips if either FastAPI server is not reachable, so CI machines without
the systemd units don't false-fail. The probe uses a 1.5s connect timeout so
a missing server doesn't stall the test session.
"""

from __future__ import annotations

import os

import httpx
import pytest

from aawazz_mcp.backends.remote import RemoteBackend


pytestmark = pytest.mark.remote


MOUTH_HEALTH = os.environ.get("AAWAZZ_MOUTH_HEALTH_URL", "http://127.0.0.1:7861/health")
EARS_HEALTH = os.environ.get("AAWAZZ_EARS_HEALTH_URL", "http://127.0.0.1:7862/health")
MOUTH_URL = "http://127.0.0.1:7861/tts"
EARS_URL = "http://127.0.0.1:7862/transcribe"


def _server_up(url: str) -> bool:
    try:
        with httpx.Client(timeout=httpx.Timeout(2.0, connect=1.5)) as c:
            return c.get(url).is_success
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
        return False


@pytest.fixture(scope="module")
def both_servers_up() -> bool:
    if not _server_up(MOUTH_HEALTH):
        pytest.skip(f"aawazz-mouth not reachable at {MOUTH_HEALTH}")
    if not _server_up(EARS_HEALTH):
        pytest.skip(f"aawazz-ears not reachable at {EARS_HEALTH}")
    return True


@pytest.mark.asyncio
async def test_remote_speak_round_trip(both_servers_up: bool) -> None:  # noqa: ARG001
    backend = RemoteBackend(MOUTH_URL, EARS_URL)
    try:
        out = await backend.speak(text="aawazz remote smoke test", voice="MALE", speed=1.0)
    finally:
        await backend.aclose()

    assert "error" not in out, out
    assert out["backend"] == "remote"
    assert out["audio_path"]
    # tiny-tts emits 22050 natively; the FastAPI server resamples to 44100
    # before returning. Accept either; verify it's positive.
    assert out["sample_rate"] in (22050, 44100)
    assert out["text_hash"]


@pytest.mark.asyncio
async def test_remote_speak_then_transcribe(both_servers_up: bool) -> None:  # noqa: ARG001
    backend = RemoteBackend(MOUTH_URL, EARS_URL)
    try:
        spoken = await backend.speak(
            text="the quick brown fox jumps over the lazy dog", voice="MALE", speed=1.0
        )
        assert "error" not in spoken, spoken
        audio_path = spoken["audio_path"]
        assert audio_path and audio_path.endswith(".wav")

        heard = await backend.transcribe(
            audio_path=audio_path, language="en", model_arch="tiny_streaming"
        )
    finally:
        await backend.aclose()

    assert "error" not in heard, heard
    assert heard["backend"] == "remote"
    assert heard["text"], "expected non-empty transcription"
    # Loose check — moonshine can drop articles. "fox" is the dominant noun.
    assert "fox" in heard["text"].lower()
