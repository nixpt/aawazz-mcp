"""Tests for AawazzConfig resolution + Dispatcher routing.

Coverage:

- Config: env-only / args-vs-env precedence / split mode / URL normalisation.
- Dispatcher: listen-always-local override, structured-error on unreachable.

LocalBackend is heavy (pulls tiny_tts + moonshine_voice). We never construct
one in these tests — instead we monkeypatch ``Dispatcher._ensure_local`` to
return a fake stub. This keeps the suite fast (<1s) and avoids requiring the
ML stack just to test routing logic.
"""

from __future__ import annotations

import argparse
from unittest.mock import patch

import httpx
import pytest

from aawazz_mcp.backends.base import Backend
from aawazz_mcp.backends.remote import RemoteBackend
from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.dispatcher import Dispatcher


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeLocal(Backend):
    """Stub LocalBackend — records calls, never loads a model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def speak(self, **kwargs) -> dict:
        self.calls.append(("speak", kwargs))
        return {"backend": "local", "audio_path": "/tmp/local.wav"}

    async def transcribe(self, **kwargs) -> dict:
        self.calls.append(("transcribe", kwargs))
        return {"backend": "local", "text": "fake transcription"}

    async def listen(self, **kwargs) -> dict:
        self.calls.append(("listen", kwargs))
        return {"backend": "local", "text": "fake listen"}

    async def warm(self) -> None:
        self.calls.append(("warm", {}))


def _patch_local(dispatcher: Dispatcher) -> FakeLocal:
    """Replace ``dispatcher._ensure_local`` with a fake-returning closure."""
    fake = FakeLocal()
    dispatcher._local = fake  # type: ignore[assignment]
    # Keep _ensure_local idempotent on the fake.
    dispatcher._ensure_local = lambda: fake  # type: ignore[method-assign]
    return fake


# ---------------------------------------------------------------------------
# AawazzConfig.from_env
# ---------------------------------------------------------------------------

def test_from_env_default_local(clean_aawazz_env: None) -> None:
    cfg = AawazzConfig.from_env()
    assert cfg.mode == "local"
    assert cfg.remote_mouth_url is None
    assert cfg.remote_ears_url is None


def test_from_env_remote_joint(clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://10.0.0.5:7861")
    cfg = AawazzConfig.from_env()
    assert cfg.mode == "remote"
    # Joint base — both sides get derived URLs (mouth=/tts, ears=/transcribe).
    assert cfg.remote_mouth_url == "http://10.0.0.5:7861/tts"
    assert cfg.remote_ears_url == "http://10.0.0.5:7861/transcribe"


def test_from_env_remote_joint_comma_split(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "AAWAZZ_REMOTE_URL", "http://mouth.local:7861,http://ears.local:7862"
    )
    cfg = AawazzConfig.from_env()
    assert cfg.remote_mouth_url == "http://mouth.local:7861/tts"
    assert cfg.remote_ears_url == "http://ears.local:7862/transcribe"


def test_from_env_per_service_split_only_mouth(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Split mode: only mouth set — mouth=remote, ears=None (local fallback)."""
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://127.0.0.1:7861")
    cfg = AawazzConfig.from_env()
    assert cfg.mode == "remote"
    assert cfg.remote_mouth_url == "http://127.0.0.1:7861/tts"
    assert cfg.remote_ears_url is None


def test_from_env_per_service_split_only_ears(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_EARS_URL", "http://127.0.0.1:7862/transcribe")
    cfg = AawazzConfig.from_env()
    assert cfg.mode == "remote"
    assert cfg.remote_mouth_url is None
    assert cfg.remote_ears_url == "http://127.0.0.1:7862/transcribe"


def test_from_env_per_service_overrides_joint(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://joint.local:7860")
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://specific-mouth:7861")
    cfg = AawazzConfig.from_env()
    # Per-service mouth wins; ears falls back to the joint URL.
    assert cfg.remote_mouth_url == "http://specific-mouth:7861/tts"
    assert cfg.remote_ears_url == "http://joint.local:7860/transcribe"


def test_from_env_url_with_path_left_intact(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the URL already includes /tts or /transcribe, do not double-append."""
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://127.0.0.1:7861/tts")
    cfg = AawazzConfig.from_env()
    assert cfg.remote_mouth_url == "http://127.0.0.1:7861/tts"


def test_from_env_empty_string_treated_as_unset(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "  ")
    cfg = AawazzConfig.from_env()
    assert cfg.mode == "local"
    assert cfg.remote_mouth_url is None


# ---------------------------------------------------------------------------
# AawazzConfig.from_args precedence
# ---------------------------------------------------------------------------

def test_from_args_overrides_env(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://env.local:7861")
    monkeypatch.setenv("AAWAZZ_EARS_URL", "http://env.local:7862")
    args = argparse.Namespace(remote="http://cli.local:9000", warm=True)
    cfg = AawazzConfig.from_args(args)
    assert cfg.remote_mouth_url == "http://cli.local:9000/tts"
    assert cfg.remote_ears_url == "http://cli.local:9000/transcribe"
    assert cfg.warm is True


def test_from_args_no_remote_falls_through_to_env(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://env.local:7861")
    args = argparse.Namespace(remote=None)
    cfg = AawazzConfig.from_args(args)
    assert cfg.remote_mouth_url == "http://env.local:7861/tts"


def test_from_args_default_local_when_nothing_set(clean_aawazz_env: None) -> None:
    args = argparse.Namespace(remote=None)
    cfg = AawazzConfig.from_args(args)
    assert cfg.mode == "local"


def test_summary_local(clean_aawazz_env: None) -> None:
    cfg = AawazzConfig.from_env()
    assert "mode=local" in cfg.summary()


def test_summary_remote_full(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://r:7860")
    cfg = AawazzConfig.from_env()
    s = cfg.summary()
    assert "mode=remote" in s
    assert "split" not in s


def test_summary_remote_split(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://r:7861")
    cfg = AawazzConfig.from_env()
    assert "split" in cfg.summary()


# ---------------------------------------------------------------------------
# Dispatcher routing
# ---------------------------------------------------------------------------

def test_dispatcher_local_mode_no_remote_instance(clean_aawazz_env: None) -> None:
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)
    assert d._remote is None  # pure local — no httpx client constructed.


def test_dispatcher_remote_mode_constructs_remote(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://r:7860")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)
    assert isinstance(d._remote, RemoteBackend)
    assert d._local is None  # lazy — not constructed until listen() or split fallback.


@pytest.mark.asyncio
async def test_listen_always_routes_to_local_even_when_remote(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://r:7860")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)
    fake = _patch_local(d)
    out = await d.listen(duration_s=1.0)
    assert out["backend"] == "local"
    assert fake.calls == [("listen", {"duration_s": 1.0})]


@pytest.mark.asyncio
async def test_speak_unreachable_returns_structured_error(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://127.0.0.1:7861")
    monkeypatch.setenv("AAWAZZ_EARS_URL", "http://127.0.0.1:7862")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)

    def boom(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(boom)
    # Inject a mocked client into the lazy slot.
    d._remote._client = httpx.AsyncClient(transport=transport)  # type: ignore[union-attr]

    out = await d.speak(text="hello", voice="MALE", speed=1.0)

    assert "error" in out
    assert out["backend"] == "remote"
    assert out["url"] == "http://127.0.0.1:7861/tts"
    assert "unreachable" in out["error"]
    assert "Connection refused" in out["error"]
    assert "systemctl --user status aawazz-mouth" in out["hint"]


@pytest.mark.asyncio
async def test_transcribe_5xx_returns_structured_error(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_EARS_URL", "http://127.0.0.1:7862")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)

    def server_error(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="moonshine model load failed")

    d._remote._client = httpx.AsyncClient(transport=httpx.MockTransport(server_error))  # type: ignore[union-attr]

    out = await d.transcribe(audio_path="/tmp/x.wav")
    assert "error" in out
    assert out["status"] == 503
    assert "moonshine model load failed" in out["error"]


@pytest.mark.asyncio
async def test_speak_success_normalises_response(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://127.0.0.1:7861")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)

    def ok(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/tts"
        return httpx.Response(
            200,
            json={
                "audio_path": "/tmp/out.wav",
                "duration_s": 1.23,
                "sample_rate": 22050,
                "latency_ms": 42,
                "voice": "MALE",
                "speed": 1.0,
                "text_hash": "deadbeef",
            },
        )

    d._remote._client = httpx.AsyncClient(transport=httpx.MockTransport(ok))  # type: ignore[union-attr]

    out = await d.speak(text="hi", voice="MALE", speed=1.0)
    assert out["backend"] == "remote"
    assert out["audio_path"] == "/tmp/out.wav"
    assert out["text_hash"] == "deadbeef"
    assert out["played"] is False


@pytest.mark.asyncio
async def test_speak_success_computes_text_hash_when_server_omits(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://127.0.0.1:7861")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)

    def ok_no_hash(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"audio_path": "/tmp/x.wav", "duration_s": 1.0, "sample_rate": 22050},
        )

    d._remote._client = httpx.AsyncClient(transport=httpx.MockTransport(ok_no_hash))  # type: ignore[union-attr]

    out = await d.speak(text="hello", voice="MALE", speed=1.0)
    # sha1("hello")[:8] == "aaf4c61d"
    assert out["text_hash"] == "aaf4c61d"


@pytest.mark.asyncio
async def test_split_mode_routes_unset_side_to_local(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only mouth set -> speak=remote, transcribe=local fallback."""
    monkeypatch.setenv("AAWAZZ_MOUTH_URL", "http://127.0.0.1:7861")
    cfg = AawazzConfig.from_env()
    assert cfg.remote_ears_url is None
    d = Dispatcher(cfg)
    fake = _patch_local(d)

    out = await d.transcribe(audio_path="/tmp/x.wav")
    assert out["backend"] == "local"
    assert fake.calls == [("transcribe", {"audio_path": "/tmp/x.wav"})]


@pytest.mark.asyncio
async def test_warm_remote_only_does_not_touch_local(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pure-remote setup: warm() must not instantiate LocalBackend."""
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://r:7860")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)

    # Sentinel — if _ensure_local fires, we'd see this raise.
    def explode():
        raise AssertionError("LocalBackend should not be instantiated in pure-remote warm")

    d._ensure_local = explode  # type: ignore[method-assign]
    await d.warm()  # must complete without exploding.


@pytest.mark.asyncio
async def test_voices_list_metadata_only(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)

    # Stub the capability probes so the test doesn't depend on host audio devices.
    with patch("aawazz_mcp.audio.capture.has_input_device", return_value=True), \
            patch("aawazz_mcp.audio.playback.has_player", return_value=True):
        meta = await d.voices_list()

    assert meta["tts"]["voices"][0]["id"] == "MALE"
    assert "tiny_streaming" in meta["stt"]["model_archs"]
    assert meta["capabilities"]["backend_mode"] == "local"
    assert meta["capabilities"]["listen"] is True


@pytest.mark.asyncio
async def test_health_payload_shape(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://r:7860")
    cfg = AawazzConfig.from_env()
    d = Dispatcher(cfg)

    with patch("aawazz_mcp.audio.capture.has_input_device", return_value=False), \
            patch("aawazz_mcp.audio.playback.has_player", return_value=True):
        h = await d.health()

    assert h["mode"] == "remote"
    assert h["remote_url"]["mouth"] == "http://r:7860/tts"
    assert h["remote_url"]["ears"] == "http://r:7860/transcribe"
    assert h["models_loaded"] == {"tts": False, "stt_archs": []}
    assert h["capabilities"]["listen"] is False
    assert h["capabilities"]["play"] is True
    assert "version" in h
