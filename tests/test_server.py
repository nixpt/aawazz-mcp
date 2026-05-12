"""Tests for the assembled FastMCP server.

In-memory exercises only — never call ``mcp.run()`` (it blocks on stdio).
We poke ``mcp._tool_manager`` / ``mcp._resource_manager`` to inspect
registration; that's an internal API, but it's the only way to assert the
tool surface without spinning a transport. Runtime code does NOT depend on
those internals.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from aawazz_mcp import __version__
from aawazz_mcp.config import AawazzConfig
from aawazz_mcp.server import build_server


# ---------------------------------------------------------------------------
# Tool + resource registration
# ---------------------------------------------------------------------------


def test_build_server_registers_7_tools_and_health_resource(
    clean_aawazz_env: None,
) -> None:
    """v1.0 contract tools + v1.4 ``respond`` + Termux ``say`` + ``capture_photo`` + health resource."""
    mcp = build_server(AawazzConfig.from_env())

    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert tool_names == {
        "speak",
        "say",
        "transcribe",
        "listen",
        "voices_list",
        "respond",
        "capture_photo",
    }

    resource_uris = {str(r.uri) for r in mcp._resource_manager.list_resources()}
    assert "aawazz://health" in resource_uris


def test_build_server_remote_mode_registers_same_tool_surface(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool surface is mode-independent — agents see the same names always."""
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://r:7860")
    cfg = AawazzConfig.from_env()
    assert cfg.mode == "remote"

    mcp = build_server(cfg)
    tool_names = {t.name for t in mcp._tool_manager.list_tools()}
    assert tool_names == {
        "speak",
        "say",
        "transcribe",
        "listen",
        "voices_list",
        "respond",
        "capture_photo",
    }


def test_tool_descriptions_carry_docstrings(clean_aawazz_env: None) -> None:
    """Tool docstrings ARE the MCP tool descriptions — verify they survive registration."""
    mcp = build_server(AawazzConfig.from_env())
    tools = {t.name: t for t in mcp._tool_manager.list_tools()}

    assert "tiny-tts" in tools["speak"].description.lower()
    assert "moonshine" in tools["transcribe"].description.lower()
    assert "microphone" in tools["listen"].description.lower()
    assert "does not load any models" in tools["voices_list"].description.lower()


# ---------------------------------------------------------------------------
# voices_list: cheap-probe contract — must not load any models
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_voices_list_does_not_load_models(clean_aawazz_env: None) -> None:
    """Calling voices_list() must not instantiate any TtsLoader / SttLoader."""
    mcp = build_server(AawazzConfig.from_env())
    voices = mcp._tool_manager.get_tool("voices_list")
    assert voices is not None

    # Stub capability probes so we don't depend on host audio stack.
    with patch("aawazz_mcp.audio.capture.has_input_device", return_value=True), \
            patch("aawazz_mcp.audio.playback.has_player", return_value=False):
        result = await voices.fn()

    # Shape contract.
    assert result["tts"]["voices"][0]["id"] == "MALE"
    assert "tiny_streaming" in result["stt"]["model_archs"]
    assert result["capabilities"]["backend_mode"] == "local"
    assert result["capabilities"]["listen"] is True
    assert result["capabilities"]["play"] is False

    # Cheap-probe contract: dispatcher must not have built a LocalBackend.
    # We inspect the closed-over dispatcher via the registered speak tool —
    # it shares the same dispatcher with voices_list (build_server creates one).
    # The dispatcher is captured in tool function closures; pull it out.
    speak_tool = mcp._tool_manager.get_tool("speak")
    closure_cells = speak_tool.fn.__closure__ or ()
    dispatcher = next(
        (c.cell_contents for c in closure_cells
         if c.cell_contents.__class__.__name__ == "Dispatcher"),
        None,
    )
    assert dispatcher is not None, "dispatcher should be captured in tool closure"
    assert dispatcher._local is None, "voices_list must not have triggered LocalBackend"


# ---------------------------------------------------------------------------
# Health resource
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_resource_returns_valid_json(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aawazz://health resolves to JSON with version + mode + nested keys."""
    monkeypatch.setenv("AAWAZZ_REMOTE_URL", "http://r:7860")
    mcp = build_server(AawazzConfig.from_env())

    resource = await mcp._resource_manager.get_resource("aawazz://health")
    assert resource is not None

    with patch("aawazz_mcp.audio.capture.has_input_device", return_value=False), \
            patch("aawazz_mcp.audio.playback.has_player", return_value=True):
        body = await resource.read()

    assert isinstance(body, str)
    payload = json.loads(body)

    assert payload["version"] == __version__
    assert payload["mode"] == "remote"
    assert payload["remote_url"]["mouth"] == "http://r:7860/tts"
    assert payload["remote_url"]["ears"] == "http://r:7860/transcribe"
    assert payload["models_loaded"] == {"tts": False, "stt_archs": []}
    assert payload["capabilities"]["listen"] is False
    assert payload["capabilities"]["play"] is True


@pytest.mark.asyncio
async def test_health_resource_local_mode(clean_aawazz_env: None) -> None:
    """Local mode: remote_url has both sides None."""
    mcp = build_server(AawazzConfig.from_env())
    resource = await mcp._resource_manager.get_resource("aawazz://health")
    assert resource is not None

    with patch("aawazz_mcp.audio.capture.has_input_device", return_value=True), \
            patch("aawazz_mcp.audio.playback.has_player", return_value=True):
        body = await resource.read()

    payload = json.loads(body)
    assert payload["mode"] == "local"
    assert payload["remote_url"] == {"mouth": None, "ears": None}


# ---------------------------------------------------------------------------
# Lifespan: warm-on-enter + aclose-on-exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lifespan_warms_when_cfg_warm(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When cfg.warm is True, the lifespan ctx mgr calls dispatcher.warm() before yield."""
    import argparse

    args = argparse.Namespace(remote=None, warm=True, transport="stdio",
                              host="127.0.0.1", port=7860)
    cfg = AawazzConfig.from_args(args)
    assert cfg.warm is True

    calls: list[str] = []

    async def fake_warm(self) -> None:  # noqa: ANN001
        calls.append("warm")

    async def fake_aclose(self) -> None:  # noqa: ANN001
        calls.append("aclose")

    monkeypatch.setattr("aawazz_mcp.dispatcher.Dispatcher.warm", fake_warm)
    monkeypatch.setattr("aawazz_mcp.dispatcher.Dispatcher.aclose", fake_aclose)

    mcp = build_server(cfg)

    # Invoke the lifespan context manager directly. FastMCP wires it through
    # ``settings.lifespan`` — that's the bare async ctx mgr we registered.
    lifespan = mcp.settings.lifespan
    assert lifespan is not None

    async with lifespan(mcp) as ctx:
        # Inside the context: warm should have run, aclose should not have.
        assert calls == ["warm"]
        assert "dispatcher" in ctx
        assert "cfg" in ctx

    # After context exit: aclose ran in finally.
    assert calls == ["warm", "aclose"]


@pytest.mark.asyncio
async def test_lifespan_skips_warm_when_cfg_warm_false(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cfg.warm=False (default): dispatcher.warm() must NOT fire."""
    cfg = AawazzConfig.from_env()
    assert cfg.warm is False

    calls: list[str] = []

    async def fake_warm(self) -> None:  # noqa: ANN001
        calls.append("warm")

    async def fake_aclose(self) -> None:  # noqa: ANN001
        calls.append("aclose")

    monkeypatch.setattr("aawazz_mcp.dispatcher.Dispatcher.warm", fake_warm)
    monkeypatch.setattr("aawazz_mcp.dispatcher.Dispatcher.aclose", fake_aclose)

    mcp = build_server(cfg)
    async with mcp.settings.lifespan(mcp):  # type: ignore[misc]
        pass

    assert calls == ["aclose"]  # no warm, but aclose still runs.


@pytest.mark.asyncio
async def test_lifespan_calls_aclose_even_on_exception(
    clean_aawazz_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """aclose runs in finally — guaranteed even if the body raises."""
    cfg = AawazzConfig.from_env()
    calls: list[str] = []

    async def fake_aclose(self) -> None:  # noqa: ANN001
        calls.append("aclose")

    monkeypatch.setattr("aawazz_mcp.dispatcher.Dispatcher.aclose", fake_aclose)

    mcp = build_server(cfg)

    with pytest.raises(RuntimeError, match="boom"):
        async with mcp.settings.lifespan(mcp):  # type: ignore[misc]
            raise RuntimeError("boom")

    assert calls == ["aclose"]
