"""LocalBackend regression tests that avoid host audio probes at import time."""

from __future__ import annotations

import pytest

from aawazz_mcp.backends.local import LocalBackend
from aawazz_mcp.config import AawazzConfig


@pytest.mark.asyncio
async def test_listen_returns_capture_error_before_transcribe(
    clean_aawazz_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture error should not continue into STT as a misleading missing file."""

    def fake_record_hard_timeout(
        duration_s: float,
        output_path: str,
        timeout_s: float,
    ) -> dict:
        return {
            "audio_path": None,
            "error": "mic capture failed: device unavailable",
            "hint": "test capture hint",
        }

    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.record_to_wav_hard_timeout",
        fake_record_hard_timeout,
    )

    backend = LocalBackend(AawazzConfig.from_env())
    result = await backend.listen(duration_s=1.0, save_audio=False)

    assert result["backend"] == "local"
    assert "mic capture failed" in result["error"]
    assert "audio file not found" not in result["error"]
    assert result["hint"] == "test capture hint"


@pytest.mark.asyncio
async def test_listen_returns_timeout_error_when_capture_wedges(
    clean_aawazz_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hard-timeout variant should surface a timeout-shape error to listen.

    Regression for the s146 listen-hang bug: when the mic enumerates but never
    yields samples, ``record_to_wav_hard_timeout`` returns a timeout payload
    rather than blocking forever inside ``sd.wait``. Listen must propagate
    that as a structured error, not retry or swallow it.
    """

    def fake_timeout(
        duration_s: float,
        output_path: str,
        timeout_s: float,
    ) -> dict:
        return {
            "audio_path": None,
            "error": "mic capture timed out (no samples arrived)",
            "hint": (
                f"device enumerated but produced no samples in {timeout_s:.1f}s. "
                "Check: OS mute, UEFI mute, PulseAudio/PipeWire source routing, "
                "container audio passthrough."
            ),
        }

    monkeypatch.setattr(
        "aawazz_mcp.audio.capture.record_to_wav_hard_timeout",
        fake_timeout,
    )

    backend = LocalBackend(AawazzConfig.from_env())
    result = await backend.listen(duration_s=1.0, save_audio=False)

    assert result["backend"] == "local"
    assert "timed out" in result["error"]
    assert "OS mute" in result["hint"]
