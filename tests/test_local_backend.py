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

    async def fake_record_to_wav(
        duration_s: float,
        output_path: str,
        sample_rate: int = 16000,
    ) -> dict:
        return {
            "audio_path": None,
            "error": "mic capture failed: device unavailable",
            "hint": "test capture hint",
        }

    monkeypatch.setattr("aawazz_mcp.audio.capture.record_to_wav", fake_record_to_wav)

    backend = LocalBackend(AawazzConfig.from_env())
    result = await backend.listen(duration_s=1.0, save_audio=False)

    assert result["backend"] == "local"
    assert "mic capture failed" in result["error"]
    assert "audio file not found" not in result["error"]
    assert result["hint"] == "test capture hint"
