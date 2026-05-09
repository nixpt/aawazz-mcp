"""Round-trip smoke for the local backend.

speak("Hello aawazz") → transcribe(wav) → assert "hello" in lower(text).

Marked ``slow`` because tiny-tts + Moonshine cold-load is ~3-8s on CPU. Run::

    pytest -m slow tests/test_smoke_local.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest


@pytest.mark.slow
def test_speak_then_transcribe_roundtrip(
    tmp_aawazz_home: Path, tmp_path: Path
) -> None:
    """Synthesize 'Hello aawazz', transcribe the WAV, expect 'hello' in output.

    Uses the loaders directly (skips :class:`LocalBackend`) to keep the smoke
    independent of dispatcher / config wiring.
    """
    # Defer import so pytest collection doesn't pull torch in for unrelated
    # test sessions.
    from aawazz_mcp.models.stt_loader import SttLoader
    from aawazz_mcp.models.tts_loader import TtsLoader, stdout_to_stderr

    out_wav = tmp_path / "smoke.wav"

    async def _run() -> dict:
        tts = TtsLoader()
        # stdout_to_stderr is a no-op outside the speak path here, but verify
        # the helper imports and is usable as a context manager.
        with stdout_to_stderr():
            pass
        meta = await tts.synthesize(
            text="Hello aawazz",
            output_path=str(out_wav),
            voice="MALE",
            speed=1.0,
        )
        assert out_wav.exists(), "tts.synthesize did not write the WAV"
        assert meta["sample_rate"] > 0
        assert meta["duration_s"] > 0

        stt = SttLoader()
        result = await stt.transcribe(
            audio_path=str(out_wav),
            language="en",
            model_arch="tiny_streaming",
        )
        return result

    result = asyncio.run(_run())
    text = (result.get("text") or "").lower()
    assert "hello" in text, f"expected 'hello' in transcript, got {text!r}"


@pytest.mark.slow
def test_local_backend_speak_validates_voice(tmp_aawazz_home: Path) -> None:
    """LocalBackend should reject unknown voices with a structured error
    listing all valid voice IDs (MALE + DSP profiles)."""
    # Construct a minimal config substitute. We only touch fields LocalBackend
    # uses for ``speak`` — output_dir + default lang/arch — so we sidestep the
    # AawazzConfig URL-resolution path that this test doesn't exercise.
    from aawazz_mcp.audio.dsp import VOICE_PROFILES
    from aawazz_mcp.backends.local import LocalBackend

    class _FakeCfg:
        output_dir = tmp_aawazz_home / "mouth"
        default_language = "en"
        default_model_arch = "tiny_streaming"

    backend = LocalBackend(_FakeCfg())  # type: ignore[arg-type]

    async def _run():
        return await backend.speak(text="hi", voice="FEMALE")

    result = asyncio.run(_run())
    assert "error" in result
    assert result["backend"] == "local"
    assert set(result["available_voices"]) == set(VOICE_PROFILES)
    assert result["requested_voice"] == "FEMALE"
