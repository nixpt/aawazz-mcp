"""Lazy tiny-tts loader with stdout-redirect contextmanager.

CRITICAL — stdout safety:
    ``tiny_tts.TinyTTS.speak()`` does ``print(f"Synthesizing: ...")`` and
    ``print(f"Saved audio to ...")`` directly to stdout. Under FastMCP stdio
    transport stdout carries JSON-RPC frames; any rogue ``print`` corrupts
    the wire and hangs the runtime. EVERY ``self._tts.speak(...)``
    invocation MUST be wrapped in :func:`stdout_to_stderr`.

CRITICAL — concurrent calls:
    ``TinyTTS`` is not thread-safe (mutates a torch model in-place). FastMCP
    serves tools concurrently; an ``asyncio.Lock`` per loader serializes
    invocations.

Cache reuse: tiny-tts uses ``~/.cache/huggingface/hub/`` (override ``HF_HOME``).
NLTK data lives in ``~/nltk_data/`` (override ``NLTK_DATA``). Don't invent
``~/.cache/aawazz/``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from pathlib import Path
from typing import Any

import soundfile as sf

log = logging.getLogger("aawazz_mcp.tts_loader")


@contextlib.contextmanager
def stdout_to_stderr():
    """Redirect ``sys.stdout`` to ``sys.stderr`` for the duration of the block.

    The bundled ``tiny_tts`` package prints diagnostics to stdout. Under FastMCP
    stdio transport stdout is reserved for JSON-RPC frames; rogue prints
    corrupt the protocol. Wrap every ``tts.speak(...)`` call in this manager.

    We can't use ``contextlib.redirect_stdout(sys.stderr)`` directly with
    Python's ``print`` because tiny-tts is otherwise well-behaved — but going
    through ``redirect_stdout`` is the canonical idiom and is what we use here.
    Exposed as a top-level helper so other modules (LocalBackend) can wrap
    calls without re-importing tiny-tts.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


def _ensure_nltk_data() -> None:
    """Download NLTK packages tiny-tts needs (g2p_en).

    Idempotent — NLTK skips already-present packages. Bounded so a flaky
    network at first-call doesn't hang the runtime; on failure we log and let
    the synth attempt surface the real error.
    """
    try:
        import nltk

        for pkg in (
            "averaged_perceptron_tagger_eng",
            "averaged_perceptron_tagger",
            "cmudict",
        ):
            try:
                nltk.download(pkg, quiet=True, raise_on_error=False)
            except Exception as e:  # noqa: BLE001 — NLTK raises a few classes
                log.warning("nltk.download(%s) failed: %s", pkg, e)
    except ImportError:
        log.warning("nltk not installed — tiny-tts may fail on first synth")
    except Exception as e:  # noqa: BLE001
        log.warning("nltk preload failed; first synth may error: %s", e)


class TtsLoader:
    """Lazy TinyTTS singleton.

    First :meth:`load` (or first :meth:`synthesize`) imports tiny-tts, runs
    ``_ensure_nltk_data()``, and constructs the model. Subsequent calls reuse
    the cached instance.
    """

    def __init__(self) -> None:
        self._tts: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def loaded(self) -> bool:
        return self._tts is not None

    async def load(self) -> None:
        """Eager-load the model (used by ``--warm`` and prefetch script)."""
        async with self._lock:
            if self._tts is not None:
                return
            await asyncio.to_thread(self._load_blocking)

    def _load_blocking(self) -> None:
        """Heavy import + model init. Must run off the event loop."""
        _ensure_nltk_data()
        log.info("loading TinyTTS model (first call)")
        t0 = time.time()
        # Import inside the function so a fresh ``import aawazz_mcp`` doesn't
        # drag torch in. Stdout-redirect because tiny-tts may print download
        # progress on first run.
        with stdout_to_stderr():
            from tiny_tts import TinyTTS

            self._tts = TinyTTS()
        log.info("TinyTTS loaded in %.2fs", time.time() - t0)

    async def synthesize(
        self,
        text: str,
        output_path: str,
        voice: str,
        speed: float,
    ) -> dict:
        """Synthesize ``text`` to ``output_path``. Returns soundfile metadata.

        Caller is responsible for voice validation; this method passes ``voice``
        straight to ``tiny_tts.TinyTTS.speak``. :class:`LocalBackend` rejects
        unknown voices with a structured error rather than silently downgrading.

        Returns:
            ``{audio_path, duration_s, sample_rate, latency_ms}``.
        """
        # Lazy-load: load() does its own lock; synthesize() holds the lock for
        # the speak call so concurrent tool invocations serialize through tiny-tts.
        if self._tts is None:
            await self.load()

        async with self._lock:
            t0 = time.time()
            await asyncio.to_thread(
                self._speak_blocking, text, output_path, voice, speed
            )
            latency_ms = int((time.time() - t0) * 1000)

        out = Path(output_path)
        if not out.exists():
            raise RuntimeError(
                f"tiny-tts.speak claimed success but no audio file at {output_path}"
            )

        info = sf.info(str(out))
        return {
            "audio_path": str(out),
            "duration_s": float(info.duration),
            "sample_rate": int(info.samplerate),
            "latency_ms": latency_ms,
        }

    def _speak_blocking(
        self, text: str, output_path: str, voice: str, speed: float
    ) -> None:
        """Run tiny-tts under stdout-redirect on a worker thread."""
        # Re-affirmed at the call site: tiny-tts prints to stdout.
        with stdout_to_stderr():
            self._tts.speak(
                text, output_path=output_path, speaker=voice, speed=speed
            )
