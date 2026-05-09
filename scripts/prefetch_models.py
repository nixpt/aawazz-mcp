"""Pre-fetch tiny-tts + Moonshine model weights so first-call latency is amortized.

Run::

    python -m aawazz_mcp.scripts.prefetch_models
    # or directly:
    python scripts/prefetch_models.py

Useful for offline boxes — runs the loaders once, surfaces download progress
to stderr (NOT stdout — keeps this script safe to pipe into tooling), and
exits 0 if both load successfully.

Exit codes:
    0 — both loaders warmed.
    1 — TTS load failed.
    2 — STT load failed.
    3 — both failed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time

# Force logs to stderr — stdio MCP servers reserve stdout for JSON-RPC.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)


async def _main() -> int:
    # Imported here so script invocation doesn't drag torch in until needed.
    from aawazz_mcp.models.stt_loader import SttLoader
    from aawazz_mcp.models.tts_loader import TtsLoader

    language = os.environ.get("AAWAZZ_PREFETCH_LANG", "en")
    arch = os.environ.get("AAWAZZ_PREFETCH_ARCH", "tiny_streaming")

    rc = 0

    print("[1/2] tiny-tts: loading...", file=sys.stderr, flush=True)
    t0 = time.time()
    try:
        await TtsLoader().load()
        print(
            f"[1/2] tiny-tts: ready ({time.time() - t0:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[1/2] tiny-tts: FAILED — {e}", file=sys.stderr, flush=True)
        rc |= 1

    print(
        f"[2/2] moonshine ({language}/{arch}): loading...",
        file=sys.stderr,
        flush=True,
    )
    t0 = time.time()
    try:
        await SttLoader().load(language=language, model_arch=arch)
        print(
            f"[2/2] moonshine: ready ({time.time() - t0:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[2/2] moonshine: FAILED — {e}", file=sys.stderr, flush=True)
        rc |= 2

    if rc == 0:
        print("aawazz-mcp prefetch: ALL OK", file=sys.stderr, flush=True)
    return rc


def main() -> int:
    return asyncio.run(_main())


if __name__ == "__main__":
    sys.exit(main())
