"""Pre-fetch tiny-tts + Moonshine model weights so first-call latency is amortized.

Run: ``python -m aawazz_mcp.scripts.prefetch_models``

Useful for offline boxes — runs the loaders once, surfaces download progress to
stderr, then exits 0 if both load successfully.

Wave 1A owns the body (it knows the loaders). Wave 0 ships the entry point.
"""

from __future__ import annotations

import sys


def main() -> int:
    raise NotImplementedError(
        "Wave 1A: import TtsLoader + SttLoader; await load() on each; report stderr."
    )


if __name__ == "__main__":
    sys.exit(main())
