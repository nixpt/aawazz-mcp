"""Path helpers — default output dir, tempdir fallback, hash-stamped naming.

Wave 0 ships :func:`default_output_dir` because Wave 1A and Wave 1B both need
the same answer; centralizing keeps them aligned. The full implementation here
covers naming + tempdir fallback so Wave 1A only needs to read.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def default_output_dir() -> Path:
    """Resolve the default output dir, falling back to tempdir if unwritable.

    Resolution:
        1. ``$AAWAZZ_HOME/mouth`` (default ``~/.local/share/aawazz/mouth``)
        2. ``$TMPDIR/aawazz/mouth`` (sandboxed runtimes that don't allow ~).
    """
    base = Path(os.environ.get("AAWAZZ_HOME", str(Path.home() / ".local/share/aawazz")))
    candidate = base / "mouth"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        # Probe writability — some sandboxes allow mkdir but block writes.
        probe = candidate / ".aawazz-write-probe"
        probe.touch()
        probe.unlink()
        return candidate
    except OSError:
        fallback = Path(tempfile.gettempdir()) / "aawazz" / "mouth"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def hashed_wav_name(text: str) -> str:
    """Naming convention matching the s144 mouth server: ``<utc-ts>-<sha8>.wav``.

    Format: ``YYYYMMDDTHHMMSSZ-<8-char-sha1-hex>.wav`` — sortable, idempotent
    per (text, second).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{ts}-{digest}.wav"


def text_hash(text: str) -> str:
    """8-char sha1 hex of the input text. Used in ``speak`` response payload."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
