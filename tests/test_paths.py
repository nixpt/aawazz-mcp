"""Coverage for :mod:`aawazz_mcp.audio.paths`."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest


def test_default_output_dir_happy_path(
    tmp_aawazz_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``$AAWAZZ_HOME`` is writable, return ``$AAWAZZ_HOME/mouth``."""
    from aawazz_mcp.audio.paths import default_output_dir

    out = default_output_dir()
    assert out == tmp_aawazz_home / "mouth"
    assert out.is_dir()


def test_default_output_dir_fallback_to_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``$AAWAZZ_HOME`` is unwritable, fall back to ``$TMPDIR/aawazz/mouth``.

    Sandboxed runtimes (e.g. some MCP-host subprocesses) may allow ``mkdir``
    but block writes. ``default_output_dir`` probes by touching a file; on
    OSError it falls back to ``tempfile.gettempdir()``.
    """
    from aawazz_mcp.audio.paths import default_output_dir

    # Build a read-only AAWAZZ_HOME so the touch-probe inside the helper fails.
    home = tmp_path / "ro_aawazz"
    home.mkdir()
    monkeypatch.setenv("AAWAZZ_HOME", str(home))

    # Pre-create the candidate as 0o500 (r-x) so mkdir succeeds (already
    # exists) but the probe write OSErrors.
    candidate = home / "mouth"
    candidate.mkdir()
    candidate.chmod(stat.S_IRUSR | stat.S_IXUSR)

    try:
        # Skip when running as root (chmod doesn't enforce write block).
        if os.geteuid() == 0:
            pytest.skip("read-only mode is no-op as root")

        out = default_output_dir()
        assert out != candidate, (
            "should have fallen back, but kept the read-only candidate"
        )
        assert out.is_dir()
        # Fallback path lives under tempdir (per implementation).
        import tempfile

        assert str(out).startswith(tempfile.gettempdir())
    finally:
        # Restore perms so pytest can clean up tmp_path.
        candidate.chmod(stat.S_IRWXU)


def test_hashed_wav_name_format() -> None:
    """``hashed_wav_name`` returns ``<utc-ts>-<sha8>.wav``."""
    from aawazz_mcp.audio.paths import hashed_wav_name

    name = hashed_wav_name("Hello aawazz")
    assert name.endswith(".wav")
    # 16 ts chars + 1 dash + 8 sha + 4 ext = 29 chars total (e.g.
    # ``20260509T123456Z-abcd1234.wav``).
    assert len(name) == len("20260509T123456Z-abcd1234.wav")


def test_text_hash_stable() -> None:
    """``text_hash`` is deterministic and 8 hex chars."""
    from aawazz_mcp.audio.paths import text_hash

    h = text_hash("Hello aawazz")
    assert len(h) == 8
    assert all(c in "0123456789abcdef" for c in h)
    assert text_hash("Hello aawazz") == h  # idempotent
