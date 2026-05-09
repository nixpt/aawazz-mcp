"""Test fixtures — env isolation + tmp output dir."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def tmp_aawazz_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set AAWAZZ_HOME to a tmp dir for the duration of one test.

    Use this in any test that calls ``default_output_dir()`` or invokes a tool
    that writes a WAV — keeps the user's real ``~/.local/share/aawazz/`` clean.
    """
    home = tmp_path / "aawazz"
    home.mkdir()
    monkeypatch.setenv("AAWAZZ_HOME", str(home))
    return home


@pytest.fixture
def clean_aawazz_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear AAWAZZ_* env vars so config resolution is deterministic per-test."""
    for key in list(os.environ):
        if key.startswith("AAWAZZ_"):
            monkeypatch.delenv(key, raising=False)
