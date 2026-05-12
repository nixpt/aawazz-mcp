"""Test fixtures — env isolation + tmp output dir."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_routing_config(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Point ``$AAWAZZ_ROUTING_FILE`` at a tmp path for every test.

    Without this, any test that builds a Dispatcher / LocalBackend /
    Router reads the operator's real ``~/.config/aawazz/aawazz.toml``
    (or ``$XDG_CONFIG_HOME/aawazz/aawazz.toml``) — see issue #20. The
    config file resolver falls through to built-in defaults when the
    path doesn't exist, so leaving the file absent gives every test
    a clean baseline.

    Tests that intentionally exercise file-based routing should write
    a TOML to ``$AAWAZZ_ROUTING_FILE`` (already set to a writable tmp
    path here).
    """
    routing_dir = tmp_path_factory.mktemp("aawazz-routing")
    monkeypatch.setenv("AAWAZZ_ROUTING_FILE", str(routing_dir / "aawazz.toml"))


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
    """Clear AAWAZZ_* env vars so config resolution is deterministic per-test.

    Preserves ``AAWAZZ_ROUTING_FILE`` — that's managed by the autouse
    :func:`_isolate_routing_config` fixture, and clearing it here would
    let the operator's real ``~/.config/aawazz/aawazz.toml`` leak in
    (issue #20).
    """
    for key in list(os.environ):
        if key.startswith("AAWAZZ_") and key != "AAWAZZ_ROUTING_FILE":
            monkeypatch.delenv(key, raising=False)
