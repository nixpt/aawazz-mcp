"""Frozen configuration assembled from CLI args + environment.

Wave 1B owns the full implementation. Wave 0 ships the dataclass shape so
Wave 1A/1C can import the type without circular blocking.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BackendMode = Literal["local", "remote"]


@dataclass(frozen=True)
class AawazzConfig:
    """Resolved configuration. Construct via :meth:`from_args` or :meth:`from_env`.

    Resolution priority (Wave 1B):
        1. CLI ``--remote`` flag (sets mode=remote, joint URL).
        2. Env ``AAWAZZ_REMOTE_URL`` (joint).
        3. Per-service env ``AAWAZZ_MOUTH_URL`` / ``AAWAZZ_EARS_URL`` (overrides joint).
        4. Default: mode=local, both remote URLs None.
    """

    mode: BackendMode = "local"
    remote_mouth_url: str | None = None
    remote_ears_url: str | None = None

    transport: Literal["stdio", "streamable-http"] = "stdio"
    host: str = "127.0.0.1"
    port: int = 7860
    warm: bool = False

    # Default output dir for synthesized WAVs; resolved at startup, may be tempdir
    # if AAWAZZ_HOME is unwritable (sandboxed runtimes).
    output_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("AAWAZZ_HOME", str(Path.home() / ".local/share/aawazz"))
    ) / "mouth")

    # Defaults for STT
    default_language: str = "en"
    default_model_arch: str = "tiny_streaming"

    @classmethod
    def from_args(cls, args) -> "AawazzConfig":  # noqa: ANN001 — argparse.Namespace
        """Wave 1B: implement full resolution."""
        raise NotImplementedError("Wave 1B: env + args → AawazzConfig")

    @classmethod
    def from_env(cls) -> "AawazzConfig":
        """Wave 1B: pure-env resolution (used by tests)."""
        raise NotImplementedError("Wave 1B: env-only AawazzConfig")
