"""Frozen configuration assembled from CLI args + environment.

Resolution priority (highest first):

1. CLI flag ``--remote http://mouth-url[,http://ears-url]`` (sets ``mode=remote``;
   one URL = joint base for both services, two = explicit per-service).
2. Env ``AAWAZZ_REMOTE_URL`` (joint base, sets ``mode=remote``).
3. Per-service env ``AAWAZZ_MOUTH_URL`` / ``AAWAZZ_EARS_URL`` (overrides any
   joint base; sets ``mode=remote`` if **either** is present).
4. Default: ``mode=local``, both remote URLs ``None``.

**Split mode.** When only one of ``AAWAZZ_MOUTH_URL`` / ``AAWAZZ_EARS_URL`` is
set, that side goes remote and the other side falls back to the local backend.
``mode`` is reported as ``"remote"`` because *some* tool will route through
httpx; the dispatcher per-tool routing handles the asymmetry. ``cfg.summary()``
reports the split explicitly so Wave 2's startup banner can surface it.

URL normalisation lives in :func:`_normalise_mouth_url` /
:func:`_normalise_ears_url`: a bare ``http://host:port`` gets the canonical
endpoint suffix appended, while a URL that already includes the path is left
alone. This mirrors the existing Rust arm in ``joker-mcp::modalities`` so the
two backends are wire-compatible against the same FastAPI servers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


BackendMode = Literal["local", "remote"]


def _strip(value: str | None) -> str | None:
    """Normalise empty / whitespace-only strings to ``None``."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _normalise_mouth_url(url: str | None) -> str | None:
    """Append ``/tts`` if the URL is a bare host:port (no path component)."""
    url = _strip(url)
    if url is None:
        return None
    # Trailing slash makes the path-suffix check below misfire.
    url = url.rstrip("/")
    # Strip scheme to inspect the path portion.
    after_scheme = url.split("://", 1)[-1]
    if "/" in after_scheme:
        return url  # caller already supplied a path (e.g. /tts).
    return f"{url}/tts"


def _normalise_ears_url(url: str | None) -> str | None:
    """Append ``/transcribe`` if the URL is a bare host:port (no path component)."""
    url = _strip(url)
    if url is None:
        return None
    url = url.rstrip("/")
    after_scheme = url.split("://", 1)[-1]
    if "/" in after_scheme:
        return url
    return f"{url}/transcribe"


def _split_remote_flag(value: str | None) -> tuple[str | None, str | None]:
    """Parse ``--remote`` arg / ``AAWAZZ_REMOTE_URL`` into (mouth_base, ears_base).

    Comma-separated form: ``http://mouth,http://ears``. Single URL = joint.
    """
    value = _strip(value)
    if value is None:
        return (None, None)
    if "," in value:
        parts = [p.strip() for p in value.split(",", 1)]
        return (parts[0] or None, parts[1] or None)
    return (value, value)


@dataclass(frozen=True)
class AawazzConfig:
    """Resolved configuration. Construct via :meth:`from_args` or :meth:`from_env`."""

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

    # ----- Resolution helpers ------------------------------------------------

    @classmethod
    def _resolve(
        cls,
        *,
        cli_remote: str | None,
        env: dict[str, str] | None = None,
        transport: Literal["stdio", "streamable-http"] = "stdio",
        host: str = "127.0.0.1",
        port: int = 7860,
        warm: bool = False,
    ) -> "AawazzConfig":
        """Apply the four-tier resolution order. Used by both classmethods."""
        env = env if env is not None else os.environ

        # Tier 1: CLI --remote.
        cli_mouth, cli_ears = _split_remote_flag(cli_remote)

        # Tier 2: AAWAZZ_REMOTE_URL (joint base).
        env_joint_mouth, env_joint_ears = _split_remote_flag(env.get("AAWAZZ_REMOTE_URL"))

        # Tier 3: per-service env overrides joint env.
        env_mouth = _strip(env.get("AAWAZZ_MOUTH_URL"))
        env_ears = _strip(env.get("AAWAZZ_EARS_URL"))

        # CLI wins over env entirely. Within env, per-service overrides joint.
        if cli_mouth is not None or cli_ears is not None:
            mouth = cli_mouth
            ears = cli_ears
        else:
            mouth = env_mouth if env_mouth is not None else env_joint_mouth
            ears = env_ears if env_ears is not None else env_joint_ears

        mouth_url = _normalise_mouth_url(mouth)
        ears_url = _normalise_ears_url(ears)

        mode: BackendMode = "remote" if (mouth_url or ears_url) else "local"

        return cls(
            mode=mode,
            remote_mouth_url=mouth_url,
            remote_ears_url=ears_url,
            transport=transport,
            host=host,
            port=port,
            warm=warm,
        )

    @classmethod
    def from_args(cls, args) -> "AawazzConfig":  # noqa: ANN001 — argparse.Namespace
        """Build config from an ``argparse.Namespace`` (CLI > env > default).

        Recognised attributes (all optional): ``remote``, ``transport``, ``host``,
        ``port``, ``warm``. Missing attributes fall back to the dataclass default.
        """
        cli_remote = getattr(args, "remote", None)
        return cls._resolve(
            cli_remote=cli_remote,
            transport=getattr(args, "transport", "stdio"),
            host=getattr(args, "host", "127.0.0.1"),
            port=getattr(args, "port", 7860),
            warm=bool(getattr(args, "warm", False)),
        )

    @classmethod
    def from_env(cls) -> "AawazzConfig":
        """Pure-env resolution (used by tests + Wave 2 lifespan setup)."""
        return cls._resolve(cli_remote=None)

    # ----- Diagnostics -------------------------------------------------------

    def summary(self) -> str:
        """One-line human summary; Wave 2 uses this in the startup banner."""
        if self.mode == "local":
            return "aawazz-mcp mode=local (bundled tiny-tts + moonshine)"
        # Remote — note split if only one side is set.
        mouth = self.remote_mouth_url or "<local fallback>"
        ears = self.remote_ears_url or "<local fallback>"
        if self.remote_mouth_url and self.remote_ears_url:
            return f"aawazz-mcp mode=remote mouth={mouth} ears={ears}"
        return (
            f"aawazz-mcp mode=remote (split) mouth={mouth} ears={ears} "
            "(unset side served by bundled backend)"
        )
