"""CLI entry point for aawazz-mcp.

Parses argv into an :class:`aawazz_mcp.config.AawazzConfig`, builds the FastMCP
server via :func:`aawazz_mcp.server.build_server`, and runs it on the requested
transport.
"""

from __future__ import annotations

import argparse
import logging
import sys


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aawazz-mcp",
        description=(
            "Portable local-CPU TTS + STT MCP server. "
            "Bundles tiny-tts + Moonshine; optional --remote mode delegates "
            "to a separately-running aawazz-mouth/ears FastAPI pair."
        ),
    )
    p.add_argument(
        "--remote",
        metavar="MOUTH_URL[,EARS_URL]",
        help=(
            "Comma-separated URLs for an existing aawazz-mouth + aawazz-ears pair. "
            "Single URL is treated as the joint base; per-service env overrides "
            "AAWAZZ_MOUTH_URL / AAWAZZ_EARS_URL still apply."
        ),
    )
    p.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="MCP transport (default: stdio).",
    )
    p.add_argument("--host", default="127.0.0.1", help="streamable-http host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=7860, help="streamable-http port (default: 7860)")
    p.add_argument(
        "--warm",
        action="store_true",
        help="Eagerly load tiny-tts + Moonshine models at startup. Default: lazy first-call.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Log level (default: INFO).",
    )

    # ── v1.3 routing layer ────────────────────────────────────────────────
    p.add_argument(
        "--routing-config",
        metavar="FILE",
        default=None,
        help=(
            "TOML file with [tts.routing] / [stt.routing] tables. Defaults to "
            "$AAWAZZ_ROUTING_FILE or ~/.config/aawazz/aawazz.toml."
        ),
    )
    p.add_argument(
        "--tts-default",
        metavar="NAME",
        default=None,
        help=(
            "Provider name for the default TTS chain (overrides config + env). "
            "Built-in: gtts. Per-call ``tts_provider=`` still wins."
        ),
    )
    p.add_argument(
        "--stt-default",
        metavar="NAME",
        default=None,
        help=(
            "Provider name for the default STT chain (overrides config + env). "
            "Built-in: moonshine. Per-call ``stt_provider=`` still wins."
        ),
    )
    p.add_argument(
        "--llm-default",
        metavar="NAME",
        default=None,
        help=(
            "Provider name for the default LLM chain (overrides config + env). "
            "Built-in: pipefish. Per-call ``llm_provider=`` still wins. "
            "Only meaningful when the [llm] extra is installed."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Parse args → AawazzConfig → build_server(cfg) → mcp.run()."""
    args = _build_parser().parse_args(argv)
    # NEVER log to stdout under stdio transport — corrupts the MCP frame stream.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s aawazz-mcp %(message)s",
    )

    # Imported lazily so `--help` doesn't drag in the FastMCP / Dispatcher tree.
    from aawazz_mcp.config import AawazzConfig  # noqa: PLC0415
    from aawazz_mcp.server import build_server  # noqa: PLC0415

    cfg = AawazzConfig.from_args(args)
    log = logging.getLogger("aawazz-mcp")
    log.info("%s transport=%s", cfg.summary(), cfg.transport)

    mcp = build_server(cfg)

    if cfg.transport == "streamable-http":
        # FastMCP reads host/port from settings, not run() kwargs (mcp 1.24+).
        mcp.settings.host = cfg.host
        mcp.settings.port = cfg.port
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
