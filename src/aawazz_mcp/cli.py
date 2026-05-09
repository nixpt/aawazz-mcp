"""CLI entry point for aawazz-mcp.

Wave 0: argparse skeleton with the flags Wave 1B/2 will read from
:class:`aawazz_mcp.config.AawazzConfig`.

Wave 2: full implementation — build the FastMCP server via
:func:`aawazz_mcp.server.build_server` and run it.
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
    return p


def main(argv: list[str] | None = None) -> int:
    """Wave 2: parse args → AawazzConfig → build_server(cfg) → mcp.run()."""
    args = _build_parser().parse_args(argv)
    # NEVER log to stdout under stdio transport — corrupts the MCP frame stream.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s aawazz-mcp %(message)s",
    )
    raise NotImplementedError(
        "Wave 2: wire AawazzConfig.from_args(args) → build_server(cfg) → mcp.run(transport=args.transport)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
