"""CLI entry point for the ``slavv parity`` subcommand group."""

from __future__ import annotations

import sys

from slavv_python.analytics.parity.commands import build_parity_parser


def main(argv: list[str] | None = None) -> None:
    """Main entry point for parity CLI."""
    parser = build_parity_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()
        sys.exit(0)


__all__ = ["main"]
