"""CLI entrypoint for the grouped SLAVV CLI package."""

from __future__ import annotations

from .dispatch import dispatch_cli_command
from .parser import _build_cli_parser

__all__ = ["main"]


def main(argv=None):
    """CLI entrypoint."""
    parser = _build_cli_parser()
    args = parser.parse_args(argv)

    if args.version:
        from slavv_python import __version__

        print(f"slavv_python {__version__}")
        return

    dispatch_cli_command(parser, args)


if __name__ == "__main__":
    main()
