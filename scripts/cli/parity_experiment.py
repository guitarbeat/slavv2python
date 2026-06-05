"""Developer CLI wrapper for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from slavv_python.analytics.parity.commands import build_parity_parser  # noqa: E402

if TYPE_CHECKING:
    import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the developer parity experiment parser."""
    return build_parity_parser()


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "handler"):
        args.handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
