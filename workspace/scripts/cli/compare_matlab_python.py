#!/usr/bin/env python3
"""Backward-compatible wrapper for the packaged parity CLI."""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root / "source"))


def main() -> int:
    from slavv.apps.parity_cli import main as packaged_main

    return int(packaged_main())


if __name__ == "__main__":
    raise SystemExit(main())
