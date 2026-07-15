#!/usr/bin/env python3
"""Deprecated entry point — use generate_parity_claim_figures.py.

Kept so older docs/commands still regenerate the four claim-driven figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from generate_parity_claim_figures import main  # noqa: E402

if __name__ == "__main__":
    paths = main()
    print(f"Generated {len(paths)} claim-driven figures (via legacy entry point).")
