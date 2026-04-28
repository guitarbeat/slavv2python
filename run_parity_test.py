#!/usr/bin/env python
"""Wrapper to run parity experiment with proper imports."""
import sys
from pathlib import Path

# Add source to path
REPO_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = REPO_ROOT / "source"
sys.path.insert(0, str(SOURCE_DIR))

# Now run the parity experiment
exec(open("dev/scripts/cli/parity_experiment.py").read())
