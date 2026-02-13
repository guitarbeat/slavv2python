"""Pytest configuration and fixtures.

Ensures the slavv package is importable when running tests from the repo root,
whether or not `pip install -e .` has been run.
"""
import sys
from pathlib import Path

# Add source/ to path so slavv is importable
repo_root = Path(__file__).resolve().parent.parent
source_dir = repo_root / "source"
if source_dir.exists() and str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))
