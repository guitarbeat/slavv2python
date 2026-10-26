"""Unit tests for Phase 2 profiling and benchmark CLI scripts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_profile_edges_detailed_help():
    """Verify profile_edges_detailed.py CLI help exits cleanly."""
    script = REPO_ROOT / "scripts" / "profiling" / "profile_edges_detailed.py"
    res = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "Detailed Phase 2 Performance Profiling for Edges" in res.stdout
    assert "--run-dir" in res.stdout
    assert "--out-dir" in res.stdout


def test_benchmark_full_edges_help():
    """Verify benchmark_full_edges.py CLI help exits cleanly."""
    script = REPO_ROOT / "scripts" / "profiling" / "benchmark_full_edges.py"
    res = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0
    assert "Full-Volume Phase 2 Benchmark for Edges Stage" in res.stdout
    assert "--run-dir" in res.stdout
    assert "--out-dir" in res.stdout
