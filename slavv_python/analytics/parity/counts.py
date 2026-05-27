"""Canonical run-count helpers for exact parity reports and checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.engine.state import load_json_dict
from slavv_python.schema.results import EdgeSet, NetworkResult, VertexSet
from slavv_python.utils.safe_unpickle import safe_load

from .constants import CHECKPOINTS_DIR, RUN_SNAPSHOT_PATH
from .models import RunCounts

if TYPE_CHECKING:
    from pathlib import Path


def extract_matlab_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract MATLAB-side counts from a comparison or proof report."""
    if "matlab" in report_payload:
        matlab = report_payload.get("matlab", {})
        return RunCounts(
            vertices=int(matlab.get("vertices_count", 0)),
            edges=int(matlab.get("edges_count", 0)),
            strands=int(matlab.get("strand_count", 0)),
        )
    counts = report_payload.get("matlab_counts", {})
    return RunCounts(
        vertices=int(counts.get("vertices", 0)),
        edges=int(counts.get("edges", 0)),
        strands=int(counts.get("strands", 0)),
    )


def extract_source_python_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract Python-side counts from a comparison or proof report."""
    if "python" in report_payload:
        python = report_payload.get("python", {})
        return RunCounts(
            vertices=int(python.get("vertices_count", 0)),
            edges=int(python.get("edges_count", 0)),
            strands=int(python.get("network_strands_count", python.get("strand_count", 0))),
        )
    counts = report_payload.get("python_counts", {})
    return RunCounts(
        vertices=int(counts.get("vertices", 0)),
        edges=int(counts.get("edges", 0)),
        strands=int(counts.get("strands", 0)),
    )


def read_python_counts_from_run(run_root: Path) -> RunCounts:
    """Read vertex, edge, and strand counts from a processed run directory."""
    snapshot = load_json_dict(run_root / RUN_SNAPSHOT_PATH)
    if snapshot:
        counts = snapshot.get("counts", {})
        if counts:
            return RunCounts(
                vertices=int(counts.get("vertices", 0)),
                edges=int(counts.get("edges", 0)),
                strands=int(counts.get("strands", 0)),
            )

    checkpoints_dir = run_root / CHECKPOINTS_DIR
    return RunCounts(
        vertices=_count_vertices(checkpoints_dir / "checkpoint_vertices.pkl"),
        edges=_count_edges(checkpoints_dir / "checkpoint_edges.pkl"),
        strands=_count_strands(checkpoints_dir / "checkpoint_network.pkl"),
    )


def _count_vertices(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(VertexSet.load(path).positions)
    except Exception:
        payload = safe_load(path)
        if isinstance(payload, dict):
            return len(payload.get("positions", []))
        return len(getattr(payload, "positions", []))


def _count_edges(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(EdgeSet.load(path).connections)
    except Exception:
        payload = safe_load(path)
        if isinstance(payload, dict):
            return len(payload.get("connections", []))
        return len(getattr(payload, "connections", []))


def _count_strands(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        return len(NetworkResult.load(path).strands)
    except Exception:
        payload = safe_load(path)
        if isinstance(payload, dict):
            return len(payload.get("strands", []))
        return len(getattr(payload, "strands", []))
