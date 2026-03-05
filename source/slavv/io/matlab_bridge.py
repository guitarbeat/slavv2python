"""
MATLAB ↔ Python Bridge for SLAVV.

Converts MATLAB batch_* output folders into Python checkpoint pickles so
the Python pipeline can resume from MATLAB-curated vertices, edges, or
network data.

Usage:
    from slavv.io.matlab_bridge import import_matlab_batch

    # Point at a MATLAB batch folder and a Python checkpoint directory
    import_matlab_batch("path/to/batch_260210-101213", "my_checkpoints/")

    # Now run the pipeline — it will skip any step that has a checkpoint
    from slavv import SLAVVProcessor
    processor = SLAVVProcessor()
    results = processor.process_image(image, params, checkpoint_dir="my_checkpoints/")
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np

from slavv.io.matlab_parser import (
    extract_edges,
    extract_network_data,
    extract_vertices,
    find_batch_folder,
    load_mat_file_safe,
)

logger = logging.getLogger(__name__)


def _mat_vertices_to_python(mat_vertices: dict[str, Any]) -> dict[str, Any]:
    """Convert matlab_parser vertex dict to pipeline-compatible dict."""
    positions = mat_vertices.get("positions", np.array([]))
    radii = mat_vertices.get("radii", np.array([]))

    # Ensure positions is a list-of-lists (pipeline convention)
    if isinstance(positions, np.ndarray) and positions.ndim == 2:
        positions = positions.tolist()
    elif isinstance(positions, np.ndarray):
        positions = []

    if isinstance(radii, np.ndarray) and radii.ndim >= 1:
        radii = radii.flatten().tolist()
    else:
        radii = []

    return {
        "positions": positions,
        "radii": radii,
        "count": len(positions),
    }


def _mat_edges_to_python(mat_edges: dict[str, Any]) -> dict[str, Any]:
    """Convert matlab_parser edge dict to pipeline-compatible dict."""
    connections = mat_edges.get("connections", np.array([]))
    traces_raw = mat_edges.get("traces", [])

    # connections: Nx2 array of vertex index pairs (already 0-based from parser)
    if isinstance(connections, np.ndarray) and connections.ndim == 2 and connections.shape[1] == 2:
        origin_indices = connections[:, 0].tolist()
        terminal_indices = connections[:, 1].tolist()
    else:
        origin_indices = []
        terminal_indices = []

    # traces: list of Nx3 arrays → list of lists
    traces = []
    for t in traces_raw:
        if isinstance(t, np.ndarray) and t.size > 0:
            traces.append(t.tolist())
        elif isinstance(t, list):
            traces.append(t)

    return {
        "traces": traces,
        "origin_indices": origin_indices,
        "terminal_indices": terminal_indices,
        "count": max(len(traces), len(origin_indices)),
        "total_length": mat_edges.get("total_length", 0.0),
    }


def import_matlab_batch(
    batch_folder: Union[str, Path],
    checkpoint_dir: Union[str, Path],
    *,
    stages: Optional[list] = None,
) -> dict[str, str]:
    """
    Import MATLAB batch output and write Python-compatible checkpoint files.

    Parameters
    ----------
    batch_folder : str | Path
        Path to a MATLAB ``batch_*`` folder (or a parent dir containing one).
    checkpoint_dir : str | Path
        Directory where Python checkpoint pickles will be written.
    stages : list, optional
        Which stages to import.  Defaults to all available:
        ``['energy', 'vertices', 'edges', 'network']``.

    Returns
    -------
    dict
        Mapping of stage name → checkpoint file path that was written.
    """
    batch_path = Path(batch_folder)

    # If user pointed at a parent directory, auto-discover batch subfolder
    if not (batch_path / "vectors").exists():
        found = find_batch_folder(batch_path)
        if found is None:
            raise FileNotFoundError(f"No MATLAB batch_* folder found in {batch_folder}")
        batch_path = found

    vectors_dir = batch_path / "vectors"
    data_dir = batch_path / "data"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if stages is None:
        stages = ["energy", "vertices", "edges", "network"]

    written = {}

    # ── Energy ────────────────────────────────────────────────────────────
    if "energy" in stages:
        energy_files = sorted(data_dir.glob("energy*.mat")) if data_dir.exists() else []
        if not energy_files:
            energy_files = sorted(vectors_dir.glob("energy*.mat"))
        if energy_files:
            mat = load_mat_file_safe(energy_files[-1])
            if mat:
                # The energy image is typically stored as a 3D array
                for key in ("energy", "energy_data", "energy_image"):
                    if key in mat and isinstance(mat[key], np.ndarray):
                        out = os.path.join(checkpoint_dir, "checkpoint_energy.pkl")
                        joblib.dump({"energy_image": mat[key]}, out)
                        written["energy"] = out
                        logger.info("Wrote energy checkpoint: %s", out)
                        break

    # ── Vertices ──────────────────────────────────────────────────────────
    if "vertices" in stages:
        v_files = sorted(vectors_dir.glob("vertices_*.mat"))
        if v_files:
            mat = load_mat_file_safe(v_files[-1])
            if mat:
                raw = extract_vertices(mat)
                py_verts = _mat_vertices_to_python(raw)
                out = os.path.join(checkpoint_dir, "checkpoint_vertices.pkl")
                joblib.dump(py_verts, out)
                written["vertices"] = out
                logger.info("Wrote vertices checkpoint (%d verts): %s", py_verts["count"], out)

    # ── Edges ─────────────────────────────────────────────────────────────
    if "edges" in stages:
        e_files = sorted(vectors_dir.glob("edges_*.mat"))
        if e_files:
            mat = load_mat_file_safe(e_files[-1])
            if mat:
                raw = extract_edges(mat)
                py_edges = _mat_edges_to_python(raw)
                out = os.path.join(checkpoint_dir, "checkpoint_edges.pkl")
                joblib.dump(py_edges, out)
                written["edges"] = out
                logger.info("Wrote edges checkpoint (%d edges): %s", py_edges["count"], out)

    # ── Network ───────────────────────────────────────────────────────────
    if "network" in stages:
        n_files = sorted(vectors_dir.glob("network_*.mat"))
        if n_files:
            mat = load_mat_file_safe(n_files[-1])
            if mat:
                net = extract_network_data(mat)
                out = os.path.join(checkpoint_dir, "checkpoint_network.pkl")
                joblib.dump(net, out)
                written["network"] = out
                logger.info("Wrote network checkpoint: %s", out)

    if not written:
        logger.warning("No MATLAB data files found in %s", batch_path)
    else:
        logger.info(
            "Imported %d MATLAB stage(s) into %s: %s",
            len(written),
            checkpoint_dir,
            list(written.keys()),
        )

    return written
