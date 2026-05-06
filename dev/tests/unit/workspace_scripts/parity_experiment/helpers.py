"""Helper functions for the developer parity experiment runner tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from dev.tests.support.run_state_builders import (
    materialize_checkpoint_surface,
)
from scipy.io import savemat

if TYPE_CHECKING:
    from pathlib import Path


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_experiment_root(tmp_path: Path) -> Path:
    root = tmp_path / "live-parity"
    for name in ("datasets", "oracles", "reports", "runs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _build_source_run_root(tmp_path: Path) -> Path:
    run_root = tmp_path / "source-run"
    materialize_checkpoint_surface(
        run_root,
        stages=("energy", "vertices", "edges", "network"),
    )
    _write_json(
        run_root / "03_Analysis" / "comparison_report.json",
        {
            "matlab": {
                "vertices_count": 4,
                "edges_count": 5,
                "strand_count": 3,
            },
            "python": {
                "vertices_count": 4,
                "edges_count": 2,
                "network_strands_count": 1,
            },
            "vertices": {
                "matlab_count": 4,
                "python_count": 4,
            },
            "edges": {
                "matlab_count": 5,
                "python_count": 2,
            },
            "network": {
                "matlab_strand_count": 3,
                "python_strand_count": 1,
            },
        },
    )
    _write_json(
        run_root / "99_Metadata" / "validated_params.json",
        {"number_of_edges_per_vertex": 4},
    )
    return run_root


def _materialize_exact_matlab_batch(run_root: Path) -> Path:
    batch_dir = run_root / "01_Input" / "matlab_results" / "batch_260421-151654"
    data_dir = batch_dir / "data"
    vectors_dir = batch_dir / "vectors"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        data_dir / "energy_260421.mat",
        {
            "energy": np.zeros((2, 2, 2), dtype=np.float64),
            "scale_indices": np.ones((2, 2, 2), dtype=np.int16),
            "energy_4d": np.zeros((2, 2, 2, 1), dtype=np.float64),
            "lumen_radius_microns": np.array([1.0], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "vertices_260421.mat",
        {
            "vertex_space_subscripts": [[1.0, 2.0, 3.0]],
            "vertex_scale_subscripts": [2],
            "vertex_energies": [-1.0],
        },
    )
    savemat(
        vectors_dir / "edges_260421.mat",
        {
            "edges2vertices": [[1, 1]],
            "edge_space_subscripts": [],
            "edge_scale_subscripts": [],
            "edge_energies": [],
            "mean_edge_energies": [],
        },
    )
    savemat(
        vectors_dir / "network_260421.mat",
        {
            "strands2vertices": [],
            "bifurcation_vertices": [],
            "strand_subscripts": [],
            "strand_energies": [],
            "mean_strand_energies": [],
            "vessel_directions": [],
        },
    )
    return batch_dir
