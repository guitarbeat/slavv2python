"""Shared test support for parity experiment runner tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import savemat


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _build_experiment_root(tmp_path: Path) -> Path:
    root = tmp_path / "live-parity"
    for name in ("datasets", "oracles", "reports", "runs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def _cell(items: list[np.ndarray]) -> np.ndarray:
    cell = np.empty((len(items),), dtype=object)
    for index, item in enumerate(items):
        cell[index] = item
    return cell


def _materialize_exact_matlab_batch(run_root: Path) -> Path:
    batch_dir = run_root / "01_Input" / "matlab_results" / "batch_260421-151654"
    data_dir = batch_dir / "data"
    vectors_dir = batch_dir / "vectors"
    data_dir.mkdir(parents=True, exist_ok=True)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        data_dir / "energy_260421.mat",
        {
            "size_of_image": np.array([2, 2, 2], dtype=np.uint16),
            "intensity_limits": np.array([0, 1], dtype=np.uint16),
            "energy_runtime_in_seconds": np.array([1.0], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "vertices_260421.mat",
        {
            "vertex_space_subscripts": np.array(
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64
            ),
            "vertex_scale_subscripts": np.array([2, 3], dtype=np.int16),
            "vertex_energies": np.array([-2.0, -1.0], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "edges_260421.mat",
        {
            "edges2vertices": np.array([[1, 2]], dtype=np.int16),
            "edge_space_subscripts": _cell(
                [
                    np.array(
                        [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ],
                        dtype=np.float64,
                    )
                ]
            ),
            "edge_scale_subscripts": _cell([np.array([2.0, 2.5], dtype=np.float64)]),
            "edge_energies": _cell([np.array([-4.0, -3.0], dtype=np.float64)]),
            "mean_edge_energies": np.array([-3.5], dtype=np.float64),
        },
    )
    savemat(
        vectors_dir / "network_260421.mat",
        {
            "strands2vertices": np.array([[1, 2]], dtype=np.int16),
            "bifurcation_vertices": np.empty((0,), dtype=np.int16),
            "strand_subscripts": _cell(
                [
                    np.array(
                        [
                            [1.0, 2.0, 3.0, 2.0],
                            [4.0, 5.0, 6.0, 2.5],
                        ],
                        dtype=np.float64,
                    )
                ]
            ),
            "strand_energies": _cell([np.array([-4.0, -3.0], dtype=np.float64)]),
            "mean_strand_energies": np.array([-3.5], dtype=np.float64),
            "vessel_directions": _cell(
                [np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)]
            ),
        },
    )
    return batch_dir


def _exact_vertex_payload() -> dict[str, object]:
    return {
        "positions": np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
        "scales": np.array([1, 2], dtype=np.int16),
        "energies": np.array([-2.0, -1.0], dtype=np.float32),
    }


def _exact_edge_payload(*, energies: np.ndarray | None = None) -> dict[str, object]:
    return {
        "connections": np.array([[0, 1]], dtype=np.int32),
        "traces": [np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)],
        "scale_traces": [np.array([1.0, 1.5], dtype=np.float32)],
        "energy_traces": [np.array([-4.0, -3.0], dtype=np.float32)],
        "energies": np.array([-3.5], dtype=np.float32) if energies is None else energies,
        "bridge_vertex_positions": np.empty((0, 3), dtype=np.float32),
        "bridge_vertex_scales": np.empty((0,), dtype=np.int16),
        "bridge_vertex_energies": np.empty((0,), dtype=np.float32),
        "bridge_edges": {
            "connections": np.empty((0, 2), dtype=np.int32),
            "traces": [],
            "scale_traces": [],
            "energy_traces": [],
            "energies": np.empty((0,), dtype=np.float32),
        },
    }


def _exact_network_payload() -> dict[str, object]:
    return {
        "strands": [[0, 1]],
        "bifurcations": np.empty((0,), dtype=np.int32),
        "strand_subscripts": [
            np.array(
                [
                    [0.0, 1.0, 2.0, 1.0],
                    [3.0, 4.0, 5.0, 1.5],
                ],
                dtype=np.float32,
            )
        ],
        "strand_energy_traces": [np.array([-4.0, -3.0], dtype=np.float32)],
        "mean_strand_energies": np.array([-3.5], dtype=np.float32),
        "vessel_directions": [np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)],
    }


def _pass_report() -> dict[str, object]:
    return {"passed": True, "first_failure": None}


def _fail_report(field_path: str) -> dict[str, object]:
    return {
        "passed": False,
        "first_failure": {"field_path": field_path},
    }


def _exact_validated_params(**overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "comparison_exact_network": True,
        "direction_method": "hessian",
        "discrete_tracing": False,
        "edge_method": "tracing",
        "energy_method": "hessian",
        "energy_projection_mode": "matlab",
    }
    params.update(overrides)
    return params
