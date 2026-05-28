"""Load and normalize Python exact-route checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np

from slavv_python.engine.state import atomic_joblib_dump
from slavv_python.utils.safe_unpickle import safe_load

from slavv_python.analytics.parity.array_normalization import (
    _normalize_connection_array,
    _normalize_float_array,
    _normalize_float_matrix,
    _normalize_float_matrix_list,
    _normalize_float_vector,
    _normalize_float_vector_list,
    _normalize_int_array,
    _normalize_int_vector,
    _normalize_python_bridge_payload,
    _normalize_python_strands,
    _normalize_spatial_matrix_list,
)
from slavv_python.analytics.parity.exact_proof_contract import EXACT_STAGE_ORDER
from slavv_python.analytics.parity.matlab_vector_loader import load_normalized_matlab_vectors


def load_normalized_python_checkpoints(
    checkpoints_dir: Path,
    stages: tuple[str, ...] = EXACT_STAGE_ORDER,
) -> dict[str, dict[str, Any]]:
    """Load and normalize the requested Python checkpoint payloads."""
    normalized: dict[str, dict[str, Any]] = {}
    for stage in stages:
        checkpoint_path = checkpoints_dir / f"checkpoint_{stage}.pkl"
        if not checkpoint_path.is_file():
            raise ValueError(f"missing Python checkpoint for exact proof: {checkpoint_path}")
        payload = safe_load(checkpoint_path)
        if not isinstance(payload, dict):
            raise ValueError(f"expected mapping payload in {checkpoint_path}")
        normalized[stage] = normalize_python_stage_payload(stage, cast("dict[str, Any]", payload))
    return normalized


def sync_exact_vertex_checkpoint_from_matlab(
    checkpoint_path: Path,
    batch_dir: Path,
) -> dict[str, Any]:
    """Overwrite vertex checkpoint parity fields from the canonical MATLAB vector surface."""
    payload = safe_load(checkpoint_path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected mapping payload in {checkpoint_path}")

    normalized_vertices = load_normalized_matlab_vectors(batch_dir, ("vertices",))["vertices"]
    updated = dict(cast("dict[str, Any]", payload))

    updated["positions"] = np.asarray(normalized_vertices["positions"], dtype=np.float64)
    updated["scales"] = np.asarray(normalized_vertices["scales"], dtype=np.int64)
    updated["energies"] = np.asarray(normalized_vertices["energies"], dtype=np.float64)
    updated["count"] = len(updated["positions"])
    atomic_joblib_dump(updated, checkpoint_path)
    return updated


def normalize_python_stage_payload(stage: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Python checkpoint payload into the exact-proof contract."""
    if stage == "energy":
        return {
            "energy": _normalize_float_array(payload.get("energy")),
            "scale_indices": _normalize_int_array(payload.get("scale_indices")),
            "energy_4d": _normalize_float_array(payload.get("energy_4d")),
            "lumen_radius_microns": _normalize_float_vector(payload.get("lumen_radius_microns")),
        }
    if stage == "vertices":
        return {
            "positions": _normalize_float_matrix(payload.get("positions"), columns=3),
            "scales": _normalize_int_vector(payload.get("scales")),
            "energies": _normalize_float_vector(payload.get("energies")),
        }
    if stage == "edges":
        return {
            "connections": _normalize_connection_array(payload.get("connections")),
            "traces": _normalize_float_matrix_list(payload.get("traces"), columns=3),
            "scale_traces": _normalize_float_vector_list(payload.get("scale_traces")),
            "energy_traces": _normalize_float_vector_list(payload.get("energy_traces")),
            "energies": _normalize_float_vector(payload.get("energies")),
            "bridge_vertex_positions": _normalize_float_matrix(
                payload.get("bridge_vertex_positions"),
                columns=3,
            ),
            "bridge_vertex_scales": _normalize_int_vector(payload.get("bridge_vertex_scales")),
            "bridge_vertex_energies": _normalize_float_vector(
                payload.get("bridge_vertex_energies")
            ),
            "bridge_edges": _normalize_python_bridge_payload(payload.get("bridge_edges")),
        }
    if stage == "network":
        return {
            "strands": _normalize_python_strands(payload.get("strands")),
            "bifurcations": _normalize_int_vector(payload.get("bifurcations")),
            "strand_subscripts": _normalize_float_matrix_list(
                payload.get("strand_subscripts"),
                columns=4,
            ),
            "strand_energy_traces": _normalize_float_vector_list(
                payload.get("strand_energy_traces"),
            ),
            "mean_strand_energies": _normalize_float_vector(payload.get("mean_strand_energies")),
            "vessel_directions": _normalize_float_matrix_list(
                payload.get("vessel_directions"),
                columns=3,
            ),
        }
    raise ValueError(f"unsupported exact-proof stage: {stage}")


__all__ = [
    "load_normalized_python_checkpoints",
    "normalize_python_stage_payload",
    "sync_exact_vertex_checkpoint_from_matlab",
]
