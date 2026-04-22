"""Post-choice MATLAB edge math helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def _matlab_edge_endpoint_energy(edge_energy_trace: np.ndarray) -> float:
    """Return MATLAB's endpoint energy normalizer for one chosen edge."""
    trace = np.asarray(edge_energy_trace, dtype=np.float32).reshape(-1)
    if trace.size == 0:
        return -1.0
    endpoint_product = float(trace[0]) * float(trace[-1])
    endpoint_magnitude = float(np.sqrt(max(endpoint_product, 0.0)))
    if endpoint_magnitude <= 1e-12:
        return -1.0
    return -endpoint_magnitude


def normalize_edges_matlab_style(chosen_edges: dict[str, Any]) -> dict[str, Any]:
    """Apply MATLAB's post-choice edge-energy normalization formulas."""
    energy_traces = [
        np.asarray(trace, dtype=np.float32).copy()
        for trace in chosen_edges.get("energy_traces", [])
    ]
    if not energy_traces:
        return chosen_edges

    raw_energies = np.asarray(
        chosen_edges.get("energies", np.zeros((0,), dtype=np.float32)),
        dtype=np.float32,
    ).copy()
    endpoint_energies = np.asarray(
        [_matlab_edge_endpoint_energy(trace) for trace in energy_traces],
        dtype=np.float32,
    )
    safe_endpoint_energies = endpoint_energies.copy()
    safe_endpoint_energies[np.abs(safe_endpoint_energies) < 1e-12] = -1.0

    normalized_energy_traces = [
        (-trace / safe_endpoint_energies[index]).astype(np.float32, copy=False)
        for index, trace in enumerate(energy_traces)
    ]
    normalized_energies = (
        -raw_energies / safe_endpoint_energies[: len(raw_energies)]
    ).astype(np.float32, copy=False)

    chosen_edges["raw_energies"] = raw_energies
    chosen_edges["raw_energy_traces"] = energy_traces
    chosen_edges["edge_endpoint_energies"] = endpoint_energies
    chosen_edges["energies"] = normalized_energies
    chosen_edges["energy_traces"] = normalized_energy_traces
    return cast("dict[str, Any]", chosen_edges)
