"""Skimage label-adjacency helpers for the non-certification watershed path.

Do not confuse with Watershed Discovery (Exact Route MATLAB global watershed).
See ``discovery.WatershedDiscovery`` for the Certification Edge Discovery strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed


@dataclass
class NaiveWatershedLabelUnit:
    """Edges discovered for one watershed label and its higher-index neighbors."""

    origin_index: int
    traces: list[np.ndarray] = field(default_factory=list)
    connections: list[list[int]] = field(default_factory=list)
    metrics: list[float] = field(default_factory=list)
    connection_source: str = "fallback"

    def to_unit_payload(self) -> dict[str, Any]:
        """Serialize to the resumable per-label checkpoint shape."""
        return {
            "origin_index": self.origin_index,
            "candidate_source": self.connection_source,
            "traces": self.traces,
            "connections": self.connections,
            "metrics": self.metrics,
            "energy_traces": [
                np.asarray([energy_value], dtype=np.float64) for energy_value in self.metrics
            ],
            "scale_traces": [np.zeros((len(trace),), dtype=np.int16) for trace in self.traces],
            "origin_indices": [self.origin_index] * len(self.traces),
            "connection_sources": [self.connection_source] * len(self.traces),
        }


def paint_vertex_watershed_markers(
    vertex_positions: np.ndarray,
    energy_shape: tuple[int, ...],
) -> np.ndarray:
    """Paint 1-based watershed markers at floored vertex voxel coordinates."""
    markers: np.ndarray = np.zeros(energy_shape, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(energy_shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)
    return markers


def run_skimage_watershed_labels(
    energy: np.ndarray,
    markers: np.ndarray,
    *,
    energy_sign: float,
) -> np.ndarray:
    """Run skimage watershed and return integer region labels."""
    return cast("np.ndarray", np.asarray(watershed(-energy_sign * energy, markers), dtype=np.int32))


def collect_naive_watershed_label_unit(
    label: int,
    labels: np.ndarray,
    energy: np.ndarray,
    structure: np.ndarray,
    seen_pairs: set[tuple[int, int]],
    *,
    coord_dtype: type = np.float64,
) -> NaiveWatershedLabelUnit:
    """Collect boundary traces for one label against higher-index neighbors."""
    unit = NaiveWatershedLabelUnit(origin_index=label - 1)
    region = labels == label
    dilated = ndi.binary_dilation(region, structure)
    neighbors = np.unique(labels[dilated & (labels != label)])

    for neighbor in neighbors:
        if neighbor <= label or neighbor == 0:
            continue
        pair = (label - 1, neighbor - 1)
        if pair in seen_pairs:
            continue
        boundary = (ndi.binary_dilation(labels == neighbor, structure) & region) | (
            ndi.binary_dilation(region, structure) & (labels == neighbor)
        )
        coords = np.argwhere(boundary)
        if coords.size == 0:
            continue
        coords = coords.astype(coord_dtype)
        idx = np.floor(coords).astype(int)
        energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
        unit.traces.append(coords)
        unit.connections.append([label - 1, neighbor - 1])
        unit.metrics.append(float(np.mean(energies)))
        seen_pairs.add(pair)

    return unit


__all__ = [
    "NaiveWatershedLabelUnit",
    "collect_naive_watershed_label_unit",
    "paint_vertex_watershed_markers",
    "run_skimage_watershed_labels",
]
