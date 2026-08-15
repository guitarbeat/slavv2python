"""Strict MATLAB↔Python spatial compare helpers for the synthetic complexity ladder.

Mirrors quantization / 0- vs 1-based handling from the tiny dual-run experiment, but
exposes a *strict* first-break surface (vertices → edges → strands) for ladder stop.
Graded ``first_big_break`` labels from the tiny script are not used as the stop predicate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np

FirstBreakSurface = Literal["vertices", "edges", "strands"]


@dataclass(frozen=True)
class PairStats:
    n_left: int
    n_right: int
    n_intersection: int
    n_only_left: int
    n_only_right: int
    overlap_pct_of_union: float
    overlap_pct_of_left: float
    overlap_pct_of_right: float


class NonComparableArtifactsError(ValueError):
    """Raised when artifact dicts lack fields required for a strict compare."""


def undirected_pairs(connections: np.ndarray) -> set[tuple[int, int]]:
    arr = np.asarray(connections, dtype=np.int64)
    if arr.size == 0:
        return set()
    arr = arr.reshape(-1, 2)
    out: set[tuple[int, int]] = set()
    for a, b in arr:
        if a == b:
            continue
        out.add((int(min(a, b)), int(max(a, b))))
    return out


def pair_stats(left: set[tuple[int, int]], right: set[tuple[int, int]]) -> PairStats:
    inter = left & right
    only_l = left - right
    only_r = right - left
    union = left | right
    n_union = max(len(union), 1)
    return PairStats(
        n_left=len(left),
        n_right=len(right),
        n_intersection=len(inter),
        n_only_left=len(only_l),
        n_only_right=len(only_r),
        overlap_pct_of_union=100.0 * len(inter) / n_union,
        overlap_pct_of_left=100.0 * len(inter) / max(len(left), 1),
        overlap_pct_of_right=100.0 * len(inter) / max(len(right), 1),
    )


def quantize_positions(positions: np.ndarray) -> np.ndarray:
    """Round spatial positions to integer voxel keys for cross-runtime matching."""
    pos = np.asarray(positions, dtype=np.float64)
    if pos.size == 0:
        return np.zeros((0, 3), dtype=np.int64)
    pos = pos.reshape(-1, 3)
    return np.rint(pos).astype(np.int64)


def position_keys_one_based(
    positions: np.ndarray,
    *,
    positions_are_one_based: bool,
) -> list[tuple[int, int, int]]:
    q = quantize_positions(positions)
    if not positions_are_one_based:
        q = q + 1
    return [(int(r[0]), int(r[1]), int(r[2])) for r in q]


def spatial_pair_set(
    positions: np.ndarray,
    connections: np.ndarray,
    *,
    positions_are_one_based: bool,
    connection_indices_one_based: bool,
) -> set[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    keys = position_keys_one_based(positions, positions_are_one_based=positions_are_one_based)
    pairs = undirected_pairs(connections)
    spatial: set[tuple[tuple[int, int, int], tuple[int, int, int]]] = set()
    n = len(keys)
    for a, b in pairs:
        ia = a - 1 if connection_indices_one_based else a
        ib = b - 1 if connection_indices_one_based else b
        if ia < 0 or ib < 0 or ia >= n or ib >= n:
            continue
        ka, kb = keys[ia], keys[ib]
        spatial.add((ka, kb) if ka <= kb else (kb, ka))
    return spatial


def _require_side(side: dict[str, Any], label: str) -> None:
    if not isinstance(side, dict):
        raise NonComparableArtifactsError(f"{label} artifacts must be a dict")
    if not side.get("ok", True):
        raise NonComparableArtifactsError(f"{label} artifacts not ok")
    for key in ("positions", "connections", "n_strands"):
        if key not in side:
            raise NonComparableArtifactsError(f"{label} missing required field {key!r}")


def first_break_surface(
    matlab: dict[str, Any],
    python: dict[str, Any],
    *,
    matlab_positions_one_based: bool = True,
    matlab_connections_one_based: bool = True,
    python_positions_one_based: bool = False,
) -> FirstBreakSurface | None:
    """Return the first strict mismatch surface, or None on full match.

    Order: curated vertex spatial keys → spatial undirected edge pairs → strand counts.
    Empty-both-sides vertex/edge sets still compare as equal sets (match).
    """
    _require_side(matlab, "matlab")
    _require_side(python, "python")

    vertex_keys_m = set(
        position_keys_one_based(
            matlab["positions"], positions_are_one_based=matlab_positions_one_based
        )
    )
    vertex_keys_p = set(
        position_keys_one_based(
            python["positions"], positions_are_one_based=python_positions_one_based
        )
    )
    if vertex_keys_m != vertex_keys_p:
        return "vertices"

    p_conn = np.asarray(python["connections"], dtype=np.int64)
    py_conn_one_based = bool(p_conn.size) and int(p_conn.min()) >= 1

    m_spatial = spatial_pair_set(
        matlab["positions"],
        matlab["connections"],
        positions_are_one_based=matlab_positions_one_based,
        connection_indices_one_based=matlab_connections_one_based,
    )
    p_spatial = spatial_pair_set(
        python["positions"],
        python["connections"],
        positions_are_one_based=python_positions_one_based,
        connection_indices_one_based=py_conn_one_based,
    )
    if m_spatial != p_spatial:
        return "edges"

    if matlab.get("n_strands") != python.get("n_strands"):
        return "strands"
    return None


def strict_compare_summary(
    matlab: dict[str, Any],
    python: dict[str, Any],
) -> dict[str, Any]:
    """Build a ladder-oriented compare summary with strict first-break surface."""
    try:
        surface = first_break_surface(matlab, python)
    except NonComparableArtifactsError as exc:
        return {
            "comparable": False,
            "reason": str(exc),
            "first_break_surface": None,
        }

    p_conn = np.asarray(python["connections"], dtype=np.int64)
    py_conn_one_based = bool(p_conn.size) and int(p_conn.min()) >= 1
    m_spatial = spatial_pair_set(
        matlab["positions"],
        matlab["connections"],
        positions_are_one_based=True,
        connection_indices_one_based=True,
    )
    p_spatial = spatial_pair_set(
        python["positions"],
        python["connections"],
        positions_are_one_based=False,
        connection_indices_one_based=py_conn_one_based,
    )
    vertex_keys_m = set(position_keys_one_based(matlab["positions"], positions_are_one_based=True))
    vertex_keys_p = set(position_keys_one_based(python["positions"], positions_are_one_based=False))
    spatial = pair_stats(m_spatial, p_spatial)

    def _n_rows(arr: Any, width: int) -> int:
        a = np.asarray(arr)
        if a.size == 0:
            return 0
        return int(a.reshape(-1, width).shape[0])

    return {
        "comparable": True,
        "first_break_surface": surface,
        "counts": {
            "matlab_vertices": _n_rows(matlab["positions"], 3),
            "python_vertices": _n_rows(python["positions"], 3),
            "matlab_edges": _n_rows(matlab["connections"], 2),
            "python_edges": _n_rows(python["connections"], 2),
            "matlab_strands": matlab.get("n_strands"),
            "python_strands": python.get("n_strands"),
        },
        "vertex_spatial_overlap": {
            "n_matlab": len(vertex_keys_m),
            "n_python": len(vertex_keys_p),
            "n_intersection": len(vertex_keys_m & vertex_keys_p),
            "n_only_matlab": len(vertex_keys_m - vertex_keys_p),
            "n_only_python": len(vertex_keys_p - vertex_keys_m),
        },
        "spatial_pair_overlap": asdict(spatial),
        "match": surface is None,
    }
