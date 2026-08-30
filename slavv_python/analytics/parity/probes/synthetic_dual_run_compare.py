"""Strict MATLAB↔Python spatial compare helpers for the synthetic complexity ladder.

Mirrors quantization / 0- vs 1-based handling from the tiny dual-run experiment, but
exposes a *strict* first-break surface (vertices → edges → strands) for ladder stop.
Graded ``first_big_break`` labels from the tiny script are not used as the stop predicate.

Also provides stage-localization helpers (vertices → candidates → edges → strands) for
toy-rung divergence investigation. Outcomes are NOT Certification / NOT Phase 1.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal, TypeVar, cast

import numpy as np

from slavv_python.analytics.parity.proof.array_normalization import (
    _normalize_matlab_strands,
)

FirstBreakSurface = Literal["vertices", "edges", "strands"]
LocalizeStage = Literal["vertices", "candidates", "edges", "strands"]
LOCALIZATION_NON_CERTIFICATION_NOTE = (
    "Synthetic rung localization - NOT Certification / NOT Phase 1. "
    "Do not update ONE TRUTH or claim-run roots from this report."
)


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


def count_matlab_strands2vertices(value: Any) -> int | None:
    """Count MATLAB ``strands2vertices`` rows without treating Nx2 numeric as 1.

    Aligns with proof-path ``_normalize_matlab_strands`` for numeric endpoint matrices.
    Object cells / Python sequences use element count. Missing → None.
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray) and value.dtype == object:
        return int(value.size)
    if isinstance(value, (list, tuple)):
        return len(value)
    strands = _normalize_matlab_strands(value)
    return len(strands)


def _endpoint_pair_from_flat(flat: np.ndarray) -> tuple[int, int]:
    if flat.size == 0:
        return (-1, -1)
    lo, hi = sorted((int(flat[0]), int(flat[-1])))
    return (lo, hi)


def strand_endpoint_pair_multiset(
    strands: Any,
    *,
    indices_one_based: bool,
) -> Counter[tuple[int, int]]:
    """Undirected endpoint-pair multiset (ADR 0012 sense) from strand payloads.

    MATLAB ``strands2vertices`` is typically an ``(N, 2)`` endpoint matrix (or object
    cell of endpoint rows). Python network strands are vertex-index chains; only the
    chain ends are used.
    """
    if strands is None:
        return Counter()

    pairs: list[tuple[int, int]] = []
    if isinstance(strands, np.ndarray) and strands.dtype == object:
        items: Sequence[Any] = list(strands.reshape(-1))
    elif isinstance(strands, (list, tuple)):
        items = list(strands)
    else:
        arr = np.asarray(strands)
        if arr.size == 0:
            return Counter()
        # Numeric Nx2 (or length-2 vector) → one pair per row.
        rows = np.atleast_2d(arr).reshape(-1, 2)
        items = list(rows)

    for item in items:
        flat = np.asarray(item).ravel()
        if indices_one_based and flat.size:
            flat = flat.astype(np.int64, copy=True)
            positive = flat > 0
            flat[positive] -= 1
            flat[~positive] = -1
        pairs.append(_endpoint_pair_from_flat(np.asarray(flat, dtype=np.int64)))
    return Counter(pairs)


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


TPair = TypeVar("TPair", bound=Hashable)


def pair_stats(left: set[TPair], right: set[TPair]) -> PairStats:
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
    return cast("np.ndarray", np.rint(pos).astype(np.int64))


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

    m_n = matlab.get("n_strands")
    p_n = python.get("n_strands")
    if m_n != p_n:
        return "strands"

    # When both sides expose strand payloads, also require endpoint-pair multisets.
    m_raw = matlab.get("strands2vertices", matlab.get("strands"))
    p_raw = python.get("strands")
    if m_raw is not None and p_raw is not None:
        m_pairs = strand_endpoint_pair_multiset(m_raw, indices_one_based=True)
        p_pairs = strand_endpoint_pair_multiset(p_raw, indices_one_based=False)
        if m_pairs != p_pairs:
            return "strands"
    return None


def _candidate_pair_set(side: dict[str, Any]) -> set[tuple[int, int]] | None:
    """Return undirected candidate pairs when present; None if unavailable."""
    raw = side.get("candidate_connections")
    if raw is None:
        return None
    return undirected_pairs(np.asarray(raw, dtype=np.int64))


def _ordered_stage_statuses(
    first_diff: LocalizeStage | None,
    *,
    candidates_status: Literal["unavailable", "match", "mismatch"],
) -> dict[str, str]:
    order: tuple[LocalizeStage, ...] = ("vertices", "candidates", "edges", "strands")
    out: dict[str, str] = {}
    for name in order:
        if name == "candidates":
            out[name] = candidates_status
            continue
        if first_diff is None:
            out[name] = "match"
            continue
        if name == first_diff:
            out[name] = "mismatch"
        elif order.index(name) < order.index(first_diff):
            out[name] = "match"
        else:
            out[name] = "not_reached"
    return out


def first_diff_stage(
    matlab: dict[str, Any],
    python: dict[str, Any],
    *,
    matlab_positions_one_based: bool = True,
    matlab_connections_one_based: bool = True,
    python_positions_one_based: bool = False,
) -> LocalizeStage | None:
    """First differing stage: vertices → candidates (if both) → edges → strands.

    Missing candidates on either side skips that stage (unavailable), never a
    discovery residual from Python candidates vs MATLAB finals.
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

    cand_m = _candidate_pair_set(matlab)
    cand_p = _candidate_pair_set(python)
    if cand_m is not None and cand_p is not None and cand_m != cand_p:
        return "candidates"

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

    m_n = matlab.get("n_strands")
    p_n = python.get("n_strands")
    if m_n != p_n:
        return "strands"

    m_raw = matlab.get("strands2vertices", matlab.get("strands"))
    p_raw = python.get("strands")
    if m_raw is not None and p_raw is not None:
        m_pairs = strand_endpoint_pair_multiset(m_raw, indices_one_based=True)
        p_pairs = strand_endpoint_pair_multiset(p_raw, indices_one_based=False)
        if m_pairs != p_pairs:
            return "strands"
    elif m_n is None or p_n is None:
        raise NonComparableArtifactsError("strand counts missing on one or both sides")
    return None


def localize_stage_compare(
    matlab: dict[str, Any],
    python: dict[str, Any],
    *,
    previously_reported_strand_break: bool = False,
) -> dict[str, Any]:
    """Build a durable localization payload for one dual-run rung.

    Never offers Python candidates vs MATLAB finals as a discovery verdict.
    """
    try:
        stage = first_diff_stage(matlab, python)
    except NonComparableArtifactsError as exc:
        return {
            "comparable": False,
            "reason": str(exc),
            "first_diff_stage": None,
            "outcome": "inconclusive",
            "note": LOCALIZATION_NON_CERTIFICATION_NOTE,
        }

    cand_m = _candidate_pair_set(matlab)
    cand_p = _candidate_pair_set(python)
    candidates_status: Literal["unavailable", "match", "mismatch"]
    if cand_m is None or cand_p is None:
        candidates_status = "unavailable"
    elif cand_m == cand_p:
        candidates_status = "match"
    else:
        candidates_status = "mismatch"

    m_raw = matlab.get("strands2vertices", matlab.get("strands"))
    p_raw = python.get("strands")
    m_pairs = (
        strand_endpoint_pair_multiset(m_raw, indices_one_based=True)
        if m_raw is not None
        else Counter()
    )
    p_pairs = (
        strand_endpoint_pair_multiset(p_raw, indices_one_based=False)
        if p_raw is not None
        else Counter()
    )
    inter = sum((m_pairs & p_pairs).values())
    union = sum((m_pairs | p_pairs).values()) or 1

    if stage is None:
        outcome = "measurement_fixed_match" if previously_reported_strand_break else "match"
    else:
        outcome = "first_diff"

    return {
        "comparable": True,
        "first_diff_stage": stage,
        "outcome": outcome,
        "note": LOCALIZATION_NON_CERTIFICATION_NOTE,
        "stages": _ordered_stage_statuses(stage, candidates_status=candidates_status),
        "counts": {
            "matlab_strands": matlab.get("n_strands"),
            "python_strands": python.get("n_strands"),
        },
        "strand_endpoint_multiset": {
            "n_matlab": sum(m_pairs.values()),
            "n_python": sum(p_pairs.values()),
            "n_intersection": inter,
            "n_only_matlab": sum((m_pairs - p_pairs).values()),
            "n_only_python": sum((p_pairs - m_pairs).values()),
            "overlap_pct_of_union": 100.0 * inter / union,
        },
        "same_class_guard": (
            "candidates compared only when both sides expose candidate_connections; "
            "Python candidates are never compared to MATLAB final edges"
        ),
    }


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
