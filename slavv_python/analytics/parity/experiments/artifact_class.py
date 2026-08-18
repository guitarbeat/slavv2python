"""Classify edge artifacts and compare only the same class."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from slavv_python.analytics.parity.constants import EDGE_CANDIDATE_CHECKPOINT_PATH
from slavv_python.pipeline.edges.candidate_manifest import endpoint_pairs_from_connections

if TYPE_CHECKING:
    from pathlib import Path


class ArtifactClass(Enum):
    """On-disk class for an edge-shaped artifact."""

    RAW_CANDIDATE_SET = "raw_candidate_set"
    EDGE_SET = "edge_set"


class ArtifactClassError(ValueError):
    """Raised when a path cannot be classified or a compare mixes classes."""


@dataclass(frozen=True)
class PairSetCompare:
    """Same-class undirected pair-set difference."""

    left_class: ArtifactClass
    right_class: ArtifactClass
    n_left: int
    n_right: int
    n_intersection: int
    n_only_left: int
    n_only_right: int


@dataclass(frozen=True)
class CoverageCompare:
    """How many Edge Set pairs appear in a raw Candidate Set."""

    n_raw: int
    n_final: int
    n_covered: int
    n_missing_from_raw: int
    n_extra_raw: int


_EDGE_SET_NAMES = frozenset(
    {
        "edges.pkl",
        "checkpoint_edges.pkl",
        "chosen_edges.pkl",
        "edges.mat",
    }
)
_RAW_NAMES = frozenset(
    {
        "candidates.pkl",
        "checkpoint_edge_candidates.pkl",
    }
)


def classify_edge_artifact(path: Path) -> ArtifactClass:
    """Return the artifact class for an on-disk edge-shaped file.

    Unknown names raise. Callers that need a different class must pass
    ``left_class`` / ``right_class`` explicitly to the compare functions.
    """
    name = path.name.lower()
    if name in _RAW_NAMES:
        return ArtifactClass.RAW_CANDIDATE_SET
    if "candidate" in name and name.endswith((".pkl", ".mat")):
        return ArtifactClass.RAW_CANDIDATE_SET
    if name in _EDGE_SET_NAMES:
        return ArtifactClass.EDGE_SET
    if name.startswith("edges_") and name.endswith(".mat"):
        return ArtifactClass.EDGE_SET
    raise ArtifactClassError(f"cannot classify edge artifact: {path.name}")


def _as_pair_set(
    connections: np.ndarray | set[tuple[int, int]],
) -> set[tuple[int, int]]:
    if isinstance(connections, set):
        return connections
    return set(endpoint_pairs_from_connections(np.asarray(connections)))


def compare_same_class_pair_sets(
    left: np.ndarray | set[tuple[int, int]],
    right: np.ndarray | set[tuple[int, int]],
    *,
    left_class: ArtifactClass,
    right_class: ArtifactClass,
) -> PairSetCompare:
    """Compare two undirected pair sets that share an artifact class.

    Mixed Candidate Set vs Edge Set compares raise. That is the raw-vs-final
    trap: it invents a “MATLAB never emits” story from cleaned finals.
    """
    if left_class is not right_class:
        raise ArtifactClassError(
            f"refuse mixed-class pair-set compare: {left_class.value} vs {right_class.value}"
        )
    left_pairs = _as_pair_set(left)
    right_pairs = _as_pair_set(right)
    only_left = left_pairs - right_pairs
    only_right = right_pairs - left_pairs
    return PairSetCompare(
        left_class=left_class,
        right_class=right_class,
        n_left=len(left_pairs),
        n_right=len(right_pairs),
        n_intersection=len(left_pairs & right_pairs),
        n_only_left=len(only_left),
        n_only_right=len(only_right),
    )


def coverage_of_finals_by_raw(
    raw_pairs: np.ndarray | set[tuple[int, int]],
    final_pairs: np.ndarray | set[tuple[int, int]],
) -> CoverageCompare:
    """The only legal Candidate Set vs Edge Set compare.

    This is coverage of finals by raw emission, not pair-set equality and not
    Certification.
    """
    raw = _as_pair_set(raw_pairs)
    finals = _as_pair_set(final_pairs)
    return CoverageCompare(
        n_raw=len(raw),
        n_final=len(finals),
        n_covered=len(raw & finals),
        n_missing_from_raw=len(finals - raw),
        n_extra_raw=len(raw - finals),
    )


def resolve_candidate_set_path(run_root: Path) -> Path | None:
    """Return the on-disk Candidate Set, preferring the Edges Artifact.

    ``04_Edges/candidates.pkl`` is the production write. The checkpoint name is
    a read-only fallback for dests that still have the former dual-write.
    """
    artifact = run_root / "04_Edges" / "candidates.pkl"
    if artifact.is_file():
        return artifact
    checkpoint = run_root / EDGE_CANDIDATE_CHECKPOINT_PATH
    if checkpoint.is_file():
        return checkpoint
    return None
