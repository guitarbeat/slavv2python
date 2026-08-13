"""Load an edge-shaped artifact and return connections plus its class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
from scipy.io import loadmat

from slavv_python.analytics.parity.experiments.artifact_class import (
    ArtifactClass,
    ArtifactClassError,
    classify_edge_artifact,
)
from slavv_python.utils.safe_unpickle import safe_load

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class EdgeArtifact:
    """Connections loaded from one classified on-disk artifact."""

    path: Path
    artifact_class: ArtifactClass
    connections: np.ndarray


def _normalize_endpoints(raw: np.ndarray, *, one_based: bool) -> np.ndarray:
    values = np.asarray(raw, dtype=np.int64)
    if values.size == 0:
        return np.zeros((0, 2), dtype=np.int64)
    if values.ndim != 2:
        raise ArtifactClassError(f"endpoint array must be 2-D: {values.shape}")
    if values.shape[0] == 2 and values.shape[1] != 2:
        values = values.T
    if values.shape[1] != 2:
        raise ArtifactClassError(f"endpoint array must be (N, 2): {values.shape}")
    if one_based:
        values = values - 1
    return cast("np.ndarray", np.asarray(values, dtype=np.int64))


def _load_mat_endpoints(path: Path) -> np.ndarray:
    try:
        with h5py.File(path, "r") as handle:
            if "edges2vertices" not in handle:
                raise ArtifactClassError(f"MATLAB artifact missing edges2vertices: {path}")
            return _normalize_endpoints(handle["edges2vertices"], one_based=True)
    except OSError:
        payload = loadmat(path, squeeze_me=False, struct_as_record=False)
        if "edges2vertices" not in payload:
            raise ArtifactClassError(f"MATLAB artifact missing edges2vertices: {path}") from None
        return _normalize_endpoints(payload["edges2vertices"], one_based=True)


def _load_pickle_endpoints(path: Path) -> np.ndarray:
    payload = safe_load(path)
    if not isinstance(payload, dict):
        raise ArtifactClassError(f"pickle artifact is not a mapping: {path}")
    if "connections" in payload:
        return _normalize_endpoints(payload["connections"], one_based=False)
    if "edges2vertices" in payload:
        return _normalize_endpoints(payload["edges2vertices"], one_based=False)
    raise ArtifactClassError(f"pickle artifact missing connections: {path}")


def load_edge_artifact(path: Path) -> EdgeArtifact:
    """Load connections and refuse unknown names.

    MATLAB ``.mat`` dumps are treated as 1-based ``edges2vertices``.
    Python pickles are 0-based ``connections``.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ArtifactClassError(f"edge artifact not found: {resolved}")
    artifact_class = classify_edge_artifact(resolved)
    suffix = resolved.suffix.lower()
    if suffix == ".mat":
        connections = _load_mat_endpoints(resolved)
    elif suffix == ".pkl":
        connections = _load_pickle_endpoints(resolved)
    else:
        raise ArtifactClassError(f"unsupported edge artifact suffix: {resolved.name}")
    return EdgeArtifact(
        path=resolved,
        artifact_class=artifact_class,
        connections=connections,
    )
