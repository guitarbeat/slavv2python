from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.utils.safe_unpickle import safe_load

logger = logging.getLogger(__name__)


def _to_2d_array(value: Any, dtype: Any = float) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype)
    if arr.size == 0:
        return np.empty((0, 0), dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def _to_1d_array(value: Any, dtype: Any = int) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=dtype).reshape(-1)


def _pick(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _load_training_payload(file_path: Path) -> dict[str, Any] | None:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".npz":
            with np.load(file_path, allow_pickle=False) as npz:
                return {key: npz[key] for key in npz.files}
        if suffix == ".json":
            with file_path.open(encoding="utf-8-sig") as handle:
                loaded = json.load(handle)
            return loaded if isinstance(loaded, dict) else None
        if suffix in {".pkl", ".pickle"}:
            loaded = safe_load(file_path)
            return loaded if isinstance(loaded, dict) else None
    except Exception as exc:
        logger.warning("Failed to load training data from %s: %s", file_path, exc)
        return None

    logger.debug("Skipping unsupported training data file: %s", file_path)
    return None


def load_aggregated_training_data(
    data_dir: str | Path, file_pattern: str = "*.npz"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and aggregate training data from supported feature payloads."""
    dir_path = Path(data_dir)
    if not dir_path.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    files = sorted(set(dir_path.rglob(file_pattern)))
    if not files:
        return np.array([]), np.array([]), np.array([]), np.array([])

    vertex_feature_chunks: list[np.ndarray] = []
    vertex_label_chunks: list[np.ndarray] = []
    edge_feature_chunks: list[np.ndarray] = []
    edge_label_chunks: list[np.ndarray] = []

    for file_path in files:
        data = _load_training_payload(file_path)
        if data is None:
            continue

        vertex_features = _to_2d_array(
            _pick(data, ["vertex_features", "v_features", "v_feat"]), float
        )
        vertex_labels = _to_1d_array(_pick(data, ["vertex_labels", "v_labels"]), int)
        edge_features = _to_2d_array(_pick(data, ["edge_features", "e_features", "e_feat"]), float)
        edge_labels = _to_1d_array(_pick(data, ["edge_labels", "e_labels"]), int)

        if vertex_features is not None and vertex_labels is not None:
            if len(vertex_features) == len(vertex_labels):
                vertex_feature_chunks.append(vertex_features)
                vertex_label_chunks.append(vertex_labels)
            else:
                logger.warning(
                    "Skipping mismatched vertex arrays in %s: %s features vs %s labels",
                    file_path,
                    len(vertex_features),
                    len(vertex_labels),
                )

        if edge_features is not None and edge_labels is not None:
            if len(edge_features) == len(edge_labels):
                edge_feature_chunks.append(edge_features)
                edge_label_chunks.append(edge_labels)
            else:
                logger.warning(
                    "Skipping mismatched edge arrays in %s: %s features vs %s labels",
                    file_path,
                    len(edge_features),
                    len(edge_labels),
                )

    return (
        np.vstack(vertex_feature_chunks) if vertex_feature_chunks else np.array([]),
        np.hstack(vertex_label_chunks) if vertex_label_chunks else np.array([]),
        np.vstack(edge_feature_chunks) if edge_feature_chunks else np.array([]),
        np.hstack(edge_label_chunks) if edge_label_chunks else np.array([]),
    )
