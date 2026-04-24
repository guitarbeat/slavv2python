"""Training-data helpers for SLAVV ML curator workflows."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.safe_unpickle import safe_load

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
    return None if value is None else np.asarray(value, dtype=dtype).reshape(-1)


def _pick(data: dict[str, Any], keys: list[str]) -> Any:
    return next((data[key] for key in keys if key in data), None)


def _empty_training_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return np.array([]), np.array([]), np.array([]), np.array([])


def _load_training_payload(file_path: Path) -> dict[str, Any] | None:
    loaded: dict[str, Any] | None = None
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".npz":
            with np.load(file_path, allow_pickle=False) as npz:
                loaded = {k: npz[k] for k in npz.files}
        elif suffix == ".json":
            with open(file_path, encoding="utf-8-sig") as f:
                loaded = json.load(f)
        elif suffix in {".pkl", ".pickle"}:
            maybe = safe_load(file_path)
            if isinstance(maybe, dict):
                loaded = maybe
        else:
            logger.debug(f"Skipping unsupported training data file: {file_path}")
    except Exception as exc:
        logger.warning(f"Skipping unreadable training data file {file_path}: {exc}")
        return None
    if not isinstance(loaded, dict):
        logger.warning(f"Skipping non-dictionary training payload in {file_path}")
        return None
    return loaded


def _collect_training_arrays(
    loaded: dict[str, Any],
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    return (
        _to_2d_array(_pick(loaded, ["vertex_features", "v_features", "v_feat"]), float),
        _to_1d_array(_pick(loaded, ["vertex_labels", "v_labels"]), int),
        _to_2d_array(_pick(loaded, ["edge_features", "e_features", "e_feat"]), float),
        _to_1d_array(_pick(loaded, ["edge_labels", "e_labels"]), int),
    )


def _append_matching_arrays(
    feature_chunks: list[np.ndarray],
    label_chunks: list[np.ndarray],
    features: np.ndarray | None,
    labels: np.ndarray | None,
    file_path: Path,
    label_name: str,
) -> None:
    if features is None or labels is None:
        return
    if len(features) != len(labels):
        logger.warning(
            f"Skipping mismatched {label_name} arrays in {file_path}: "
            f"{len(features)} features vs {len(labels)} labels"
        )
        return
    feature_chunks.append(features)
    label_chunks.append(labels)


def _combine_training_chunks(
    feature_chunks: list[np.ndarray], label_chunks: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    features = np.vstack(feature_chunks) if feature_chunks else np.array([])
    labels = np.hstack(label_chunks) if label_chunks else np.array([])
    return features, labels


def load_aggregated_training_data(
    data_dir: str | Path,
    file_pattern: str = "*_results.json",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and aggregate training features from multiple result payloads."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning(f"Training data directory does not exist: {data_dir}")
        return _empty_training_arrays()

    files = sorted(data_dir.rglob(file_pattern))
    if not files:
        logger.warning(f"No training data files matched pattern '{file_pattern}' in {data_dir}")
        return _empty_training_arrays()

    vertex_feature_chunks: list[np.ndarray] = []
    vertex_label_chunks: list[np.ndarray] = []
    edge_feature_chunks: list[np.ndarray] = []
    edge_label_chunks: list[np.ndarray] = []

    for file_path in files:
        loaded = _load_training_payload(file_path)
        if loaded is None:
            continue

        v_feat, v_lab, e_feat, e_lab = _collect_training_arrays(loaded)
        _append_matching_arrays(
            vertex_feature_chunks, vertex_label_chunks, v_feat, v_lab, file_path, "vertex"
        )
        _append_matching_arrays(
            edge_feature_chunks, edge_label_chunks, e_feat, e_lab, file_path, "edge"
        )

    v_feat, v_lab = _combine_training_chunks(vertex_feature_chunks, vertex_label_chunks)
    e_feat, e_lab = _combine_training_chunks(edge_feature_chunks, edge_label_chunks)
    logger.info(
        f"Aggregated training data from {len(files)} files: "
        f"{len(v_lab)} vertex labels, {len(e_lab)} edge labels"
    )
    return v_feat, v_lab, e_feat, e_lab
