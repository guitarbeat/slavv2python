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


def load_aggregated_training_data(
    data_dir: str | Path,
    file_pattern: str = "*_results.json",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and aggregate training features from multiple result payloads."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning(f"Training data directory does not exist: {data_dir}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    files = sorted(data_dir.rglob(file_pattern))
    if not files:
        logger.warning(f"No training data files matched pattern '{file_pattern}' in {data_dir}")
        return np.array([]), np.array([]), np.array([]), np.array([])

    vertex_feature_chunks: list[np.ndarray] = []
    vertex_label_chunks: list[np.ndarray] = []
    edge_feature_chunks: list[np.ndarray] = []
    edge_label_chunks: list[np.ndarray] = []

    for file_path in files:
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
                continue
        except Exception as exc:
            logger.warning(f"Skipping unreadable training data file {file_path}: {exc}")
            continue

        if not isinstance(loaded, dict):
            logger.warning(f"Skipping non-dictionary training payload in {file_path}")
            continue

        v_feat = _to_2d_array(_pick(loaded, ["vertex_features", "v_features", "v_feat"]), float)
        v_lab = _to_1d_array(_pick(loaded, ["vertex_labels", "v_labels"]), int)
        e_feat = _to_2d_array(_pick(loaded, ["edge_features", "e_features", "e_feat"]), float)
        e_lab = _to_1d_array(_pick(loaded, ["edge_labels", "e_labels"]), int)

        if v_feat is not None and v_lab is not None:
            if len(v_feat) == len(v_lab):
                vertex_feature_chunks.append(v_feat)
                vertex_label_chunks.append(v_lab)
            else:
                logger.warning(
                    f"Skipping mismatched vertex arrays in {file_path}: "
                    f"{len(v_feat)} features vs {len(v_lab)} labels"
                )

        if e_feat is not None and e_lab is not None:
            if len(e_feat) == len(e_lab):
                edge_feature_chunks.append(e_feat)
                edge_label_chunks.append(e_lab)
            else:
                logger.warning(
                    f"Skipping mismatched edge arrays in {file_path}: "
                    f"{len(e_feat)} features vs {len(e_lab)} labels"
                )

    v_feat = np.vstack(vertex_feature_chunks) if vertex_feature_chunks else np.array([])
    v_lab = np.hstack(vertex_label_chunks) if vertex_label_chunks else np.array([])
    e_feat = np.vstack(edge_feature_chunks) if edge_feature_chunks else np.array([])
    e_lab = np.hstack(edge_label_chunks) if edge_label_chunks else np.array([])
    logger.info(
        f"Aggregated training data from {len(files)} files: "
        f"{len(v_lab)} vertex labels, {len(e_lab)} edge labels"
    )
    return v_feat, v_lab, e_feat, e_lab
