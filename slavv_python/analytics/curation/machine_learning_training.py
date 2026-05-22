from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_aggregated_training_data(
    data_dir: str | Path, file_pattern: str = "*.npz"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and aggregate training data (features and labels) from multiple files.
    Supports .npz (numpy) and .json formats.
    """
    dir_path = Path(data_dir)
    if not dir_path.exists():
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_v_feat, all_v_lab = [], []
    all_e_feat, all_e_lab = [], []

    # Handle both .npz and .json if pattern allows
    search_pattern = file_pattern
    if "*" not in search_pattern:
        search_pattern = f"*{search_pattern}*"

    # We want to find both .npz and .json if possible
    files = list(dir_path.glob(search_pattern))
    # If the user provided a pattern like 'chunk_*', we also want to look for .json
    if "." not in search_pattern:
        files.extend(list(dir_path.glob(f"{search_pattern}.json")))
        files.extend(list(dir_path.glob(f"{search_pattern}.npz")))

    # Deduplicate files
    files = sorted(set(files))

    for f in files:
        try:
            if f.suffix == ".npz":
                data = np.load(f)
                all_v_feat.append(data["vertex_features"])
                all_v_lab.append(data["vertex_labels"])
                all_e_feat.append(data["edge_features"])
                all_e_lab.append(data["edge_labels"])
            elif f.suffix == ".json":
                with open(f, encoding="utf-8") as jf:
                    data = json.load(jf)
                all_v_feat.append(np.array(data["vertex_features"]))
                all_v_lab.append(np.array(data["vertex_labels"]))
                all_e_feat.append(np.array(data["edge_features"]))
                all_e_lab.append(np.array(data["edge_labels"]))
        except Exception as e:
            logger.warning(f"Failed to load training data from {f}: {e}")

    def _concat(list_of_arrs):
        if not list_of_arrs:
            return np.array([])
        # Ensure they are all at least 2D for features if not empty
        return np.concatenate(list_of_arrs, axis=0)

    return (
        _concat(all_v_feat),
        _concat(all_v_lab),
        _concat(all_e_feat),
        _concat(all_e_lab),
    )
