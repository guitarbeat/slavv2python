"""Shared crop TIFF / params IO for stretch isolation operators.

Path names live in ``slavv_python.analytics.parity.constants``. Isolation
probes import these helpers instead of copying dest lookup.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from slavv_python.analytics.parity.constants import (
    CROP_TIF_NAME,
    EXPERIMENT_PARAMS_DIR,
    EXPERIMENT_REFS_DIR,
    VALIDATED_PARAMS_PATH,
)


def load_dest_params(dest: Path) -> dict[str, Any]:
    """Load ``validated_params.json`` from metadata or ``01_Params``."""
    dest = Path(dest)
    candidates = (
        dest / VALIDATED_PARAMS_PATH,
        dest / EXPERIMENT_PARAMS_DIR / "validated_params.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return cast("dict[str, Any]", json.loads(candidate.read_text(encoding="utf-8")))
    raise FileNotFoundError(f"validated_params.json missing under {dest}")


def find_crop_tif(dest: Path, *, repo_root: Path, crop_tif_name: str = CROP_TIF_NAME) -> Path:
    """Prefer dest ``00_Refs`` TIFF; else first matching dataset input."""
    dest = Path(dest)
    refs = dest / EXPERIMENT_REFS_DIR / crop_tif_name
    if refs.is_file():
        return Path(refs)
    datasets = Path(repo_root) / "workspace" / "datasets"
    if datasets.is_dir():
        matches = sorted(datasets.glob(f"*/01_Input/{crop_tif_name}"))
        if matches:
            return Path(matches[0])
    raise FileNotFoundError(
        f"crop TIFF missing: expected {refs} (not 01_Input) or workspace/datasets/"
    )


def reorient_image_to_energy(image: np.ndarray, energy_shape: tuple[int, ...]) -> np.ndarray:
    """Transpose a 3D volume so its shape matches Energy ZYX."""
    if tuple(int(v) for v in image.shape) == tuple(int(v) for v in energy_shape):
        return image
    for perm in itertools.permutations((0, 1, 2)):
        reordered = tuple(int(image.shape[i]) for i in perm)
        if reordered == tuple(int(v) for v in energy_shape):
            return cast("np.ndarray", np.transpose(image, perm))
    raise ValueError(f"cannot reorient image {image.shape} to energy {energy_shape}")


__all__ = [
    "find_crop_tif",
    "load_dest_params",
    "reorient_image_to_energy",
]
