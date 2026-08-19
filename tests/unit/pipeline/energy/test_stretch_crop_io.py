"""CI-safe tests for shared stretch crop TIFF / params IO."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import numpy as np
import pytest

from slavv_python.analytics.parity.constants import (
    CROP_TIF_NAME,
    EXPERIMENT_REFS_DIR,
    VALIDATED_PARAMS_PATH,
)
from slavv_python.pipeline.energy.stretch.crop_io import (
    find_crop_tif,
    load_dest_params,
    reorient_image_to_energy,
)


def test_find_crop_tif_prefers_dest_refs(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    refs = dest / EXPERIMENT_REFS_DIR
    refs.mkdir(parents=True)
    tif = refs / CROP_TIF_NAME
    tif.write_bytes(b"tif")
    assert find_crop_tif(dest, repo_root=tmp_path) == tif


def test_load_dest_params_from_metadata(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    params_path = dest / VALIDATED_PARAMS_PATH
    params_path.parent.mkdir(parents=True)
    params_path.write_text('{"ok": true}', encoding="utf-8")
    assert load_dest_params(dest) == {"ok": True}


def test_reorient_image_to_energy_transposes() -> None:
    image = np.zeros((2, 3, 4), dtype=np.float64)
    out = reorient_image_to_energy(image, (4, 2, 3))
    assert out.shape == (4, 2, 3)


def test_reorient_image_to_energy_rejects_mismatch() -> None:
    image = np.zeros((2, 3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="cannot reorient"):
        reorient_image_to_energy(image, (5, 5, 5))
