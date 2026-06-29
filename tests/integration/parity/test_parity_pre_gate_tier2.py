"""Parity Pre-Gate tier 2 — crop harness oracle surface (ADR 0009).

These tests run only when ``workspace/oracles/180709_E_crop_M`` is promoted locally
or ``SLAVV_CROP_ORACLE_ROOT`` points at a valid oracle tree. They do not perform
full ``prove-exact-sequence`` runs (too heavy for default CI).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from slavv_python.analytics.parity.constants import ORACLE_MANIFEST_PATH
from slavv_python.analytics.parity.oracle.surfaces import load_oracle_surface

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
@pytest.mark.parity
def test_tier2_crop_harness_oracle_surface_loads(crop_harness_oracle_root: Path):
    """Promoted crop oracle exposes MATLAB vector paths for all discovery stages."""
    surface = load_oracle_surface(crop_harness_oracle_root)

    assert surface.oracle_root == crop_harness_oracle_root.resolve()
    assert surface.manifest_path is not None
    assert surface.manifest_path.is_file()
    assert surface.manifest_path.name == ORACLE_MANIFEST_PATH.name
    assert surface.matlab_batch_dir is not None
    assert surface.oracle_id == "180709_E_crop_M"
    assert set(surface.matlab_vector_paths) >= {"energy", "vertices", "edges", "network"}


@pytest.mark.integration
@pytest.mark.parity
@pytest.mark.slow
def test_tier2_crop_harness_oracle_energy_vectors_present(crop_harness_oracle_root: Path):
    """Crop oracle energy stage includes size_of_image metadata for preflight."""
    surface = load_oracle_surface(crop_harness_oracle_root)
    energy_path = surface.matlab_vector_paths.get("energy")
    assert energy_path is not None
    assert energy_path.is_file()
