"""E14 policy: whole-crop get_energy_V202 must not overwrite protected roots."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from slavv_python.analytics.parity.constants import PROTECTED_DEST_NAMES
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEnginePolicyError,
    refuse_matlab_only_energy_checkpoint_as_stretch_success,
    refuse_protected_stretch_energy_dest,
)


@pytest.mark.parametrize("dest_name", list(PROTECTED_DEST_NAMES))
def test_e14_refuses_protected_dest_roots(tmp_path: Path, dest_name: str) -> None:
    dest = tmp_path / "workspace" / "runs" / "oracle_180709_E" / dest_name
    with pytest.raises(MatlabEnginePolicyError, match="protected root"):
        refuse_protected_stretch_energy_dest(dest)


def test_e14_allows_scratch_dest(tmp_path: Path) -> None:
    dest = tmp_path / "workspace" / "scratch" / "e14_whole_crop_get_energy_v202"
    refuse_protected_stretch_energy_dest(dest)


def test_e14_matlab_only_checkpoint_is_not_stretch_success() -> None:
    with pytest.raises(MatlabEnginePolicyError, match="R6"):
        refuse_matlab_only_energy_checkpoint_as_stretch_success()
