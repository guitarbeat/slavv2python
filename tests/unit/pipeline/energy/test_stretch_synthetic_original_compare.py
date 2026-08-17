"""CI-safe tests for tiny synthetic helper vs original Energy isolation."""

from __future__ import annotations

import numpy as np
import pytest

from slavv_python.analytics.parity.constants import PROTECTED_DEST_NAMES
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEnginePolicyError,
    refuse_protected_stretch_energy_dest,
)
from slavv_python.pipeline.energy.stretch_helper_body_isolation import (
    h5py_c_order_to_matlab_yxz,
)
from slavv_python.pipeline.energy.stretch_synthetic_original_compare import (
    INTENSITY_LOGNORMAL,
    INTERPRET_CLAMP_CLASS,
    INTERPRET_SYNTHETIC_MATCH,
    INTERPRET_TINY_ULP,
    SEED,
    SHAPE_ZYX,
    classify_synthetic_compare,
    energy_h5py_plane_to_zyx,
    matlab_yxz_to_h5py_c_order,
    seeded_volume_zyx,
    volume_zyx_to_matlab_yxz,
)


def test_seeded_volume_is_deterministic() -> None:
    first = seeded_volume_zyx()
    second = seeded_volume_zyx(seed=SEED, shape_zyx=SHAPE_ZYX)
    assert first.shape == SHAPE_ZYX
    assert np.array_equal(first, second)
    assert first.dtype == np.float64
    larger = seeded_volume_zyx(shape_zyx=(32, 32, 32))
    assert larger.shape == (32, 32, 32)
    assert larger.dtype == np.float64
    lognormal = seeded_volume_zyx(intensity=INTENSITY_LOGNORMAL)
    assert lognormal.shape == SHAPE_ZYX
    assert np.all(lognormal > 0.0)
    assert not np.array_equal(first, lognormal)


def test_h5py_roundtrip_preserves_yxz() -> None:
    image = seeded_volume_zyx()
    yxz = volume_zyx_to_matlab_yxz(image)
    c_order = matlab_yxz_to_h5py_c_order(yxz)
    restored = h5py_c_order_to_matlab_yxz(c_order)
    assert np.array_equal(yxz, restored)
    zyx = energy_h5py_plane_to_zyx(c_order)
    assert np.array_equal(image, zyx)


def test_classify_bit_match_is_not_stretch_success() -> None:
    volume = seeded_volume_zyx()
    payload = classify_synthetic_compare(helper_energy=volume, original_energy=volume)
    assert payload["result"] == "pass_fixture"
    assert payload["stretch_complete"] is False
    assert payload["not_stretch_success"] is True
    assert INTERPRET_SYNTHETIC_MATCH in payload["interpretation"]


def test_classify_names_clamp_vs_tiny_ulp() -> None:
    helper = np.array([[[-0.2, 0.0]]], dtype=np.float64)
    original_clamp = np.array([[[-0.2, 0.4]]], dtype=np.float64)
    clamp_payload = classify_synthetic_compare(helper_energy=helper, original_energy=original_clamp)
    assert clamp_payload["result"] == "named_clamp"
    assert INTERPRET_CLAMP_CLASS in clamp_payload["interpretation"]

    original_ulp = np.array([[[-0.2, -0.2 + 1e-10]]], dtype=np.float64)
    helper_ulp = np.array([[[-0.2, -0.2]]], dtype=np.float64)
    ulp_payload = classify_synthetic_compare(helper_energy=helper_ulp, original_energy=original_ulp)
    assert ulp_payload["result"] == "blocked_float_path"
    assert INTERPRET_TINY_ULP in ulp_payload["interpretation"]
    assert ulp_payload["max_abs_delta"] == pytest.approx(1e-10)


@pytest.mark.parametrize("dest_name", list(PROTECTED_DEST_NAMES))
def test_synthetic_dest_refuses_protected_roots(tmp_path, dest_name: str) -> None:
    dest = tmp_path / "workspace" / "runs" / "oracle_180709_E" / dest_name
    with pytest.raises(MatlabEnginePolicyError, match="protected root"):
        refuse_protected_stretch_energy_dest(dest)
