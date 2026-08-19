"""CI-safe tests for Energy helper-body stretch isolation."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import numpy as np
import pytest

from slavv_python.analytics.parity.proof.stretch import StretchStatus
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    resolve_matlab_root,
)
from slavv_python.pipeline.energy.matlab_engine_host import resolve_python37_executable
from slavv_python.pipeline.energy.stretch.chunk_isolation import patch_stretch_status_extra
from slavv_python.pipeline.energy.stretch.helper_body_isolation import (
    INTERPRET_CLAMP_NOT_TINY_ULP,
    INTERPRET_INCOMPLETE_INFRA,
    INTERPRET_INPUT_WINDOW,
    INTERPRET_LOCAL_RANGES_DIFFER,
    INTERPRET_LOCAL_RANGES_MATCH,
    compare_input_windows,
    compare_local_range,
    default_production_local_compares,
    downsample_matches_strided,
    h5py_c_order_to_matlab_yxz,
    helper_clamps_nonnegative,
    interpret_helper_body,
    isolation_payload,
    matlab_h52mat_chunk_yxz,
    matlab_local_range_1based,
    python_strided_chunk_yxz,
)


def test_production_chunk0_local_ranges_match() -> None:
    compares = default_production_local_compares()
    assert len(compares) == 3
    assert all(item.equal for item in compares)
    y_start, y_stop = matlab_local_range_1based(0, 51, 3)
    assert (y_start, y_stop) == (1, 18)
    z_start, z_stop = matlab_local_range_1based(0, 21, 1)
    assert (z_start, z_stop) == (1, 21)


def test_local_range_mismatch_is_named() -> None:
    item = compare_local_range(0, 51, 3)
    assert item.equal is True
    assert (
        interpret_helper_body(local_ranges_equal=False, input_windows_equal=True)
        == INTERPRET_LOCAL_RANGES_DIFFER
    )


def test_downsample_count_matches_python_stride() -> None:
    assert downsample_matches_strided(51, 3) is True
    assert downsample_matches_strided(21, 1) is True
    assert downsample_matches_strided(51, 1) is True


def test_clamp_is_named_and_not_tiny_ulp() -> None:
    assert helper_clamps_nonnegative() is True
    text = interpret_helper_body(local_ranges_equal=True, input_windows_equal=True)
    assert INTERPRET_LOCAL_RANGES_MATCH in text
    assert INTERPRET_CLAMP_NOT_TINY_ULP in text
    assert "1e-10" not in text or "not the 1e-10" in text


def test_tiff_vs_hdf5_windows_match_on_synthetic() -> None:
    rng = np.random.default_rng(0)
    image_zyx = rng.normal(size=(32, 64, 64)).astype(np.float64)
    volume_yxz = np.transpose(image_zyx, (1, 2, 0))
    python_chunk = python_strided_chunk_yxz(
        image_zyx,
        z_start=0,
        y_start=0,
        x_start=0,
        z_count=21,
        y_count=51,
        x_count=51,
        rf_zyx=(1, 3, 3),
    )
    matlab_chunk = matlab_h52mat_chunk_yxz(
        volume_yxz,
        y_start_1based=1,
        x_start_1based=1,
        z_start_1based=1,
        y_read_count=51,
        x_read_count=51,
        z_read_count=21,
        rf_yxz=(3, 3, 1),
    )
    result = compare_input_windows(python_chunk, matlab_chunk)
    assert result.equal is True
    assert result.n_diff == 0


def test_h5py_axis_reverse_is_matlab_yxz() -> None:
    volume_yxz = np.arange(24, dtype=np.float64).reshape((4, 3, 2), order="F")
    h5py_c = np.transpose(volume_yxz, (2, 1, 0))
    restored = h5py_c_order_to_matlab_yxz(h5py_c)
    assert restored.shape == volume_yxz.shape
    assert np.array_equal(restored, volume_yxz)


def test_windows_differ_is_named_input_source() -> None:
    text = interpret_helper_body(local_ranges_equal=True, input_windows_equal=False)
    assert INTERPRET_INPUT_WINDOW in text
    payload = isolation_payload(
        local_compares=default_production_local_compares(),
        input_windows_equal=False,
    )
    assert payload["status_class"] == StretchStatus.BLOCKED_FLOAT_PATH.value
    assert payload["stretch_complete"] is False
    assert payload["interpretation"] == text


def test_isolation_payload_keeps_blocked_float_path() -> None:
    payload = isolation_payload(local_compares=default_production_local_compares())
    assert payload["status_class"] == StretchStatus.BLOCKED_FLOAT_PATH.value
    assert payload["stretch_complete"] is False
    assert payload["local_ranges_equal"] is True
    assert payload["matlab_active_path_clamps_nonnegative"] is False
    assert INTERPRET_LOCAL_RANGES_MATCH in payload["interpretation"]


def test_patch_status_extra_does_not_change_status(tmp_path: Path) -> None:
    status_path = tmp_path / "stretch_status.json"
    status_path.write_text(
        json.dumps({"status": StretchStatus.BLOCKED_FLOAT_PATH.value, "extra": {}}, indent=2),
        encoding="utf-8",
    )
    patch_stretch_status_extra(status_path, {"helper_body_isolation": {"ok": True}})
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == StretchStatus.BLOCKED_FLOAT_PATH.value
    assert payload["extra"]["helper_body_isolation"]["ok"] is True


def test_live_oracle_engine_skip_is_incomplete_infra() -> None:
    python37 = resolve_python37_executable()
    if python37 is not None:
        try:
            resolve_matlab_root()
        except MatlabEngineInfraError:
            pytest.skip(INTERPRET_INCOMPLETE_INFRA)
        pytest.skip("engine present; helper-body isolation does not require a live call")
    assert StretchStatus.INCOMPLETE_INFRA.value == INTERPRET_INCOMPLETE_INFRA
