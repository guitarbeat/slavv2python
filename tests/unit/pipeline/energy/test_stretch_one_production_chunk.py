"""CI-safe tests for one production crop Energy chunk isolation."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003

import numpy as np
import pytest

from slavv_python.analytics.parity.proof.stretch import StretchStatus
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    resolve_matlab_root,
)
from slavv_python.pipeline.energy.matlab_engine_host import resolve_python37_executable
from slavv_python.pipeline.energy.parity_energy_voxel_probe import resolve_write_chunk_idx_for_voxel
from slavv_python.pipeline.energy.stretch_chunk_isolation import (
    DEFAULT_MISMATCH_VOXEL_ZYX,
    DEFAULT_WINNER_SCALE,
    INTERPRET_HELPER_ORACLE,
    INTERPRET_OTHER_CHUNKS,
    INTERPRET_PACKAGING,
    INTERPRET_WINDOW_MATCHES_ALL,
    build_octave_chunk_lattice,
    chunk_index_for_voxel_zyx,
    compare_three_way,
    interpret_three_way,
    octave_for_scale,
    octave_owned_mask,
    patch_stretch_status_extra,
    run_stretch_chunk_v202,
)
from slavv_python.utils.validation import validate_parameters

_SMALL_PARAMS = {
    "energy_method": "hessian",
    "comparison_exact_network": True,
    "n_jobs": 1,
    "radius_of_smallest_vessel_in_microns": 1.5,
    "radius_of_largest_vessel_in_microns": 60.0,
    "scales_per_octave": 6.0,
    "max_voxels_per_node_energy": 64.0,
    "gaussian_to_ideal_ratio": 0.5,
    "spherical_to_annular_ratio": 0.5,
    "approximating_PSF": True,
    "numerical_aperture": 0.95,
    "excitation_wavelength_in_microns": 0.95,
    "sample_index_of_refraction": 1.33,
    "microns_per_voxel": [0.916, 0.916, 1.99688],
    "energy_axis_permutation": [2, 0, 1],
}


def _small_config(shape: tuple[int, int, int] = (16, 16, 16)) -> dict:
    image = np.zeros(shape, dtype=np.float64)
    params = validate_parameters(dict(_SMALL_PARAMS))
    return _prepare_energy_config(image, params)


def test_interpret_three_way_helper_oracle() -> None:
    assert (
        interpret_three_way(
            rerun_equals_v2=True,
            rerun_equals_oracle=False,
            v2_equals_oracle=False,
        )
        == INTERPRET_HELPER_ORACLE
    )


def test_interpret_three_way_packaging() -> None:
    assert (
        interpret_three_way(
            rerun_equals_v2=False,
            rerun_equals_oracle=False,
            v2_equals_oracle=False,
        )
        == INTERPRET_PACKAGING
    )
    assert (
        interpret_three_way(
            rerun_equals_v2=False,
            rerun_equals_oracle=True,
            v2_equals_oracle=False,
        )
        == INTERPRET_PACKAGING
    )


def test_interpret_three_way_other_chunks_or_all_match() -> None:
    assert (
        interpret_three_way(
            rerun_equals_v2=True,
            rerun_equals_oracle=True,
            v2_equals_oracle=False,
        )
        == INTERPRET_OTHER_CHUNKS
    )
    assert (
        interpret_three_way(
            rerun_equals_v2=True,
            rerun_equals_oracle=True,
            v2_equals_oracle=True,
        )
        == INTERPRET_WINDOW_MATCHES_ALL
    )


def test_compare_three_way_array_equal() -> None:
    rerun = np.array([1.0, 2.0], dtype=np.float64)
    v2 = np.array([1.0, 2.0], dtype=np.float64)
    oracle = np.array([1.0, 3.0], dtype=np.float64)
    result = compare_three_way(rerun, v2, oracle)
    assert result.rerun_equals_v2 is True
    assert result.rerun_equals_oracle is False
    assert result.v2_equals_oracle is False
    assert result.interpretation == INTERPRET_HELPER_ORACLE
    assert result.n_v2_ne_oracle == 1


def test_octave_owned_mask_selects_this_octave() -> None:
    scales = np.array([[43, 10], [43, 44]], dtype=np.int16)
    mask = octave_owned_mask(scales, (42, 43, 44))
    assert mask.tolist() == [[True, False], [True, True]]


def test_chunk_index_matches_existing_write_resolver() -> None:
    config = _small_config()
    octave_at_scales = np.asarray(config["octave_at_scales"])
    assert int(octave_at_scales.size) > DEFAULT_WINNER_SCALE
    voxel = (1, 2, 3)
    hit = chunk_index_for_voxel_zyx(config, voxel, winner_scale=DEFAULT_WINNER_SCALE)
    expected = resolve_write_chunk_idx_for_voxel(
        config,
        voxel_zyx=voxel,
        target_rf_zyx=hit.rf_zyx,
    )
    assert hit.chunk_index == expected
    z0, y0, x0 = hit.write_start_zyx
    dz, dy, dx = hit.write_count_zyx
    assert z0 <= voxel[0] < z0 + dz
    assert y0 <= voxel[1] < y0 + dy
    assert x0 <= voxel[2] < x0 + dx
    assert hit.octave == octave_for_scale(config, DEFAULT_WINNER_SCALE)
    y_idx, x_idx, z_idx = hit.lattice_indices_yxz
    rebuilt = int(
        np.ravel_multi_index((y_idx, x_idx, z_idx), hit.lattice_dimensions_yxz, order="F")
    )
    assert rebuilt == hit.chunk_index


def test_default_mismatch_voxel_maps_on_crop_shaped_lattice() -> None:
    config = _small_config((32, 32, 32))
    hit = chunk_index_for_voxel_zyx(
        config,
        DEFAULT_MISMATCH_VOXEL_ZYX,
        winner_scale=DEFAULT_WINNER_SCALE,
    )
    assert hit.number_of_chunks >= 1
    z0, y0, x0 = hit.write_start_zyx
    dz, dy, dx = hit.write_count_zyx
    assert z0 <= 13 < z0 + dz
    assert y0 <= 0 < y0 + dy
    assert x0 <= 0 < x0 + dx


def test_fortran_unravel_is_yxz_lattice() -> None:
    config = _small_config()
    lattice = build_octave_chunk_lattice(config, octave_for_scale(config, DEFAULT_WINNER_SCALE))
    if lattice.number_of_chunks < 2:
        pytest.skip("synthetic lattice did not produce multiple chunks")
    y_idx, x_idx, z_idx = np.unravel_index(1, lattice.lattice_dimensions_yxz, order="F")
    assert (int(y_idx), int(x_idx), int(z_idx)) == (1, 0, 0) or lattice.lattice_dimensions_yxz[
        0
    ] == 1


def test_run_stretch_chunk_monkeypatched_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_chunk_helper(
        session: object,
        chunk: np.ndarray,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del session
        calls.append(int(np.asarray(chunk).size))
        y_w = int(kwargs["y_write_count"])
        x_w = int(kwargs["x_write_count"])
        z_w = int(kwargs["z_write_count"])
        energy = np.full((y_w, x_w, z_w), -1.0, dtype=np.float64)
        scale = np.ones((y_w, x_w, z_w), dtype=np.float64)
        return energy, scale

    monkeypatch.setattr(
        "slavv_python.pipeline.energy.stretch_chunk_isolation.energy_chunk_v202_from_spatial",
        fake_chunk_helper,
    )
    image = np.zeros((16, 16, 16), dtype=np.float64)
    config = _small_config(image.shape)
    hit = chunk_index_for_voxel_zyx(config, (0, 0, 0), winner_scale=DEFAULT_WINNER_SCALE)
    lattice = build_octave_chunk_lattice(config, hit.octave)
    energy, scales, slices = run_stretch_chunk_v202(
        image, config, lattice, hit.chunk_index, session=object()
    )
    assert calls
    assert energy.shape == tuple(hit.write_count_zyx)
    assert scales.shape == energy.shape
    assert slices == hit.write_slices_zyx


def test_run_stretch_chunk_without_session_is_incomplete_infra() -> None:
    image = np.zeros((16, 16, 16), dtype=np.float64)
    config = _small_config(image.shape)
    hit = chunk_index_for_voxel_zyx(config, (0, 0, 0), winner_scale=0)
    lattice = build_octave_chunk_lattice(config, hit.octave)
    with pytest.raises(MatlabEngineInfraError, match="incomplete_infra"):
        run_stretch_chunk_v202(image, config, lattice, hit.chunk_index, session=None)


def test_live_engine_one_chunk_skips_without_py37_matlab() -> None:
    python37 = resolve_python37_executable()
    if python37 is None:
        pytest.skip("incomplete_infra: isolated Python 3.7 stretch env missing")
    try:
        resolve_matlab_root()
    except MatlabEngineInfraError as exc:
        pytest.skip(f"incomplete_infra: {exc}")


def test_patch_stretch_status_extra_keeps_blocked_float_path(tmp_path: Path) -> None:
    status_path = tmp_path / "stretch_status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": StretchStatus.BLOCKED_FLOAT_PATH.value,
                "note": "crop Energy not bit-equal",
                "phase1_claim_untouched": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    payload = patch_stretch_status_extra(
        status_path,
        {"one_production_chunk": {"interpretation": INTERPRET_HELPER_ORACLE}},
    )
    assert payload["status"] == StretchStatus.BLOCKED_FLOAT_PATH.value
    reloaded = json.loads(status_path.read_text(encoding="utf-8"))
    assert reloaded["status"] == StretchStatus.BLOCKED_FLOAT_PATH.value
    assert reloaded["extra"]["one_production_chunk"]["interpretation"] == INTERPRET_HELPER_ORACLE
    assert reloaded["note"] == "crop Energy not bit-equal"


def test_patch_stretch_status_refuses_non_blocked_status(tmp_path: Path) -> None:
    status_path = tmp_path / "stretch_status.json"
    status_path.write_text(
        json.dumps({"status": StretchStatus.STRETCH_COMPLETE.value}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="blocked_float_path"):
        patch_stretch_status_extra(status_path, {"one_production_chunk": {}})
    reloaded = json.loads(status_path.read_text(encoding="utf-8"))
    assert reloaded["status"] == StretchStatus.STRETCH_COMPLETE.value
    assert "extra" not in reloaded
