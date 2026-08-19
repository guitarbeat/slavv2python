"""CI-safe tests for Energy lattice/params stretch isolation."""

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
from slavv_python.pipeline.energy.stretch.chunk_isolation import patch_stretch_status_extra
from slavv_python.pipeline.energy.stretch.lattice_params_isolation import (
    CROP_YXZ,
    CROP_ZYX,
    EXPECTED_MATLAB_OCTAVE2_CHUNKS,
    EXPECTED_PYTHON_OCTAVE2_CHUNKS,
    INTERPRET_BODY,
    INTERPRET_INCOMPLETE_INFRA,
    INTERPRET_LATTICE_OR_PARAMS,
    LatticeRecord,
    compare_param_fields,
    interpret_lattice_params,
    isolation_payload,
    lattice_from_rf,
    lattices_match_by_rf,
    matlab_derived_scales_per_octave,
    matlab_formula_lattices,
    matlab_h5_size_from_h5py_shape,
    python_lattices_from_config,
    record_at_octave,
    values_equal,
)
from slavv_python.utils.validation import validate_parameters

_CROP_PARAMS = {
    "energy_method": "hessian",
    "comparison_exact_network": True,
    "n_jobs": 1,
    "radius_of_smallest_vessel_in_microns": 1.5,
    "radius_of_largest_vessel_in_microns": 60.0,
    "scales_per_octave": 6.0,
    "max_voxels_per_node_energy": 6000.0,
    "gaussian_to_ideal_ratio": 0.5,
    "spherical_to_annular_ratio": 0.5,
    "approximating_PSF": True,
    "numerical_aperture": 0.95,
    "excitation_wavelength_in_microns": 0.95,
    "sample_index_of_refraction": 1.33,
    "microns_per_voxel": [0.916, 0.916, 1.99688],
    "energy_axis_permutation": [2, 0, 1],
}


def _crop_config() -> dict:
    image = np.zeros(CROP_ZYX, dtype=np.float64)
    params = validate_parameters(dict(_CROP_PARAMS))
    return _prepare_energy_config(image, params)


def test_h5py_shape_reverses_to_matlab_size() -> None:
    assert matlab_h5_size_from_h5py_shape((64, 256, 256)) == CROP_YXZ


def test_no_downsample_crop_lattice_is_726() -> None:
    record = lattice_from_rf(
        size_of_image=np.array(CROP_YXZ, dtype=float),
        rf=np.array([1.0, 1.0, 1.0]),
        microns=np.array([0.916, 0.916, 1.99688]),
        max_voxels=6000.0,
        octave=2,
        frame="matlab_yxz",
    )
    assert record.number_of_chunks == EXPECTED_MATLAB_OCTAVE2_CHUNKS
    assert record.lattice_dimensions == (11, 11, 6)


def test_python_octave2_crop_lattice_is_75() -> None:
    config = _crop_config()
    records = python_lattices_from_config(config)
    octave2 = record_at_octave(records, 2)
    assert octave2 is not None
    assert octave2.number_of_chunks == EXPECTED_PYTHON_OCTAVE2_CHUNKS
    assert octave2.lattice_dimensions == (5, 5, 3)


def test_interpret_lattice_or_params_when_75_vs_726() -> None:
    python_oct2 = LatticeRecord(
        octave=2,
        rf=(3, 3, 1),
        approx_size=(85, 85, 64),
        lattice_dimensions=(5, 5, 3),
        number_of_chunks=75,
        frame="python_yxz",
    )
    matlab_oct2 = LatticeRecord(
        octave=2,
        rf=(1, 1, 1),
        approx_size=CROP_YXZ,
        lattice_dimensions=(11, 11, 6),
        number_of_chunks=726,
        frame="matlab_yxz",
    )
    assert (
        interpret_lattice_params(
            params_equal=True,
            python_octave2=python_oct2,
            matlab_octave2=matlab_oct2,
        )
        == INTERPRET_LATTICE_OR_PARAMS
    )
    assert (
        interpret_lattice_params(
            params_equal=False,
            python_octave2=python_oct2,
            matlab_octave2=python_oct2,
        )
        == INTERPRET_LATTICE_OR_PARAMS
    )


def test_rf_matched_lattices_are_body_even_if_octave_ids_differ() -> None:
    python_records = [
        LatticeRecord(1, (1, 1, 1), CROP_YXZ, (11, 11, 6), 726, "python_yxz"),
        LatticeRecord(2, (3, 3, 1), (85, 85, 64), (5, 5, 3), 75, "python_yxz"),
    ]
    matlab_records = [
        LatticeRecord(2, (1, 1, 1), CROP_YXZ, (11, 11, 6), 726, "matlab_yxz"),
        LatticeRecord(3, (3, 3, 1), (85, 85, 64), (5, 5, 3), 75, "matlab_yxz"),
    ]
    assert lattices_match_by_rf(python_records, matlab_records) is True
    assert (
        interpret_lattice_params(
            params_equal=False,
            python_octave2=python_records[1],
            matlab_octave2=matlab_records[0],
            lattices_match_by_rf=True,
            params_core_equal=True,
        )
        == INTERPRET_BODY
    )
    shared = LatticeRecord(
        octave=2,
        rf=(3, 3, 1),
        approx_size=(85, 85, 64),
        lattice_dimensions=(5, 5, 3),
        number_of_chunks=75,
        frame="python_yxz",
    )
    assert (
        interpret_lattice_params(
            params_equal=True,
            python_octave2=shared,
            matlab_octave2=shared,
        )
        == INTERPRET_BODY
    )


def test_compare_param_fields_equality() -> None:
    radii = np.array([1.5, 1.6], dtype=np.float64)
    dest = {
        "lumen_radius_microns": radii,
        "microns_per_voxel_raw": np.array([0.916, 0.916, 1.99688]),
        "microns_per_voxel_working": np.array([1.99688, 0.916, 0.916]),
        "pixels_per_sigma_PSF": np.array([1.0, 1.0, 1.0]),
        "max_voxels": 6000.0,
        "gaussian_to_ideal_ratio": 0.5,
        "spherical_to_annular_ratio": 0.5,
        "scales_per_octave": 6.0,
    }
    oracle = {
        "radii": radii,
        "microns_per_voxel": dest["microns_per_voxel_raw"],
        "pixels_per_sigma_psf": dest["pixels_per_sigma_PSF"],
        "max_voxels": 6000.0,
        "gaussian_to_ideal_ratio": 0.5,
        "spherical_to_annular_ratio": 0.5,
        "scales_per_octave_derived": 6.0,
    }
    fields = compare_param_fields(dest, oracle)
    by_name = {field.name: field for field in fields}
    assert by_name["max_voxels"].equal is True
    assert values_equal(6000.0, 6000)
    assert by_name["lumen_radius_in_microns_range"].equal is True
    assert by_name["microns_per_voxel_raw"].equal is True
    assert by_name["microns_per_voxel_working"].equal is False
    derived = matlab_derived_scales_per_octave(np.array([1.5, 1.5 * 2 ** (1.0 / 18.0)]))
    assert derived == pytest.approx(6.0)


def test_isolation_payload_keeps_blocked_float_path() -> None:
    python_oct2 = LatticeRecord(
        octave=2,
        rf=(3, 3, 1),
        approx_size=(85, 85, 64),
        lattice_dimensions=(5, 5, 3),
        number_of_chunks=75,
        frame="python_yxz",
    )
    matlab_oct2 = LatticeRecord(
        octave=2,
        rf=(1, 1, 1),
        approx_size=CROP_YXZ,
        lattice_dimensions=(11, 11, 6),
        number_of_chunks=726,
        frame="matlab_yxz",
    )
    payload = isolation_payload(
        param_fields=[],
        python_lattices=[python_oct2],
        matlab_lattices=[matlab_oct2],
    )
    assert payload["status_class"] == StretchStatus.BLOCKED_FLOAT_PATH.value
    assert payload["stretch_complete"] is False
    assert payload["interpretation"] == INTERPRET_LATTICE_OR_PARAMS
    assert payload["octave2"]["python_matches_expected_75"] is True
    assert payload["octave2"]["matlab_matches_expected_726"] is True


def test_patch_status_extra_does_not_change_status(tmp_path: Path) -> None:
    status_path = tmp_path / "stretch_status.json"
    status_path.write_text(
        json.dumps({"status": StretchStatus.BLOCKED_FLOAT_PATH.value, "extra": {}}, indent=2),
        encoding="utf-8",
    )
    patch_stretch_status_extra(status_path, {"lattice_params_isolation": {"ok": True}})
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["status"] == StretchStatus.BLOCKED_FLOAT_PATH.value
    assert payload["extra"]["lattice_params_isolation"]["ok"] is True


def test_matlab_formula_lattices_include_octave_2() -> None:
    radii = np.array([1.5 * (2 ** (i / 18.0)) for i in range(54)], dtype=np.float64)
    records = matlab_formula_lattices(
        size_of_image_yxz=np.array(CROP_YXZ, dtype=float),
        radii=radii,
        microns_yxz=np.array([0.916, 0.916, 1.99688]),
        max_voxels=6000.0,
        scales_per_octave=6.0,
    )
    octave2 = record_at_octave(records, 2)
    assert octave2 is not None
    assert octave2.number_of_chunks >= 1


def test_live_oracle_engine_skip_is_incomplete_infra() -> None:
    python37 = resolve_python37_executable()
    if python37 is not None:
        try:
            resolve_matlab_root()
        except MatlabEngineInfraError:
            pytest.skip(INTERPRET_INCOMPLETE_INFRA)
        pytest.skip("engine present; lattice isolation does not require a live call")
    assert StretchStatus.INCOMPLETE_INFRA.value == INTERPRET_INCOMPLETE_INFRA
