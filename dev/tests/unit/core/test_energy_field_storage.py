import numpy as np
import numpy.testing as npt
import pytest
from source.core import SLAVVProcessor
from source.core import energy as energy_module
from source.core._energy import backends as energy_backends
from source.runtime import RunContext
from source.utils import get_chunking_lattice, validate_parameters


def test_energy_field_no_full_storage():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters({})
    proc = SLAVVProcessor()
    result = proc.calculate_energy_field(img, params)
    assert "energy_4d" not in result
    assert result["energy"].shape == img.shape


def test_energy_field_with_full_storage():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters({"return_all_scales": True})
    proc = SLAVVProcessor()
    result = proc.calculate_energy_field(img, params)
    assert "energy_4d" in result
    energy_4d = result["energy_4d"]
    assert energy_4d.shape[:3] == img.shape
    assert energy_4d.shape[3] == len(result["lumen_radius_pixels"])


def test_energy_scale_range_matches_matlab_ordination():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters(
        {
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 50.0,
            "scales_per_octave": 1.5,
        }
    )

    result = SLAVVProcessor().calculate_energy_field(img, params)

    largest_per_smallest_volume_ratio = (50.0 / 1.5) ** 3
    final_scale = int(np.round(np.log2(largest_per_smallest_volume_ratio) * 1.5))
    expected_ordinates = np.arange(-1, final_scale + 2, dtype=float)
    expected_radii = 1.5 * 2 ** (expected_ordinates / 1.5 / 3.0)

    assert len(result["lumen_radius_microns"]) == 26
    npt.assert_allclose(result["lumen_radius_microns"], expected_radii)


def test_direct_and_resumable_hessian_energy_match(tmp_path):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, :, 3] = 1.0
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
            "approximating_PSF": False,
        }
    )

    direct = SLAVVProcessor().calculate_energy_field(image, params)
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="energy")
    resumable = energy_module.calculate_energy_field_resumable(
        image,
        params,
        run_context.stage("energy"),
        get_chunking_lattice,
    )

    npt.assert_allclose(direct["energy"], resumable["energy"])
    npt.assert_array_equal(direct["scale_indices"], resumable["scale_indices"])
    assert resumable["energy_origin"] == direct["energy_origin"]


def test_direct_and_resumable_hessian_paper_projection_match(tmp_path):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, :, 3] = 1.0
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "energy_projection_mode": "paper",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
            "approximating_PSF": False,
        }
    )

    direct = SLAVVProcessor().calculate_energy_field(image, params)
    run_context = RunContext(run_dir=tmp_path / "run-paper", target_stage="energy")
    resumable = energy_module.calculate_energy_field_resumable(
        image,
        params,
        run_context.stage("energy"),
        get_chunking_lattice,
    )

    assert "energy_4d" not in direct
    assert "energy_4d" not in resumable
    npt.assert_allclose(direct["energy"], resumable["energy"])
    npt.assert_array_equal(direct["scale_indices"], resumable["scale_indices"])
    assert resumable["energy_origin"] == direct["energy_origin"]


def test_resumable_energy_can_store_large_arrays_in_zarr(tmp_path):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, :, 3] = 1.0
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "energy_storage_format": "zarr",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
            "approximating_PSF": False,
            "return_all_scales": True,
        }
    )

    direct = SLAVVProcessor().calculate_energy_field(image, params)
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="energy")
    resumable = energy_module.calculate_energy_field_resumable(
        image,
        params,
        run_context.stage("energy"),
        get_chunking_lattice,
    )

    stage_dir = run_context.stage("energy").stage_dir
    assert (stage_dir / "best_energy.zarr").exists()
    assert (stage_dir / "best_scale.zarr").exists()
    assert (stage_dir / "energy_4d.zarr").exists()
    assert not (stage_dir / "best_energy.npy").exists()
    npt.assert_allclose(direct["energy"], resumable["energy"])
    npt.assert_array_equal(direct["scale_indices"], resumable["scale_indices"])
    npt.assert_allclose(direct["energy_4d"], resumable["energy_4d"])
    assert resumable["energy_origin"] == direct["energy_origin"]


def test_resumable_energy_zarr_storage_requires_optional_dependency(monkeypatch, tmp_path):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "energy_storage_format": "zarr",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
            "approximating_PSF": False,
        }
    )
    monkeypatch.setattr(energy_backends, "zarr", None)
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="energy")

    with pytest.raises(RuntimeError, match="slavv\\[zarr\\]"):
        energy_module.calculate_energy_field_resumable(
            image,
            params,
            run_context.stage("energy"),
            get_chunking_lattice,
        )
