import numpy as np
import numpy.testing as npt

from slavv.core import SLAVVProcessor
from slavv.core import energy as energy_module
from slavv.runtime import RunContext
from slavv.utils import get_chunking_lattice, validate_parameters


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
    finite_mask = np.isfinite(direct["energy"])
    npt.assert_array_equal(
        direct["scale_indices"][finite_mask],
        resumable["scale_indices"][finite_mask],
    )
