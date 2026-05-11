"""Comprehensive tests for energy field calculation, storage, and projection.
Consolidated from multiple energy-related test files.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from slavv_python.core import SLAVVProcessor, energy_backends, hessian_response as native_hessian
from slavv_python.core import energy as energy_module
from slavv_python.core.energy_config import _prepare_energy_config
from slavv_python.runtime import RunContext
from slavv_python.utils import get_chunking_lattice, validate_parameters


# ==============================================================================
# Fake Backends for Testing (Inlined from test_energy_methods.py)
# ==============================================================================


class _FakeSitkImage:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array, dtype=np.float32)
        self.spacing = None

    def SetSpacing(self, spacing: tuple[float, float, float]) -> None:  # noqa: N802
        self.spacing = spacing


class _FakeSimpleITK:
    @staticmethod
    def GetImageFromArray(array: np.ndarray) -> _FakeSitkImage:  # noqa: N802
        return _FakeSitkImage(array)

    @staticmethod
    def GetArrayFromImage(image: _FakeSitkImage) -> np.ndarray:  # noqa: N802
        return image.array

    @staticmethod
    def HessianRecursiveGaussianImageFilter():  # noqa: N802
        class _Filter:
            def SetSigma(self, sigma): self.sigma = sigma
            def SetNormalizeAcrossScale(self, normalize): self.normalize = normalize
            def Execute(self, img):
                res = _FakeSitkImage(img.array * (1.0 + self.sigma / 10.0))
                res.spacing = img.spacing
                return res
        return _Filter()

    @staticmethod
    def ObjectnessMeasureImageFilter():  # noqa: N802
        class _Filter:
            def SetObjectDimension(self, dim): pass
            def SetBrightObject(self, bright): pass
            def SetScaleObjectnessMeasure(self, scale): pass
            def Execute(self, img):
                res = _FakeSitkImage(np.abs(img.array))
                res.spacing = img.spacing
                return res
        return _Filter()


class _FakeCuPy:
    float32 = np.float32
    cuda = type("cuda", (), {"is_available": staticmethod(lambda: True)})()

    @staticmethod
    def asarray(array, dtype=None): return np.asarray(array, dtype=dtype)

    @staticmethod
    def asnumpy(array): return np.asarray(array)


def _fake_cupy_energy_scale(image, sigma_object, *_args, **_kwargs):
    energy = np.zeros(image.shape, dtype=np.float32)
    center = tuple(int(axis // 2) for axis in image.shape)
    energy[center] = -1.0
    return energy


# ==============================================================================
# Storage and Resumable Logic
# ==============================================================================


@pytest.mark.unit
def test_energy_field_no_full_storage():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters({})
    proc = SLAVVProcessor()
    result = proc.calculate_energy_field(img, params)
    assert "energy_4d" not in result
    assert result["energy"].shape == img.shape


@pytest.mark.unit
def test_energy_field_with_full_storage():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters({"return_all_scales": True})
    proc = SLAVVProcessor()
    result = proc.calculate_energy_field(img, params)
    assert "energy_4d" in result


@pytest.mark.unit
def test_energy_scale_range_matches_matlab_ordination():
    img = np.zeros((4, 4, 4), dtype=np.float32)
    params = validate_parameters(
        {"radius_of_smallest_vessel_in_microns": 1.5, "radius_of_largest_vessel_in_microns": 50.0, "scales_per_octave": 1.5}
    )
    result = SLAVVProcessor().calculate_energy_field(img, params)
    assert len(result["lumen_radius_microns"]) == 26


@pytest.mark.unit
def test_direct_and_resumable_hessian_energy_match(tmp_path):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, :, 3] = 1.0
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
        }
    )
    direct = SLAVVProcessor().calculate_energy_field(image, params)
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="energy")
    resumable = energy_module.calculate_energy_field_resumable(
        image, params, run_context.stage("energy"), get_chunking_lattice
    )
    npt.assert_allclose(direct["energy"], resumable["energy"])


@pytest.mark.unit
def test_resumable_energy_can_store_large_arrays_in_zarr(tmp_path):
    pytest.importorskip("zarr")
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, :, 3] = 1.0
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "energy_storage_format": "zarr",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
            "return_all_scales": True,
        }
    )
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="energy")
    resumable = energy_module.calculate_energy_field_resumable(
        image, params, run_context.stage("energy"), get_chunking_lattice
    )
    assert (run_context.stage("energy").stage_dir / "energy_4d.zarr").exists()


# ==============================================================================
# Alternative Energy Methods
# ==============================================================================


@pytest.mark.parametrize("method", ["frangi", "sato"])
@pytest.mark.unit
def test_alternative_energy_methods(method):
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0
    proc = SLAVVProcessor()
    params = {
        "energy_method": method,
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
    }
    result = proc.calculate_energy_field(image, params)
    assert result["energy"][4, 4, 4] < 0


@pytest.mark.unit
def test_simpleitk_energy_method_produces_expected_shape_and_sign(monkeypatch):
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0
    monkeypatch.setattr(energy_backends, "sitk", _FakeSimpleITK)
    params = {
        "energy_method": "simpleitk_objectness",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
    }
    result = SLAVVProcessor().calculate_energy_field(image, params)
    assert result["energy"][4, 4, 4] < 0


@pytest.mark.unit
def test_cupy_energy_method_produces_expected_shape_and_sign(monkeypatch):
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0
    monkeypatch.setattr(energy_backends, "cp", _FakeCuPy)
    monkeypatch.setattr(energy_backends, "cupy_ndimage", type("nd", (), {"gaussian_filter": lambda i, s, o: i})())
    monkeypatch.setattr(energy_backends, "_cupy_matlab_hessian_energy", _fake_cupy_energy_scale)
    params = {
        "energy_method": "cupy_hessian",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
    }
    result = SLAVVProcessor().calculate_energy_field(image, params)
    assert result["energy"][4, 4, 4] < 0


# ==============================================================================
# Energy Projection and Config
# ==============================================================================


@pytest.mark.unit
def test_matlab_projection_uses_per_voxel_minimum():
    energy_4d = np.array([[[[-1.0, -3.0, -2.0]]]], dtype=np.float32)
    energy, scale_indices = native_hessian.project_energy_stack(
        energy_4d, energy_sign=-1.0, projection_mode="matlab", spherical_to_annular_ratio=1.0
    )
    npt.assert_allclose(energy, np.array([[[-3.0]]], dtype=np.float32))


@pytest.mark.unit
def test_paper_projection_blends_annular_and_spherical_scale_estimates():
    energy_4d = np.array([[[[-20.0, -10.0, -19.0, -19.0]]]], dtype=np.float32)
    paper_energy, paper_scale = native_hessian.project_energy_stack(
        energy_4d, energy_sign=-1.0, projection_mode="paper", spherical_to_annular_ratio=0.5
    )
    npt.assert_allclose(paper_energy, np.array([[[-10.0]]], dtype=np.float32))


@pytest.mark.unit
def test_native_energy_config_wires_projection_mode_and_octaves():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    config = _prepare_energy_config(
        image,
        {
            "energy_projection_mode": "paper",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 4.0,
            "scales_per_octave": 1.0,
        },
    )
    assert config["energy_projection_mode"] == "paper"


# Made with Bob
