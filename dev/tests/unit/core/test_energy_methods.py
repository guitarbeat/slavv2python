"""Consolidated tests for alternative energy methods."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from source.core import SLAVVProcessor
from source.core import energy as energy_module
from source.core._energy import backends as energy_backends
from source.runtime import RunContext
from source.utils import get_chunking_lattice


class _FakeSitkImage:
    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array, dtype=np.float32)
        self.spacing = None

    def SetSpacing(self, spacing: tuple[float, float, float]) -> None:  # noqa: N802
        self.spacing = spacing


class _FakeHessianRecursiveGaussianImageFilter:
    def __init__(self):
        self.sigma = 1.0
        self.normalize_across_scale = False

    def SetSigma(self, sigma: float) -> None:  # noqa: N802
        self.sigma = sigma

    def SetNormalizeAcrossScale(self, normalize: bool) -> None:  # noqa: N802
        self.normalize_across_scale = normalize

    def Execute(self, image: _FakeSitkImage) -> _FakeSitkImage:  # noqa: N802
        response = image.array * np.float32(1.0 + self.sigma / 10.0)
        output = _FakeSitkImage(response)
        output.spacing = image.spacing
        return output


class _FakeObjectnessMeasureImageFilter:
    def __init__(self):
        self.object_dimension = None
        self.bright_object = True
        self.scale_objectness_measure = False

    def SetObjectDimension(self, object_dimension: int) -> None:  # noqa: N802
        self.object_dimension = object_dimension

    def SetBrightObject(self, bright_object: bool) -> None:  # noqa: N802
        self.bright_object = bright_object

    def SetScaleObjectnessMeasure(self, scale_objectness_measure: bool) -> None:  # noqa: N802
        self.scale_objectness_measure = scale_objectness_measure

    def Execute(self, image: _FakeSitkImage) -> _FakeSitkImage:  # noqa: N802
        response = np.abs(image.array)
        output = _FakeSitkImage(response)
        output.spacing = image.spacing
        return output


class _FakeSimpleITK:
    @staticmethod
    def GetImageFromArray(array: np.ndarray) -> _FakeSitkImage:  # noqa: N802
        return _FakeSitkImage(array)

    @staticmethod
    def GetArrayFromImage(image: _FakeSitkImage) -> np.ndarray:  # noqa: N802
        return image.array

    @staticmethod
    def HessianRecursiveGaussianImageFilter(  # noqa: N802
    ) -> _FakeHessianRecursiveGaussianImageFilter:
        return _FakeHessianRecursiveGaussianImageFilter()

    @staticmethod
    def ObjectnessMeasureImageFilter(  # noqa: N802
    ) -> _FakeObjectnessMeasureImageFilter:
        return _FakeObjectnessMeasureImageFilter()


class _FakeCuPyCuda:
    @staticmethod
    def is_available() -> bool:
        return True


class _FakeCuPy:
    float32 = np.float32
    cuda = _FakeCuPyCuda()

    @staticmethod
    def asarray(array: np.ndarray, dtype=None) -> np.ndarray:
        return np.asarray(array, dtype=dtype)

    @staticmethod
    def asnumpy(array: np.ndarray) -> np.ndarray:
        return np.asarray(array)


class _FakeCuPyNdimage:
    @staticmethod
    def gaussian_filter(image: np.ndarray, sigma, order) -> np.ndarray:
        return np.asarray(image, dtype=np.float32)


def _fake_cupy_energy_scale(
    image: np.ndarray,
    sigma_object: np.ndarray,
    sigma_background: np.ndarray | None,
    spherical_to_annular_ratio: float,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
) -> np.ndarray:
    energy = np.zeros(image.shape, dtype=np.float32)
    center = tuple(int(axis // 2) for axis in image.shape)
    energy[center] = energy_sign
    return energy


@pytest.mark.parametrize("method", ["frangi", "sato"])
def test_alternative_energy_methods(method):
    """Test that Frangi and Sato energy methods produce valid output."""
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

    assert result["energy"].shape == image.shape
    assert result["scale_indices"].shape == image.shape
    assert result["energy"][4, 4, 4] < 0


def test_default_hessian_energy_produces_negative_tubular_response():
    """Default Hessian energy should mark the vessel center as a local minimum."""
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0

    proc = SLAVVProcessor()
    params = {
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
    }
    result = proc.calculate_energy_field(image, params)

    assert result["energy"][4, 4, 4] < 0
    assert result["energy"][4, 4, 4] == result["energy"].min()
    assert result["energy_origin"] == "python_native_hessian"


def test_simpleitk_energy_method_requires_optional_dependency(monkeypatch):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    params = {
        "energy_method": "simpleitk_objectness",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
    }
    monkeypatch.setattr(energy_backends, "sitk", None)

    with pytest.raises(RuntimeError, match="slavv\\[sitk\\]"):
        SLAVVProcessor().calculate_energy_field(image, params)


def test_simpleitk_energy_method_produces_expected_shape_and_sign(monkeypatch):
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0
    params = {
        "energy_method": "simpleitk_objectness",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
    }
    monkeypatch.setattr(energy_backends, "sitk", _FakeSimpleITK)

    result = SLAVVProcessor().calculate_energy_field(image, params)

    assert result["energy"].shape == image.shape
    assert result["scale_indices"].shape == image.shape
    assert result["energy"][4, 4, 4] < 0


def test_simpleitk_direct_and_resumable_paths_match(monkeypatch, tmp_path):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, :, 3] = 1.0
    params = {
        "energy_method": "simpleitk_objectness",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
        "return_all_scales": True,
    }
    monkeypatch.setattr(energy_backends, "sitk", _FakeSimpleITK)

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
    npt.assert_allclose(direct["energy_4d"], resumable["energy_4d"])


def test_simpleitk_energy_method_preserves_axis_alignment_for_anisotropic_spacing(monkeypatch):
    image = np.zeros((5, 7, 9), dtype=np.float32)
    image[2, :, 4] = 1.0
    params = {
        "energy_method": "simpleitk_objectness",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
        "microns_per_voxel": [2.0, 1.0, 3.0],
    }
    monkeypatch.setattr(energy_backends, "sitk", _FakeSimpleITK)

    result = SLAVVProcessor().calculate_energy_field(image, params)

    assert result["energy"].shape == image.shape
    assert result["energy"][2, 3, 4] < 0
    assert result["energy"][2, 1, 4] < 0
    assert result["energy"][0, 0, 0] == 0.0


def test_cupy_energy_method_requires_optional_dependency(monkeypatch):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    params = {
        "energy_method": "cupy_hessian",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
    }
    monkeypatch.setattr(energy_backends, "cp", None)
    monkeypatch.setattr(energy_backends, "cupy_ndimage", None)

    with pytest.raises(RuntimeError, match="CuPy"):
        SLAVVProcessor().calculate_energy_field(image, params)


def test_cupy_energy_method_produces_expected_shape_and_sign(monkeypatch):
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0
    params = {
        "energy_method": "cupy_hessian",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
        "approximating_PSF": False,
    }
    monkeypatch.setattr(energy_backends, "cp", _FakeCuPy)
    monkeypatch.setattr(energy_backends, "cupy_ndimage", _FakeCuPyNdimage)
    monkeypatch.setattr(energy_backends, "_cupy_matlab_hessian_energy", _fake_cupy_energy_scale)

    result = SLAVVProcessor().calculate_energy_field(image, params)

    assert result["energy"].shape == image.shape
    assert result["scale_indices"].shape == image.shape
    assert result["energy"][4, 4, 4] < 0


def test_cupy_direct_and_resumable_paths_match(monkeypatch, tmp_path):
    image = np.zeros((7, 7, 7), dtype=np.float32)
    image[3, :, 3] = 1.0
    params = {
        "energy_method": "cupy_hessian",
        "radius_of_smallest_vessel_in_microns": 1.0,
        "radius_of_largest_vessel_in_microns": 2.0,
        "scales_per_octave": 1.0,
        "approximating_PSF": False,
        "return_all_scales": True,
    }
    monkeypatch.setattr(energy_backends, "cp", _FakeCuPy)
    monkeypatch.setattr(energy_backends, "cupy_ndimage", _FakeCuPyNdimage)
    monkeypatch.setattr(energy_backends, "_cupy_matlab_hessian_energy", _fake_cupy_energy_scale)

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
    npt.assert_allclose(direct["energy_4d"], resumable["energy_4d"])


def test_anisotropic_structuring_element():
    strel_iso = energy_module.spherical_structuring_element(1, np.array([1.0, 1.0, 1.0]))
    strel_aniso = energy_module.spherical_structuring_element(1, np.array([1.0, 1.0, 2.0]))
    assert strel_iso.shape == (3, 3, 3)
    assert strel_iso[1, 1, 2]
    assert not strel_aniso[1, 1, 2]

