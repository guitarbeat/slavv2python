"""
Energy field calculations for SLAVV.
Includes Hessian-based vessel enhancement (Frangi/Sato) and Numba-accelerated gradient computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.ndimage import gaussian_filter

from slavv.core import energy_storage as _energy_storage

if TYPE_CHECKING:
    from slavv.runtime import StageController

try:
    from skimage.filters import frangi, sato
except ImportError:
    try:
        from skimage.filters import frangi

        sato = None
    except ImportError:
        frangi = None
        sato = None

try:
    import SimpleITK as sitk  # noqa: N813
except ImportError:
    sitk = None

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cupy_ndimage
except ImportError:
    cp = None
    cupy_ndimage = None

try:
    import zarr
except ImportError:
    zarr = None

# Optional Numba acceleration for hot gradient helpers.
try:
    from numba import njit
except ImportError:
    njit = None

_NUMBA_AVAILABLE = njit is not None

logger = logging.getLogger(__name__)

_SIMPLEITK_INSTALL_HINT = (
    "SimpleITK is required for energy_method='simpleitk_objectness'. "
    'Install it with `pip install -e ".[sitk]"`, `pip install slavv[sitk]`, '
    "or `pip install SimpleITK>=2.4.0`."
)
_CUPY_INSTALL_HINT = (
    "CuPy with a matching CUDA build is required for energy_method='cupy_hessian'. "
    "Install a compatible package such as `pip install cupy-cuda12x` or add an "
    "appropriate CuPy wheel for the target GPU runtime."
)
_ZARR_INSTALL_HINT = (
    "Zarr is required for energy_storage_format='zarr'. "
    'Install it with `pip install -e ".[zarr]"`, `pip install slavv[zarr]`, '
    "or `pip install zarr>=2.12.0`."
)
_NUMBA_FAILURE_MESSAGE = (
    "Numba gradient acceleration is unavailable in this environment; "
    "falling back to the pure-Python helpers."
)
_CUPY_PARAMETER_WARNING = (
    "CuPy Hessian backend accelerates the Gaussian/Hessian derivative work; "
    "remaining eigendecomposition and aggregation stay on CPU in this v1 path."
)


def _compute_gradient_impl_python(energy, pos_int, microns_per_voxel):
    """Compute local energy gradient via central differences."""
    grad = np.zeros(3)

    pos_y, pos_x, pos_z = pos_int
    shape_y, shape_x, shape_z = energy.shape

    if shape_y < 3 or shape_x < 3 or shape_z < 3:
        return np.zeros(3)
    if pos_y < 1:
        pos_y = 1
    elif pos_y > shape_y - 2:
        pos_y = shape_y - 2

    if pos_x < 1:
        pos_x = 1
    elif pos_x > shape_x - 2:
        pos_x = shape_x - 2

    if pos_z < 1:
        pos_z = 1
    elif pos_z > shape_z - 2:
        pos_z = shape_z - 2

    grad[0] = (energy[pos_y + 1, pos_x, pos_z] - energy[pos_y - 1, pos_x, pos_z]) / (
        2.0 * microns_per_voxel[0]
    )
    grad[1] = (energy[pos_y, pos_x + 1, pos_z] - energy[pos_y, pos_x - 1, pos_z]) / (
        2.0 * microns_per_voxel[1]
    )
    grad[2] = (energy[pos_y, pos_x, pos_z + 1] - energy[pos_y, pos_x, pos_z - 1]) / (
        2.0 * microns_per_voxel[2]
    )

    return grad


def _compute_gradient_fast_python(energy, p0, p1, p2, inv_mpv_2x):
    """Optimized gradient computation avoiding position-array allocations."""
    s0, s1, s2 = energy.shape

    if s0 < 3 or s1 < 3 or s2 < 3:
        return np.zeros(3)

    if p0 < 1:
        p0 = 1
    elif p0 > s0 - 2:
        p0 = s0 - 2

    if p1 < 1:
        p1 = 1
    elif p1 > s1 - 2:
        p1 = s1 - 2

    if p2 < 1:
        p2 = 1
    elif p2 > s2 - 2:
        p2 = s2 - 2

    grad = np.empty(3)
    grad[0] = (energy[p0 + 1, p1, p2] - energy[p0 - 1, p1, p2]) * inv_mpv_2x[0]
    grad[1] = (energy[p0, p1 + 1, p2] - energy[p0, p1 - 1, p2]) * inv_mpv_2x[1]
    grad[2] = (energy[p0, p1, p2 + 1] - energy[p0, p1, p2 - 1]) * inv_mpv_2x[2]

    return grad


if _NUMBA_AVAILABLE:
    _compute_gradient_impl_numba = cast("Any", njit(cache=False)(_compute_gradient_impl_python))
    _compute_gradient_fast_numba = cast("Any", njit(cache=False)(_compute_gradient_fast_python))
else:
    _compute_gradient_impl_numba = None
    _compute_gradient_fast_numba = None

_NUMBA_ACCELERATION_ENABLED = _NUMBA_AVAILABLE


def compute_gradient_impl(
    energy: np.ndarray,
    pos_int: np.ndarray | tuple[int, int, int],
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Compute a local energy gradient with optional Numba acceleration."""
    global _NUMBA_ACCELERATION_ENABLED

    if _NUMBA_ACCELERATION_ENABLED and _compute_gradient_impl_numba is not None:
        try:
            return cast(
                "np.ndarray", _compute_gradient_impl_numba(energy, pos_int, microns_per_voxel)
            )
        except Exception as exc:  # pragma: no cover - depends on local numba build
            logger.warning("%s Detail: %s", _NUMBA_FAILURE_MESSAGE, exc)
            _NUMBA_ACCELERATION_ENABLED = False

    return cast("np.ndarray", _compute_gradient_impl_python(energy, pos_int, microns_per_voxel))


def compute_gradient_fast(
    energy: np.ndarray,
    p0: int,
    p1: int,
    p2: int,
    inv_mpv_2x: np.ndarray,
) -> np.ndarray:
    """Compute a local energy gradient without allocating a position array."""
    global _NUMBA_ACCELERATION_ENABLED

    if _NUMBA_ACCELERATION_ENABLED and _compute_gradient_fast_numba is not None:
        try:
            return cast("np.ndarray", _compute_gradient_fast_numba(energy, p0, p1, p2, inv_mpv_2x))
        except Exception as exc:  # pragma: no cover - depends on local numba build
            logger.warning("%s Detail: %s", _NUMBA_FAILURE_MESSAGE, exc)
            _NUMBA_ACCELERATION_ENABLED = False

    return cast("np.ndarray", _compute_gradient_fast_python(energy, p0, p1, p2, inv_mpv_2x))


def is_numba_acceleration_enabled() -> bool:
    """Return whether gradient helpers are currently using Numba-compiled paths."""
    return _NUMBA_ACCELERATION_ENABLED


# --- Helper Functions ---


def spherical_structuring_element(radius: int, microns_per_voxel: np.ndarray) -> np.ndarray:
    """
    Create a 3D spherical structuring element accounting for voxel spacing.

    The ``radius`` is interpreted in voxel units along the smallest physical
    dimension.  For anisotropic data the resulting footprint becomes an
    ellipsoid so that voxels within ``radius`` microns of the origin are
    included regardless of spacing differences along ``y, x, z``.
    """
    microns_per_voxel = np.asarray(microns_per_voxel, dtype=float)
    r_phys = float(radius) * microns_per_voxel.min()
    ranges = [
        np.arange(-int(np.ceil(r_phys / s)), int(np.ceil(r_phys / s)) + 1)
        for s in microns_per_voxel
    ]
    yy, xx, zz = np.meshgrid(*ranges, indexing="ij")
    dist2 = (
        (yy * microns_per_voxel[0]) ** 2
        + (xx * microns_per_voxel[1]) ** 2
        + (zz * microns_per_voxel[2]) ** 2
    )
    return (dist2 <= r_phys**2).astype(bool)  # type: ignore[no-any-return]


def _matlab_lumen_radius_range(
    radius_smallest: float, radius_largest: float, scales_per_octave: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-aligned scale ordinates and lumen radii.

    MATLAB defines ``scales_per_octave`` per doubling of the vessel radius cubed
    and pads the requested range by one scale on each side so 4D extrema can be
    detected during vertex extraction.
    """
    largest_per_smallest_volume_ratio = (radius_largest / radius_smallest) ** 3
    final_scale = int(np.round(np.log2(largest_per_smallest_volume_ratio) * scales_per_octave))
    scale_ordinates: np.ndarray = np.arange(-1, final_scale + 2, dtype=float)
    scale_factors: np.ndarray = np.power(2.0, scale_ordinates / scales_per_octave / 3.0)
    return scale_ordinates, radius_smallest * scale_factors


def _matched_filter_derivative(
    image: np.ndarray,
    sigma_object: np.ndarray,
    sigma_background: np.ndarray | None,
    spherical_to_annular_ratio: float,
    order: tuple[int, int, int],
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Evaluate a matched-kernel derivative in physical units."""
    derivative = gaussian_filter(image, sigma=tuple(sigma_object), order=order)
    if sigma_background is not None and spherical_to_annular_ratio < 1.0:
        background = gaussian_filter(image, sigma=tuple(sigma_background), order=order)
        derivative = spherical_to_annular_ratio * derivative + (
            1.0 - spherical_to_annular_ratio
        ) * (derivative - background)
    scale = np.prod(np.power(microns_per_voxel, order))
    if scale > 0:
        derivative = derivative / scale
    return derivative.astype(np.float32, copy=False)  # type: ignore[no-any-return]


def _matlab_hessian_energy(
    image: np.ndarray,
    sigma_object: np.ndarray,
    sigma_background: np.ndarray | None,
    spherical_to_annular_ratio: float,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
) -> np.ndarray:
    """Approximate MATLAB's curvature-weighted Hessian energy response."""
    grad_y = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 0, 0),
        microns_per_voxel,
    )
    grad_x = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 1, 0),
        microns_per_voxel,
    )
    grad_z = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 0, 1),
        microns_per_voxel,
    )
    h_yy = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (2, 0, 0),
        microns_per_voxel,
    )
    h_xx = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 2, 0),
        microns_per_voxel,
    )
    h_zz = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 0, 2),
        microns_per_voxel,
    )
    h_yx = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 1, 0),
        microns_per_voxel,
    )
    h_xz = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 1, 1),
        microns_per_voxel,
    )
    h_yz = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 0, 1),
        microns_per_voxel,
    )

    laplacian: np.ndarray = h_yy + h_xx + h_zz
    if energy_sign < 0:
        valid: np.ndarray = laplacian < 0
        energy = np.full(image.shape, np.inf, dtype=np.float32)
    else:
        valid = laplacian > 0
        energy = np.full(image.shape, -np.inf, dtype=np.float32)
    if not np.any(valid):
        return cast("np.ndarray", energy)

    grad_valid = np.stack([grad_y[valid], grad_x[valid], grad_z[valid]], axis=1).astype(np.float64)
    hessian_valid = np.empty((grad_valid.shape[0], 3, 3), dtype=np.float64)
    hessian_valid[:, 0, 0] = h_yy[valid]
    hessian_valid[:, 0, 1] = h_yx[valid]
    hessian_valid[:, 0, 2] = h_yz[valid]
    hessian_valid[:, 1, 0] = h_yx[valid]
    hessian_valid[:, 1, 1] = h_xx[valid]
    hessian_valid[:, 1, 2] = h_xz[valid]
    hessian_valid[:, 2, 0] = h_yz[valid]
    hessian_valid[:, 2, 1] = h_xz[valid]
    hessian_valid[:, 2, 2] = h_zz[valid]

    eigvals, eigvecs = np.linalg.eigh(hessian_valid)
    projections = np.einsum("ni,nik->nk", grad_valid, eigvecs)
    denom = np.where(np.abs(eigvals) > 1e-12, eigvals, np.where(eigvals >= 0, 1e-12, -1e-12))
    principal_energy = eigvals * np.exp(-0.5 * np.square(projections / denom))

    if energy_sign < 0:
        principal_energy[:, 2] = np.minimum(principal_energy[:, 2], 0.0)
        energy_valid = principal_energy.sum(axis=1)
        energy_valid[~np.isfinite(energy_valid)] = np.inf
        energy_valid[energy_valid >= 0] = np.inf
    else:
        principal_energy[:, 0] = np.maximum(principal_energy[:, 0], 0.0)
        energy_valid = principal_energy.sum(axis=1)
        energy_valid[~np.isfinite(energy_valid)] = -np.inf
        energy_valid[energy_valid <= 0] = -np.inf

    energy[valid] = energy_valid.astype(np.float32, copy=False)
    return cast("np.ndarray", energy)


def _warn_simpleitk_parameter_mismatches(params: dict[str, Any]) -> None:
    """Warn when SimpleITK backend ignores MATLAB-style tuning controls."""
    if (
        not bool(params.get("approximating_PSF", True))
        or float(params.get("spherical_to_annular_ratio", 1.0)) != 1.0
        or float(params.get("gaussian_to_ideal_ratio", 1.0)) != 1.0
    ):
        logger.warning(
            "SimpleITK objectness backend uses its own objectness path; "
            "approximating_PSF, spherical_to_annular_ratio, and gaussian_to_ideal_ratio "
            "are not applied in this mode."
        )


def _require_cupy_backend() -> Any:
    """Return the CuPy module after validating GPU availability."""
    if cp is None or cupy_ndimage is None:
        raise RuntimeError(_CUPY_INSTALL_HINT)

    try:
        if not bool(cp.cuda.is_available()):
            raise RuntimeError(
                "CuPy is installed but no CUDA-capable NVIDIA GPU is available for "
                "energy_method='cupy_hessian'."
            )
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(
            "CuPy is installed, but CUDA runtime initialization failed for "
            "energy_method='cupy_hessian'."
        ) from exc

    return cp


def _warn_cupy_parameter_mismatches(params: dict[str, Any]) -> None:
    """Warn about the current CuPy backend scope."""
    logger.info(_CUPY_PARAMETER_WARNING)
    if params.get("energy_method") == "cupy_hessian" and params.get("return_all_scales", False):
        logger.debug(
            "CuPy Hessian backend stores per-scale outputs after transferring chunk results back to CPU."
        )


def _require_zarr_backend() -> Any:
    """Return the Zarr module or raise a clear install error."""
    if zarr is None:
        raise RuntimeError(_ZARR_INSTALL_HINT)
    return zarr


def _select_energy_storage_format(config: dict[str, Any], total_voxels: int) -> str:
    """Choose the resumable energy array storage backend."""
    storage_format = _energy_storage.select_energy_storage_format(
        str(config.get("energy_storage_format", "auto")),
        total_voxels=total_voxels,
        max_voxels=int(config["max_voxels"]),
        require_zarr_backend=_require_zarr_backend,
    )
    return cast("str", storage_format)


def _remove_storage_path(path: Any) -> None:
    """Remove a file or directory-backed storage artifact."""
    _energy_storage.remove_storage_path(path)


def _zarr_chunks_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return conservative chunk sizes for energy arrays."""
    chunks = _energy_storage.zarr_chunks_for_shape(shape)
    return cast("tuple[int, ...]", chunks)


def _open_energy_storage_array(
    path: Any,
    *,
    mode: str,
    dtype: Any,
    shape: tuple[int, ...],
    fill_value: float | int | None = None,
    storage_format: str,
) -> Any:
    """Open a resumable energy array in either NPY memmap or Zarr format."""
    return _energy_storage.open_energy_storage_array(
        path,
        mode=mode,
        dtype=dtype,
        shape=shape,
        fill_value=fill_value,
        storage_format=storage_format,
        require_zarr_backend=_require_zarr_backend,
    )


def _cupy_matched_filter_derivative(
    image: np.ndarray,
    sigma_object: np.ndarray,
    sigma_background: np.ndarray | None,
    spherical_to_annular_ratio: float,
    order: tuple[int, int, int],
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Evaluate matched-kernel derivatives with CuPy ndimage kernels."""
    cupy_module = _require_cupy_backend()
    image_gpu = cupy_module.asarray(image, dtype=cupy_module.float32)
    derivative_gpu = cupy_ndimage.gaussian_filter(image_gpu, sigma=tuple(sigma_object), order=order)
    if sigma_background is not None and spherical_to_annular_ratio < 1.0:
        background_gpu = cupy_ndimage.gaussian_filter(
            image_gpu,
            sigma=tuple(sigma_background),
            order=order,
        )
        derivative_gpu = spherical_to_annular_ratio * derivative_gpu + (
            1.0 - spherical_to_annular_ratio
        ) * (derivative_gpu - background_gpu)
    scale = np.prod(np.power(microns_per_voxel, order))
    if scale > 0:
        derivative_gpu = derivative_gpu / scale
    derivative = cupy_module.asnumpy(derivative_gpu)
    return derivative.astype(np.float32, copy=False)  # type: ignore[no-any-return]


def _cupy_matlab_hessian_energy(
    image: np.ndarray,
    sigma_object: np.ndarray,
    sigma_background: np.ndarray | None,
    spherical_to_annular_ratio: float,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
) -> np.ndarray:
    """Approximate MATLAB Hessian energy using CuPy for derivative kernels."""
    grad_y = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 0, 0),
        microns_per_voxel,
    )
    grad_x = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 1, 0),
        microns_per_voxel,
    )
    grad_z = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 0, 1),
        microns_per_voxel,
    )
    h_yy = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (2, 0, 0),
        microns_per_voxel,
    )
    h_xx = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 2, 0),
        microns_per_voxel,
    )
    h_zz = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 0, 2),
        microns_per_voxel,
    )
    h_yx = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 1, 0),
        microns_per_voxel,
    )
    h_xz = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 1, 1),
        microns_per_voxel,
    )
    h_yz = _cupy_matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 0, 1),
        microns_per_voxel,
    )

    laplacian: np.ndarray = h_yy + h_xx + h_zz
    if energy_sign < 0:
        valid: np.ndarray = laplacian < 0
        energy = np.full(image.shape, np.inf, dtype=np.float32)
    else:
        valid = laplacian > 0
        energy = np.full(image.shape, -np.inf, dtype=np.float32)
    if not np.any(valid):
        return cast("np.ndarray", energy)

    grad_valid = np.stack([grad_y[valid], grad_x[valid], grad_z[valid]], axis=1).astype(np.float64)
    hessian_valid = np.empty((grad_valid.shape[0], 3, 3), dtype=np.float64)
    hessian_valid[:, 0, 0] = h_yy[valid]
    hessian_valid[:, 0, 1] = h_yx[valid]
    hessian_valid[:, 0, 2] = h_yz[valid]
    hessian_valid[:, 1, 0] = h_yx[valid]
    hessian_valid[:, 1, 1] = h_xx[valid]
    hessian_valid[:, 1, 2] = h_xz[valid]
    hessian_valid[:, 2, 0] = h_yz[valid]
    hessian_valid[:, 2, 1] = h_xz[valid]
    hessian_valid[:, 2, 2] = h_zz[valid]

    eigvals, eigvecs = np.linalg.eigh(hessian_valid)
    projections = np.einsum("ni,nik->nk", grad_valid, eigvecs)
    denom = np.where(np.abs(eigvals) > 1e-12, eigvals, np.where(eigvals >= 0, 1e-12, -1e-12))
    principal_energy = eigvals * np.exp(-0.5 * np.square(projections / denom))

    if energy_sign < 0:
        principal_energy[:, 2] = np.minimum(principal_energy[:, 2], 0.0)
        energy_valid = principal_energy.sum(axis=1)
        energy_valid[~np.isfinite(energy_valid)] = np.inf
        energy_valid[energy_valid >= 0] = np.inf
    else:
        principal_energy[:, 0] = np.maximum(principal_energy[:, 0], 0.0)
        energy_valid = principal_energy.sum(axis=1)
        energy_valid[~np.isfinite(energy_valid)] = -np.inf
        energy_valid[energy_valid <= 0] = -np.inf

    energy[valid] = energy_valid.astype(np.float32, copy=False)
    return cast("np.ndarray", energy)


def _require_simpleitk_backend() -> Any:
    """Return the SimpleITK module or raise a clear install error."""
    if sitk is None:
        raise RuntimeError(_SIMPLEITK_INSTALL_HINT)
    return sitk


def _simpleitk_objectness_energy(
    image: np.ndarray,
    sigma_world: float,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
) -> np.ndarray:
    """Compute one scale of spacing-aware vessel objectness with SimpleITK."""
    sitk_module = _require_simpleitk_backend()
    sitk_image = sitk_module.GetImageFromArray(np.transpose(image, (2, 0, 1)))
    sitk_image.SetSpacing(
        (
            float(microns_per_voxel[1]),
            float(microns_per_voxel[0]),
            float(microns_per_voxel[2]),
        )
    )

    hessian_filter = sitk_module.HessianRecursiveGaussianImageFilter()
    hessian_filter.SetSigma(sigma_world)
    hessian_filter.SetNormalizeAcrossScale(True)
    hessian_image = hessian_filter.Execute(sitk_image)

    objectness_filter = sitk_module.ObjectnessMeasureImageFilter()
    objectness_filter.SetObjectDimension(1)
    objectness_filter.SetBrightObject(energy_sign < 0)
    objectness_filter.SetScaleObjectnessMeasure(False)
    objectness = objectness_filter.Execute(hessian_image)

    response = sitk_module.GetArrayFromImage(objectness).astype(np.float32, copy=False)
    return cast(
        "np.ndarray",
        energy_sign * np.transpose(response, (1, 2, 0)),
    )


def calculate_energy_field(
    image: np.ndarray, params: dict[str, Any], get_chunking_lattice_func=None
) -> dict[str, Any]:
    """
    Calculate multi-scale energy field using Hessian-based filtering.

    This implements the energy calculation from ``get_energy_V202`` in
    MATLAB, including PSF prefiltering and configurable Gaussian/annular
    ratios. Set ``energy_method='frangi'`` or ``'sato'`` in ``params`` to use
    scikit-image's :func:`~skimage.filters.frangi` or
    :func:`~skimage.filters.sato` vesselness filters as alternative backends.
    """
    logger.info("Calculating energy field")
    image = image.astype(np.float32, copy=False)
    config = _prepare_energy_config(image, params)
    lattice = _energy_lattice(
        image.shape,
        int(config["max_voxels"]),
        int(config["margin"]),
        get_chunking_lattice_func,
    )
    if len(lattice) > 1:
        return _calculate_energy_field_chunked(
            image,
            params,
            config,
            lattice,
            get_chunking_lattice_func,
        )
    energy_3d, scale_indices, energy_4d = _compute_direct_energy_outputs(image, config)
    return _energy_result_payload(config, image.shape, energy_3d, scale_indices, energy_4d)


def _prepare_energy_config(image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute scale and PSF metadata for resumable energy evaluation."""
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    radius_smallest = float(params.get("radius_of_smallest_vessel_in_microns", 1.5))
    radius_largest = float(params.get("radius_of_largest_vessel_in_microns", 50.0))
    scales_per_octave = float(params.get("scales_per_octave", 1.5))
    gaussian_to_ideal_ratio = float(params.get("gaussian_to_ideal_ratio", 1.0))
    spherical_to_annular_ratio = float(params.get("spherical_to_annular_ratio", 1.0))
    approximating_PSF = bool(params.get("approximating_PSF", True))
    energy_sign = float(params.get("energy_sign", -1.0))
    energy_method = params.get("energy_method", "hessian")
    return_all_scales = bool(params.get("return_all_scales", False))
    max_voxels = int(params.get("max_voxels_per_node_energy", 1e5))
    if energy_method == "simpleitk_objectness":
        _require_simpleitk_backend()
        _warn_simpleitk_parameter_mismatches(params)
    if energy_method == "cupy_hessian":
        _require_cupy_backend()
        _warn_cupy_parameter_mismatches(params)

    if approximating_PSF:
        numerical_aperture = params.get("numerical_aperture", 0.95)
        excitation_wavelength = params.get("excitation_wavelength_in_microns", 1.3)
        sample_index_of_refraction = params.get("sample_index_of_refraction", 1.33)
        if numerical_aperture <= 0.7:
            coefficient, exponent = 0.320, 1.0
        else:
            coefficient, exponent = 0.325, 0.91
        microns_per_sigma_PSF = np.array(
            [
                excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
                excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
                excitation_wavelength
                / (2**0.5)
                * 0.532
                / (
                    sample_index_of_refraction
                    - (sample_index_of_refraction**2 - numerical_aperture**2) ** 0.5
                ),
            ],
            dtype=float,
        )
    else:
        microns_per_sigma_PSF = np.zeros(3, dtype=float)

    pixels_per_sigma_PSF = microns_per_sigma_PSF / microns_per_voxel
    scale_ordinates, lumen_radius_microns = _matlab_lumen_radius_range(
        radius_smallest,
        radius_largest,
        scales_per_octave,
    )
    lumen_radius_pixels_axes = lumen_radius_microns[:, None] / microns_per_voxel[None, :]
    lumen_radius_pixels = lumen_radius_pixels_axes.mean(axis=1)

    max_sigma = (lumen_radius_microns[-1] / microns_per_voxel) / max(
        gaussian_to_ideal_ratio,
        1e-12,
    )
    if approximating_PSF:
        max_sigma = np.sqrt(max_sigma**2 + pixels_per_sigma_PSF**2)
    margin = int(np.ceil(np.max(max_sigma)))

    if energy_method == "sato" and sato is None:
        logger.warning(
            "Sato filter unavailable (requires scikit-image>=0.19). Falling back to Hessian."
        )
        energy_method = "hessian"
    if energy_method == "frangi" and frangi is None:
        logger.warning("Frangi filter unavailable. Falling back to Hessian.")
        energy_method = "hessian"

    return {
        "image_shape": tuple(image.shape),
        "image_dtype": str(image.dtype),
        "microns_per_voxel": microns_per_voxel,
        "energy_storage_format": str(params.get("energy_storage_format", "auto")).strip(),
        "gaussian_to_ideal_ratio": gaussian_to_ideal_ratio,
        "spherical_to_annular_ratio": spherical_to_annular_ratio,
        "approximating_PSF": approximating_PSF,
        "energy_sign": energy_sign,
        "energy_method": energy_method,
        "return_all_scales": return_all_scales,
        "max_voxels": max_voxels,
        "margin": margin,
        "scale_ordinates": scale_ordinates,
        "lumen_radius_microns": lumen_radius_microns,
        "lumen_radius_pixels": lumen_radius_pixels,
        "lumen_radius_pixels_axes": lumen_radius_pixels_axes,
        "pixels_per_sigma_PSF": pixels_per_sigma_PSF,
        "microns_per_sigma_PSF": microns_per_sigma_PSF,
    }


def _energy_lattice(
    image_shape: tuple[int, ...],
    max_voxels: int,
    margin: int,
    get_chunking_lattice_func,
) -> list[
    tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], tuple[slice, slice, slice]]
]:
    total_voxels = int(np.prod(image_shape))
    if total_voxels > max_voxels:
        return cast(
            "list[tuple[tuple[slice, slice, slice], tuple[slice, slice, slice], tuple[slice, slice, slice]]]",
            get_chunking_lattice_func(image_shape, max_voxels, margin),
        )
    return [
        (
            (slice(0, image_shape[0]), slice(0, image_shape[1]), slice(0, image_shape[2])),
            (slice(0, image_shape[0]), slice(0, image_shape[1]), slice(0, image_shape[2])),
            (slice(0, image_shape[0]), slice(0, image_shape[1]), slice(0, image_shape[2])),
        )
    ]


def _energy_result_payload(
    config: dict[str, Any],
    image_shape: tuple[int, ...],
    energy_3d: np.ndarray,
    scale_indices: np.ndarray,
    energy_4d: np.ndarray | None = None,
) -> dict[str, Any]:
    result = {
        "energy": energy_3d,
        "scale_indices": scale_indices,
        "lumen_radius_microns": config["lumen_radius_microns"],
        "lumen_radius_pixels": config["lumen_radius_pixels"],
        "lumen_radius_pixels_axes": config["lumen_radius_pixels_axes"],
        "pixels_per_sigma_PSF": config["pixels_per_sigma_PSF"],
        "microns_per_sigma_PSF": config["microns_per_sigma_PSF"],
        "energy_sign": config["energy_sign"],
        "image_shape": image_shape,
    }
    if energy_4d is not None:
        result["energy_4d"] = energy_4d
    return result


def _best_energy_outputs(
    image_shape: tuple[int, ...],
    energy_sign: float,
    n_scales: int,
    return_all_scales: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    fill_value = np.inf if energy_sign < 0 else -np.inf
    energy_3d = np.full(image_shape, fill_value, dtype=np.float32)
    scale_indices: np.ndarray = np.zeros(image_shape, dtype=np.int16)
    energy_4d = np.zeros((*image_shape, n_scales), dtype=np.float32) if return_all_scales else None
    return energy_3d, scale_indices, energy_4d


def _update_best_energy(
    energy_3d: np.ndarray,
    scale_indices: np.ndarray,
    energy_scale: np.ndarray,
    scale_idx: int,
    energy_sign: float,
) -> None:
    mask = energy_scale < energy_3d if energy_sign < 0 else energy_scale > energy_3d
    energy_3d[mask] = energy_scale[mask]
    scale_indices[mask] = scale_idx


def _compute_direct_energy_outputs(
    image: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    n_scales = len(config["lumen_radius_microns"])
    energy_3d, scale_indices, energy_4d = _best_energy_outputs(
        image.shape,
        float(config["energy_sign"]),
        n_scales,
        bool(config["return_all_scales"]),
    )
    for scale_idx in range(n_scales):
        energy_scale = _compute_energy_scale(image, config, scale_idx)
        if energy_4d is not None:
            energy_4d[..., scale_idx] = energy_scale
        _update_best_energy(
            energy_3d,
            scale_indices,
            energy_scale,
            scale_idx,
            float(config["energy_sign"]),
        )
    return energy_3d, scale_indices, energy_4d


def _calculate_energy_field_chunked(
    image: np.ndarray,
    params: dict[str, Any],
    config: dict[str, Any],
    lattice,
    get_chunking_lattice_func,
) -> dict[str, Any]:
    if config["return_all_scales"]:
        n_scales = len(config["lumen_radius_microns"])
        energy_4d = np.zeros((*image.shape, n_scales), dtype=np.float32)
        for chunk_slice, out_slice, inner_slice in lattice:
            chunk_img = image[chunk_slice]
            sub_params = params.copy()
            sub_params["max_voxels_per_node_energy"] = chunk_img.size + 1
            sub_params["return_all_scales"] = True
            chunk_data = calculate_energy_field(chunk_img, sub_params, get_chunking_lattice_func)
            energy_4d[(*out_slice, slice(None))] = chunk_data["energy_4d"][
                (*inner_slice, slice(None))
            ]
        if config["energy_sign"] < 0:
            energy_3d = np.min(energy_4d, axis=3)
            scale_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
        else:
            energy_3d = np.max(energy_4d, axis=3)
            scale_indices = np.argmax(energy_4d, axis=3).astype(np.int16)
        return _energy_result_payload(config, image.shape, energy_3d, scale_indices, energy_4d)

    energy_3d = np.empty(image.shape, dtype=np.float32)
    scale_indices = np.empty(image.shape, dtype=np.int16)
    for chunk_slice, out_slice, inner_slice in lattice:
        chunk_img = image[chunk_slice]
        sub_params = params.copy()
        sub_params["max_voxels_per_node_energy"] = chunk_img.size + 1
        sub_params["return_all_scales"] = False
        chunk_data = calculate_energy_field(chunk_img, sub_params, get_chunking_lattice_func)
        energy_3d[out_slice] = chunk_data["energy"][inner_slice]
        scale_indices[out_slice] = chunk_data["scale_indices"][inner_slice]
    return _energy_result_payload(config, image.shape, energy_3d, scale_indices)


def _compute_energy_scale(image: np.ndarray, config: dict[str, Any], scale_idx: int) -> np.ndarray:
    """Compute a single-scale energy response for a chunk."""
    image = image.astype(np.float32, copy=False)
    energy_method = config["energy_method"]
    energy_sign = config["energy_sign"]
    sigma_scale = config["lumen_radius_microns"][scale_idx] / config["microns_per_voxel"]
    sigma_scale = sigma_scale / max(config["gaussian_to_ideal_ratio"], 1e-12)
    sigma_scale = np.asarray(sigma_scale, dtype=float)

    if config["approximating_PSF"]:
        sigma_object = np.sqrt(sigma_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
    else:
        sigma_object = sigma_scale

    if energy_method in ("frangi", "sato"):
        sigma = float(config["lumen_radius_pixels"][scale_idx])
        if energy_method == "frangi":
            vesselness = frangi(image, sigmas=[sigma], black_ridges=(energy_sign > 0))
        else:
            vesselness = sato(image, sigmas=[sigma], black_ridges=(energy_sign > 0))
        return energy_sign * vesselness.astype(np.float32)  # type: ignore[no-any-return]

    if energy_method == "simpleitk_objectness":
        return _simpleitk_objectness_energy(
            image,
            float(config["lumen_radius_microns"][scale_idx]),
            np.asarray(config["microns_per_voxel"], dtype=float),
            float(energy_sign),
        )
    if energy_method == "cupy_hessian":
        if config["spherical_to_annular_ratio"] < 1.0:
            annular_scale = sigma_scale * 1.5
            if config["approximating_PSF"]:
                sigma_background = np.sqrt(annular_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
            else:
                sigma_background = annular_scale
        else:
            sigma_background = None
        return _cupy_matlab_hessian_energy(
            image,
            sigma_object,
            sigma_background,
            float(config["spherical_to_annular_ratio"]),
            np.asarray(config["microns_per_voxel"], dtype=float),
            float(energy_sign),
        )

    if config["spherical_to_annular_ratio"] < 1.0:
        annular_scale = sigma_scale * 1.5
        if config["approximating_PSF"]:
            sigma_background = np.sqrt(annular_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
        else:
            sigma_background = annular_scale
    else:
        sigma_background = None

    return _matlab_hessian_energy(
        image,
        sigma_object,
        sigma_background,
        config["spherical_to_annular_ratio"],
        config["microns_per_voxel"],
        energy_sign,
    )


def calculate_energy_field_resumable(
    image: np.ndarray,
    params: dict[str, Any],
    stage_controller: StageController,
    get_chunking_lattice_func=None,
) -> dict[str, Any]:
    """Compute energy with resumable chunk/scale units backed by memmaps."""
    config = _prepare_energy_config(image, params)
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "params": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in config.items()
                    if k not in {"image_shape", "image_dtype"}
                },
                "shape": list(config["image_shape"]),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    total_voxels = int(np.prod(image.shape))
    storage_format = _select_energy_storage_format(config, total_voxels)
    lattice = _energy_lattice(
        image.shape,
        int(config["max_voxels"]),
        int(config["margin"]),
        get_chunking_lattice_func,
    )

    energy_suffix = ".zarr" if storage_format == "zarr" else ".npy"
    energy_path = stage_controller.artifact_path(f"best_energy{energy_suffix}")
    scale_path = stage_controller.artifact_path(f"best_scale{energy_suffix}")
    energy4d_path = stage_controller.artifact_path(f"energy_4d{energy_suffix}")
    state = stage_controller.load_state()
    completed_units = set(state.get("completed_units", []))
    if state.get("config_hash") not in (None, config_hash):
        completed_units = set()
        for stale_path in (energy_path, scale_path, energy4d_path):
            _remove_storage_path(stale_path)
    for legacy_path in (
        stage_controller.artifact_path("best_energy.npy"),
        stage_controller.artifact_path("best_scale.npy"),
        stage_controller.artifact_path("energy_4d.npy"),
        stage_controller.artifact_path("best_energy.zarr"),
        stage_controller.artifact_path("best_scale.zarr"),
        stage_controller.artifact_path("energy_4d.zarr"),
    ):
        if legacy_path not in (energy_path, scale_path, energy4d_path):
            _remove_storage_path(legacy_path)

    n_scales = len(config["lumen_radius_microns"])
    total_units = len(lattice) * n_scales
    resumed = bool(completed_units)
    stage_controller.begin(
        detail="Computing resumable energy field",
        units_total=total_units,
        units_completed=len(completed_units),
        substage="scale_chunks",
        resumed=resumed,
    )

    best_energy = _open_energy_storage_array(
        energy_path,
        mode="r+" if energy_path.exists() else "w",
        dtype=np.float32,
        shape=tuple(image.shape),
        fill_value=np.inf if config["energy_sign"] < 0 else -np.inf,
        storage_format=storage_format,
    )

    best_scale = _open_energy_storage_array(
        scale_path,
        mode="r+" if scale_path.exists() else "w",
        dtype=np.int16,
        shape=tuple(image.shape),
        fill_value=0,
        storage_format=storage_format,
    )

    energy_4d = None
    if config["return_all_scales"]:
        energy_4d = _open_energy_storage_array(
            energy4d_path,
            mode="r+" if energy4d_path.exists() else "w",
            dtype=np.float32,
            shape=(*image.shape, n_scales),
            fill_value=0.0,
            storage_format=storage_format,
        )

    for chunk_idx, (chunk_slice, out_slice, inner_slice) in enumerate(lattice):
        chunk_img = image[chunk_slice]
        for scale_idx in range(n_scales):
            unit_id = f"{chunk_idx}:{scale_idx}"
            if unit_id in completed_units:
                continue

            energy_scale = _compute_energy_scale(chunk_img, config, scale_idx)
            chunk_inner = energy_scale[inner_slice]
            target_view = best_energy[out_slice]
            if config["energy_sign"] < 0:
                mask = chunk_inner < target_view
            else:
                mask = chunk_inner > target_view
            target_view[mask] = chunk_inner[mask]
            best_energy[out_slice] = target_view
            scale_view = best_scale[out_slice]
            scale_view[mask] = scale_idx
            best_scale[out_slice] = scale_view
            if energy_4d is not None:
                energy_4d[(*out_slice, scale_idx)] = chunk_inner

            completed_units.add(unit_id)
            state = {
                "config_hash": config_hash,
                "completed_units": sorted(completed_units),
                "total_units": total_units,
                "n_chunks": len(lattice),
                "n_scales": n_scales,
                "storage_format": storage_format,
            }
            stage_controller.save_state(state)
            stage_controller.update(
                units_total=total_units,
                units_completed=len(completed_units),
                detail=(
                    f"Energy volume tile {chunk_idx + 1}/{len(lattice)}, "
                    f"vessel scale {scale_idx + 1}/{n_scales}"
                ),
                substage="scale_chunks",
                resumed=resumed,
            )

    result = {
        "energy": np.asarray(best_energy),
        "scale_indices": np.asarray(best_scale),
        "lumen_radius_microns": config["lumen_radius_microns"],
        "lumen_radius_pixels": config["lumen_radius_pixels"],
        "lumen_radius_pixels_axes": config["lumen_radius_pixels_axes"],
        "pixels_per_sigma_PSF": config["pixels_per_sigma_PSF"],
        "microns_per_sigma_PSF": config["microns_per_sigma_PSF"],
        "energy_sign": config["energy_sign"],
        "image_shape": image.shape,
    }
    if energy_4d is not None:
        result["energy_4d"] = np.asarray(energy_4d)
    return result


__all__ = [
    "calculate_energy_field",
    "compute_gradient_fast",
    "compute_gradient_impl",
    "is_numba_acceleration_enabled",
    "spherical_structuring_element",
]
