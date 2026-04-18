from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
from scipy.ndimage import gaussian_filter

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
_CUPY_PARAMETER_WARNING = (
    "CuPy Hessian backend accelerates the Gaussian/Hessian derivative work; "
    "remaining eigendecomposition and aggregation stay on CPU in this v1 path."
)


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


def _require_simpleitk_backend() -> Any:
    """Return the SimpleITK module or raise a clear install error."""
    if sitk is None:
        raise RuntimeError(_SIMPLEITK_INSTALL_HINT)
    return sitk


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
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (1, 0, 0), microns_per_voxel
    )
    grad_x = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 1, 0), microns_per_voxel
    )
    grad_z = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 0, 1), microns_per_voxel
    )
    h_yy = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (2, 0, 0), microns_per_voxel
    )
    h_xx = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 2, 0), microns_per_voxel
    )
    h_zz = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 0, 2), microns_per_voxel
    )
    h_yx = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (1, 1, 0), microns_per_voxel
    )
    h_xz = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 1, 1), microns_per_voxel
    )
    h_yz = _matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (1, 0, 1), microns_per_voxel
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
            image_gpu, sigma=tuple(sigma_background), order=order
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
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (1, 0, 0), microns_per_voxel
    )
    grad_x = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 1, 0), microns_per_voxel
    )
    grad_z = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 0, 1), microns_per_voxel
    )
    h_yy = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (2, 0, 0), microns_per_voxel
    )
    h_xx = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 2, 0), microns_per_voxel
    )
    h_zz = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 0, 2), microns_per_voxel
    )
    h_yx = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (1, 1, 0), microns_per_voxel
    )
    h_xz = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (0, 1, 1), microns_per_voxel
    )
    h_yz = _cupy_matched_filter_derivative(
        image, sigma_object, sigma_background, spherical_to_annular_ratio, (1, 0, 1), microns_per_voxel
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
    return cast("np.ndarray", energy_sign * np.transpose(response, (1, 2, 0)))
