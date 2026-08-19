"""Compare v2 Python Energy lattices vs original MATLAB ``get_energy_V202``.

Isolation only: names the leftover helper/oracle split (lattice/params vs
chunk body). Does **not** change production chunking, emit an Energy unlock,
or relaunch a writer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from slavv_python.analytics.parity.proof.stretch import (
    INTERPRET_INCOMPLETE_INFRA,
    StretchStatus,
)
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    get_chunking_lattice_v190,
)
from slavv_python.pipeline.energy.stretch.chunk_isolation import build_octave_chunk_lattice

logger = logging.getLogger(__name__)

EXPECTED_PYTHON_OCTAVE2_CHUNKS = 75
EXPECTED_MATLAB_OCTAVE2_CHUNKS = 726
CROP_YXZ = (256, 256, 64)
CROP_ZYX = (64, 256, 256)
_WORST_RESOLUTION_TO_DOWNSAMPLE = 1.0 / 2.5

INTERPRET_LATTICE_OR_PARAMS = (
    "Lattices or params differ → named source is helper/oracle lattice or "
    "settings, not merge. Still blocked_float_path."
)
INTERPRET_BODY = (
    "Lattices and params match → residual is helper body vs original MATLAB "
    "chunk math on the same lattice; still blocked_float_path; do not relaunch E14."
)

_STATUS_BLOCKED = StretchStatus.BLOCKED_FLOAT_PATH.value


@dataclass(frozen=True)
class LatticeRecord:
    """One octave's ``get_chunking_lattice_v190`` result."""

    octave: int
    rf: tuple[int, int, int]
    approx_size: tuple[int, int, int]
    lattice_dimensions: tuple[int, int, int]
    number_of_chunks: int
    frame: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "octave": self.octave,
            "rf": list(self.rf),
            "approx_size": list(self.approx_size),
            "lattice_dimensions": list(self.lattice_dimensions),
            "number_of_chunks": self.number_of_chunks,
            "frame": self.frame,
        }


@dataclass(frozen=True)
class ParamFieldCompare:
    """One dest vs oracle scalar/vector compare."""

    name: str
    equal: bool
    dest: Any
    oracle: Any

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "equal": self.equal,
            "dest": _jsonable(self.dest),
            "oracle": _jsonable(self.oracle),
        }


def matlab_round_positive(values: np.ndarray) -> np.ndarray:
    """MATLAB ``round`` for non-negative values (half away from zero)."""
    result: np.ndarray = np.floor(np.asarray(values, dtype=float) + 0.5)
    return result


def matlab_derived_scales_per_octave(radii: np.ndarray) -> float:
    """``get_energy_V202``: ``log(2)/log(r2/r1)/3``."""
    radii_f = np.asarray(radii, dtype=np.float64).reshape(-1)
    if radii_f.size < 2:
        raise ValueError("need at least two lumen radii to derive scales_per_octave")
    return float(np.log(2.0) / np.log(radii_f[1] / radii_f[0]) / 3.0)


def matlab_h5_size_from_h5py_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """MATLAB ``Dataspace.Size`` from h5py C-order shape (axes reversed)."""
    return tuple(int(v) for v in reversed(shape))


def lattice_from_rf(
    *,
    size_of_image: np.ndarray,
    rf: np.ndarray,
    microns: np.ndarray,
    max_voxels: float,
    octave: int,
    frame: str,
) -> LatticeRecord:
    """``approx_size = round(size ./ rf)`` then ``get_chunking_lattice_v190``."""
    size_f = np.asarray(size_of_image, dtype=float).reshape(3)
    rf_f = np.maximum(np.asarray(rf, dtype=float).reshape(3), 1.0)
    microns_f = np.asarray(microns, dtype=float).reshape(3)
    approx_size = matlab_round_positive(size_f / rf_f)
    microns_per_pixel = microns_f * rf_f
    voxel_aspect = 1.0 / microns_per_pixel
    dims, n_chunks = get_chunking_lattice_v190(voxel_aspect, float(max_voxels), approx_size)
    dim_list = [int(v) for v in np.asarray(dims).tolist()]
    rf_list = [int(v) for v in rf_f.tolist()]
    approx_list = [int(v) for v in approx_size.tolist()]
    return LatticeRecord(
        octave=int(octave),
        rf=(rf_list[0], rf_list[1], rf_list[2]),
        approx_size=(approx_list[0], approx_list[1], approx_list[2]),
        lattice_dimensions=(dim_list[0], dim_list[1], dim_list[2]),
        number_of_chunks=int(n_chunks),
        frame=frame,
    )


def python_lattices_from_config(config: dict[str, Any]) -> list[LatticeRecord]:
    """Production v2 lattices (ZYX working image, MATLAB YXZ lattice)."""
    octave_at_scales = np.asarray(config["octave_at_scales"])
    records: list[LatticeRecord] = []
    for octave in np.unique(octave_at_scales):
        built = build_octave_chunk_lattice(config, int(octave))
        rf_matlab = (built.rf_zyx[1], built.rf_zyx[2], built.rf_zyx[0])
        image_shape = np.asarray(config["image_shape"], dtype=float)
        matlab_image_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
        rf_m = np.array([built.rf_zyx[1], built.rf_zyx[2], built.rf_zyx[0]], dtype=float)
        approx = matlab_round_positive(matlab_image_shape / np.maximum(rf_m, 1.0))
        approx_l = [int(v) for v in approx.tolist()]
        records.append(
            LatticeRecord(
                octave=built.octave,
                rf=rf_matlab,
                approx_size=(approx_l[0], approx_l[1], approx_l[2]),
                lattice_dimensions=built.lattice_dimensions_yxz,
                number_of_chunks=built.number_of_chunks,
                frame="python_yxz",
            )
        )
    return records


def matlab_formula_lattices(
    *,
    size_of_image_yxz: np.ndarray,
    radii: np.ndarray,
    microns_yxz: np.ndarray,
    max_voxels: float,
    scales_per_octave: float | None = None,
) -> list[LatticeRecord]:
    """Lattices from ``get_energy_V202.m`` ~109-153 (oracle settings, MATLAB YXZ)."""
    radii_f = np.asarray(radii, dtype=np.float64).reshape(-1)
    microns_f = np.asarray(microns_yxz, dtype=float).reshape(3)
    size_f = np.asarray(size_of_image_yxz, dtype=float).reshape(3)
    spo = (
        float(scales_per_octave)
        if scales_per_octave is not None
        else matlab_derived_scales_per_octave(radii_f)
    )
    n_scales = int(radii_f.size)
    scale_subscripts: np.ndarray = np.arange(1, n_scales + 1, dtype=float)
    octave_at_scales = np.ceil(scale_subscripts / spo / 3.0).astype(np.int32)
    octave_range = np.unique(octave_at_scales)
    worst = float(_WORST_RESOLUTION_TO_DOWNSAMPLE)
    max_oct = int(octave_range.max())
    rf_by_octave: np.ndarray = np.zeros((max_oct, 3), dtype=float)
    for current_octave in octave_range:
        smallest = min(
            n_scales,
            int(np.floor((int(current_octave) - 1) * spo * 3.0)) + 1,
        )
        resolutions = np.minimum(
            microns_f / float(radii_f[smallest - 1]),
            np.full(3, worst, dtype=float),
        )
        rf_by_octave[int(current_octave) - 1, :] = matlab_round_positive(worst / resolutions)
    rf_matrix = np.stack([rf_by_octave[int(o) - 1] for o in octave_range], axis=0)
    _unique_rf, inverse = np.unique(rf_matrix, axis=0, return_inverse=True)
    ia_last: np.ndarray = np.zeros(int(_unique_rf.shape[0]), dtype=int)
    for idx, row_id in enumerate(inverse):
        ia_last[int(row_id)] = int(idx)
    octave_range_kept = octave_range[ia_last]
    records: list[LatticeRecord] = []
    for current_octave in octave_range_kept:
        rf = np.maximum(rf_by_octave[int(current_octave) - 1], 1.0)
        records.append(
            lattice_from_rf(
                size_of_image=size_f,
                rf=rf,
                microns=microns_f,
                max_voxels=max_voxels,
                octave=int(current_octave),
                frame="matlab_yxz",
            )
        )
    return records


def record_at_octave(records: list[LatticeRecord], octave: int) -> LatticeRecord | None:
    """Return the lattice whose octave id equals ``octave``."""
    for record in records:
        if record.octave == int(octave):
            return record
    return None


def values_equal(left: Any, right: Any) -> bool:
    """Scalar or ndarray equality for isolation compares."""
    left_arr = np.asarray(left)
    right_arr = np.asarray(right)
    if left_arr.shape != right_arr.shape:
        return False
    if left_arr.dtype.kind in "fc" or right_arr.dtype.kind in "fc":
        return bool(np.array_equal(left_arr.astype(np.float64), right_arr.astype(np.float64)))
    return bool(np.array_equal(left_arr, right_arr))


def _allclose(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return False
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    return left_arr.shape == right_arr.shape and bool(np.allclose(left_arr, right_arr))


def compare_param_fields(
    dest: dict[str, Any],
    oracle: dict[str, Any],
) -> list[ParamFieldCompare]:
    """Dest vs oracle Energy settings (radii, microns, PSF, max_voxels, ratios)."""
    pairs = (
        ("lumen_radius_in_microns_range", dest.get("lumen_radius_microns"), oracle.get("radii")),
        (
            "microns_per_voxel_raw",
            dest.get("microns_per_voxel_raw"),
            oracle.get("microns_per_voxel"),
        ),
        (
            "microns_per_voxel_working",
            dest.get("microns_per_voxel_working"),
            oracle.get("microns_per_voxel"),
        ),
        (
            "pixels_per_sigma_PSF",
            dest.get("pixels_per_sigma_PSF"),
            oracle.get("pixels_per_sigma_psf"),
        ),
        (
            "pixels_per_sigma_PSF_yxz",
            dest.get("pixels_per_sigma_PSF_yxz"),
            oracle.get("pixels_per_sigma_psf"),
        ),
        ("max_voxels", dest.get("max_voxels"), oracle.get("max_voxels")),
        (
            "gaussian_to_ideal_ratio",
            dest.get("gaussian_to_ideal_ratio"),
            oracle.get("gaussian_to_ideal_ratio"),
        ),
        (
            "spherical_to_annular_ratio",
            dest.get("spherical_to_annular_ratio"),
            oracle.get("spherical_to_annular_ratio"),
        ),
        (
            "scales_per_octave",
            dest.get("scales_per_octave"),
            oracle.get("scales_per_octave_derived"),
        ),
    )
    fields = [
        ParamFieldCompare(name=name, equal=values_equal(left, right), dest=left, oracle=right)
        for name, left, right in pairs
    ]
    dest_radii = dest.get("lumen_radius_microns")
    oracle_radii = oracle.get("radii")
    fields.append(
        ParamFieldCompare(
            name="lumen_radius_allclose",
            equal=_allclose(dest_radii, oracle_radii),
            dest=dest_radii,
            oracle=oracle_radii,
        )
    )
    dest_psf_yxz = dest.get("pixels_per_sigma_PSF_yxz")
    oracle_psf = oracle.get("pixels_per_sigma_psf")
    for idx, field in enumerate(fields):
        if field.name == "pixels_per_sigma_PSF_yxz":
            fields[idx] = ParamFieldCompare(
                name=field.name,
                equal=_allclose(dest_psf_yxz, oracle_psf),
                dest=dest_psf_yxz,
                oracle=oracle_psf,
            )
    return fields


_CORE_PARAM_NAMES = frozenset(
    {
        "microns_per_voxel_raw",
        "pixels_per_sigma_PSF_yxz",
        "max_voxels",
        "gaussian_to_ideal_ratio",
        "spherical_to_annular_ratio",
        "scales_per_octave",
        "lumen_radius_allclose",
    }
)


def zyx_to_yxz(values: np.ndarray) -> np.ndarray:
    """Permute a ZYX vector to MATLAB YXZ."""
    arr = np.asarray(values, dtype=np.float64).reshape(3)
    result: np.ndarray = arr[[1, 2, 0]]
    return result


def lattices_match_by_rf(
    python_lattices: list[LatticeRecord],
    matlab_lattices: list[LatticeRecord],
) -> bool:
    """True when each rf tuple has the same chunk count and lattice dims."""
    python_by_rf = {
        record.rf: (record.number_of_chunks, record.lattice_dimensions)
        for record in python_lattices
    }
    matlab_by_rf = {
        record.rf: (record.number_of_chunks, record.lattice_dimensions)
        for record in matlab_lattices
    }
    return python_by_rf == matlab_by_rf


def interpret_lattice_params(
    *,
    params_equal: bool,
    python_octave2: LatticeRecord | None,
    matlab_octave2: LatticeRecord | None,
    lattices_match_by_rf: bool | None = None,
    params_core_equal: bool | None = None,
) -> str:
    """Name lattice/params vs same-lattice helper body. Never stretch_complete."""
    python_n = python_octave2.number_of_chunks if python_octave2 is not None else None
    matlab_n = matlab_octave2.number_of_chunks if matlab_octave2 is not None else None
    index_match = (
        python_octave2 is not None
        and matlab_octave2 is not None
        and python_octave2.number_of_chunks == matlab_octave2.number_of_chunks
        and python_octave2.lattice_dimensions == matlab_octave2.lattice_dimensions
        and python_octave2.rf == matlab_octave2.rf
    )
    rf_ok = index_match if lattices_match_by_rf is None else bool(lattices_match_by_rf)
    core_ok = params_equal if params_core_equal is None else bool(params_core_equal)
    if core_ok and rf_ok:
        return INTERPRET_BODY
    logger.info(
        "lattice/params isolation: params_equal=%s rf_match=%s python_oct2=%s matlab_oct2=%s",
        params_equal,
        rf_ok,
        python_n,
        matlab_n,
    )
    return INTERPRET_LATTICE_OR_PARAMS


def isolation_payload(
    *,
    param_fields: list[ParamFieldCompare],
    python_lattices: list[LatticeRecord],
    matlab_lattices: list[LatticeRecord],
    extra: dict[str, Any] | None = None,
    params_core_equal: bool | None = None,
) -> dict[str, Any]:
    """Assemble scratch JSON; status class stays blocked_float_path."""
    python_oct2 = record_at_octave(python_lattices, 2)
    matlab_oct2 = record_at_octave(matlab_lattices, 2)
    params_equal = all(field.equal for field in param_fields)
    rf_match = lattices_match_by_rf(python_lattices, matlab_lattices)
    by_name = {field.name: field.equal for field in param_fields}
    core_ok = (
        bool(params_core_equal)
        if params_core_equal is not None
        else all(by_name.get(name, False) for name in _CORE_PARAM_NAMES)
    )
    interpretation = interpret_lattice_params(
        params_equal=params_equal,
        python_octave2=python_oct2,
        matlab_octave2=matlab_oct2,
        lattices_match_by_rf=rf_match,
        params_core_equal=core_ok,
    )
    payload: dict[str, Any] = {
        "result": "ok",
        "status_class": _STATUS_BLOCKED,
        "interpretation": interpretation,
        "isolation_only": True,
        "not_stretch_success": True,
        "stretch_complete": False,
        "params_equal": params_equal,
        "params_core_equal": core_ok,
        "lattices_match_by_rf": rf_match,
        "param_fields": [field.to_dict() for field in param_fields],
        "python_lattices": [record.to_dict() for record in python_lattices],
        "matlab_formula_lattices": [record.to_dict() for record in matlab_lattices],
        "octave2": {
            "python": None if python_oct2 is None else python_oct2.to_dict(),
            "matlab_formula": None if matlab_oct2 is None else matlab_oct2.to_dict(),
            "expected_python_chunks": EXPECTED_PYTHON_OCTAVE2_CHUNKS,
            "expected_matlab_chunks": EXPECTED_MATLAB_OCTAVE2_CHUNKS,
            "python_matches_expected_75": (
                python_oct2 is not None
                and python_oct2.number_of_chunks == EXPECTED_PYTHON_OCTAVE2_CHUNKS
            ),
            "matlab_matches_expected_726": (
                matlab_oct2 is not None
                and matlab_oct2.number_of_chunks == EXPECTED_MATLAB_OCTAVE2_CHUNKS
            ),
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, (bool, int, float, str)):
            return item
        return float(item) if np.issubdtype(array.dtype, np.number) else str(item)
    return array.astype(np.float64, copy=False).tolist()


__all__ = [
    "CROP_YXZ",
    "CROP_ZYX",
    "EXPECTED_MATLAB_OCTAVE2_CHUNKS",
    "EXPECTED_PYTHON_OCTAVE2_CHUNKS",
    "INTERPRET_BODY",
    "INTERPRET_INCOMPLETE_INFRA",
    "INTERPRET_LATTICE_OR_PARAMS",
    "LatticeRecord",
    "ParamFieldCompare",
    "compare_param_fields",
    "interpret_lattice_params",
    "isolation_payload",
    "lattice_from_rf",
    "lattices_match_by_rf",
    "matlab_derived_scales_per_octave",
    "matlab_formula_lattices",
    "matlab_h5_size_from_h5py_shape",
    "matlab_round_positive",
    "python_lattices_from_config",
    "record_at_octave",
    "values_equal",
    "zyx_to_yxz",
]
