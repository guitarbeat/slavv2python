"""Compare stretch helper body vs original MATLAB ``get_energy_V202`` chunk math.

Isolation only. Does not unlock Energy or change production chunking.
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
    _matlab_coarse_local_slices,
)

logger = logging.getLogger(__name__)

# Octave-2 chunk 0 write window on crop_M (ZYX [0:21, 0:51, 0:51] → YXZ).
DEFAULT_CHUNK0_OFFSETS_YXZ = (0, 0, 0)
DEFAULT_CHUNK0_WRITE_YXZ = (51, 51, 21)
DEFAULT_CHUNK0_STRIDES_YXZ = (3, 3, 1)

INTERPRET_LOCAL_RANGES_MATCH = (
    "local_ranges match MATLAB floor/ceil; residual is not the coarse-slice formula."
)
INTERPRET_LOCAL_RANGES_DIFFER = (
    "local_ranges differ from MATLAB floor/ceil — named helper-body source."
)
INTERPRET_CLAMP_NOT_TINY_ULP = (
    "Helper clamps energy>=0 to 0; MATLAB active path does not. "
    "That is not the 1e-10 negative-energy ULP class."
)
INTERPRET_INPUT_WINDOW = "Python TIFF window vs oracle HDF5 window on the same lattice hit."

_STATUS_BLOCKED = StretchStatus.BLOCKED_FLOAT_PATH.value


@dataclass(frozen=True)
class LocalRangeCompare:
    """One-axis MATLAB 1-based range vs Python 0-based slice."""

    offset: int
    write_count: int
    stride: int
    matlab_start_1based: int
    matlab_stop_1based: int
    python_start_0based: int
    python_stop_0based: int
    equal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "offset": self.offset,
            "write_count": self.write_count,
            "stride": self.stride,
            "matlab_start_1based": self.matlab_start_1based,
            "matlab_stop_1based": self.matlab_stop_1based,
            "python_start_0based": self.python_start_0based,
            "python_stop_0based": self.python_stop_0based,
            "equal": self.equal,
        }


def matlab_local_range_1based(offset: int, write_count: int, stride: int) -> tuple[int, int]:
    """``get_energy_V202``: ``1+floor(off/rf) : 1+ceil((off+n-1)/rf)`` inclusive."""
    start = 1 + int(np.floor(offset / stride))
    stop = 1 + int(np.ceil((offset + write_count - 1) / stride))
    return int(start), int(stop)


def python_local_slice(offset: int, write_count: int, stride: int, padded_extent: int) -> slice:
    """Production ``_matlab_coarse_local_slices`` for one axis."""
    y, _x, _z = _matlab_coarse_local_slices(
        offsets=(offset, 0, 0),
        write_counts=(write_count, 1, 1),
        strides=(stride, 1, 1),
        padded_shape=(padded_extent, 8, 8),
    )
    return y


def compare_local_range(
    offset: int, write_count: int, stride: int, *, padded_extent: int = 4096
) -> LocalRangeCompare:
    """True when Python slice equals MATLAB 1-based inclusive range."""
    m0, m1 = matlab_local_range_1based(offset, write_count, stride)
    sl = python_local_slice(offset, write_count, stride, padded_extent)
    py0 = int(sl.start or 0)
    py1 = int(sl.stop)
    equal = py0 == (m0 - 1) and py1 == m1
    return LocalRangeCompare(
        offset=int(offset),
        write_count=int(write_count),
        stride=int(stride),
        matlab_start_1based=m0,
        matlab_stop_1based=m1,
        python_start_0based=py0,
        python_stop_0based=py1,
        equal=equal,
    )


def helper_clamps_nonnegative() -> bool:
    """``stretch_energy_chunk_v202.m`` does ``energy(energy>=0)=0``; MATLAB min-path does not."""
    return True


def downsample_count(read_count: int, rf: int) -> int:
    """MATLAB ``1 + floor((reading_counts-1)./rf)`` == Python strided length."""
    return 1 + int(np.floor((int(read_count) - 1) / int(rf)))


def strided_length(count: int, stride: int) -> int:
    """``len(range(0, count, stride))``."""
    if count <= 0 or stride <= 0:
        return 0
    return len(range(0, int(count), int(stride)))


def downsample_matches_strided(read_count: int, rf: int) -> bool:
    """True when MATLAB downsample count equals Python ``[start:start+n:rf]`` length."""
    return downsample_count(read_count, rf) == strided_length(read_count, rf)


def default_production_local_compares() -> list[LocalRangeCompare]:
    """local_ranges for crop octave-2 chunk 0 write window (Y, X, Z)."""
    return [
        compare_local_range(offset, write, stride)
        for offset, write, stride in zip(
            DEFAULT_CHUNK0_OFFSETS_YXZ,
            DEFAULT_CHUNK0_WRITE_YXZ,
            DEFAULT_CHUNK0_STRIDES_YXZ,
            strict=True,
        )
    ]


def h5py_c_order_to_matlab_yxz(volume: np.ndarray) -> np.ndarray:
    """Reverse all axes: h5py C-order of MATLAB column-major ``[Y, X, Z]``."""
    arr = np.asarray(volume)
    result: np.ndarray = np.transpose(arr, tuple(range(arr.ndim - 1, -1, -1)))
    return result


def python_strided_chunk_yxz(
    image_zyx: np.ndarray,
    *,
    z_start: int,
    y_start: int,
    x_start: int,
    z_count: int,
    y_count: int,
    x_count: int,
    rf_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Production TIFF window: strided ZYX slice, then transpose to MATLAB YXZ."""
    stride_z, stride_y, stride_x = (int(v) for v in rf_zyx)
    chunk_zyx = np.asarray(image_zyx)[
        z_start : z_start + z_count : stride_z,
        y_start : y_start + y_count : stride_y,
        x_start : x_start + x_count : stride_x,
    ]
    result: np.ndarray = np.transpose(chunk_zyx, (1, 2, 0)).astype(np.float64, copy=False)
    return result


def matlab_h52mat_chunk_yxz(
    volume_yxz: np.ndarray,
    *,
    y_start_1based: int,
    x_start_1based: int,
    z_start_1based: int,
    y_read_count: int,
    x_read_count: int,
    z_read_count: int,
    rf_yxz: tuple[int, int, int],
) -> np.ndarray:
    """``h52mat`` window: 1-based starts, downsample counts, stride ``rf``."""
    rf_y, rf_x, rf_z = (int(v) for v in rf_yxz)
    y0 = int(y_start_1based) - 1
    x0 = int(x_start_1based) - 1
    z0 = int(z_start_1based) - 1
    y_n = downsample_count(int(y_read_count), rf_y)
    x_n = downsample_count(int(x_read_count), rf_x)
    z_n = downsample_count(int(z_read_count), rf_z)
    result: np.ndarray = np.asarray(volume_yxz, dtype=np.float64)[
        y0 : y0 + y_n * rf_y : rf_y,
        x0 : x0 + x_n * rf_x : rf_x,
        z0 : z0 + z_n * rf_z : rf_z,
    ]
    return result


@dataclass(frozen=True)
class WindowCompare:
    """TIFF vs HDF5 input window on one lattice hit."""

    equal: bool
    python_shape: tuple[int, ...]
    matlab_shape: tuple[int, ...]
    n_diff: int
    max_abs_delta: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "equal": self.equal,
            "python_shape": list(self.python_shape),
            "matlab_shape": list(self.matlab_shape),
            "n_diff": self.n_diff,
            "max_abs_delta": self.max_abs_delta,
        }


def compare_input_windows(python_yxz: np.ndarray, matlab_yxz: np.ndarray) -> WindowCompare:
    """Bit-compare Python TIFF window vs MATLAB HDF5 window in YXZ."""
    python_arr = np.asarray(python_yxz, dtype=np.float64)
    matlab_arr = np.asarray(matlab_yxz, dtype=np.float64)
    if python_arr.shape != matlab_arr.shape:
        return WindowCompare(
            equal=False,
            python_shape=tuple(int(v) for v in python_arr.shape),
            matlab_shape=tuple(int(v) for v in matlab_arr.shape),
            n_diff=-1,
            max_abs_delta=-1.0,
        )
    n_diff = int(np.count_nonzero(python_arr != matlab_arr))
    delta = float(np.max(np.abs(python_arr - matlab_arr))) if python_arr.size else 0.0
    return WindowCompare(
        equal=n_diff == 0,
        python_shape=tuple(int(v) for v in python_arr.shape),
        matlab_shape=tuple(int(v) for v in matlab_arr.shape),
        n_diff=n_diff,
        max_abs_delta=delta,
    )


def interpret_helper_body(
    *,
    local_ranges_equal: bool,
    input_windows_equal: bool | None,
) -> str:
    """Name leftover. Never stretch_complete."""
    if not local_ranges_equal:
        return INTERPRET_LOCAL_RANGES_DIFFER
    if input_windows_equal is False:
        return (
            INTERPRET_INPUT_WINDOW
            + " Windows differ — named source is TIFF vs HDF5 / h52mat, not merge."
        )
    if input_windows_equal is True:
        return (
            INTERPRET_LOCAL_RANGES_MATCH
            + " Input windows match. Residual is filter/interp3 args or MATLAB "
            "engine vs original batch internals. " + INTERPRET_CLAMP_NOT_TINY_ULP
        )
    return INTERPRET_LOCAL_RANGES_MATCH + " " + INTERPRET_CLAMP_NOT_TINY_ULP


def isolation_payload(
    *,
    local_compares: list[LocalRangeCompare],
    input_windows_equal: bool | None = None,
    window_compare: WindowCompare | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scratch JSON; status stays blocked_float_path."""
    local_ok = all(item.equal for item in local_compares)
    windows_equal = input_windows_equal
    if windows_equal is None and window_compare is not None:
        windows_equal = window_compare.equal
    payload: dict[str, Any] = {
        "result": "ok",
        "status_class": _STATUS_BLOCKED,
        "interpretation": interpret_helper_body(
            local_ranges_equal=local_ok,
            input_windows_equal=windows_equal,
        ),
        "isolation_only": True,
        "not_stretch_success": True,
        "stretch_complete": False,
        "local_ranges_equal": local_ok,
        "helper_clamps_nonnegative": helper_clamps_nonnegative(),
        "matlab_active_path_clamps_nonnegative": False,
        "input_windows_equal": windows_equal,
        "local_compares": [item.to_dict() for item in local_compares],
        "downsample_count_matches_strided": all(
            downsample_matches_strided(count, rf) for count, rf in ((51, 3), (21, 1), (51, 1))
        ),
    }
    if window_compare is not None:
        payload["window_compare"] = window_compare.to_dict()
    if extra:
        payload.update(extra)
    logger.info("helper-body isolation: local_ok=%s windows=%s", local_ok, windows_equal)
    return payload


__all__ = [
    "DEFAULT_CHUNK0_OFFSETS_YXZ",
    "DEFAULT_CHUNK0_STRIDES_YXZ",
    "DEFAULT_CHUNK0_WRITE_YXZ",
    "INTERPRET_CLAMP_NOT_TINY_ULP",
    "INTERPRET_INCOMPLETE_INFRA",
    "INTERPRET_INPUT_WINDOW",
    "INTERPRET_LOCAL_RANGES_DIFFER",
    "INTERPRET_LOCAL_RANGES_MATCH",
    "LocalRangeCompare",
    "WindowCompare",
    "compare_input_windows",
    "compare_local_range",
    "default_production_local_compares",
    "downsample_count",
    "downsample_matches_strided",
    "h5py_c_order_to_matlab_yxz",
    "helper_clamps_nonnegative",
    "interpret_helper_body",
    "isolation_payload",
    "matlab_h52mat_chunk_yxz",
    "matlab_local_range_1based",
    "python_local_slice",
    "python_strided_chunk_yxz",
    "strided_length",
]
