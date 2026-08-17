"""E12: py37 worker ``matlab.double`` list-marshalling bit-identity."""

from __future__ import annotations

import numpy as np


def _e12_payload() -> np.ndarray:
    rng = np.random.default_rng(12)
    arr = np.asfortranarray(rng.standard_normal((5, 4, 3), dtype=np.float64))
    arr[0, 0, 0] = np.inf
    arr[1, 0, 0] = -np.inf
    arr[2, 0, 0] = np.nan
    arr[0, 1, 0] = -0.0
    return arr


def _assert_payload_identity(original: np.ndarray, restored: np.ndarray) -> None:
    assert restored.shape == original.shape
    assert restored.dtype == np.float64
    finite = np.isfinite(original) & np.isfinite(restored)
    assert int(np.count_nonzero(finite)) >= 1
    assert np.array_equal(original[finite], restored[finite])
    assert np.array_equal(np.isnan(original), np.isnan(restored))
    assert np.array_equal(np.isposinf(original), np.isposinf(restored))
    assert np.array_equal(np.isneginf(original), np.isneginf(restored))


def _worker_float_list_roundtrip(array: np.ndarray) -> np.ndarray:
    """Python-side npy → Fortran ravel → float list → reshape (no matlab.double)."""
    arr = np.asfortranarray(np.asarray(array, dtype=np.float64))
    flat = [float(v) for v in np.ravel(arr, order="F")]
    return np.reshape(np.asarray(flat, dtype=np.float64), arr.shape, order="F")


def test_e12_fortran_float_list_roundtrip_without_engine() -> None:
    original = _e12_payload()
    restored = _worker_float_list_roundtrip(original)
    _assert_payload_identity(original, restored)


def test_e12_worker_matlab_double_roundtrip_bit_identical(stretch_py37_worker) -> None:
    original = _e12_payload()
    restored = stretch_py37_worker.roundtrip_float64(original)
    _assert_payload_identity(original, restored)
    finite = np.isfinite(original)
    n_finite = int(np.count_nonzero(finite))
    n_bit_identical = int(np.count_nonzero(original[finite] == restored[finite]))
    assert n_bit_identical == n_finite, (
        f"E12 finite bit-identity {n_bit_identical}/{n_finite} (marshalling is not bit-identical)"
    )
