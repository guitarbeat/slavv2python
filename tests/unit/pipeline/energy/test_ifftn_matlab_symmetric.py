"""Unit tests for accelerated _ifftn_matlab_symmetric in energy filtering.

Verifies:
1. Exact mathematical parity across even, odd, and anisotropic 3D volumes.
2. Index cache memoization across calls for identical shapes.
3. Multi-worker vs single-worker numerical equivalence.
4. Safe in-place overwrite mode (overwrite_input=True).
5. Self-partner voxel identification (DC and Nyquist frequencies).
"""

from __future__ import annotations

import numpy as np

from slavv_python.pipeline.energy.matlab_energy_filter_v200 import (
    _SYMMETRIC_PARTNER_CACHE,
    _get_symmetric_partner_indices,
    _ifftn_matlab_symmetric,
)


def _reference_ifftn_matlab_symmetric(spectrum: np.ndarray) -> np.ndarray:
    """Original un-cached, full-broadcast reference implementation."""
    spectrum_arr = np.asarray(spectrum)
    ny, nx, nz = spectrum_arr.shape
    symmetric_spectrum = np.array(spectrum_arr, copy=True)

    p_idx_y = (-np.arange(ny)) % ny
    p_idx_x = (-np.arange(nx)) % nx
    p_idx_z = (-np.arange(nz)) % nz

    lin_y = np.arange(ny)
    lin_x = np.arange(nx)
    lin_z = np.arange(nz)

    plin = p_idx_y[:, None, None] + ny * p_idx_x[None, :, None] + ny * nx * p_idx_z[None, None, :]
    lin = lin_y[:, None, None] + ny * lin_x[None, :, None] + ny * nx * lin_z[None, None, :]

    mask = lin > plin
    self_partner_mask = lin == plin

    if np.any(mask):
        flat_mask = mask.ravel(order="F")
        masked_lin_flat = np.where(flat_mask)[0]

        iy = masked_lin_flat % ny
        tmp = masked_lin_flat // ny
        ix = tmp % nx
        iz = tmp // nx

        py = (-iy) % ny
        px = (-ix) % nx
        pz = (-iz) % nz

        symmetric_spectrum[iy, ix, iz] = np.conj(spectrum_arr[py, px, pz])

    if np.any(self_partner_mask):
        symmetric_spectrum[self_partner_mask] = symmetric_spectrum[self_partner_mask].real

    return np.fft.ifftn(symmetric_spectrum).real


def test_ifftn_matlab_symmetric_parity_across_shapes():
    """Verify bit-level parity between accelerated and reference implementation."""
    test_shapes = [
        (16, 16, 16),
        (17, 19, 21),  # Odd dimensions
        (32, 16, 8),   # Powers of two
        (64, 48, 32),  # Anisotropic
    ]

    rng = np.random.default_rng(42)
    for shape in test_shapes:
        spec = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        ref_out = _reference_ifftn_matlab_symmetric(spec)
        acc_out_single = _ifftn_matlab_symmetric(spec, workers=1)
        acc_out_multi = _ifftn_matlab_symmetric(spec, workers=-1)
        acc_out_inplace = _ifftn_matlab_symmetric(spec.copy(), workers=1, overwrite_input=True)

        # Delta should be at or near floating-point machine epsilon (<= 1e-15)
        np.testing.assert_allclose(ref_out, acc_out_single, atol=1e-15, rtol=1e-15)
        np.testing.assert_allclose(ref_out, acc_out_multi, atol=1e-15, rtol=1e-15)
        np.testing.assert_allclose(ref_out, acc_out_inplace, atol=1e-15, rtol=1e-15)


def test_symmetric_partner_indices_cache_hit():
    """Ensure partner index cache returns identical references without recomputation."""
    shape = (24, 28, 30)
    dest1, src1, self1 = _get_symmetric_partner_indices(shape)
    assert shape in _SYMMETRIC_PARTNER_CACHE

    dest2, src2, self2 = _get_symmetric_partner_indices(shape)
    assert dest1 is dest2
    assert src1 is src2
    assert self1 is self2


def test_self_partner_analytic_counts():
    """Verify self-partner index count matches analytic count (2^k where k = even axes)."""
    # All even dimensions -> 2^3 = 8 self-partner frequencies
    _, _, self_even = _get_symmetric_partner_indices((16, 16, 16))
    assert self_even[0].size == 8

    # Two even, one odd -> 2^2 = 4 self-partner frequencies
    _, _, self_mixed = _get_symmetric_partner_indices((16, 16, 15))
    assert self_mixed[0].size == 4

    # All odd dimensions -> only DC frequency (0, 0, 0) is self-partner
    _, _, self_odd = _get_symmetric_partner_indices((15, 17, 19))
    assert self_odd[0].size == 1
    assert self_odd[0][0] == 0
    assert self_odd[1][0] == 0
    assert self_odd[2][0] == 0
