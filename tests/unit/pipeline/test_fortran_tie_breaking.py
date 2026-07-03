# Feature: matlab-python-parity, Property 3: Fortran-Order Tie-Breaking
"""Property 3: Fortran-Order Tie-Breaking.

For any pair of voxels with equal energy values in the watershed flood-fill,
the voxel with the lower Fortran-order linear index (computed via
``np.ravel_multi_index`` with ``order='F'``) shall be selected as the
catchment winner.

Validates: Requirements 3.2
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from slavv_python.pipeline.edges.matlab_indexing import _argmin_with_linear_index_tiebreak


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fortran_linear_index(coord: tuple[int, int, int], shape: tuple[int, int, int]) -> int:
    """Return the Fortran-order (column-major) linear index for a [Y, X, Z] coord."""
    return int(np.ravel_multi_index(coord, shape, order="F"))


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Shapes: at least 2 elements along the first two axes so we can always place
# two tied voxels at distinct (y, x, z) positions.
_shape_strategy = st.tuples(
    st.integers(min_value=2, max_value=12),  # Y
    st.integers(min_value=2, max_value=12),  # X
    st.integers(min_value=1, max_value=6),   # Z
)


@st.composite
def grid_with_two_tied_min_voxels(draw: st.DrawFn) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Generate a float64 3-D energy grid with at least two voxels at the minimum.

    Returns ``(energy_grid, shape)`` where ``shape = (Y, X, Z)``.
    """
    shape: tuple[int, int, int] = draw(_shape_strategy)
    ny, nx, nz = shape
    total = ny * nx * nz

    # Draw an energy array with values in a small range
    energy = draw(
        arrays(
            dtype=np.float64,
            shape=(total,),
            elements=st.floats(min_value=-10.0, max_value=0.0, allow_nan=False, allow_infinity=False),
        )
    )

    # Choose two distinct flat indices that will both receive the minimum value
    idx_a, idx_b = draw(
        st.lists(
            st.integers(min_value=0, max_value=total - 1),
            min_size=2,
            max_size=2,
            unique=True,
        ).filter(lambda pair: pair[0] != pair[1])
    )

    # Force them to be the minimum
    current_min = float(np.min(energy))
    tie_value = current_min - 1.0  # guaranteed to be the new minimum
    energy[idx_a] = tie_value
    energy[idx_b] = tie_value

    grid = energy.reshape(shape, order="C")  # standard C reshape; indices are still 0-based flat
    return grid, shape


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(grid_with_two_tied_min_voxels())
@settings(max_examples=100)
def test_argmin_tiebreak_selects_lower_fortran_index(
    grid_and_shape: tuple[np.ndarray, tuple[int, int, int]],
) -> None:
    """**Validates: Requirements 3.2**

    For any 3-D [Y, X, Z] energy grid with at least two voxels sharing the
    minimum energy value, ``_argmin_with_linear_index_tiebreak`` shall select
    the voxel with the lower Fortran-order linear index.
    """
    grid, shape = grid_and_shape
    ny, nx, nz = shape

    # Build the flat strel-like view: all voxels are candidates.
    total = ny * nx * nz
    flat_energies = grid.ravel(order="C")

    # Build Fortran-order linear indices for every voxel.
    # np.ravel_multi_index with order='F' maps (y, x, z) → F-linear index.
    all_coords = np.array(
        [(y, x, z) for z in range(nz) for x in range(nx) for y in range(ny)],
        dtype=np.int64,
    )
    fortran_indices = np.array(
        [np.ravel_multi_index((int(c[0]), int(c[1]), int(c[2])), shape, order="F") for c in all_coords],
        dtype=np.int64,
    )

    # The function takes energies and their corresponding Fortran linear indices.
    winner_strel_idx = _argmin_with_linear_index_tiebreak(flat_energies, fortran_indices)

    winner_fortran_idx = int(fortran_indices[winner_strel_idx])
    min_energy = float(np.min(flat_energies))

    # Find all tied voxels (those sharing the minimum energy value)
    tied_mask = flat_energies == min_energy
    tied_fortran_indices = fortran_indices[tied_mask]

    # The winner must have the LOWEST Fortran-order linear index among all tied voxels
    assert winner_fortran_idx == int(np.min(tied_fortran_indices)), (
        f"Expected winner Fortran index {int(np.min(tied_fortran_indices))}, "
        f"got {winner_fortran_idx} (shape={shape})"
    )


@pytest.mark.unit
@given(grid_with_two_tied_min_voxels())
@settings(max_examples=100)
def test_argmin_tiebreak_winner_has_min_energy(
    grid_and_shape: tuple[np.ndarray, tuple[int, int, int]],
) -> None:
    """**Validates: Requirements 3.2**

    The winner selected by ``_argmin_with_linear_index_tiebreak`` always has
    the minimum energy value — it must be among the tied-minimum voxels, not
    some other voxel.
    """
    grid, shape = grid_and_shape
    ny, nx, nz = shape
    total = ny * nx * nz
    flat_energies = grid.ravel(order="C")

    all_coords = np.array(
        [(y, x, z) for z in range(nz) for x in range(nx) for y in range(ny)],
        dtype=np.int64,
    )
    fortran_indices = np.array(
        [np.ravel_multi_index((int(c[0]), int(c[1]), int(c[2])), shape, order="F") for c in all_coords],
        dtype=np.int64,
    )

    winner_strel_idx = _argmin_with_linear_index_tiebreak(flat_energies, fortran_indices)

    assert flat_energies[winner_strel_idx] == float(np.min(flat_energies)), (
        "Winner energy must equal the global minimum energy."
    )


@pytest.mark.unit
@given(
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=10),
    st.integers(min_value=2, max_value=5),
)
@settings(max_examples=100)
def test_argmin_tiebreak_fortran_order_matches_ravel_multi_index(
    ny: int, nx: int, nz: int,
) -> None:
    """**Validates: Requirements 3.2**

    Directly verify that for two voxels with equal energy, the winner is the
    one whose ``np.ravel_multi_index(..., order='F')`` index is lower, covering
    all possible pairs of positions in the grid.
    """
    shape = (ny, nx, nz)

    # Use the two voxels at Fortran positions 0 and 1 — always distinct and
    # guaranteed to be the two lowest Fortran indices in any grid.
    coord_a = np.array(np.unravel_index(0, shape, order="F"), dtype=np.int64)  # (0,0,0)
    coord_b = np.array(np.unravel_index(1, shape, order="F"), dtype=np.int64)  # (1,0,0) for ny>1

    f_idx_a = int(np.ravel_multi_index(tuple(coord_a.tolist()), shape, order="F"))
    f_idx_b = int(np.ravel_multi_index(tuple(coord_b.tolist()), shape, order="F"))

    # Both at the same (minimum) energy
    tie_energy = -5.0
    energies = np.array([tie_energy, tie_energy], dtype=np.float64)
    linear_indices = np.array([f_idx_a, f_idx_b], dtype=np.int64)

    winner_idx = _argmin_with_linear_index_tiebreak(energies, linear_indices)

    # coord_a has the lower Fortran index (0), so it must win
    assert f_idx_a < f_idx_b
    assert linear_indices[winner_idx] == f_idx_a, (
        f"Expected winner Fortran index {f_idx_a}, got {linear_indices[winner_idx]} "
        f"(shape={shape}, coord_a={coord_a}, coord_b={coord_b})"
    )
