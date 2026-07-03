# Feature: matlab-python-parity, Property 1: Float64 Computation Invariant
"""Property 1: Float64 Computation Invariant.

For any 3D input volume and any pipeline stage (Energy, Vertices, Edges, Network)
on the Exact Route, all continuous quantity arrays (energies, coordinates, radii,
distance penalties, suppression factors, strand geometry) shall have dtype float64
during computation, before any persistence coercion.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4**
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slavv_python.pipeline.edges.matlab_get_edges_by_watershed import (
    _coords_from_linear_trace,
    _matlab_global_watershed_unit_vectors,
)
from slavv_python.pipeline.edges.matlab_get_edges_v300_geometry import (
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_frontier_directional_suppression_factors,
)
from slavv_python.pipeline.energy.matlab_energy_filter_v200 import (
    _matched_hessian_intermediates,
)
from slavv_python.pipeline.energy.matlab_principal_energy import compute_principal_energy
from slavv_python.pipeline.network.operations import (
    _matlab_get_strand_objects,
    _matlab_smooth_edges_v2,
)
from slavv_python.pipeline.policy import PipelinePolicy
from slavv_python.pipeline.vertices.detection import matlab_vertex_candidates_in_chunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXACT_POLICY = PipelinePolicy(
    precision=np.dtype(np.float64),
    internal_grid_alignment="matlab",
    rounding_mode="half-up",
    energy_engine="incremental",
)


def _make_volume(ny: int, nx: int, nz: int) -> np.ndarray:
    """Synthetic float64 volume with a tubular intensity gradient."""
    rng = np.random.default_rng(seed=42)
    vol = rng.random((ny, nx, nz), dtype=np.float64).astype(np.float64)
    # Introduce a mild tubular blob to generate valid energy voxels.
    cy, cx, cz = ny // 2, nx // 2, nz // 2
    vol[cy, cx, cz] = 2.0
    return vol


def _make_synthetic_edge_traces(
    n_edges: int,
    trace_len: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Build minimal float64 edge trace lists for network-stage tests."""
    rng = np.random.default_rng(seed=7)
    space = [rng.random((trace_len, 3), dtype=np.float64) for _ in range(n_edges)]
    scale = [rng.random((trace_len,), dtype=np.float64) for _ in range(n_edges)]
    energy = [-rng.random((trace_len,), dtype=np.float64) for _ in range(n_edges)]
    return space, scale, energy


# ---------------------------------------------------------------------------
# Property 1a — Energy stage: compute_principal_energy returns float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_energy_principal_energy_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: compute_principal_energy always returns float64.

    Energy is the innermost reduction step of the Exact Route energy filter.
    Regardless of input volume shape, the principal-energy eigendecomposition
    must accumulate in float64.
    """
    n_voxels = max(1, ny * nx * nz // 4)
    rng = np.random.default_rng(seed=(ny * 1000 + nx * 100 + nz))
    gradients = rng.standard_normal((n_voxels, 3)).astype(np.float64)
    curvatures = rng.standard_normal((n_voxels, 6)).astype(np.float64)
    # Ensure valid negative-definite-ish diagonal dominance for some voxels.
    curvatures[:, :3] -= 2.0

    result = compute_principal_energy(gradients, curvatures, energy_sign=-1.0, dtype=np.float64)

    assert result.dtype == np.float64, (
        f"compute_principal_energy returned {result.dtype}, expected float64 "
        f"(shape ({ny},{nx},{nz}), n_voxels={n_voxels})"
    )


# ---------------------------------------------------------------------------
# Property 1b — Energy stage: _matched_hessian_intermediates returns float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_energy_hessian_intermediates_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: Hessian intermediates (energy, laplacian) are float64 on Exact Route.

    _matched_hessian_intermediates is the inner filter step of the exact-route
    energy engine. Its output energy array must be float64 before any persistence.
    """
    image = _make_volume(ny, nx, nz)
    result = _matched_hessian_intermediates(
        image,
        radius_of_lumen_in_microns=1.5,
        microns_per_pixel=np.array([0.4, 0.4, 1.0], dtype=np.float64),
        pixels_per_sigma_psf=np.array([0.5, 0.5, 0.5], dtype=np.float64),
        gaussian_to_ideal_ratio=0.5,
        spherical_to_annular_ratio=0.5,
    )

    for field_name in ("energy", "laplacian"):
        arr = result[field_name]
        assert arr.dtype == np.float64, (
            f"_matched_hessian_intermediates['{field_name}'] has dtype {arr.dtype}, "
            f"expected float64 (volume shape ({ny},{nx},{nz}))"
        )


# ---------------------------------------------------------------------------
# Property 1c — Vertex stage: candidate energies are float64 on Exact Route
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100, deadline=None)
def test_vertex_candidate_energies_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: vertex candidate energies use float64 on the Exact Route.

    matlab_vertex_candidates_in_chunk returns accepted_energies with the policy
    precision. The Exact Route policy is float64, so the vertex scan must
    accumulate and return float64 energies during computation.
    """
    rng = np.random.default_rng(seed=(ny + nx * 17 + nz * 31))
    energy = rng.random((ny, nx, nz), dtype=np.float64) * -1.0
    scale_indices = np.zeros((ny, nx, nz), dtype=np.int16)

    _positions, _scales, accepted_energies = matlab_vertex_candidates_in_chunk(
        energy=energy,
        scale_indices=scale_indices,
        energy_sign=-1.0,
        energy_upper_bound=0.0,
        space_strel_apothem=1,
        policy=_EXACT_POLICY,
    )

    assert accepted_energies.dtype == np.float64, (
        f"vertex accepted_energies has dtype {accepted_energies.dtype}, "
        f"expected float64 (volume shape ({ny},{nx},{nz}))"
    )


# ---------------------------------------------------------------------------
# Property 1d — Edge stage: coordinate traces from linear indices are float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100, deadline=None)
def test_edge_coordinate_trace_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: watershed coordinate traces are float64 during edge computation.

    _coords_from_linear_trace converts Fortran-order linear indices to [Z,Y,X]
    coordinate arrays. These coordinates are continuous quantities and must be
    float64 before any persistence coercion.
    """
    shape = (ny, nx, nz)
    n_voxels = ny * nx * nz
    # Build a short linear trace using valid Fortran-order indices.
    max_trace = min(8, n_voxels)
    rng = np.random.default_rng(seed=(ny * 7 + nx * 13 + nz))
    linear_trace = sorted(rng.choice(n_voxels, size=max_trace, replace=False).tolist())

    coords = _coords_from_linear_trace(linear_trace, shape)

    assert coords.dtype == np.float64, (
        f"_coords_from_linear_trace returned dtype {coords.dtype}, expected float64 (shape {shape})"
    )
    assert coords.shape == (len(linear_trace), 3), (
        f"Expected ({len(linear_trace)}, 3), got {coords.shape}"
    )


# ---------------------------------------------------------------------------
# Property 1e — Edge stage: adjusted neighbour energies (penalties) are float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_edge_adjusted_neighbour_energies_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: edge distance-penalty and size-penalty arrays are float64.

    _matlab_frontier_adjusted_neighbor_energies applies size, distance, and
    direction penalties. All intermediate and output penalty arrays must be float64
    to preserve sub-ULP precision at watershed decision boundaries.
    """
    n_neighbors = max(4, ny)
    rng = np.random.default_rng(seed=(nz * 41 + nx))
    raw_energies = -rng.random((n_neighbors,), dtype=np.float64)
    offsets = rng.integers(-2, 3, size=(n_neighbors, 3)).astype(np.int32)
    r_over_R = rng.random((n_neighbors,)).astype(np.float64)
    scale_indices = rng.integers(0, 3, size=(n_neighbors,)).astype(np.float64)
    lumen_radius_microns = np.array([1.0, 2.0, 4.0], dtype=np.float64)
    microns_per_voxel = np.array([0.4, 0.4, 1.0], dtype=np.float64)

    result = _matlab_frontier_adjusted_neighbor_energies(
        raw_energies,
        neighbor_offsets=offsets,
        neighbor_r_over_R=r_over_R,
        neighbor_scale_indices=scale_indices,
        propagated_scale_index=1,
        current_d_over_r=0.5,
        origin_radius_microns=2.0,
        current_forward_unit=None,
        microns_per_voxel=microns_per_voxel,
        lumen_radius_microns=lumen_radius_microns,
    )

    assert result.dtype == np.float64, (
        f"adjusted_neighbor_energies returned {result.dtype}, expected float64 "
        f"(shape ({ny},{nx},{nz}))"
    )


# ---------------------------------------------------------------------------
# Property 1f — Edge stage: directional suppression factors are float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_edge_directional_suppression_factors_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: directional suppression factors are float64 during edge computation.

    Suppression factors gate multi-seed branching in the watershed. They are
    continuous quantities and must carry full float64 precision.
    """
    n_neighbors = max(4, ny)
    rng = np.random.default_rng(seed=(ny * 3 + nz * 5))
    offsets = rng.integers(-2, 3, size=(n_neighbors, 3)).astype(np.int32)
    microns_per_voxel = np.array([0.4, 0.4, 1.0], dtype=np.float64)

    result = _matlab_frontier_directional_suppression_factors(
        offsets,
        selected_index=0,
        microns_per_voxel=microns_per_voxel,
    )

    assert result.dtype == np.float64, (
        f"directional_suppression_factors returned {result.dtype}, expected float64 "
        f"(shape ({ny},{nx},{nz}))"
    )


# ---------------------------------------------------------------------------
# Property 1g — Edge stage: unit vectors are float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_edge_unit_vectors_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: watershed strel unit vectors are float64 during edge computation.

    Unit vectors normalise the strel neighbourhood directions for directional
    penalty calculation. They are continuous geometry arrays and must be float64.
    """
    n_neighbors = max(4, ny)
    rng = np.random.default_rng(seed=(nx * 11 + nz * 7))
    offsets = rng.integers(-2, 3, size=(n_neighbors, 3)).astype(np.int32)
    microns_per_voxel = np.array([0.4, 0.4, 1.0], dtype=np.float64)

    result = _matlab_global_watershed_unit_vectors(offsets, microns_per_voxel)

    assert result.dtype == np.float64, (
        f"_matlab_global_watershed_unit_vectors returned {result.dtype}, "
        f"expected float64 (n_neighbors={n_neighbors})"
    )


# ---------------------------------------------------------------------------
# Property 1h — Network stage: _matlab_get_strand_objects output is float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_network_strand_objects_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: strand geometry (space traces) from _matlab_get_strand_objects is float64.

    Strand coordinates are continuous spatial quantities computed at network stage.
    They must be float64 during assembly, before any persistence coercion.
    """
    # Build 2-4 synthetic edges forming a chain: 0-1, 1-2
    n_edges = max(2, (ny + nz) % 4 + 1)
    trace_len = max(3, nx)
    space_traces, scale_traces, energy_traces = _make_synthetic_edge_traces(n_edges, trace_len)

    # Strand: one strand using edges 0..(n_edges-1) in order, none backwards
    edge_indices_in_strands = [np.arange(n_edges, dtype=np.int32)]
    edge_backwards_in_strands = [np.zeros(n_edges, dtype=bool)]

    strand_space, _strand_scale, strand_energy = _matlab_get_strand_objects(
        space_traces,
        scale_traces,
        energy_traces,
        edge_indices_in_strands,
        edge_backwards_in_strands,
    )

    assert len(strand_space) >= 1, "Expected at least one strand"
    for i, arr in enumerate(strand_space):
        assert arr.dtype == np.float64, (
            f"strand_space_traces[{i}] has dtype {arr.dtype}, expected float64 "
            f"(volume shape ({ny},{nx},{nz}))"
        )
    for i, arr in enumerate(strand_energy):
        assert arr.dtype == np.float64, (
            f"strand_energy_traces[{i}] has dtype {arr.dtype}, expected float64 "
            f"(volume shape ({ny},{nx},{nz}))"
        )


# ---------------------------------------------------------------------------
# Property 1i — Network stage: _matlab_smooth_edges_v2 output is float64
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_network_smooth_edges_dtype_is_float64(ny: int, nx: int, nz: int) -> None:
    """Property 1: smoothed edge geometry from _matlab_smooth_edges_v2 is float64.

    Edge smoothing interpolates continuous spatial and energy quantities along
    strand traces. All output arrays must be float64 before persistence.
    """
    n_edges = max(2, (nx + nz) % 3 + 1)
    trace_len = max(4, ny)
    space_traces, scale_traces, energy_traces = _make_synthetic_edge_traces(n_edges, trace_len)

    # Provide a non-empty lumen radius range so the smoothing branch executes.
    lumen_radius_range = np.array([1.0, 1.5, 2.0], dtype=np.float64)
    microns_per_voxel = np.array([0.4, 0.4, 1.0], dtype=np.float64)

    smoothed_space, _smoothed_scale, smoothed_energy = _matlab_smooth_edges_v2(
        space_traces,
        scale_traces,
        energy_traces,
        smoothing_kernel_sigma_to_lumen_radius_ratio=0.5,
        lumen_radius_in_microns_range=lumen_radius_range,
        microns_per_voxel=microns_per_voxel,
    )

    for i, arr in enumerate(smoothed_space):
        assert arr.dtype == np.float64, (
            f"smoothed_space[{i}] has dtype {arr.dtype}, expected float64 "
            f"(volume shape ({ny},{nx},{nz}))"
        )
    for i, arr in enumerate(smoothed_energy):
        assert arr.dtype == np.float64, (
            f"smoothed_energy[{i}] has dtype {arr.dtype}, expected float64 "
            f"(volume shape ({ny},{nx},{nz}))"
        )


# ---------------------------------------------------------------------------
# Property 1j — Cross-stage: compute_principal_energy rejects float32 input
#               (regression guard — the Exact Route must NOT silently downcast)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    ny=st.integers(2, 16),
    nx=st.integers(2, 16),
    nz=st.integers(2, 8),
)
@settings(max_examples=100)
def test_energy_principal_energy_preserves_float64_when_input_is_float32(
    ny: int, nx: int, nz: int
) -> None:
    """Property 1: compute_principal_energy upcasts float32 inputs to float64 when dtype=float64.

    Even if caller accidentally passes float32 gradients, the computation must
    produce float64 output when invoked with dtype=np.float64 (Exact Route).
    This guards against silent downcasting in the eigen step.
    """
    n_voxels = max(1, ny * nx * nz // 4)
    rng = np.random.default_rng(seed=(ny * 1001 + nx * 101 + nz + 1))
    # Deliberately pass float32 to simulate a caller that forgot to cast.
    gradients = rng.standard_normal((n_voxels, 3)).astype(np.float32)
    curvatures = rng.standard_normal((n_voxels, 6)).astype(np.float32)
    curvatures[:, :3] -= 2.0

    result = compute_principal_energy(gradients, curvatures, energy_sign=-1.0, dtype=np.float64)

    assert result.dtype == np.float64, (
        f"compute_principal_energy with float32 inputs returned {result.dtype}, "
        f"expected float64 (Exact Route dtype=float64 must upcast inputs)"
    )
