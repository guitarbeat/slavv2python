"""Comprehensive tests for the consolidated, deep Vertex stage interface.
Verifies the new hybrid functional entry points and resumability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import pytest

from slavv_python.engine import SlavvPipeline
from slavv_python.engine.state import RunContext
from slavv_python.pipeline.vertices import (
    extract_vertices,
    paint_vertices,
    VertexManager,
)
from slavv_python.schema.results import EnergyResult, VertexSet

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def dummy_energy_data() -> EnergyResult:
    """Fixture returning a simple 3D energy field with a localized minimum."""
    energy = np.zeros((10, 10, 10), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)
    
    # Create localized minimum at center (5, 5, 5)
    energy[5, 5, 5] = -10.0
    energy[6, 5, 5] = -8.0
    scale_indices[5, 5, 5] = 1
    scale_indices[6, 5, 5] = 1
    
    return EnergyResult.create(
        energy=energy,
        scale_indices=scale_indices,
        lumen_radius_pixels=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        lumen_radius_microns=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        image_shape=(10, 10, 10),
        lumen_radius_pixels_axes=np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        pixels_per_sigma_PSF=1.0,
        microns_per_sigma_PSF=1.0,
        energy_sign=-1.0,
        energy_origin="python_native_hessian",
    )


@pytest.mark.unit
def test_extract_vertices_hybrid_default(dummy_energy_data):
    """Verify that extract_vertices works with default arguments (zero-config)."""
    vertices = extract_vertices(dummy_energy_data)
    assert isinstance(vertices, VertexSet)
    assert len(vertices.positions) == 1
    assert np.allclose(vertices.positions[0], [5, 5, 5])
    assert vertices.scales[0] == 1


@pytest.mark.unit
def test_extract_vertices_hybrid_with_params(dummy_energy_data):
    """Verify that extract_vertices works with pipeline parameter dictionary."""
    params = {
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "length_dilation_ratio": 1.0,
        "max_voxels_per_node": 6000,
    }
    vertices = extract_vertices(dummy_energy_data, params)
    assert len(vertices.positions) == 1
    assert np.allclose(vertices.positions[0], [5, 5, 5])


@pytest.mark.unit
def test_extract_vertices_hybrid_with_overrides(dummy_energy_data):
    """Verify that kwargs overrides standard params."""
    params = {
        "energy_upper_bound": -20.0,  # too low to detect our -10.0 energy vertex
        "space_strel_apothem": 1,
    }
    # Standard call with too low bound -> 0 vertices
    vertices_none = extract_vertices(dummy_energy_data, params)
    assert len(vertices_none.positions) == 0
    
    # Kwarg override -> resets bound -> detects vertex!
    vertices_overridden = extract_vertices(dummy_energy_data, params, energy_upper_bound=0.0)
    assert len(vertices_overridden.positions) == 1


@pytest.mark.unit
def test_extract_vertices_with_stage_controller(dummy_energy_data, tmp_path: Path):
    """Verify that resumable extraction correctly checkpoints and recovers."""
    run_context = RunContext(run_dir=tmp_path / "run", target_stage="vertices")
    controller = run_context.stage("vertices")
    
    # 1. First run: computes and saves checkpoints
    vertices = extract_vertices(dummy_energy_data, stage_controller=controller)
    assert len(vertices.positions) == 1
    assert (controller.stage_dir / "candidates.pkl").exists()
    
    # 2. Second run: recovers from checkpoint
    vertices_cached = extract_vertices(dummy_energy_data, stage_controller=controller)
    assert len(vertices_cached.positions) == 1
    assert np.allclose(vertices_cached.positions[0], vertices.positions[0])


@pytest.mark.unit
def test_paint_vertices_body_and_center(dummy_energy_data):
    """Verify that paint_vertices correctly renders vertex ellipsoid body and center spikes."""
    vertices = extract_vertices(dummy_energy_data)
    
    # Paint body ellipsoids
    body_mask = paint_vertices(vertices, (10, 10, 10), mode="body")
    assert body_mask.shape == (10, 10, 10)
    # The center voxel should be painted with vertex index + 1 = 1
    assert body_mask[5, 5, 5] == 1
    # Because scale=1 has radius=2, neighboring voxels should be painted too
    assert body_mask[5, 6, 5] == 1
    
    # Paint coordinate centers
    center_mask = paint_vertices(vertices, (10, 10, 10), mode="center")
    assert center_mask[5, 5, 5] == 1
    # Neighboring voxels should remain empty (0)
    assert center_mask[5, 6, 5] == 0


@pytest.mark.unit
def test_paint_vertices_invalid_mode(dummy_energy_data):
    """Verify paint_vertices raises descriptive error for invalid mode."""
    vertices = extract_vertices(dummy_energy_data)
    with pytest.raises(ValueError, match="Unknown painting mode"):
        paint_vertices(vertices, (10, 10, 10), mode="invalid_mode")  # type: ignore
