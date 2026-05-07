import numpy as np
import pytest
from slavv_python.core.edge_candidates_internal.global_watershed import (
    _generate_edge_candidates_matlab_global_watershed,
)
from slavv_python.core.edge_candidates_internal.tracing import ExecutionTracer


class MockTracer(ExecutionTracer):
    def __init__(self):
        self.iteration_starts = []
        self.seed_selected = []
        self.joins = []

    def on_iteration_start(self, iteration, current_linear, current_energy):
        self.iteration_starts.append((iteration, current_linear, current_energy))

    def on_seed_selected(self, seed_idx, selected_linear, selected_energy):
        self.seed_selected.append((seed_idx, selected_linear, selected_energy))

    def on_join(self, start_vertex, end_vertex, half_1, half_2):
        self.joins.append((start_vertex, end_vertex, half_1, half_2))


@pytest.mark.unit
def test_global_watershed_execution_tracing():
    # Simple 3x3x3 volume with 2 vertices that should connect
    energy = np.zeros((3, 3, 3), dtype=np.float32)
    # V1 at (1,0,1), V2 at (1,2,1)
    # Path: (1,0,1) -> (1,1,1) -> (1,2,1)
    energy[1, 0, 1] = -10.0
    energy[1, 1, 1] = -5.0
    energy[1, 2, 1] = -10.0

    vertex_positions = np.array([[1.0, 0.0, 1.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    vertex_scales = np.zeros((2,), dtype=np.int32)
    lumen_radius_microns = np.array([1.0], dtype=np.float32)
    microns_per_voxel = np.ones((3,), dtype=np.float32)

    tracer = MockTracer()
    params = {
        "edge_number_tolerance": 1,
        "energy_tolerance": 1.0,
        "step_size_per_origin_radius": 1.0,
    }

    _generate_edge_candidates_matlab_global_watershed(
        energy,
        None,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        np.zeros_like(energy),
        params,
        tracer=tracer,
    )

    # Verify tracer captured events
    assert len(tracer.iteration_starts) > 0
    assert len(tracer.seed_selected) > 0
    assert len(tracer.joins) == 1

    # Check join details
    v_start, v_end, h1, h2 = tracer.joins[0]
    assert {v_start, v_end} == {1, 2}
    assert len(h1) + len(h2) >= 3  # Should be at least 3 points total for this path
