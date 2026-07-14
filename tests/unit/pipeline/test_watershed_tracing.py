import json

import numpy as np
import pytest

from slavv_python.pipeline.edges.execution_tracing import ExecutionTracer, JsonExecutionTracer
from slavv_python.pipeline.edges.matlab_get_edges_by_watershed import (
    _generate_edge_candidates_matlab_global_watershed,
)


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

    def on_join_skipped(
        self,
        start_vertex,
        end_vertex,
        *,
        reason,
        iteration,
        current_linear,
    ):
        return None

    def on_strel_state(self, **kwargs):
        return None

    def on_frontier_state(self, **kwargs):
        return None

    def on_frontier_action(self, **kwargs):
        return None

    def frontier_state_targets(self):
        return set()


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


@pytest.mark.unit
def test_strel_state_trace_separates_before_and_after_claim(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = JsonExecutionTracer(trace_path, state_iterations=[7])

    tracer.on_strel_state(
        iteration=7,
        current_linear=10,
        current_vertex_index=3,
        current_scale_label=2,
        current_pointer_value=0,
        current_d_over_r=0.0,
        strel_linear=np.array([10, 11], dtype=np.int64),
        strel_pointer_indices=np.array([1, 2], dtype=np.uint64),
        strel_r_over_R=np.array([0.0, 0.5], dtype=np.float64),
        raw_energies=np.array([-5.0, -4.0], dtype=np.float64),
        adjusted_energies=np.array([-5.0, -4.0], dtype=np.float64),
        vertices_of_current_strel=np.array([3, 0], dtype=np.uint32),
        is_without_vertex=np.array([False, True]),
        pointer_values=np.array([0, 0], dtype=np.uint64),
        d_over_r_values=np.array([0.0, 0.0], dtype=np.float64),
        size_values=np.array([2, 0], dtype=np.int16),
        vertex_values_after_claim=np.array([3, 3], dtype=np.uint32),
        pointer_values_after_claim=np.array([0, 2], dtype=np.uint64),
        d_over_r_values_after_claim=np.array([0.0, 0.5], dtype=np.float64),
        size_values_after_claim=np.array([2, 2], dtype=np.int16),
    )

    payload = trace_path.read_text(encoding="utf-8").strip()
    assert payload
    row = next(item for item in json.loads(payload)["top_adjusted"] if item["linear"] == 11)
    assert row["vertex_index_before_claim"] == 0
    assert row["pointer_before_claim"] == 0
    assert row["d_over_r_before_claim"] == 0.0
    assert row["size_before_claim"] == 0
    assert row["vertex_index_after_claim"] == 3
    assert row["pointer_after_claim"] == 2
    assert row["d_over_r_after_claim"] == 0.5
    assert row["size_after_claim"] == 2


@pytest.mark.unit
def test_frontier_action_trace_filters_to_target_locations(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = JsonExecutionTracer(trace_path, state_linear_targets=[42])

    tracer.on_frontier_action(
        iteration=3,
        action="push",
        current_linear=10,
        locations=np.array([41, 42, 43], dtype=np.int64),
        details={"seed_idx": 1},
    )
    tracer.on_frontier_action(
        iteration=4,
        action="push",
        current_linear=10,
        locations=np.array([41, 43], dtype=np.int64),
        details={"seed_idx": 2},
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["event"] == "frontier_action"
    assert rows[0]["target_locations"] == [42]
    assert rows[0]["details"] == {"seed_idx": 1}


@pytest.mark.unit
def test_frontier_snapshot_trace_can_be_disabled_for_target_actions(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = JsonExecutionTracer(
        trace_path,
        state_linear_targets=[42],
        trace_frontier_snapshots=False,
    )

    tracer.on_frontier_state(
        iteration=3,
        label="after_push",
        current_linear=10,
        snapshot={"target_counts": {"42": 1}},
    )
    tracer.on_frontier_action(
        iteration=3,
        action="push",
        current_linear=10,
        locations=np.array([42], dtype=np.int64),
        details={"seed_idx": 1},
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in rows] == ["frontier_action"]


@pytest.mark.unit
def test_core_event_trace_can_be_disabled_for_state_only_runs(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    tracer = JsonExecutionTracer(
        trace_path,
        state_iterations=[3],
        trace_core_events=False,
    )

    tracer.on_iteration_start(3, 10, -5.0)
    tracer.on_seed_selected(1, 11, -4.0)
    tracer.on_join(1, 2, [10], [11])
    tracer.on_join_skipped(1, 1, reason="same", iteration=3, current_linear=10)
    tracer.on_strel_state(
        iteration=3,
        current_linear=10,
        current_vertex_index=3,
        current_scale_label=2,
        current_pointer_value=0,
        current_d_over_r=0.0,
        strel_linear=np.array([10], dtype=np.int64),
        strel_pointer_indices=np.array([1], dtype=np.uint64),
        strel_r_over_R=np.array([0.0], dtype=np.float64),
        raw_energies=np.array([-5.0], dtype=np.float64),
        adjusted_energies=np.array([-5.0], dtype=np.float64),
        vertices_of_current_strel=np.array([3], dtype=np.uint32),
        is_without_vertex=np.array([False]),
        pointer_values=np.array([0], dtype=np.uint64),
        d_over_r_values=np.array([0.0], dtype=np.float64),
        size_values=np.array([2], dtype=np.int16),
        vertex_values_after_claim=np.array([3], dtype=np.uint32),
        pointer_values_after_claim=np.array([0], dtype=np.uint64),
        d_over_r_values_after_claim=np.array([0.0], dtype=np.float64),
        size_values_after_claim=np.array([2], dtype=np.int16),
    )

    rows = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [row["event"] for row in rows] == ["strel_state"]
