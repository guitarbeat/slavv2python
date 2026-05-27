from __future__ import annotations

import numpy as np
import pytest

from slavv_python.engine.constants import STATUS_FAILED
from slavv_python.engine.context import RunContext, StageController
from slavv_python.engine.executor import StageExecutor
from slavv_python.engine.state.run_state import RunState
from slavv_python.schema.results import EnergyResult, VertexSet


def make_dummy_energy() -> EnergyResult:
    """Helper to create a valid EnergyResult for testing."""
    energy = np.zeros((3, 3, 3), dtype=np.float32)
    scale_indices = np.zeros((3, 3, 3), dtype=np.int16)
    lumen_radius_pixels = np.array([1.0, 2.0], dtype=np.float32)
    lumen_radius_microns = np.array([1.5, 3.0], dtype=np.float32)
    return EnergyResult.create(energy, scale_indices, lumen_radius_pixels, lumen_radius_microns)


def make_dummy_vertices() -> VertexSet:
    """Helper to create a valid VertexSet for testing."""
    positions = np.zeros((1, 3), dtype=np.float32)
    scales = np.array([0], dtype=np.int16)
    energies = np.array([0.5], dtype=np.float32)
    lumen_radius_pixels = np.array([1.0, 2.0], dtype=np.float32)
    lumen_radius_microns = np.array([1.5, 3.0], dtype=np.float32)
    return VertexSet.create(positions, scales, energies, lumen_radius_pixels, lumen_radius_microns)


def test_stage_executor_ephemeral_flow():
    """Verify that StageExecutor handles ephemeral runs when run_context is None."""
    executor = StageExecutor(
        run_context=None,
        progress_callback=None,
        run_state=RunState(parameters={}),
    )

    compute_called = False
    dummy_payload = make_dummy_energy()

    def dummy_compute(controller):
        nonlocal compute_called
        compute_called = True
        assert controller is None
        return dummy_payload

    payload = executor.execute(
        stage_name="energy",
        result_key="energy_data",
        progress_fraction=0.4,
        compute_fn=dummy_compute,
        schema_class=EnergyResult,
    )

    assert compute_called
    assert isinstance(payload, EnergyResult)
    assert payload == dummy_payload
    assert executor.run_state.energy_data == payload


def test_stage_executor_ephemeral_flow_with_fallback():
    """Verify ephemeral execution uses the fallback_fn when provided."""
    executor = StageExecutor(
        run_context=None,
        progress_callback=None,
        run_state=RunState(parameters={}),
    )

    fallback_called = False
    dummy_payload = make_dummy_vertices()

    def dummy_fallback():
        nonlocal fallback_called
        fallback_called = True
        return dummy_payload

    payload = executor.execute(
        stage_name="vertices",
        result_key="vertices",
        progress_fraction=0.6,
        compute_fn=lambda c: make_dummy_vertices(),
        fallback_fn=dummy_fallback,
        schema_class=VertexSet,
    )

    assert fallback_called
    assert payload == dummy_payload
    assert executor.run_state.vertices == payload


def test_stage_executor_resumable_fresh_compute(tmp_path):
    """Verify StageExecutor runs compute_fn and persists result when no checkpoint exists."""
    dummy_image = np.zeros((5, 5, 5), dtype=np.uint8)
    params = {"pipeline_profile": "paper"}

    # Prepare RunContext
    _, context, force_rerun = RunContext.prepare(
        dummy_image,
        params,
        run_dir=tmp_path,
        stop_after="energy",
    )
    assert context is not None

    progress_calls = []

    def progress_cb(fraction, stage):
        progress_calls.append((fraction, stage))

    run_state = RunState(parameters=params)
    executor = StageExecutor(
        run_context=context,
        progress_callback=progress_cb,
        run_state=run_state,
    )

    compute_called = False
    dummy_payload = make_dummy_energy()

    def dummy_compute(controller):
        nonlocal compute_called
        compute_called = True
        assert isinstance(controller, StageController)
        return dummy_payload

    payload = executor.execute(
        stage_name="energy",
        result_key="energy_data",
        progress_fraction=0.4,
        compute_fn=dummy_compute,
        force_rerun=force_rerun["energy"],
        schema_class=EnergyResult,
    )

    assert compute_called
    assert isinstance(payload, EnergyResult)
    assert run_state.energy_data == payload

    # Verify checkpoint file is written on disk
    controller = context.stage("energy")
    assert controller.checkpoint_path.exists()
    assert controller.manifest_path.exists()

    # Check progress is emitted
    assert progress_calls == [(0.4, "energy")]


def test_stage_executor_resumable_cached_load(tmp_path):
    """Verify StageExecutor loads result from checkpoint when it exists on disk."""
    dummy_image = np.zeros((5, 5, 5), dtype=np.uint8)
    params = {"pipeline_profile": "paper"}

    # Prepare RunContext
    _, context, _force_rerun = RunContext.prepare(
        dummy_image,
        params,
        run_dir=tmp_path,
        stop_after="energy",
    )
    assert context is not None

    # Pre-populate a checkpoint using StageController
    controller = context.stage("energy")
    dummy_payload = make_dummy_energy()
    controller.save_checkpoint(dummy_payload.to_dict())

    run_state = RunState(parameters=params)
    executor = StageExecutor(
        run_context=context,
        progress_callback=None,
        run_state=run_state,
    )

    compute_called = False

    def dummy_compute(controller):
        nonlocal compute_called
        compute_called = True
        return make_dummy_energy()

    payload = executor.execute(
        stage_name="energy",
        result_key="energy_data",
        progress_fraction=0.4,
        compute_fn=dummy_compute,
        force_rerun=False,
        schema_class=EnergyResult,
    )

    # Compute should not be called
    assert not compute_called
    assert isinstance(payload, EnergyResult)
    # Check loaded fields match
    np.testing.assert_array_equal(payload.energy, dummy_payload.energy)
    np.testing.assert_array_equal(payload.scale_indices, dummy_payload.scale_indices)
    assert run_state.energy_data == payload


def test_stage_executor_resumable_force_rerun(tmp_path):
    """Verify StageExecutor re-computes fresh even if checkpoint exists when force_rerun is True."""
    dummy_image = np.zeros((5, 5, 5), dtype=np.uint8)
    params = {"pipeline_profile": "paper"}

    _, context, _ = RunContext.prepare(
        dummy_image,
        params,
        run_dir=tmp_path,
        stop_after="energy",
    )
    assert context is not None

    # Pre-populate a checkpoint
    controller = context.stage("energy")
    controller.save_checkpoint(make_dummy_energy().to_dict())

    run_state = RunState(parameters=params)
    executor = StageExecutor(
        run_context=context,
        progress_callback=None,
        run_state=run_state,
    )

    compute_called = False
    fresh_payload = make_dummy_energy()
    # Modify fresh payload slightly to distinguish from cache
    fresh_payload.energy[0, 0, 0] = 999.0

    def dummy_compute(controller):
        nonlocal compute_called
        compute_called = True
        return fresh_payload

    payload = executor.execute(
        stage_name="energy",
        result_key="energy_data",
        progress_fraction=0.4,
        compute_fn=dummy_compute,
        force_rerun=True,
        schema_class=EnergyResult,
    )

    assert compute_called
    assert isinstance(payload, EnergyResult)
    assert payload.energy[0, 0, 0] == 999.0
    assert run_state.energy_data == payload


def test_stage_executor_failure_propagation(tmp_path):
    """Verify that compute_fn failures correctly update stage status and propagate the exception."""
    dummy_image = np.zeros((5, 5, 5), dtype=np.uint8)
    params = {"pipeline_profile": "paper"}

    _, context, _ = RunContext.prepare(
        dummy_image,
        params,
        run_dir=tmp_path,
        stop_after="energy",
    )
    assert context is not None

    executor = StageExecutor(
        run_context=context,
        progress_callback=None,
        run_state=RunState(parameters=params),
    )

    def failing_compute(controller):
        controller.begin(detail="Choosing edges", substage="choose_edges")
        raise ValueError("Computation failed")

    with pytest.raises(ValueError, match="Computation failed"):
        executor.execute(
            stage_name="energy",
            result_key="energy_data",
            progress_fraction=0.4,
            compute_fn=failing_compute,
        )

    # Verify stage is marked failed in run context
    assert context.snapshot.stages["energy"].status == STATUS_FAILED
    error = context.snapshot.errors[-1]
    assert error["substage"] == "choose_edges"
    assert "Traceback" in error["traceback"]
