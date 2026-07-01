import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

from slavv_python.engine.state.run_ledger import RunContext
from slavv_python.pipeline.energy.resumable import calculate_energy_field_resumable
from slavv_python.schema.results import EnergyResult


@pytest.fixture
def dummy_run_context(tmp_path):
    return RunContext(
        run_dir=tmp_path,
        input_fingerprint="dummy_input",
        params_fingerprint="dummy_params",
        target_stage="energy",
    )


@pytest.fixture
def exact_energy_params():
    return {
        "comparison_exact_network": True,
        "octave_at_scales": np.array([0, 0, 1, 1], dtype=np.int32),
        "scale_resolution_factors": np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]),
        "microns_per_voxel": np.array([1.0, 1.0, 1.0]),
        "pixels_per_sigma_PSF": np.array([0.5, 0.5, 0.5]),
        "lumen_radius_microns": np.array([1.5, 2.0, 2.5, 3.0]),
        "max_voxels": 1000,
        "energy_sign": -1,
        "energy_method": "hessian",
        "gaussian_to_ideal_ratio": 0.5,
        "spherical_to_annular_ratio": 0.5,
    }


def test_exact_route_octave_level_resume(dummy_run_context, exact_energy_params):
    """
    Test that exact-route energy calculation:
    1. Saves octave-level checkpoints after each octave.
    2. Resumes from the last completed octave checkpoint upon failure.
    3. Produces the identical final result as a non-interrupted run.
    4. Deletes octave checkpoints upon successful full run completion.
    """
    image = np.ones((8, 8, 8), dtype=np.float32) * 1000.0
    stage_controller = dummy_run_context.stage("energy")

    # Step 1: Run completely without interruption to get reference result
    ref_result = calculate_energy_field_resumable(
        image, exact_energy_params, stage_controller
    )
    assert isinstance(ref_result, EnergyResult)
    assert ref_result.energy.shape == image.shape
    assert ref_result.scale_indices.shape == image.shape

    # Reset run context for the resume test
    run_dir = dummy_run_context.run_root
    stage_controller.remove_state()
    # Clear any output files
    for p in Path(run_dir).rglob("*"):
        if p.is_file() and p.name not in ("snapshot.json", "run_ledger.json"):
            p.unlink()

    # Step 2: Run with an artificial crash after the first octave
    from slavv_python.pipeline.energy.config import _prepare_energy_config
    config = _prepare_energy_config(image, exact_energy_params)
    octave_range = np.unique(config["octave_at_scales"])
    assert len(octave_range) >= 2, f"Expected at least 2 octaves, got {octave_range}"
    first_octave = int(octave_range[0])
    second_octave = int(octave_range[1])

    from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import compute_exact_parity_energy_single_octave

    call_count = 0
    original_single_octave = compute_exact_parity_energy_single_octave

    def mock_single_octave_crash(*args, **kwargs):
        nonlocal call_count
        current_octave = kwargs.get("current_octave") or args[2]
        print(f"DEBUG mock_single_octave_crash: octave={current_octave}, call_count={call_count}")
        if current_octave == second_octave:
            raise RuntimeError(f"Simulated crash at octave {second_octave}")
        call_count += 1
        res = original_single_octave(*args, **kwargs)
        print(f"DEBUG mock_single_octave_crash completed for octave={current_octave}")
        return res

    with patch(
        "slavv_python.pipeline.energy.matlab_get_energy_v202_chunked.compute_exact_parity_energy_single_octave",
        side_effect=mock_single_octave_crash,
    ):
        with pytest.raises(RuntimeError, match=f"Simulated crash at octave {second_octave}"):
            calculate_energy_field_resumable(image, exact_energy_params, stage_controller)

    # Verify that the first octave checkpoint was created and saved in state
    state = stage_controller.load_state()
    print(f"DEBUG state_path={stage_controller.state_path}, exists={stage_controller.state_path.exists()}, state={state}")
    assert state is not None
    assert state.get("completed_octaves") == [first_octave]

    octave_first_energy_file = stage_controller.artifact_path(f"octave_energy_{first_octave}.npy")
    octave_first_scale_file = stage_controller.artifact_path(f"octave_scale_{first_octave}.npy")
    assert octave_first_energy_file.exists()
    assert octave_first_scale_file.exists()

    # Step 3: Resume the run
    # Mock single octave helper to count calls, ensuring first octave is SKIPPED
    resume_calls = []

    def mock_single_octave_resume(*args, **kwargs):
        current_octave = kwargs.get("current_octave") or args[2]
        resume_calls.append(current_octave)
        return original_single_octave(*args, **kwargs)

    with patch(
        "slavv_python.pipeline.energy.matlab_get_energy_v202_chunked.compute_exact_parity_energy_single_octave",
        side_effect=mock_single_octave_resume,
    ):
        resume_result = calculate_energy_field_resumable(
            image, exact_energy_params, stage_controller
        )

    # Verify resume skipped first octave and ran second octave
    assert first_octave not in resume_calls
    assert second_octave in resume_calls

    # Verify final resumed output matches reference output exactly
    np.testing.assert_allclose(ref_result.energy, resume_result.energy)
    np.testing.assert_array_equal(ref_result.scale_indices, resume_result.scale_indices)

    # Verify temporary octave files were cleaned up
    assert not octave_first_energy_file.exists()
    assert not octave_first_scale_file.exists()
