"""Unit tests for MATLAB-engine Energy adapter (U2)."""

from __future__ import annotations

import numpy as np
import pytest

from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    MatlabEnginePolicyError,
    MatlabEngineSession,
    matlab_engine_importable,
    numpy_to_matlab_double,
    refuse_matlab_only_energy_checkpoint_as_stretch_success,
    verify_matlab_engine_prerequisites,
)


def test_engine_missing_raises_infra_not_silent_numpy() -> None:
    """When engine cannot start, surface incomplete_infra — no stretch fallback."""
    if matlab_engine_importable():
        pytest.skip("matlab.engine importable on this interpreter")
    with pytest.raises(
        MatlabEngineInfraError,
        match=r"incomplete_infra|not importable|supports Python",
    ):
        verify_matlab_engine_prerequisites()


def test_refuse_matlab_only_energy_as_stretch_success() -> None:
    with pytest.raises(MatlabEnginePolicyError, match="R6"):
        refuse_matlab_only_energy_checkpoint_as_stretch_success()


def test_session_start_fails_as_infra_on_unsupported_python() -> None:
    """Repo CI Python (3.12) cannot host R2019a Engine — classify as infra."""
    with pytest.raises(MatlabEngineInfraError):
        MatlabEngineSession().start()


@pytest.mark.skipif(
    not matlab_engine_importable(),
    reason="matlab.engine unavailable; stretch CI skips engine happy-path",
)
def test_small_array_roundtrip_preserves_float64_values() -> None:
    rng = np.random.default_rng(0)
    original = rng.standard_normal((4, 3, 2), dtype=np.float64)
    with MatlabEngineSession() as session:
        restored = session.roundtrip_float64(original)
    assert restored.shape == original.shape
    assert restored.dtype == np.float64
    assert np.array_equal(restored, original)


def test_numpy_to_matlab_double_requires_matlab_package() -> None:
    if matlab_engine_importable():
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        ml = numpy_to_matlab_double(arr)
        assert ml is not None
        return
    with pytest.raises(MatlabEngineInfraError):
        numpy_to_matlab_double(np.zeros((2, 2), dtype=np.float64))
