"""Tests for stretch Energy origin stamp and dispatch gates (U3)."""

from __future__ import annotations

import pytest

from slavv_python.pipeline.energy.matlab_engine_backend import (
    MatlabEngineInfraError,
    ensure_matlab_engine_float_backend_ready,
)
from slavv_python.pipeline.energy.provenance import (
    CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN,
    STRETCH_ENGINE_ENERGY_ORIGIN,
    energy_origin_for_method,
    is_exact_compatible_energy_origin,
    is_watershed_allowed_energy_origin,
    refuse_mixed_stretch_energy_origins,
)
from slavv_python.utils.validation import validate_parameters


def test_default_exact_origin_unchanged() -> None:
    assert energy_origin_for_method("hessian") == CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN
    assert is_exact_compatible_energy_origin(CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN)


def test_stretch_origin_selected_for_matlab_engine_backend() -> None:
    assert (
        energy_origin_for_method("hessian", energy_float_backend="matlab_engine")
        == STRETCH_ENGINE_ENERGY_ORIGIN
    )


def test_stretch_origin_not_in_phase1_exact_allowlist() -> None:
    assert not is_exact_compatible_energy_origin(STRETCH_ENGINE_ENERGY_ORIGIN)
    assert is_watershed_allowed_energy_origin(STRETCH_ENGINE_ENERGY_ORIGIN, stretch_mode=True)
    assert not is_watershed_allowed_energy_origin(STRETCH_ENGINE_ENERGY_ORIGIN, stretch_mode=False)


def test_mixed_origin_rejected_for_stretch() -> None:
    with pytest.raises(ValueError, match="mixed"):
        refuse_mixed_stretch_energy_origins(
            {CANONICAL_NATIVE_EXACT_ENERGY_ORIGIN, STRETCH_ENGINE_ENERGY_ORIGIN}
        )


def test_validate_energy_float_backend_flag() -> None:
    params = validate_parameters({"energy_float_backend": "matlab_engine"})
    assert params["energy_float_backend"] == "matlab_engine"
    with pytest.raises(ValueError, match="energy_float_backend"):
        validate_parameters({"energy_float_backend": "mkl"})


def test_matlab_engine_backend_refuses_numpy_float_body() -> None:
    with pytest.raises(MatlabEngineInfraError):
        ensure_matlab_engine_float_backend_ready({"energy_float_backend": "matlab_engine"})
