"""Tests for stretch MATLAB-engine Energy host (Python 3.7 worker + bind)."""

from __future__ import annotations

import numpy as np
import pytest

from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_engine_backend import MatlabEngineInfraError
from slavv_python.pipeline.energy.matlab_engine_host import (
    STRETCH_PY37_ENV,
    resolve_python37_executable,
    stretch_engine_float_body_session,
)
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    compute_exact_parity_energy_chunked,
)
from slavv_python.utils.validation import validate_parameters


def test_numpy_backend_session_is_noop() -> None:
    config = {"energy_float_backend": "numpy"}
    with stretch_engine_float_body_session(config) as bound:
        assert bound is config
        assert bound.get("_stretch_engine_float_body_bound") is not True


def test_prepare_config_copies_bound_engine_session() -> None:
    image = np.zeros((4, 4, 4), dtype=np.float64)
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "energy_float_backend": "matlab_engine",
            "comparison_exact_network": True,
        }
    )
    params["_stretch_engine_float_body_bound"] = True
    params["_stretch_engine_session"] = "sentinel-session"
    config = _prepare_energy_config(image, params)
    assert config["_stretch_engine_float_body_bound"] is True
    assert config["_stretch_engine_session"] == "sentinel-session"
    assert config["energy_float_backend"] == "matlab_engine"
    assert "scales_per_octave" in config


def test_exact_chunked_refuses_unbound_engine_numpy_body() -> None:
    image = np.zeros((8, 8, 8), dtype=np.float64)
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "energy_float_backend": "matlab_engine",
            "comparison_exact_network": True,
            "n_jobs": 6,
            "max_voxels_per_node_energy": 1e9,
        }
    )
    config = _prepare_energy_config(image, params)
    with pytest.raises(MatlabEngineInfraError):
        compute_exact_parity_energy_chunked(image, config)


def test_resolve_python37_respects_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    missing = tmp_path / "no-python.exe"
    monkeypatch.setenv(STRETCH_PY37_ENV, str(missing))
    found = resolve_python37_executable()
    if found is not None:
        assert found.is_file()
