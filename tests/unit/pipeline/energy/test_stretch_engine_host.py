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
from slavv_python.pipeline.energy.resumable import _config_hash
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
    hashed = _config_hash(config)
    config["_stretch_engine_session"] = object()
    assert _config_hash(config) == hashed


def test_engine_chunk_path_calls_interp3_helper_once_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def fake_chunk_helper(
        session: object,
        chunk: np.ndarray,
        **kwargs: object,
    ) -> tuple[np.ndarray, np.ndarray]:
        del session
        calls.append(int(np.asarray(chunk).size))
        y_w = int(kwargs["y_write_count"])
        x_w = int(kwargs["x_write_count"])
        z_w = int(kwargs["z_write_count"])
        energy = np.full((y_w, x_w, z_w), -1.0, dtype=np.float64)
        scale = np.ones((y_w, x_w, z_w), dtype=np.float64)
        return energy, scale

    monkeypatch.setattr(
        "slavv_python.pipeline.energy.matlab_get_energy_v202_chunked.energy_chunk_v202_from_spatial",
        fake_chunk_helper,
    )
    image = np.zeros((8, 8, 8), dtype=np.float64)
    params = validate_parameters(
        {
            "energy_method": "hessian",
            "energy_float_backend": "matlab_engine",
            "comparison_exact_network": True,
            "n_jobs": 1,
            "max_voxels_per_node_energy": 1e9,
        }
    )
    params["_stretch_engine_float_body_bound"] = True
    params["_stretch_engine_session"] = object()
    config = _prepare_energy_config(image, params)
    energy, scales, _extra = compute_exact_parity_energy_chunked(image, config)
    assert calls, "engine path must call stretch_energy_chunk_v202 once per chunk"
    assert energy.shape == image.shape
    assert scales.shape == image.shape


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
