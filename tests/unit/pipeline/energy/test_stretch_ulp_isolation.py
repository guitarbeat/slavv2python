"""E13: remaining Energy ULP isolation (linspace, Inf/NaN interp3, chunk vs full)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    _interp3_matlab_linear_inf,
    _matlab_zero_based_linspace_raw,
    compute_exact_parity_energy_chunked,
    get_chunking_lattice_v190,
)
from slavv_python.utils.validation import validate_parameters

REPO_ROOT = Path(__file__).resolve().parents[4]
CHUNK_HELPER = REPO_ROOT / "scripts" / "stretch" / "stretch_energy_chunk_v202.m"
GET_ENERGY = REPO_ROOT / "external" / "Vectorization-Public" / "source" / "get_energy_V202.m"


def _compact_matlab(source: str) -> str:
    return "".join(source.split())


def test_e13_linspace_formula_matches_get_energy_v202() -> None:
    if not CHUNK_HELPER.is_file():
        pytest.skip(f"incomplete_infra: missing {CHUNK_HELPER}")
    if not GET_ENERGY.is_file():
        pytest.skip("incomplete_infra: Vectorization-Public source missing")
    chunk_src = _compact_matlab(CHUNK_HELPER.read_text(encoding="utf-8"))
    full_src = _compact_matlab(GET_ENERGY.read_text(encoding="utf-8"))
    chunk_token = "linspace(1+mod("
    full_token = "linspace(1+mod("
    assert chunk_token in chunk_src, f"stretch helper missing linspace mod formula: {CHUNK_HELPER}"
    assert full_token in full_src, f"get_energy_V202 missing linspace mod formula: {GET_ENERGY}"
    assert "+(yw-1)/rf(1),yw)" in chunk_src
    assert "writing_counts" in full_src
    assert "resolution_factors" in full_src


def test_e13_python_linspace_matches_matlab_engine(stretch_py37_worker) -> None:
    cases = (
        (0, 3, 128, 0),
        (51, 3, 51, 17),
        (73, 3, 52, 24),
        (0, 2, 8, 0),
    )
    mismatches = 0
    compared = 0
    for offset, stride, count, local_start in cases:
        matlab_1based = stretch_py37_worker.linspace_1based(offset, stride, count)
        python_0based = _matlab_zero_based_linspace_raw(offset, stride, count, local_start)
        matlab_0based = matlab_1based - 1.0
        compared += int(count)
        if not np.array_equal(matlab_0based, python_0based):
            mismatches += int(np.count_nonzero(matlab_0based != python_0based))
    assert mismatches == 0, (
        f"E13 linspace isolation: {mismatches}/{compared} samples differ "
        "(named source: linspace mesh endpoints)"
    )


def _interp3_values_match(matlab_value: float, python_value: float) -> bool:
    if np.isnan(matlab_value) and np.isnan(python_value):
        return True
    if np.isposinf(matlab_value) and np.isposinf(python_value):
        return True
    if np.isneginf(matlab_value) and np.isneginf(python_value):
        return True
    return bool(
        np.isfinite(matlab_value) and np.isfinite(python_value) and matlab_value == python_value
    )


def test_e13_inf_nan_interp3_matches_matlab_engine(stretch_py37_worker) -> None:
    volume = np.full((2, 2, 2), np.inf, dtype=np.float64)
    volume[0, 0, 0] = -6.0

    queries = (
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.5, 0.5),
    )
    mismatches: list[str] = []
    for y, x, z in queries:
        matlab_out = stretch_py37_worker.interp3_probe(
            volume,
            np.array([[[x + 1.0]]], dtype=np.float64),
            np.array([[[y + 1.0]]], dtype=np.float64),
            np.array([[[z + 1.0]]], dtype=np.float64),
        )
        python_out = _interp3_matlab_linear_inf(
            volume,
            (
                np.array([[[y]]], dtype=np.float64),
                np.array([[[x]]], dtype=np.float64),
                np.array([[[z]]], dtype=np.float64),
            ),
        )
        ml = float(np.asarray(matlab_out).reshape(-1)[0])
        py = float(np.asarray(python_out).reshape(-1)[0])
        if not _interp3_values_match(ml, py):
            mismatches.append(f"y={y} x={x} z={z} matlab={ml} python={py}")
    assert not mismatches, "E13 Inf interp3 MATLAB vs Python helper diverged: " + "; ".join(
        mismatches
    )


def test_e13_nan_interp3_divergence_is_not_tiny_ulp(stretch_py37_worker) -> None:
    volume = np.zeros((2, 2, 2), dtype=np.float64)
    volume[1, 1, 1] = np.nan
    matlab_out = stretch_py37_worker.interp3_probe(
        volume,
        np.array([[[2.0]]], dtype=np.float64),
        np.array([[[2.0]]], dtype=np.float64),
        np.array([[[2.0]]], dtype=np.float64),
    )
    python_out = _interp3_matlab_linear_inf(
        volume,
        (
            np.array([[[1.0]]], dtype=np.float64),
            np.array([[[1.0]]], dtype=np.float64),
            np.array([[[1.0]]], dtype=np.float64),
        ),
    )
    ml = float(np.asarray(matlab_out).reshape(-1)[0])
    py = float(np.asarray(python_out).reshape(-1)[0])
    if _interp3_values_match(ml, py):
        return
    assert np.isnan(ml) or np.isnan(py) or np.isinf(ml) or np.isinf(py), (
        f"NaN interp3 diverged as tiny ULP matlab={ml} python={py}; "
        "that would be a v2-like residual, not an Inf/NaN class gap"
    )


def test_e13_inf_interp3_positive_weight_matches_known_python_fixture(
    stretch_py37_worker,
) -> None:
    volume = np.full((2, 2, 2), np.inf, dtype=np.float64)
    volume[0, 0, 0] = -6.0
    matlab_corner = stretch_py37_worker.interp3_probe(
        volume,
        np.array([[[1.0]]], dtype=np.float64),
        np.array([[[1.0]]], dtype=np.float64),
        np.array([[[1.0]]], dtype=np.float64),
    )
    matlab_half = stretch_py37_worker.interp3_probe(
        volume,
        np.array([[[1.0]]], dtype=np.float64),
        np.array([[[1.5]]], dtype=np.float64),
        np.array([[[1.0]]], dtype=np.float64),
    )
    python_corner = _interp3_matlab_linear_inf(
        volume,
        (
            np.array([[[0.0]]], dtype=np.float64),
            np.array([[[0.0]]], dtype=np.float64),
            np.array([[[0.0]]], dtype=np.float64),
        ),
    )
    python_half = _interp3_matlab_linear_inf(
        volume,
        (
            np.array([[[0.5]]], dtype=np.float64),
            np.array([[[0.0]]], dtype=np.float64),
            np.array([[[0.0]]], dtype=np.float64),
        ),
    )
    assert float(matlab_corner.reshape(-1)[0]) == float(python_corner.reshape(-1)[0]) == -6.0
    assert np.isposinf(matlab_half.reshape(-1)[0])
    assert np.isposinf(python_half.reshape(-1)[0])


def test_e13_chunk_vs_full_tiny_engine_body(stretch_py37_worker) -> None:
    rng = np.random.default_rng(13)
    image = np.asfortranarray(rng.random((8, 8, 8), dtype=np.float64))
    shared = {
        "energy_method": "hessian",
        "energy_float_backend": "matlab_engine",
        "comparison_exact_network": True,
        "n_jobs": 1,
        "radius_of_smallest_vessel_in_microns": 1.5,
        "radius_of_largest_vessel_in_microns": 1.6,
    }

    def _bound_config(max_voxels: float) -> dict:
        params = validate_parameters({**shared, "max_voxels_per_node_energy": max_voxels})
        params["_stretch_engine_float_body_bound"] = True
        params["_stretch_engine_session"] = stretch_py37_worker
        return _prepare_energy_config(image, params)

    config_full = _bound_config(1e9)
    config_chunked = _bound_config(64.0)
    image_shape = np.asarray(image.shape, dtype=float)
    planned_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
    microns = np.asarray(config_chunked["microns_per_voxel"], dtype=float)
    _, n_chunks = get_chunking_lattice_v190(
        1.0 / np.array([microns[1], microns[2], microns[0]]),
        float(config_chunked["max_voxels"]),
        np.round(planned_shape),
    )
    if n_chunks < 2:
        pytest.skip("tiny fixture did not produce multiple chunks")

    energy_full, scales_full, _extra_full = compute_exact_parity_energy_chunked(image, config_full)
    energy_chunked, scales_chunked, _extra_chunked = compute_exact_parity_energy_chunked(
        image, config_chunked
    )
    del _extra_full, _extra_chunked
    finite = np.isfinite(energy_full) & np.isfinite(energy_chunked)
    n_total = int(energy_full.size)
    n_bit = int(np.count_nonzero(energy_full == energy_chunked))
    n_scale = int(np.count_nonzero(scales_full != scales_chunked))
    if n_bit == n_total and n_scale == 0:
        return
    abs_delta = (
        np.abs(energy_full[finite] - energy_chunked[finite]) if np.any(finite) else np.array([])
    )
    max_abs = float(np.max(abs_delta)) if abs_delta.size else float("nan")
    pytest.fail(
        "E13 chunk-vs-full named a residual: "
        f"bit-identical {n_bit}/{n_total}, scale mismatches {n_scale}, "
        f"max abs delta {max_abs}"
    )
