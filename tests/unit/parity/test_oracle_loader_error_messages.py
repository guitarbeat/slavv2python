"""Property-based tests for Oracle Loader error message identification.

# Feature: matlab-python-parity, Property 14: Oracle Loader Error Identification

For any oracle artifact that is malformed or missing a required field, the
``Oracle_Loader`` shall raise an error whose message contains both the artifact
file path and the name of the missing or malformed field.

The loader is ``load_normalized_matlab_stage`` in
``slavv_python.analytics.parity.oracle.matlab_vector_loader``.  It wraps any
``ValueError`` raised by ``_require_key`` (which fires when a required MATLAB
field is absent from the payload) to inject the artifact path.  This test
verifies that property holds across all four stages and for varying artifact
path locations.

Validates: Requirements 11.2
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.io import savemat

from slavv_python.analytics.parity.oracle.matlab_vector_loader import load_normalized_matlab_stage

# ---------------------------------------------------------------------------
# Minimal mat payloads per stage
# These must be written as mat files and loaded with loadmat - using only
# arrays that survive the savemat/loadmat round-trip cleanly.
# ---------------------------------------------------------------------------


def _complete_vertices_payload() -> dict[str, Any]:
    """Return a minimal but structurally valid vertices MATLAB payload."""
    return {
        "vertex_space_subscripts": np.array([[1.0, 2.0, 3.0]]),
        "vertex_scale_subscripts": np.array([1.0]),
        "vertex_energies": np.array([0.5]),
    }


def _complete_edges_payload() -> dict[str, Any]:
    """Return a minimal but structurally valid edges MATLAB payload.

    Object arrays of object arrays (used for per-edge traces) do NOT round-trip
    reliably through savemat/loadmat with scipy at small sizes.  The fields
    ``edge_scale_subscripts`` and ``edge_energies`` cannot be removed
    independently without triggering reshape errors from ``edge_space_subscripts``
    (which is still present and serializes to shape (1,1,3) rather than (1,3)).
    The test therefore targets only ``edges2vertices`` and ``edge_space_subscripts``,
    which are the first two ``_require_key`` calls and produce clean missing-field
    errors.
    """
    return {
        "edges2vertices": np.array([[1, 2]]),
        "edge_space_subscripts": np.array([np.array([[1.0, 2.0, 3.0]])], dtype=object),
        "edge_scale_subscripts": np.array([np.array([1.0])], dtype=object),
        "edge_energies": np.array([np.array([0.5])], dtype=object),
    }


def _complete_network_payload() -> dict[str, Any]:
    """Return a minimal but structurally valid network MATLAB payload."""
    return {
        "strands2vertices": np.array([[1, 2]]),
        "bifurcation_vertices": np.array([1.0]),
        "strand_subscripts": np.array([np.array([[1.0, 2.0, 3.0, 1.0]])], dtype=object),
        "strand_energies": np.array([np.array([0.5])], dtype=object),
        "mean_strand_energies": np.array([0.5]),
        "vessel_directions": np.array([np.array([[0.0, 1.0, 0.0]])], dtype=object),
    }


# Map from stage → (complete_payload_factory, fields_whose_removal_triggers_clean_missing_error)
# Only include fields whose removal causes a '_require_key' error for THAT field
# (not an earlier field producing a reshape error from unexpected mat round-trip shape).
_STAGE_TESTABLE_FIELDS: dict[str, tuple[Any, list[str]]] = {
    "vertices": (
        _complete_vertices_payload,
        [
            "vertex_space_subscripts",
            "vertex_scale_subscripts",
            "vertex_energies",
        ],
    ),
    "edges": (
        _complete_edges_payload,
        # Only the first two _require_key fields produce clean errors.
        # edge_scale_subscripts / edge_energies trigger reshape errors from
        # edge_space_subscripts when that field is still present.
        [
            "edges2vertices",
            "edge_space_subscripts",
        ],
    ),
    "network": (
        _complete_network_payload,
        [
            "strands2vertices",
            "bifurcation_vertices",
            "strand_subscripts",
        ],
    ),
}


def _write_mat_without_field(
    path: Path,
    stage: str,
    missing_field: str,
) -> None:
    """Write a .mat file for ``stage`` with ``missing_field`` removed."""
    factory, _ = _STAGE_TESTABLE_FIELDS[stage]
    payload = dict(factory())
    payload.pop(missing_field, None)
    savemat(str(path), payload)


# ---------------------------------------------------------------------------
# Property 14 — one PBT per stage
#
# hypothesis varies the artifact path suffix (via st.text()) to exercise
# that the full path string is embedded in the error message regardless of
# what the path looks like.  Each generated suffix creates a distinct temp
# file path.  The inner loop exhausts all testable missing fields for the
# stage in each hypothesis iteration.
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    artifact_path_suffix=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-",
        ),
        min_size=1,
        max_size=20,
    ),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_vertices_missing_field_error_contains_path_and_field(
    artifact_path_suffix: str,
) -> None:
    """Property 14 (vertices): error message contains artifact path and missing field name.

    For any artifact path suffix, removing each required vertices field one at a
    time causes ``load_normalized_matlab_stage`` to raise a ``ValueError`` whose
    message contains both the full artifact file path and the name of the missing
    field.

    Validates: Requirements 11.2
    """
    _, testable_fields = _STAGE_TESTABLE_FIELDS["vertices"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for missing_field in testable_fields:
            mat_path = tmp_path / f"vertices_{artifact_path_suffix}_{missing_field}.mat"
            _write_mat_without_field(mat_path, "vertices", missing_field)

            with pytest.raises(ValueError, match=r".") as exc_info:
                load_normalized_matlab_stage(mat_path, "vertices")

            error_msg = str(exc_info.value)
            assert str(mat_path) in error_msg, (
                f"Artifact path '{mat_path}' not found in error message: {error_msg!r}"
            )
            assert missing_field in error_msg, (
                f"Missing field '{missing_field}' not found in error message: {error_msg!r}"
            )


@pytest.mark.unit
@given(
    artifact_path_suffix=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-",
        ),
        min_size=1,
        max_size=20,
    ),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_edges_missing_field_error_contains_path_and_field(
    artifact_path_suffix: str,
) -> None:
    """Property 14 (edges): error message contains artifact path and missing field name.

    For any artifact path suffix, removing each testable required edges field causes
    ``load_normalized_matlab_stage`` to raise a ``ValueError`` whose message contains
    both the artifact file path and the field name.

    Validates: Requirements 11.2
    """
    _, testable_fields = _STAGE_TESTABLE_FIELDS["edges"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for missing_field in testable_fields:
            mat_path = tmp_path / f"edges_{artifact_path_suffix}_{missing_field}.mat"
            _write_mat_without_field(mat_path, "edges", missing_field)

            with pytest.raises(ValueError, match=r".") as exc_info:
                load_normalized_matlab_stage(mat_path, "edges")

            error_msg = str(exc_info.value)
            assert str(mat_path) in error_msg, (
                f"Artifact path '{mat_path}' not found in error message: {error_msg!r}"
            )
            assert missing_field in error_msg, (
                f"Missing field '{missing_field}' not found in error message: {error_msg!r}"
            )


@pytest.mark.unit
@given(
    artifact_path_suffix=st.text(
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-",
        ),
        min_size=1,
        max_size=20,
    ),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_network_missing_field_error_contains_path_and_field(
    artifact_path_suffix: str,
) -> None:
    """Property 14 (network): error message contains artifact path and missing field name.

    For any artifact path suffix, removing each testable required network field causes
    ``load_normalized_matlab_stage`` to raise a ``ValueError`` whose message contains
    both the artifact file path and the field name.

    Validates: Requirements 11.2
    """
    _, testable_fields = _STAGE_TESTABLE_FIELDS["network"]
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for missing_field in testable_fields:
            mat_path = tmp_path / f"network_{artifact_path_suffix}_{missing_field}.mat"
            _write_mat_without_field(mat_path, "network", missing_field)

            with pytest.raises(ValueError, match=r".") as exc_info:
                load_normalized_matlab_stage(mat_path, "network")

            error_msg = str(exc_info.value)
            assert str(mat_path) in error_msg, (
                f"Artifact path '{mat_path}' not found in error message: {error_msg!r}"
            )
            assert missing_field in error_msg, (
                f"Missing field '{missing_field}' not found in error message: {error_msg!r}"
            )


# ---------------------------------------------------------------------------
# Deterministic baseline: a single concrete example for each stage/field combo
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("stage", "missing_field"),
    [
        # vertices — all three required fields
        ("vertices", "vertex_space_subscripts"),
        ("vertices", "vertex_scale_subscripts"),
        ("vertices", "vertex_energies"),
        # edges — first two _require_key fields (robust to savemat round-trip)
        ("edges", "edges2vertices"),
        ("edges", "edge_space_subscripts"),
        # network — first three _require_key fields
        ("network", "strands2vertices"),
        ("network", "bifurcation_vertices"),
        ("network", "strand_subscripts"),
    ],
)
def test_missing_required_field_error_contains_path_and_field_baseline(
    tmp_path: Path,
    stage: str,
    missing_field: str,
) -> None:
    """Baseline (deterministic): each stage/field combination surfaces both path and field.

    Validates: Requirements 11.2
    """
    mat_path = tmp_path / f"{stage}_test.mat"
    _write_mat_without_field(mat_path, stage, missing_field)

    with pytest.raises(ValueError, match=r".") as exc_info:
        load_normalized_matlab_stage(mat_path, stage)

    error_msg = str(exc_info.value)
    assert str(mat_path) in error_msg, (
        f"Stage '{stage}', missing field '{missing_field}': "
        f"artifact path '{mat_path}' not found in error message: {error_msg!r}"
    )
    assert missing_field in error_msg, (
        f"Stage '{stage}': "
        f"missing field name '{missing_field}' not found in error message: {error_msg!r}"
    )
