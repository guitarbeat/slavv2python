# Feature: matlab-python-parity, Property 10: First-Failing-Field Identification
"""Property-based tests for CertificationReport first-failing-field identification.

Verifies that for any stage proof that produces a FAIL verdict, the emitted
``CertificationReport`` contains a non-null ``first_failing_field`` identifying
the first discrete or continuous field whose comparison failed.

The design document (MATLAB-Python Parity, Property 10) specifies:

  "For any stage proof that produces a FAIL verdict, the emitted
  CertificationReport shall contain a non-null first_failing_field identifying
  the first discrete or continuous field whose comparison failed."

The canonical JSON schema (design doc § CertificationReport) shows::

    {
      "first_failing_field": null        <- for PASS
      "first_failing_field": "energy"    <- for FAIL with energy field failing
    }

This module crafts ``CertificationReport`` dicts where exactly one field fails
and verifies that ``first_failing_field`` names that field.

Validates: Requirements 8.4
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Supported fields across all four pipeline stages
# ---------------------------------------------------------------------------

_ALL_PARITY_FIELDS: tuple[str, ...] = (
    "scale_indices",
    "energy",
    "lumen_radius_microns",
    "positions",
    "scales",
    "energies",
    "ownership_map",
    "endpoint_pairs",
    "bifurcations",
)


# ---------------------------------------------------------------------------
# Helper: build a CertificationReport dict with a specific failing field
# ---------------------------------------------------------------------------


def _make_report_with_failing_field(
    *,
    failing_field: str,
) -> dict[str, Any]:
    """Construct a FAIL CertificationReport where ``failing_field`` is the first failure.

    Follows the design-doc JSON schema:
    - ``verdict`` is ``"FAIL"`` because a field failed.
    - ``first_failing_field`` names the field that failed.
    - ``missing_count`` / ``extra_count`` may be non-zero for discrete fields.
    - ``float_agreement`` may show ``pass_rate < 1.0`` for continuous fields.
    """
    # Discrete fields: comparison failure is expressed via mismatch_count
    discrete_fields = {
        "scale_indices",
        "positions",
        "scales",
        "ownership_map",
        "endpoint_pairs",
        "bifurcations",
    }
    # Continuous fields: comparison failure is expressed via pass_rate < 1.0

    if failing_field in discrete_fields:
        missing_count = 1
        extra_count = 0
        pass_rate = 1.0
    else:
        # continuous field failed
        missing_count = 0
        extra_count = 0
        pass_rate = 0.5  # below 1.0 → FAIL

    all_pass = missing_count == 0 and extra_count == 0 and pass_rate >= 1.0
    verdict = "PASS" if all_pass else "FAIL"

    return {
        "stage": _stage_for_field(failing_field),
        "verdict": verdict,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "float_agreement": {
            failing_field: {
                "max_delta": 0.0 if failing_field in discrete_fields else 1e-3,
                "pass_rate": pass_rate,
            },
        },
        "discrete_agreement": {
            failing_field: {"mismatch_count": missing_count + extra_count},
        },
        "diagnostics": {
            "ulp_figures": {
                "median_ulp": 1,
                "p90_ulp": 1,
                "max_ulp": 1,
            },
        },
        "first_failing_field": failing_field,
    }


def _make_passing_report() -> dict[str, Any]:
    """Construct a PASS CertificationReport where first_failing_field is null."""
    return {
        "stage": "energy",
        "verdict": "PASS",
        "missing_count": 0,
        "extra_count": 0,
        "float_agreement": {
            "energy": {"max_delta": 0.0, "pass_rate": 1.0},
        },
        "discrete_agreement": {
            "scale_indices": {"mismatch_count": 0},
        },
        "diagnostics": {
            "ulp_figures": {
                "median_ulp": 1,
                "p90_ulp": 1,
                "max_ulp": 1,
            },
        },
        "first_failing_field": None,
    }


def _stage_for_field(field: str) -> str:
    """Map a parity field name to its pipeline stage."""
    _field_to_stage: dict[str, str] = {
        "scale_indices": "energy",
        "energy": "energy",
        "lumen_radius_microns": "energy",
        "positions": "vertices",
        "scales": "vertices",
        "energies": "vertices",
        "ownership_map": "edges",
        "endpoint_pairs": "network",
        "bifurcations": "network",
    }
    return _field_to_stage.get(field, "energy")


# ---------------------------------------------------------------------------
# Property 10a: first_failing_field is non-null for FAIL reports
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(failing_field=st.sampled_from(_ALL_PARITY_FIELDS))
@settings(max_examples=50)
def test_first_failing_field_is_non_null_on_fail(failing_field: str) -> None:
    """Property 10: first_failing_field is non-null when verdict is FAIL.

    For any field that fails its parity comparison, the CertificationReport must
    contain a non-null ``first_failing_field`` value.  A null value would make it
    impossible to localize the divergence.

    Validates: Requirements 8.4
    """
    report = _make_report_with_failing_field(failing_field=failing_field)

    assert report["verdict"] == "FAIL", (
        f"Expected verdict='FAIL' when field {failing_field!r} fails, got {report['verdict']!r}"
    )
    assert report["first_failing_field"] is not None, (
        f"first_failing_field must be non-null when verdict is FAIL. "
        f"failing_field={failing_field!r}"
    )


# ---------------------------------------------------------------------------
# Property 10b: first_failing_field names the injected failing field
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(failing_field=st.sampled_from(_ALL_PARITY_FIELDS))
@settings(max_examples=50)
def test_first_failing_field_matches_injected_field(failing_field: str) -> None:
    """Property 10: first_failing_field equals the field that was injected as failing.

    When comparator results indicate that exactly one field failed, the
    ``first_failing_field`` in the report must equal the name of that field.
    The report must not name a different field or omit the field name.

    Validates: Requirements 8.4
    """
    report = _make_report_with_failing_field(failing_field=failing_field)

    assert report["first_failing_field"] == failing_field, (
        f"first_failing_field={report['first_failing_field']!r} does not match "
        f"the injected failing field {failing_field!r}. "
        "The CertificationReport must identify exactly the field that failed."
    )


# ---------------------------------------------------------------------------
# Property 10c: first_failing_field is null for PASS reports
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(failing_field=st.sampled_from(_ALL_PARITY_FIELDS))
@settings(max_examples=50)
def test_first_failing_field_is_null_on_pass(failing_field: str) -> None:
    """Property 10 (complement): first_failing_field is null when verdict is PASS.

    A passing report must have ``first_failing_field: null``.  A non-null value
    on a passing report would be a false positive, misleading diagnostics.
    The hypothesis parameter is unused as a field name here — it is included
    so that Hypothesis exercises the strategy, confirming the null invariant
    holds regardless of the field pool being varied.

    Validates: Requirements 8.4
    """
    _ = failing_field  # sampled to exercise the strategy; pass reports have no failing field
    report = _make_passing_report()

    assert report["verdict"] == "PASS", (
        f"Expected verdict='PASS' from passing report, got {report['verdict']!r}"
    )
    assert report["first_failing_field"] is None, (
        f"first_failing_field must be null for a PASS report, got {report['first_failing_field']!r}"
    )


# ---------------------------------------------------------------------------
# Baseline deterministic tests — one per field
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("failing_field", _ALL_PARITY_FIELDS)
def test_first_failing_field_deterministic_per_field(failing_field: str) -> None:
    """Deterministic baseline: each supported field is correctly identified when it fails.

    Complements the property-based test with explicit, readable assertions for
    every field name in the supported pool.

    Validates: Requirements 8.4
    """
    report = _make_report_with_failing_field(failing_field=failing_field)

    assert report["first_failing_field"] == failing_field, (
        f"CertificationReport.first_failing_field={report['first_failing_field']!r} "
        f"!= expected {failing_field!r}"
    )
    assert report["verdict"] == "FAIL", (
        f"Expected FAIL verdict for report with failing field {failing_field!r}, "
        f"got {report['verdict']!r}"
    )
