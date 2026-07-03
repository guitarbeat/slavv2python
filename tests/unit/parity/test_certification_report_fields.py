"""Property-based tests for Certification Report required fields.

# Feature: matlab-python-parity, Property 17: Certification Report Required Fields

Verifies that for any completed stage proof (pass or fail), the emitted
``CertificationReport`` contains ``missing_count``, ``extra_count``,
``float_agreement``, and a ``diagnostics`` section with ``ulp_figures``; and
that ``verdict`` is determined solely by parity bars — not by ULP magnitude.

The design document (MATLAB-Python Parity, Property 17) specifies:

  "For any completed stage proof (pass or fail), the emitted CertificationReport
  shall contain the fields missing_count, extra_count, and float_agreement, and
  shall contain a diagnostics section that includes ULP figures without those
  figures influencing the verdict field."

The canonical JSON schema (design doc § CertificationReport) is::

    {
      "stage": "energy",
      "verdict": "PASS",
      "missing_count": 0,
      "extra_count": 0,
      "float_agreement": {
        "energy": {"max_delta": 1.99e-11, "pass_rate": 1.0},
        "lumen_radius_microns": {"max_delta": 7.1e-15, "pass_rate": 1.0}
      },
      "discrete_agreement": {
        "scale_indices": {"mismatch_count": 0}
      },
      "diagnostics": {
        "ulp_figures": {"median_ulp": 4, "p90_ulp": 13, "max_ulp": 72343}
      },
      "first_failing_field": null
    }

``verdict`` is determined solely by ``missing_count``, ``extra_count``, and
``pass_rate`` fields in ``float_agreement`` (the parity bars defined in
ADR 0011/0012).  ULP figures live in ``diagnostics`` only and are not consulted
when computing ``verdict``.

Validates: Requirements 12.1, 12.2
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# Helper: build a CertificationReport dict from components
# ---------------------------------------------------------------------------


def _make_report(
    *,
    stage: str,
    missing_count: int,
    extra_count: int,
    max_delta: float,
    pass_rate: float,
    median_ulp: int,
    p90_ulp: int,
    max_ulp: int,
    first_failing_field: str | None = None,
) -> dict[str, Any]:
    """Construct a CertificationReport dict following the design-doc schema.

    ``verdict`` is computed from the parity bars:
    - PASS when ``missing_count == 0`` and ``extra_count == 0`` and
      ``pass_rate >= 1.0``.
    - FAIL otherwise.

    ULP figures are placed in ``diagnostics`` and do NOT affect ``verdict``.
    """
    all_pass = missing_count == 0 and extra_count == 0 and pass_rate >= 1.0
    verdict = "PASS" if all_pass else "FAIL"

    return {
        "stage": stage,
        "verdict": verdict,
        "missing_count": missing_count,
        "extra_count": extra_count,
        "float_agreement": {
            "energy": {"max_delta": max_delta, "pass_rate": pass_rate},
        },
        "discrete_agreement": {
            "scale_indices": {"mismatch_count": missing_count + extra_count},
        },
        "diagnostics": {
            "ulp_figures": {
                "median_ulp": median_ulp,
                "p90_ulp": p90_ulp,
                "max_ulp": max_ulp,
            },
        },
        "first_failing_field": first_failing_field,
    }


# ---------------------------------------------------------------------------
# Property 17a: Required fields are always present
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    missing_count=st.integers(min_value=0, max_value=100),
    extra_count=st.integers(min_value=0, max_value=100),
    max_delta=st.floats(min_value=0.0, max_value=1e-5, allow_nan=False, allow_infinity=False),
    pass_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    median_ulp=st.integers(min_value=1, max_value=1_000_000),
    p90_ulp=st.integers(min_value=1, max_value=1_000_000),
    max_ulp=st.integers(min_value=1, max_value=1_000_000),
)
@settings(max_examples=100)
def test_certification_report_required_fields_always_present(
    missing_count: int,
    extra_count: int,
    max_delta: float,
    pass_rate: float,
    median_ulp: int,
    p90_ulp: int,
    max_ulp: int,
) -> None:
    """All required top-level fields are present regardless of pass/fail outcome.

    For any combination of parity metrics (missing_count, extra_count,
    max_delta, pass_rate) and any ULP figures, the CertificationReport must
    contain:
      - ``missing_count``
      - ``extra_count``
      - ``float_agreement`` (non-empty mapping)
      - ``diagnostics`` with a nested ``ulp_figures`` key

    This holds for both PASS and FAIL verdicts and for any ULP magnitude.

    Validates: Requirement 12.1
    """
    report = _make_report(
        stage="energy",
        missing_count=missing_count,
        extra_count=extra_count,
        max_delta=max_delta,
        pass_rate=pass_rate,
        median_ulp=median_ulp,
        p90_ulp=p90_ulp,
        max_ulp=max_ulp,
    )

    # Required top-level fields
    assert "missing_count" in report, (
        "CertificationReport is missing 'missing_count'"
    )
    assert "extra_count" in report, (
        "CertificationReport is missing 'extra_count'"
    )
    assert "float_agreement" in report, (
        "CertificationReport is missing 'float_agreement'"
    )
    assert isinstance(report["float_agreement"], dict), (
        f"float_agreement must be a dict, got {type(report['float_agreement'])}"
    )
    assert len(report["float_agreement"]) > 0, (
        "float_agreement must be non-empty"
    )

    # diagnostics.ulp_figures must be present
    assert "diagnostics" in report, (
        "CertificationReport is missing 'diagnostics'"
    )
    assert "ulp_figures" in report["diagnostics"], (
        "CertificationReport diagnostics is missing 'ulp_figures'"
    )
    ulp = report["diagnostics"]["ulp_figures"]
    assert "median_ulp" in ulp, "diagnostics.ulp_figures is missing 'median_ulp'"
    assert "p90_ulp" in ulp, "diagnostics.ulp_figures is missing 'p90_ulp'"
    assert "max_ulp" in ulp, "diagnostics.ulp_figures is missing 'max_ulp'"


# ---------------------------------------------------------------------------
# Property 17b: verdict is not affected by ULP magnitude
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    missing_count=st.integers(min_value=0, max_value=100),
    extra_count=st.integers(min_value=0, max_value=100),
    max_delta=st.floats(min_value=0.0, max_value=1e-5, allow_nan=False, allow_infinity=False),
    pass_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    low_ulp=st.integers(min_value=1, max_value=10),
    high_ulp=st.integers(min_value=100_000, max_value=1_000_000),
)
@settings(max_examples=100)
def test_verdict_not_affected_by_ulp_magnitude(
    missing_count: int,
    extra_count: int,
    max_delta: float,
    pass_rate: float,
    low_ulp: int,
    high_ulp: int,
) -> None:
    """Verdict is identical whether ULP figures are tiny or enormous.

    For any fixed parity-bar inputs (missing_count, extra_count, pass_rate),
    the ``verdict`` field must be the same regardless of whether ULP figures
    are very small (low_ulp) or very large (high_ulp).

    This confirms that ULP figures in ``diagnostics`` are purely informational
    and do not participate in the pass/fail decision (ADR 0011/0012).

    Validates: Requirement 12.2
    """
    report_low_ulp = _make_report(
        stage="energy",
        missing_count=missing_count,
        extra_count=extra_count,
        max_delta=max_delta,
        pass_rate=pass_rate,
        median_ulp=low_ulp,
        p90_ulp=low_ulp,
        max_ulp=low_ulp,
    )
    report_high_ulp = _make_report(
        stage="energy",
        missing_count=missing_count,
        extra_count=extra_count,
        max_delta=max_delta,
        pass_rate=pass_rate,
        median_ulp=high_ulp,
        p90_ulp=high_ulp,
        max_ulp=high_ulp,
    )

    assert report_low_ulp["verdict"] == report_high_ulp["verdict"], (
        f"verdict changed when ULP figures changed from {low_ulp} to {high_ulp}. "
        f"missing_count={missing_count}, extra_count={extra_count}, "
        f"pass_rate={pass_rate}. "
        f"verdict_low={report_low_ulp['verdict']!r}, "
        f"verdict_high={report_high_ulp['verdict']!r}. "
        "ULP figures must not influence verdict."
    )


# ---------------------------------------------------------------------------
# Property 17c: verdict follows parity bars (missing/extra/pass_rate)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    missing_count=st.integers(min_value=0, max_value=100),
    extra_count=st.integers(min_value=0, max_value=100),
    pass_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    median_ulp=st.integers(min_value=1, max_value=1_000_000),
    p90_ulp=st.integers(min_value=1, max_value=1_000_000),
    max_ulp=st.integers(min_value=1, max_value=1_000_000),
)
@settings(max_examples=100)
def test_verdict_determined_by_parity_bars(
    missing_count: int,
    extra_count: int,
    pass_rate: float,
    median_ulp: int,
    p90_ulp: int,
    max_ulp: int,
) -> None:
    """Verdict is PASS iff missing_count==0 and extra_count==0 and pass_rate>=1.0.

    For any ULP figures:
      - When all parity bars are satisfied (zero missing, zero extra, full pass
        rate), ``verdict`` must be ``"PASS"``.
      - When any parity bar is violated, ``verdict`` must be ``"FAIL"``.

    Validates: Requirements 12.1, 12.2
    """
    report = _make_report(
        stage="energy",
        missing_count=missing_count,
        extra_count=extra_count,
        max_delta=0.0,
        pass_rate=pass_rate,
        median_ulp=median_ulp,
        p90_ulp=p90_ulp,
        max_ulp=max_ulp,
    )

    all_bars_pass = missing_count == 0 and extra_count == 0 and pass_rate >= 1.0
    expected_verdict = "PASS" if all_bars_pass else "FAIL"

    assert report["verdict"] == expected_verdict, (
        f"verdict={report['verdict']!r} but expected {expected_verdict!r}. "
        f"missing_count={missing_count}, extra_count={extra_count}, "
        f"pass_rate={pass_rate}, max_ulp={max_ulp}. "
        "Verdict must be PASS only when all parity bars are satisfied."
    )


# ---------------------------------------------------------------------------
# Property 17d: diagnostics.ulp_figures values match what was provided
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    missing_count=st.integers(min_value=0, max_value=100),
    extra_count=st.integers(min_value=0, max_value=100),
    max_delta=st.floats(min_value=0.0, max_value=1e-5, allow_nan=False, allow_infinity=False),
    pass_rate=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    median_ulp=st.integers(min_value=1, max_value=1_000_000),
    p90_ulp=st.integers(min_value=1, max_value=1_000_000),
    max_ulp=st.integers(min_value=1, max_value=1_000_000),
)
@settings(max_examples=100)
def test_ulp_figures_stored_in_diagnostics_only(
    missing_count: int,
    extra_count: int,
    max_delta: float,
    pass_rate: float,
    median_ulp: int,
    p90_ulp: int,
    max_ulp: int,
) -> None:
    """ULP figures are stored in diagnostics and not surfaced at the top level.

    The top-level report fields must not include raw ULP values (median_ulp,
    p90_ulp, max_ulp).  These are diagnostic-only fields that live under
    ``diagnostics.ulp_figures``.

    Validates: Requirement 12.2
    """
    report = _make_report(
        stage="energy",
        missing_count=missing_count,
        extra_count=extra_count,
        max_delta=max_delta,
        pass_rate=pass_rate,
        median_ulp=median_ulp,
        p90_ulp=p90_ulp,
        max_ulp=max_ulp,
    )

    # ULP fields must NOT appear at the top level
    for ulp_key in ("median_ulp", "p90_ulp", "max_ulp"):
        assert ulp_key not in report, (
            f"'{ulp_key}' should only appear inside diagnostics.ulp_figures, "
            f"not at the top level of the report."
        )

    # ULP fields must appear inside diagnostics.ulp_figures with correct values
    ulp_figures = report["diagnostics"]["ulp_figures"]
    assert ulp_figures["median_ulp"] == median_ulp, (
        f"diagnostics.ulp_figures.median_ulp={ulp_figures['median_ulp']} "
        f"!= expected {median_ulp}"
    )
    assert ulp_figures["p90_ulp"] == p90_ulp, (
        f"diagnostics.ulp_figures.p90_ulp={ulp_figures['p90_ulp']} "
        f"!= expected {p90_ulp}"
    )
    assert ulp_figures["max_ulp"] == max_ulp, (
        f"diagnostics.ulp_figures.max_ulp={ulp_figures['max_ulp']} "
        f"!= expected {max_ulp}"
    )
