"""Property-based tests for All-Pass Certification Verdict.

# Feature: matlab-python-parity, Property 11: All-Pass Certification Verdict

For any ``prove-exact-sequence`` result in which all four stages (energy,
vertices, edges, network) pass their respective parity bars, the harness shall
emit an overall verdict of ``CERTIFIED`` for the run.

In code, the ``CERTIFIED`` aggregate verdict is represented by the final
summary dict having ``passed=True``, written to ``EXACT_PROOF_JSON_PATH``
after all four stage proofs report ``{"passed": True}``.

The test mocks all four stage comparators (via ``ExactProofCoordinator.prove``)
to return PASS results, and varies the float-agreement and missing/extra count
values using Hypothesis to confirm the aggregate verdict is always CERTIFIED
when each stage passes.

Validates: Requirements 9.2
"""

from __future__ import annotations

import json
import tempfile
from argparse import Namespace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slavv_python.analytics.parity.cli_handlers.cli_proofs import handle_prove_exact_sequence
from slavv_python.analytics.parity.constants import EXACT_PROOF_JSON_PATH
from slavv_python.analytics.parity.proof.exact_proof_contract import EXACT_STAGE_ORDER

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# float-agreement values: pass_rate=1.0 (all pass), small positive max_delta
_pass_rate = st.just(1.0)
_max_delta = st.floats(min_value=0.0, max_value=1e-9, allow_nan=False, allow_infinity=False)
_zero_count = st.just(0)


def _float_agreement_strategy() -> st.SearchStrategy[dict[str, Any]]:
    """Generate a passing float_agreement dict: pass_rate=1.0, small max_delta."""
    return st.fixed_dictionaries(
        {
            "energy": st.fixed_dictionaries({"max_delta": _max_delta, "pass_rate": _pass_rate}),
            "lumen_radius_microns": st.fixed_dictionaries(
                {"max_delta": _max_delta, "pass_rate": _pass_rate}
            ),
        }
    )


def _pass_stage_report_strategy(stage: str) -> st.SearchStrategy[dict[str, Any]]:
    """Generate a passing stage report dict for the given stage."""
    return st.builds(
        lambda fa, mc, ec: {
            "passed": True,
            "stage": stage,
            "verdict": "PASS",
            "missing_count": mc,
            "extra_count": ec,
            "float_agreement": fa,
        },
        fa=_float_agreement_strategy(),
        mc=_zero_count,
        ec=_zero_count,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STAGES: tuple[str, ...] = EXACT_STAGE_ORDER  # ("energy", "vertices", "edges", "network")


def _build_all_pass_prove(stage_reports: dict[str, dict[str, Any]]) -> Any:
    """Return a side-effect callable that returns a per-stage passing report.

    Returns ``None`` for the json_path so ``handle_prove_exact_sequence`` skips
    the ``copy2`` call — avoiding a Windows PermissionError that occurs when the
    source and destination paths resolve to the same file (src == dst on a flat
    temp dir tree).
    """

    def _prove(
        _dest: Path,
        *,
        stage_arg: str,
        report_path_arg: str | None = None,
        **_kwargs: object,
    ) -> tuple[dict[str, Any], Path | None, Path | None]:
        report = stage_reports[stage_arg]
        # Return None for json_path: cli.py guards with `if json_path is not None`
        # before calling copy2(), so returning None safely skips the copy.
        return (report, None, None)

    return _prove


def _patch_coordinator_and_surface(
    monkeypatch: pytest.MonkeyPatch,
    coordinator: MagicMock,
) -> None:
    """Wire the stub coordinator and surface into the CLI module under test."""
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli_handlers.cli_proofs.ExactProofCoordinator",
        lambda _surface: coordinator,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli_handlers.cli_support._build_exact_proof_source_surface",
        lambda *_args, **_kwargs: MagicMock(),
    )


def _read_summary(run_dir: Path) -> dict[str, Any]:
    """Read and parse the aggregate proof summary written by the sequence handler."""
    summary_path = run_dir / EXACT_PROOF_JSON_PATH
    assert summary_path.is_file(), (
        f"Expected aggregate summary at {summary_path} but file was not written. "
        "handle_prove_exact_sequence must write the summary when all stages pass."
    )
    with summary_path.open(encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Property 11: All-Pass Certification Verdict
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    energy_report=_pass_stage_report_strategy("energy"),
    vertices_report=_pass_stage_report_strategy("vertices"),
    edges_report=_pass_stage_report_strategy("edges"),
    network_report=_pass_stage_report_strategy("network"),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_all_pass_stages_emit_certified_aggregate_verdict(
    energy_report: dict[str, Any],
    vertices_report: dict[str, Any],
    edges_report: dict[str, Any],
    network_report: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property 11: CERTIFIED aggregate verdict when all four stages pass.

    For any variation of per-stage float-agreement values (pass_rate=1.0,
    small max_delta) and zero missing/extra counts, the aggregate verdict
    written by ``prove-exact-sequence`` must be CERTIFIED (``passed=True``).

    Validates: Requirements 9.2
    """
    stage_reports = {
        "energy": energy_report,
        "vertices": vertices_report,
        "edges": edges_report,
        "network": network_report,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        coordinator = MagicMock()
        coordinator.prove.side_effect = _build_all_pass_prove(stage_reports)
        _patch_coordinator_and_surface(monkeypatch, coordinator)

        # Should NOT raise — all stages pass so no sys.exit(1)
        handle_prove_exact_sequence(
            Namespace(
                source_run_root=str(tmp_path),
                dest_run_root=str(tmp_path),
                oracle_root=None,
            )
        )

        summary = _read_summary(tmp_path)

        # --- Property 11 assertion: aggregate verdict is CERTIFIED --------
        assert summary.get("passed") is True, (
            f"Expected aggregate verdict CERTIFIED (passed=True) when all four "
            f"stages pass, but got passed={summary.get('passed')!r}. "
            f"Stage reports: {stage_reports}. Summary: {summary}."
        )

        # Confirm all four stages are recorded as passed in the summary
        stages_in_summary = {row["stage"]: row["passed"] for row in summary.get("stages", [])}
        for stage in _STAGES:
            assert stages_in_summary.get(stage) is True, (
                f"Stage '{stage}' is not recorded as passed in the aggregate summary. "
                f"summary['stages']={summary.get('stages')}."
            )


# ---------------------------------------------------------------------------
# Complementary property: all four stages are evaluated before CERTIFIED
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    energy_report=_pass_stage_report_strategy("energy"),
    vertices_report=_pass_stage_report_strategy("vertices"),
    edges_report=_pass_stage_report_strategy("edges"),
    network_report=_pass_stage_report_strategy("network"),
)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_all_four_stages_are_evaluated_before_certified_verdict(
    energy_report: dict[str, Any],
    vertices_report: dict[str, Any],
    edges_report: dict[str, Any],
    network_report: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All four stage comparators are called exactly once before CERTIFIED is emitted.

    The CERTIFIED verdict must not be emitted unless every stage has been proven.
    This verifies that ``coordinator.prove`` is called exactly four times (once
    per stage) in a passing all-pass run.

    Validates: Requirements 9.2
    """
    stage_reports = {
        "energy": energy_report,
        "vertices": vertices_report,
        "edges": edges_report,
        "network": network_report,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        coordinator = MagicMock()
        coordinator.prove.side_effect = _build_all_pass_prove(stage_reports)
        _patch_coordinator_and_surface(monkeypatch, coordinator)

        handle_prove_exact_sequence(
            Namespace(
                source_run_root=str(tmp_path),
                dest_run_root=str(tmp_path),
                oracle_root=None,
            )
        )

        # All four comparators must have been invoked before certifying
        assert coordinator.prove.call_count == 4, (
            f"Expected exactly 4 stage comparator calls for an all-pass run, "
            f"got {coordinator.prove.call_count}. The harness must prove every "
            "stage before emitting CERTIFIED."
        )

        # Verify the stage_arg order matches EXACT_STAGE_ORDER
        actual_stage_args = [c.kwargs["stage_arg"] for c in coordinator.prove.call_args_list]
        assert actual_stage_args == list(_STAGES), (
            f"Stage evaluation order incorrect: got {actual_stage_args}, expected {list(_STAGES)}."
        )


# ---------------------------------------------------------------------------
# Baseline deterministic test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_pass_deterministic_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: fixed all-pass inputs produce CERTIFIED summary with passed=True.

    Uses fixed non-hypothesis inputs to provide a clear regression baseline
    for Property 11.
    """
    fixed_float_agreement = {
        "energy": {"max_delta": 1.99e-11, "pass_rate": 1.0},
        "lumen_radius_microns": {"max_delta": 7.1e-15, "pass_rate": 1.0},
    }
    fixed_reports = {
        stage: {
            "passed": True,
            "stage": stage,
            "verdict": "PASS",
            "missing_count": 0,
            "extra_count": 0,
            "float_agreement": fixed_float_agreement,
        }
        for stage in _STAGES
    }

    coordinator = MagicMock()
    coordinator.prove.side_effect = _build_all_pass_prove(fixed_reports)
    _patch_coordinator_and_surface(monkeypatch, coordinator)

    handle_prove_exact_sequence(
        Namespace(
            source_run_root=str(tmp_path),
            dest_run_root=str(tmp_path),
            oracle_root=None,
        )
    )

    summary = _read_summary(tmp_path)

    assert summary.get("passed") is True, (
        f"Baseline all-pass run must produce CERTIFIED (passed=True), got: {summary}"
    )
    assert coordinator.prove.call_count == 4
    recorded_stages = [row["stage"] for row in summary.get("stages", [])]
    assert recorded_stages == list(_STAGES), (
        f"Summary stages must be {list(_STAGES)}, got {recorded_stages}"
    )
