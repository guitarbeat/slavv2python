"""Property 9: Downstream Blocking on Stage Failure.

# Feature: matlab-python-parity, Property 9: Downstream Blocking on Stage Failure

For any stage X in [energy, vertices, edges, network] that fails its parity bar,
all downstream stages X+1 through network shall be marked as blocked and shall
not be evaluated in that prove-exact-sequence run.

Validates: Requirements 8.2
"""

from __future__ import annotations

import contextlib
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slavv_python.analytics.parity.cli_handlers.cli_proofs import handle_prove_exact_sequence
from slavv_python.analytics.parity.proof.exact_proof_contract import EXACT_STAGE_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGES: tuple[str, ...] = EXACT_STAGE_ORDER  # ("energy", "vertices", "edges", "network")

# Strategy: pick a failing stage from the first three — "network" cannot have
# downstream stages so is excluded.  Hypothesis exercises all three options.
_failing_stage_strategy = st.sampled_from(["energy", "vertices", "edges"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_fake_prove_with_one_failure(
    failing_stage: str,
    call_log: list[str],
    proof_dir: Path,
) -> object:
    """Return a side-effect callable where exactly *failing_stage* returns FAIL.

    All stages before *failing_stage* return PASS.  Stages after *failing_stage*
    should never be called; if they are, recording their name in *call_log* will
    make the assertion fail, which is the intended signal.

    ``proof_dir`` must NOT be ``dest_run_root / "03_Analysis"`` — the CLI copies
    ``json_path`` to that location; returning a path from a sibling directory
    avoids a Windows self-copy PermissionError.
    """

    def _prove(
        _dest: Path,
        *,
        stage_arg: str,
        report_path_arg: str | None = None,
        **_kwargs: object,
    ) -> tuple[dict, Path | None, Path | None]:
        call_log.append(stage_arg)
        passed = stage_arg != failing_stage
        # Write the stub proof JSON into a staging sub-dir (not 03_Analysis) so
        # the cli.py copy2() has distinct source and destination on Windows.
        json_path = proof_dir / f"exact_proof_{stage_arg}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("{}", encoding="utf-8")
        return (
            {"passed": passed, "stage": stage_arg},
            json_path,
            None,
        )

    return _prove


def _patch_coordinator_and_surface(monkeypatch: pytest.MonkeyPatch, coordinator: MagicMock) -> None:
    """Wire the fake coordinator and a stub surface into the CLI module."""
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli_handlers.cli_proofs.ExactProofCoordinator",
        lambda _surface: coordinator,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli_handlers.cli_support._build_exact_proof_source_surface",
        lambda *_args, **_kwargs: MagicMock(),
    )


def _downstream_stages(failing_stage: str) -> list[str]:
    """Return all stages that come after *failing_stage* in EXACT_STAGE_ORDER."""
    idx = list(_STAGES).index(failing_stage)
    return list(_STAGES[idx + 1 :])


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(failing_stage=_failing_stage_strategy)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_downstream_stages_not_called_after_failure(
    failing_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property 9: when stage X fails, stages after X are never evaluated.

    The sequence must halt at the first failing stage and must NOT invoke
    coordinator.prove() for any downstream stage.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        call_log: list[str] = []
        # Use a staging sub-dir distinct from "03_Analysis" so that cli.py's
        # copy2(json_path, stage_json) has different source and destination.
        # (Windows raises PermissionError when src == dst in copy2.)
        proof_dir = tmp_path / "proof_staging"
        proof_dir.mkdir(parents=True, exist_ok=True)

        coordinator = MagicMock()
        coordinator.prove.side_effect = _build_fake_prove_with_one_failure(
            failing_stage, call_log, proof_dir
        )
        _patch_coordinator_and_surface(monkeypatch, coordinator)

        with contextlib.suppress(SystemExit):
            handle_prove_exact_sequence(
                Namespace(
                    source_run_root=str(tmp_path),
                    dest_run_root=str(tmp_path),
                    oracle_root=None,
                )
            )

        downstream = _downstream_stages(failing_stage)

        # --- Property 9 core assertion: no downstream stage was called ----------
        for blocked_stage in downstream:
            assert blocked_stage not in call_log, (
                f"Stage '{blocked_stage}' was evaluated after '{failing_stage}' failed. "
                f"Downstream stages must be BLOCKED (not evaluated). "
                f"Full call log: {call_log}"
            )

        # Confirm the failing stage itself *was* called (test sanity check).
        assert failing_stage in call_log, (
            f"Failing stage '{failing_stage}' was never called — test setup error. "
            f"Call log: {call_log}"
        )


@pytest.mark.unit
@given(failing_stage=_failing_stage_strategy)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_downstream_stages_verdict_is_blocked(
    failing_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property 9 (verdict label): downstream stages are recorded with verdict BLOCKED.

    The test builds a per-stage verdict map from the call log.  Stages that were
    never called — because the sequence halted — must be assigned verdict "BLOCKED".
    Stages that were called before the failure are PASS; the failing stage is FAIL.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        call_log: list[str] = []
        stage_reports: dict[str, dict] = {}
        proof_dir = tmp_path / "proof_staging"
        proof_dir.mkdir(parents=True, exist_ok=True)

        def _prove_and_capture(
            _dest: Path,
            *,
            stage_arg: str,
            report_path_arg: str | None = None,
            **_kwargs: object,
        ) -> tuple[dict, Path | None, Path | None]:
            call_log.append(stage_arg)
            passed = stage_arg != failing_stage
            report: dict = {
                "passed": passed,
                "stage": stage_arg,
                "verdict": "PASS" if passed else "FAIL",
            }
            stage_reports[stage_arg] = report
            json_path = proof_dir / f"exact_proof_{stage_arg}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text("{}", encoding="utf-8")
            return report, json_path, None

        coordinator = MagicMock()
        coordinator.prove.side_effect = _prove_and_capture
        _patch_coordinator_and_surface(monkeypatch, coordinator)

        with contextlib.suppress(SystemExit):
            handle_prove_exact_sequence(
                Namespace(
                    source_run_root=str(tmp_path),
                    dest_run_root=str(tmp_path),
                    oracle_root=None,
                )
            )

        # Build the full verdict map: called stages use their actual verdict,
        # uncalled stages receive "BLOCKED".
        verdict_map: dict[str, str] = {}
        for stage in _STAGES:
            if stage in stage_reports:
                verdict_map[stage] = stage_reports[stage]["verdict"]
            else:
                verdict_map[stage] = "BLOCKED"

        downstream = _downstream_stages(failing_stage)

        # --- Property 9: every downstream stage must be BLOCKED -----------------
        for blocked_stage in downstream:
            assert verdict_map[blocked_stage] == "BLOCKED", (
                f"Stage '{blocked_stage}' expected verdict 'BLOCKED' after "
                f"'{failing_stage}' failed, but got '{verdict_map[blocked_stage]}'. "
                f"Full verdict map: {verdict_map}"
            )

        # The failing stage must be FAIL, not PASS.
        assert verdict_map[failing_stage] == "FAIL", (
            f"Failing stage '{failing_stage}' expected 'FAIL', got '{verdict_map[failing_stage]}'"
        )

        # Stages before the failing stage must be PASS.
        failing_idx = list(_STAGES).index(failing_stage)
        for stage in _STAGES[:failing_idx]:
            assert verdict_map[stage] == "PASS", (
                f"Stage '{stage}' before '{failing_stage}' expected 'PASS', "
                f"got '{verdict_map[stage]}'"
            )


@pytest.mark.unit
@given(failing_stage=_failing_stage_strategy)
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_downstream_comparators_not_called(
    failing_stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property 9 (comparator guard): downstream stage comparators are never invoked.

    Each stage's ``coordinator.prove()`` is the entry point for its comparator.
    Verifies that ``coordinator.prove`` is not called for any downstream stage —
    confirming that comparators are never reached for BLOCKED stages.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        call_log: list[str] = []
        proof_dir = tmp_path / "proof_staging"
        proof_dir.mkdir(parents=True, exist_ok=True)

        coordinator = MagicMock()
        coordinator.prove.side_effect = _build_fake_prove_with_one_failure(
            failing_stage, call_log, proof_dir
        )
        _patch_coordinator_and_surface(monkeypatch, coordinator)

        with contextlib.suppress(SystemExit):
            handle_prove_exact_sequence(
                Namespace(
                    source_run_root=str(tmp_path),
                    dest_run_root=str(tmp_path),
                    oracle_root=None,
                )
            )

        downstream = _downstream_stages(failing_stage)

        # Collect the stage_arg values that coordinator.prove was actually called with.
        called_with = [call.kwargs["stage_arg"] for call in coordinator.prove.call_args_list]

        for blocked_stage in downstream:
            assert blocked_stage not in called_with, (
                f"coordinator.prove() was called for downstream stage '{blocked_stage}' "
                f"after '{failing_stage}' failed. Comparators for blocked stages must "
                f"never be invoked. "
                f"prove() was called with stage_args: {called_with}"
            )


# ---------------------------------------------------------------------------
# Deterministic baseline tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_energy_failure_blocks_vertices_edges_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: energy FAIL blocks all three downstream stages."""
    call_log: list[str] = []
    proof_dir = tmp_path / "proof_staging"
    proof_dir.mkdir(parents=True, exist_ok=True)

    coordinator = MagicMock()
    coordinator.prove.side_effect = _build_fake_prove_with_one_failure(
        "energy", call_log, proof_dir
    )
    _patch_coordinator_and_surface(monkeypatch, coordinator)

    with pytest.raises(SystemExit):
        handle_prove_exact_sequence(
            Namespace(
                source_run_root=str(tmp_path),
                dest_run_root=str(tmp_path),
                oracle_root=None,
            )
        )

    assert call_log == ["energy"], (
        f"Only 'energy' should be called when energy fails; got {call_log}"
    )
    assert coordinator.prove.call_count == 1

    for blocked in ["vertices", "edges", "network"]:
        assert blocked not in call_log


@pytest.mark.unit
def test_vertices_failure_blocks_edges_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: vertices FAIL blocks edges and network; energy was already PASS."""
    call_log: list[str] = []
    proof_dir = tmp_path / "proof_staging"
    proof_dir.mkdir(parents=True, exist_ok=True)

    coordinator = MagicMock()
    coordinator.prove.side_effect = _build_fake_prove_with_one_failure(
        "vertices", call_log, proof_dir
    )
    _patch_coordinator_and_surface(monkeypatch, coordinator)

    with pytest.raises(SystemExit):
        handle_prove_exact_sequence(
            Namespace(
                source_run_root=str(tmp_path),
                dest_run_root=str(tmp_path),
                oracle_root=None,
            )
        )

    assert call_log == ["energy", "vertices"], (
        f"Only 'energy' and 'vertices' should be called; got {call_log}"
    )
    assert coordinator.prove.call_count == 2

    for blocked in ["edges", "network"]:
        assert blocked not in call_log


@pytest.mark.unit
def test_edges_failure_blocks_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: edges FAIL blocks network; energy and vertices were already PASS."""
    call_log: list[str] = []
    proof_dir = tmp_path / "proof_staging"
    proof_dir.mkdir(parents=True, exist_ok=True)

    coordinator = MagicMock()
    coordinator.prove.side_effect = _build_fake_prove_with_one_failure("edges", call_log, proof_dir)
    _patch_coordinator_and_surface(monkeypatch, coordinator)

    with pytest.raises(SystemExit):
        handle_prove_exact_sequence(
            Namespace(
                source_run_root=str(tmp_path),
                dest_run_root=str(tmp_path),
                oracle_root=None,
            )
        )

    assert call_log == ["energy", "vertices", "edges"], (
        f"Only the first three stages should be called; got {call_log}"
    )
    assert coordinator.prove.call_count == 3

    assert "network" not in call_log
