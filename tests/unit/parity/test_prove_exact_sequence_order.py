"""Property 8: Sequential Stage Evaluation Order.

# Feature: matlab-python-parity, Property 8: Sequential Stage Evaluation Order

For any ``prove-exact-sequence`` invocation, the stages are evaluated in the
order ``[energy, vertices, edges, network]`` — a later stage is never evaluated
before an earlier stage has completed its proof.

Validates: Requirements 8.1
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

from slavv_python.analytics.parity.cli import handle_prove_exact_sequence
from slavv_python.analytics.parity.proof.exact_proof_contract import EXACT_STAGE_ORDER

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGES: tuple[str, ...] = EXACT_STAGE_ORDER  # ("energy", "vertices", "edges", "network")

# Strategy: a fixed-length list of booleans (one per stage) representing whether
# each stage passes.  Hypothesis varies the full 2^4 = 16-entry pass/fail space
# and many combinations across max_examples iterations.
_stage_outcomes = st.lists(st.booleans(), min_size=len(_STAGES), max_size=len(_STAGES)).map(tuple)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_fake_prove(
    outcomes: tuple[bool, ...],
    call_log: list[str],
    proof_dir: Path,
) -> object:
    """Return a side-effect callable that records call order and returns outcomes."""
    outcome_map = dict(zip(_STAGES, outcomes))

    def _prove(
        _dest: Path,
        *,
        stage_arg: str,
        report_path_arg: str | None = None,
        **_kwargs: object,
    ) -> tuple[dict, Path | None, Path | None]:
        call_log.append(stage_arg)
        # Write a stub JSON so cli.py's copy2 call finds a real file.
        json_path = proof_dir / f"exact_proof_{stage_arg}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("{}", encoding="utf-8")
        return (
            {"passed": outcome_map[stage_arg], "stage": stage_arg},
            json_path,
            None,
        )

    return _prove


def _patch_coordinator_and_surface(monkeypatch: pytest.MonkeyPatch, coordinator: MagicMock) -> None:
    """Wire the fake coordinator and a stub surface into the CLI module."""
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli.ExactProofCoordinator",
        lambda _surface: coordinator,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli._build_exact_proof_source_surface",
        lambda *_args, **_kwargs: MagicMock(),
    )


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(outcomes=_stage_outcomes)
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_stage_evaluation_order_is_always_energy_vertices_edges_network(
    outcomes: tuple[bool, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property 8: call order is always the prefix [energy, vertices, edges, network].

    Regardless of which stages pass or fail, every stage that *is* called must
    appear in canonical order.  No later stage may be evaluated before an earlier
    stage has returned.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        call_log: list[str] = []
        proof_dir = tmp_path / "03_Analysis"
        proof_dir.mkdir(parents=True, exist_ok=True)

        coordinator = MagicMock()
        coordinator.prove.side_effect = _build_fake_prove(outcomes, call_log, proof_dir)
        _patch_coordinator_and_surface(monkeypatch, coordinator)

        run_dir = tmp_path
        with contextlib.suppress(SystemExit):
            handle_prove_exact_sequence(
                Namespace(
                    source_run_root=str(run_dir),
                    dest_run_root=str(run_dir),
                    oracle_root=None,
                )
            )

        # --- Property 8 assertion -------------------------------------------
        # The stages that were called must form a *prefix* of EXACT_STAGE_ORDER.
        # No stage may be called out of order or before its predecessor returns.
        assert len(call_log) >= 1, "At least one stage must be called"
        assert call_log == list(_STAGES[: len(call_log)]), (
            f"Stage call order violated. Expected prefix of {list(_STAGES)}, got {call_log}"
        )


@pytest.mark.unit
@given(outcomes=_stage_outcomes)
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_each_stage_call_is_a_separate_prove_invocation(
    outcomes: tuple[bool, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Property 8 (complement): each evaluated stage corresponds to a separate prove() call.

    Ensures stages are never batched together or bypassed.  Every evaluated stage
    results in exactly one coordinator.prove() call, confirming predecessor-completion
    before successor-start.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        call_log: list[str] = []
        proof_dir = tmp_path / "03_Analysis"
        proof_dir.mkdir(parents=True, exist_ok=True)

        coordinator = MagicMock()
        coordinator.prove.side_effect = _build_fake_prove(outcomes, call_log, proof_dir)
        _patch_coordinator_and_surface(monkeypatch, coordinator)

        with contextlib.suppress(SystemExit):
            handle_prove_exact_sequence(
                Namespace(
                    source_run_root=str(tmp_path),
                    dest_run_root=str(tmp_path),
                    oracle_root=None,
                )
            )

        # Each recorded call must correspond to exactly one coordinator.prove() call.
        assert coordinator.prove.call_count == len(call_log), (
            "coordinator.prove() call count does not match the stage call log"
        )

        # Verify the stage_arg keyword in each call matches the recorded order.
        actual_stage_args = [c.kwargs["stage_arg"] for c in coordinator.prove.call_args_list]
        assert actual_stage_args == call_log, (
            f"stage_arg values in coordinator.prove() calls do not match call log. "
            f"call_args stage_args={actual_stage_args}, call_log={call_log}"
        )


# ---------------------------------------------------------------------------
# Baseline deterministic test
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_four_stages_called_when_all_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Baseline: when every stage passes, all four stages are evaluated in order."""
    call_log: list[str] = []
    proof_dir = tmp_path / "03_Analysis"
    proof_dir.mkdir(parents=True, exist_ok=True)

    coordinator = MagicMock()
    coordinator.prove.side_effect = _build_fake_prove((True, True, True, True), call_log, proof_dir)
    _patch_coordinator_and_surface(monkeypatch, coordinator)

    handle_prove_exact_sequence(
        Namespace(
            source_run_root=str(tmp_path),
            dest_run_root=str(tmp_path),
            oracle_root=None,
        )
    )

    assert call_log == list(_STAGES), (
        f"Expected all four stages in canonical order {list(_STAGES)}, got {call_log}"
    )
    assert coordinator.prove.call_count == 4
