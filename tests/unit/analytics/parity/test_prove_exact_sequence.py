"""Tests for sequential prove-exact certification CLI."""

from __future__ import annotations

from argparse import Namespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from slavv_python.analytics.parity.cli import handle_prove_exact_sequence

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.unit
def test_prove_exact_sequence_stops_on_first_failure(tmp_path: Path, monkeypatch):
    calls: list[str] = []

    def fake_prove(_dest: Path, *, stage_arg: str, report_path_arg: str | None = None, **_kwargs):
        del report_path_arg
        calls.append(stage_arg)
        passed = stage_arg != "vertices"
        return (
            {"passed": passed, "stage": stage_arg},
            tmp_path / f"proof_{stage_arg}.json",
            tmp_path / f"proof_{stage_arg}.txt",
        )

    coordinator = MagicMock()
    coordinator.prove.side_effect = fake_prove
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli.ExactProofCoordinator",
        lambda _surface: coordinator,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli._build_exact_proof_source_surface",
        lambda *_args, **_kwargs: MagicMock(),
    )

    with pytest.raises(SystemExit) as exc:
        handle_prove_exact_sequence(
            Namespace(
                source_run_root=str(tmp_path / "run"),
                dest_run_root=str(tmp_path / "run"),
                oracle_root=None,
            )
        )

    assert exc.value.code == 1
    assert calls == ["energy", "vertices"]


@pytest.mark.unit
def test_prove_exact_sequence_passes_all_stages(tmp_path: Path, monkeypatch):
    def fake_prove(_dest: Path, *, stage_arg: str, report_path_arg: str | None = None, **_kwargs):
        del report_path_arg
        json_path = tmp_path / "run" / "03_Analysis" / "exact_proof.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("{}", encoding="utf-8")
        return ({"passed": True, "stage": stage_arg}, json_path, None)

    coordinator = MagicMock()
    coordinator.prove.side_effect = fake_prove
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli.ExactProofCoordinator",
        lambda _surface: coordinator,
    )
    monkeypatch.setattr(
        "slavv_python.analytics.parity.cli._build_exact_proof_source_surface",
        lambda *_args, **_kwargs: MagicMock(),
    )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    handle_prove_exact_sequence(
        Namespace(
            source_run_root=str(run_dir),
            dest_run_root=str(run_dir),
            oracle_root=None,
        )
    )
    assert coordinator.prove.call_count == 4
    assert (run_dir / "03_Analysis" / "exact_proof_energy.json").is_file()
