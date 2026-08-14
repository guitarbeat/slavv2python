"""Parity Experiment module: artifact class, proof pairing, cheap loop."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import h5py
import joblib
import numpy as np
import pytest

from slavv_python.analytics.parity.cli_handlers.cli_diagnostics import handle_inspect_proof
from slavv_python.analytics.parity.experiments import (
    ArtifactClass,
    ArtifactClassError,
    CheapLoopError,
    ExperimentCost,
    HypothesisKind,
    ProofRecordError,
    classify_edge_artifact,
    compare_same_class_pair_sets,
    coverage_of_finals_by_raw,
    load_edge_artifact,
    load_proof_record,
    require_cheap_loop,
    require_evaluated_adr0012,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("candidates.pkl", ArtifactClass.RAW_CANDIDATE_SET),
        ("checkpoint_edge_candidates.pkl", ArtifactClass.RAW_CANDIDATE_SET),
        ("raw_full_candidates.mat", ArtifactClass.RAW_CANDIDATE_SET),
        ("raw_watershed_candidates.mat", ArtifactClass.RAW_CANDIDATE_SET),
        ("edges.pkl", ArtifactClass.EDGE_SET),
        ("checkpoint_edges.pkl", ArtifactClass.EDGE_SET),
        ("chosen_edges.pkl", ArtifactClass.EDGE_SET),
        ("edges_0001.mat", ArtifactClass.EDGE_SET),
    ],
)
def test_classify_edge_artifact_known_names(name: str, expected: ArtifactClass) -> None:
    assert classify_edge_artifact(Path(name)) is expected


@pytest.mark.unit
def test_classify_edge_artifact_unknown_raises() -> None:
    with pytest.raises(ArtifactClassError, match="cannot classify"):
        classify_edge_artifact(Path("mystery.bin"))


@pytest.mark.unit
def test_same_class_pair_set_compare_counts() -> None:
    left = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    right = np.array([[4, 3], [7, 8]], dtype=np.int32)
    report = compare_same_class_pair_sets(
        left,
        right,
        left_class=ArtifactClass.RAW_CANDIDATE_SET,
        right_class=ArtifactClass.RAW_CANDIDATE_SET,
    )
    assert report.n_left == 3
    assert report.n_right == 2
    assert report.n_intersection == 1
    assert report.n_only_left == 2
    assert report.n_only_right == 1


@pytest.mark.unit
def test_mixed_class_pair_set_compare_raises() -> None:
    raw = np.array([[1, 2]], dtype=np.int32)
    finals = np.array([[1, 2]], dtype=np.int32)
    with pytest.raises(ArtifactClassError, match="mixed-class"):
        compare_same_class_pair_sets(
            raw,
            finals,
            left_class=ArtifactClass.RAW_CANDIDATE_SET,
            right_class=ArtifactClass.EDGE_SET,
        )


@pytest.mark.unit
def test_coverage_of_finals_by_raw_is_not_equality() -> None:
    raw = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
    finals = np.array([[2, 1], [9, 8]], dtype=np.int32)
    report = coverage_of_finals_by_raw(raw, finals)
    assert report.n_raw == 3
    assert report.n_final == 2
    assert report.n_covered == 1
    assert report.n_missing_from_raw == 1
    assert report.n_extra_raw == 2


def _write_proof(run_root: Path, dest_run_root: Path, *, evaluated: bool) -> Path:
    analysis = run_root / "03_Analysis"
    analysis.mkdir(parents=True)
    path = analysis / "exact_proof_network.json"
    path.write_text(
        json.dumps(
            {
                "passed": False,
                "stages": ["network"],
                "dest_run_root": str(dest_run_root),
                "source_run_root": str(run_root),
                "edges_adr0012_gate": {"adr0012_evaluated": evaluated},
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_load_proof_record_accepts_matching_dest(tmp_path: Path) -> None:
    run_root = tmp_path / "crop_M_exact_v3"
    path = _write_proof(run_root, run_root, evaluated=True)
    record = load_proof_record(path)
    assert record.run_root == run_root.resolve()
    assert record.dest_run_root == run_root.resolve()
    assert record.passed is False
    assert record.adr0012_evaluated is True


@pytest.mark.unit
def test_load_proof_record_refuses_unpaired_dest(tmp_path: Path) -> None:
    opened = tmp_path / "crop_M_exact_v3"
    foreign = tmp_path / "crop_M_exact"
    foreign.mkdir()
    path = _write_proof(opened, foreign, evaluated=True)
    with pytest.raises(ProofRecordError, match="does not match folder"):
        load_proof_record(path)


@pytest.mark.unit
def test_require_evaluated_adr0012_refuses_unevaluated(tmp_path: Path) -> None:
    run_root = tmp_path / "crop_M_exact_v3"
    path = _write_proof(run_root, run_root, evaluated=False)
    record = load_proof_record(path)
    with pytest.raises(ProofRecordError, match="not an evaluated ADR 0012"):
        require_evaluated_adr0012(record, stage="network")


@pytest.mark.unit
@pytest.mark.parametrize(
    "hypothesis_kind",
    [
        HypothesisKind.RANKING,
        HypothesisKind.ARTIFACT_CLASS,
        HypothesisKind.PAIR_SET,
    ],
)
def test_e10_require_cheap_loop_blocks_full_writer_for_cheap_kinds(
    hypothesis_kind: HypothesisKind,
) -> None:
    """E10: RANKING / ARTIFACT_CLASS / PAIR_SET refuse FULL_WRITER."""
    with pytest.raises(CheapLoopError, match="full writer"):
        require_cheap_loop(
            hypothesis_kind=hypothesis_kind,
            requested_cost=ExperimentCost.FULL_WRITER,
        )
    require_cheap_loop(
        hypothesis_kind=hypothesis_kind,
        requested_cost=ExperimentCost.UNIT,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "hypothesis_kind",
    [HypothesisKind.GENERATION, HypothesisKind.OWNERSHIP],
)
def test_e10_generation_and_ownership_may_request_full_writer(
    hypothesis_kind: HypothesisKind,
) -> None:
    require_cheap_loop(
        hypothesis_kind=hypothesis_kind,
        requested_cost=ExperimentCost.FULL_WRITER,
    )


@pytest.mark.unit
def test_require_cheap_loop_blocks_ranking_full_writer() -> None:
    with pytest.raises(CheapLoopError, match="full writer"):
        require_cheap_loop(
            hypothesis_kind=HypothesisKind.RANKING,
            requested_cost=ExperimentCost.FULL_WRITER,
        )
    require_cheap_loop(
        hypothesis_kind=HypothesisKind.RANKING,
        requested_cost=ExperimentCost.UNIT,
    )
    require_cheap_loop(
        hypothesis_kind=HypothesisKind.GENERATION,
        requested_cost=ExperimentCost.FULL_WRITER,
    )


@pytest.mark.unit
def test_inspect_proof_cli_refuses_unpaired_dest(tmp_path: Path) -> None:
    opened = tmp_path / "crop_M_exact_v3"
    foreign = tmp_path / "crop_M_exact"
    foreign.mkdir()
    path = _write_proof(opened, foreign, evaluated=True)
    with pytest.raises(SystemExit) as raised:
        handle_inspect_proof(Namespace(path=str(path), require_evaluated=False, stage=None))
    assert raised.value.code == 1


@pytest.mark.unit
def test_load_edge_artifact_pickle_connections(tmp_path: Path) -> None:
    path = tmp_path / "candidates.pkl"
    joblib.dump({"connections": np.array([[1, 2], [4, 3]], dtype=np.int32)}, path)
    artifact = load_edge_artifact(path)
    assert artifact.artifact_class is ArtifactClass.RAW_CANDIDATE_SET
    assert artifact.connections.tolist() == [[1, 2], [4, 3]]


@pytest.mark.unit
def test_load_edge_artifact_h5_mat_transposes_fortran_layout(tmp_path: Path) -> None:
    path = tmp_path / "raw_full_candidates.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "edges2vertices",
            data=np.array([[1, 3, 5], [2, 4, 6]], dtype=np.int64),
        )
    artifact = load_edge_artifact(path)
    assert artifact.artifact_class is ArtifactClass.RAW_CANDIDATE_SET
    assert artifact.connections.tolist() == [[0, 1], [2, 3], [4, 5]]
