from __future__ import annotations

from typing import TYPE_CHECKING

from slavv_python.analytics.parity.constants import (
    EXPERIMENT_ROOT_REQUIRED_RELATIVE_PATHS,
    FULL_ORACLE_ID,
    STRETCH_CROP_ORACLE_ID,
)
from slavv_python.analytics.parity.experiments.artifact_class import resolve_candidate_set_path
from slavv_python.analytics.parity.oracle.surfaces import inspect_experiment_root

if TYPE_CHECKING:
    from pathlib import Path

_LFS_POINTER = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
    "size 123\n"
)


def _write_matlab_batch(tmp_path: Path, oracle_id: str) -> None:
    batch = (
        tmp_path
        / "workspace"
        / "oracles"
        / oracle_id
        / "01_Input"
        / "matlab_results"
        / "batch_test"
    )
    (batch / "data").mkdir(parents=True)
    (batch / "vectors").mkdir()
    (batch / "data" / "energy_test.mat").write_bytes(b"ok")
    (batch / "vectors" / "vertices_test.mat").write_bytes(b"ok")
    (batch / "vectors" / "edges_test.mat").write_bytes(b"ok")
    (batch / "vectors" / "network_test.mat").write_bytes(b"ok")


def test_inspect_experiment_root_reports_missing_on_empty_tree(tmp_path: Path) -> None:
    status = inspect_experiment_root(tmp_path)
    assert status.passed is False
    assert status.present == ()
    assert set(status.missing) >= set(EXPERIMENT_ROOT_REQUIRED_RELATIVE_PATHS)
    assert "workspace/datasets/*/01_Input/*.tif" in status.missing
    assert f"workspace/oracles/{FULL_ORACLE_ID}/01_Input/matlab_results/batch_*" in status.missing


def test_inspect_experiment_root_passes_when_required_files_exist(tmp_path: Path) -> None:
    for relative in EXPERIMENT_ROOT_REQUIRED_RELATIVE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
    tif = tmp_path / "workspace" / "datasets" / "abc" / "01_Input" / "180709_E.tif"
    tif.parent.mkdir(parents=True, exist_ok=True)
    tif.write_bytes(b"tif")
    _write_matlab_batch(tmp_path, FULL_ORACLE_ID)
    _write_matlab_batch(tmp_path, STRETCH_CROP_ORACLE_ID)

    status = inspect_experiment_root(tmp_path)
    assert status.passed is True
    assert status.missing == ()
    assert status.lfs_pointers == ()
    assert status.dataset_tifs == ("workspace/datasets/abc/01_Input/180709_E.tif",)


def test_inspect_experiment_root_rejects_lfs_pointer_stubs(tmp_path: Path) -> None:
    for relative in EXPERIMENT_ROOT_REQUIRED_RELATIVE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
    pointer_rel = EXPERIMENT_ROOT_REQUIRED_RELATIVE_PATHS[0]
    (tmp_path / pointer_rel).write_text(_LFS_POINTER, encoding="ascii")
    tif = tmp_path / "workspace" / "datasets" / "abc" / "01_Input" / "180709_E.tif"
    tif.parent.mkdir(parents=True, exist_ok=True)
    tif.write_bytes(b"tif")
    _write_matlab_batch(tmp_path, FULL_ORACLE_ID)
    _write_matlab_batch(tmp_path, STRETCH_CROP_ORACLE_ID)

    status = inspect_experiment_root(tmp_path)
    assert status.passed is False
    assert pointer_rel in status.missing
    assert pointer_rel in status.lfs_pointers


def test_resolve_candidate_set_path_prefers_edges_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "04_Edges" / "candidates.pkl"
    checkpoint = (
        tmp_path / "02_Output" / "python_results" / "checkpoints" / "checkpoint_edge_candidates.pkl"
    )
    artifact.parent.mkdir(parents=True)
    checkpoint.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact")
    checkpoint.write_bytes(b"checkpoint")
    assert resolve_candidate_set_path(tmp_path) == artifact


def test_resolve_candidate_set_path_falls_back_to_checkpoint(tmp_path: Path) -> None:
    checkpoint = (
        tmp_path / "02_Output" / "python_results" / "checkpoints" / "checkpoint_edge_candidates.pkl"
    )
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    assert resolve_candidate_set_path(tmp_path) == checkpoint
