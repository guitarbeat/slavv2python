"""Tests for promotion commands in the parity experiment runner."""

from __future__ import annotations

import importlib
import json

import pytest
from slavv_python.analysis.parity.constants import (
    DATASET_INPUT_DIR,
    DATASET_MANIFEST_PATH,
    EXPERIMENT_INDEX_PATH,
    ORACLE_MANIFEST_PATH,
    REPORT_MANIFEST_PATH,
    RUN_MANIFEST_PATH,
    SUMMARY_JSON_PATH,
)

from .support import _build_experiment_root, _write_json

parity_experiment = importlib.import_module("workspace.scripts.cli.parity_experiment")


@pytest.mark.integration
def test_promote_dataset_copies_input_and_writes_manifest(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff-payload")
    dataset_hash = parity_experiment.fingerprint_file(dataset_file)

    parity_experiment.main(
        [
            "promote-dataset",
            "--dataset-file",
            str(dataset_file),
            "--experiment-root",
            str(experiment_root),
        ]
    )

    dataset_root = experiment_root / "datasets" / dataset_hash
    manifest = json.loads((dataset_root / DATASET_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["dataset_hash"] == dataset_hash
    assert manifest["stored_input_file"] == str(
        dataset_root / DATASET_INPUT_DIR / dataset_file.name
    )
    assert (dataset_root / DATASET_INPUT_DIR / dataset_file.name).read_bytes() == (b"tiff-payload")
    assert (dataset_root / DATASET_INPUT_DIR / f"{dataset_file.name}.sha256").is_file()
    index_lines = (
        (experiment_root / EXPERIMENT_INDEX_PATH).read_text(encoding="utf-8").strip().splitlines()
    )
    assert any(
        f'"id":"{dataset_hash}"' in line and '"kind":"dataset"' in line for line in index_lines
    )


@pytest.mark.integration
def test_promote_oracle_writes_manifest_and_index(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    matlab_batch_dir = (
        tmp_path / "matlab-source" / "01_Input" / "matlab_results" / "batch_260421-151654"
    )
    matlab_batch_dir.parent.mkdir(parents=True, exist_ok=True)
    from .support import _materialize_exact_matlab_batch

    _materialize_exact_matlab_batch(tmp_path / "matlab-source")
    oracle_root = experiment_root / "oracles" / "oracle-a"
    dataset_file = tmp_path / "input.tif"
    dataset_file.write_bytes(b"tiff")

    parity_experiment.main(
        [
            "promote-oracle",
            "--matlab-batch-dir",
            str(matlab_batch_dir),
            "--oracle-root",
            str(oracle_root),
            "--dataset-file",
            str(dataset_file),
            "--oracle-id",
            "oracle-a",
        ]
    )

    manifest = json.loads((oracle_root / ORACLE_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert manifest["oracle_id"] == "oracle-a"
    assert manifest["dataset_hash"] == parity_experiment.fingerprint_file(dataset_file)
    assert (experiment_root / EXPERIMENT_INDEX_PATH).is_file()


@pytest.mark.integration
def test_promote_report_copies_analysis_and_writes_manifest(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    run_root = experiment_root / "runs" / "trial-a"
    parity_experiment.ensure_dest_run_layout(run_root)
    _write_json(run_root / SUMMARY_JSON_PATH, {"passed": True})
    _write_json(run_root / RUN_MANIFEST_PATH, {"run_id": "trial-a"})

    parity_experiment.main(
        [
            "promote-report",
            "--run-root",
            str(run_root),
        ]
    )

    report_root = experiment_root / "reports" / "trial-a"
    report_manifest = json.loads((report_root / REPORT_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert report_manifest["source_run_id"] == "trial-a"
    assert (report_root / SUMMARY_JSON_PATH).is_file()
