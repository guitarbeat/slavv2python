"""Tests for the deduplication and cleanup commands in the parity experiment runner."""

from __future__ import annotations

import importlib
import json

import pytest

from slavv_python.analysis.parity.constants import EXPERIMENT_INDEX_PATH
from slavv_python.analysis.parity.index import deduplicate_index_records

from .support import _build_experiment_root

parity_experiment = importlib.import_module("scripts.parity_experiment")


@pytest.mark.unit
def test_deduplicate_index_records_filters_stale_and_deduplicates(tmp_path):
    experiment_root = _build_experiment_root(tmp_path)
    index_path = experiment_root / EXPERIMENT_INDEX_PATH

    # Create directories that exist
    dataset_dir = experiment_root / "datasets" / "dataset_a"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Path that does not exist (stale)
    stale_dir = experiment_root / "runs" / "stale_run"

    # Define a set of records with duplicates and stale entries
    records = [
        # Record 1: dataset_a (exists)
        {
            "id": "dataset_a",
            "kind": "dataset",
            "path": str(dataset_dir),
            "status": "ready",
        },
        # Record 2: stale_run (stale path)
        {
            "id": "stale_run",
            "kind": "parity_run",
            "run_root": str(stale_dir),
            "status": "failed",
        },
        # Record 3: duplicate of dataset_a (should override older one because it is appended later)
        {
            "id": "dataset_a",
            "kind": "dataset",
            "path": str(dataset_dir),
            "status": "updated_ready",
        },
    ]

    # Write records to index.jsonl
    lines = [json.dumps(r) for r in records]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Run deduplication in dry-run mode
    removed_dry = deduplicate_index_records(experiment_root, dry_run=True)
    assert len(removed_dry) == 2  # Older duplicate of dataset_a and stale_run

    # Verify that in dry-run, the index file is not modified
    reloaded_dry = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(reloaded_dry) == 3

    # Run actual deduplication
    removed_actual = deduplicate_index_records(experiment_root, dry_run=False)
    assert len(removed_actual) == 2

    # Verify that the index file was modified
    reloaded_actual = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(reloaded_actual) == 1
    assert reloaded_actual[0]["id"] == "dataset_a"
    assert reloaded_actual[0]["status"] == "updated_ready"


@pytest.mark.integration
def test_cli_dedupe_command(tmp_path, monkeypatch, capsys):
    experiment_root = _build_experiment_root(tmp_path)
    index_path = experiment_root / EXPERIMENT_INDEX_PATH

    # Set CWD to the experiment root so that resolve_experiment_root finds it
    monkeypatch.chdir(experiment_root)

    dataset_dir = experiment_root / "datasets" / "dataset_b"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    records = [
        {
            "id": "dataset_b",
            "kind": "dataset",
            "path": str(dataset_dir),
            "status": "old",
        },
        {
            "id": "dataset_b",
            "kind": "dataset",
            "path": str(dataset_dir),
            "status": "new",
        },
    ]

    # Write to index
    lines = [json.dumps(r) for r in records]
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Call main command with dry-run
    parity_experiment.main(["dedupe", "--dry-run"])
    captured = capsys.readouterr()
    assert "[Dry Run Mode]" in captured.out
    assert "Found 1 stale or duplicate records" in captured.out

    # Call main command actually
    parity_experiment.main(["dedupe"])
    captured = capsys.readouterr()
    assert "Successfully cleaned up and updated index.jsonl!" in captured.out

    # Verify on disk
    reloaded = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(reloaded) == 1
    assert reloaded[0]["status"] == "new"
