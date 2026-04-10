"""Pytest configuration and fixtures.

Ensures the slavv package is importable when running tests from the repo root,
whether or not `pip install -e .` has been run.
"""

from __future__ import annotations

import collections
import json
import shutil
import sys
from collections import abc as collections_abc
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from scipy.io import savemat

# Keep the legacy Python 3.7 scientific stack importable without flooding test
# output with third-party deprecations from old networkx/sklearn releases.
collections.Mapping = collections_abc.Mapping
collections.Set = collections_abc.Set
collections.Iterable = collections_abc.Iterable
np.int = int
np.float = float
np.bool = np.bool_

# Add source/ to path so slavv is importable
repo_root = Path(__file__).resolve().parent.parent
source_dir = repo_root / "source"
if source_dir.exists() and str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))


def pytest_collection_modifyitems(items):
    """Auto-tag tests by folder so CI can select fast/full lanes."""
    for item in items:
        nodeid = item.nodeid.replace("\\", "/")
        if "tests/unit/" in nodeid:
            item.add_marker("unit")
        elif "tests/integration/" in nodeid:
            item.add_marker("integration")
        elif "tests/ui/" in nodeid:
            item.add_marker("ui")
        elif "tests/diagnostic/" in nodeid:
            item.add_marker("diagnostic")

        if "tests/benchmarks/" in nodeid:
            item.add_marker("slow")
        if "regression" in nodeid:
            item.add_marker("regression")


@pytest.fixture
def tmp_path():
    """Provide a writable temp directory without relying on pytest's lock-based tmpdir."""
    workspace_tmp_root = repo_root / "workspace" / "tmp_tests"
    workspace_tmp_root.mkdir(parents=True, exist_ok=True)
    path = workspace_tmp_root / f"run-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_json_fixture(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def matlab_artifact_builder():
    """Build reusable MATLAB batch/status fixtures for comparison tests."""

    def _builder(
        output_dir: Path,
        *,
        input_file: Path | None = None,
        batch_timestamp: str = "260401-120000",
        roi_name: str = "_r",
        completed_stages: tuple[str, ...] = (),
        running_status: str = "",
        log_lines: list[str] | None = None,
        partial_stage: str | None = None,
        chunk_names: tuple[str, ...] = (),
        resume_state_batch_reference: bool = False,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        effective_input = input_file or (output_dir.parent / "input.tif")
        effective_input.parent.mkdir(parents=True, exist_ok=True)
        if not effective_input.exists():
            effective_input.write_bytes(b"fake")

        batch_folder = output_dir / f"batch_{batch_timestamp}"
        settings_dir = batch_folder / "settings"
        settings_dir.mkdir(parents=True, exist_ok=True)
        savemat(
            settings_dir / "batch.mat",
            {
                "optional_input": [str(effective_input)],
                "ROI_names": [roi_name],
            },
        )

        data_dir = batch_folder / "data"
        vectors_dir = batch_folder / "vectors"
        data_dir.mkdir(parents=True, exist_ok=True)
        vectors_dir.mkdir(parents=True, exist_ok=True)

        if "energy" in completed_stages or partial_stage == "energy":
            (data_dir / f"energy_{batch_timestamp}_{roi_name}").write_text("", encoding="utf-8")
        if "vertices" in completed_stages:
            (vectors_dir / f"vertices_{batch_timestamp}_{roi_name}.mat").write_text(
                "", encoding="utf-8"
            )
            (vectors_dir / f"curated_vertices_{batch_timestamp}_{roi_name}.mat").write_text(
                "", encoding="utf-8"
            )
        if "edges" in completed_stages:
            (vectors_dir / f"edges_{batch_timestamp}_{roi_name}.mat").write_text(
                "", encoding="utf-8"
            )
            (vectors_dir / f"curated_edges_{batch_timestamp}_{roi_name}.mat").write_text(
                "", encoding="utf-8"
            )
        if "network" in completed_stages:
            (vectors_dir / f"network_{batch_timestamp}_{roi_name}.mat").write_text(
                "", encoding="utf-8"
            )

        if partial_stage == "energy":
            chunk_dir = data_dir / f"energy_{batch_timestamp}_{roi_name}_chunks_octave_2_of_6"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            for name in chunk_names:
                (chunk_dir / name).write_text("", encoding="utf-8")

        if running_status:
            _write_json_fixture(
                output_dir / "matlab_resume_state.json",
                {
                    "input_file": str(effective_input).replace("\\", "/"),
                    "output_directory": str(output_dir).replace("\\", "/"),
                    "batch_timestamp": batch_timestamp if resume_state_batch_reference else "",
                    "batch_folder": (
                        str(batch_folder).replace("\\", "/") if resume_state_batch_reference else ""
                    ),
                    "last_completed_stage": (completed_stages[-1] if completed_stages else ""),
                    "status": running_status,
                    "updated_at": "2026-04-01 12:27:34",
                },
            )

        if log_lines:
            (output_dir / "matlab_run.log").write_text("\n".join(log_lines), encoding="utf-8")

        return {
            "output_dir": output_dir,
            "input_file": effective_input,
            "batch_folder": batch_folder,
            "data_dir": data_dir,
            "vectors_dir": vectors_dir,
        }

    return _builder


@pytest.fixture
def comparison_metadata_builder():
    """Build reusable comparison metadata artifacts under ``99_Metadata``."""

    def _builder(
        run_dir: Path,
        *,
        run_snapshot: dict[str, object] | None = None,
        manifest_content: str | None = None,
        output_preflight: dict[str, object] | None = None,
        matlab_status: dict[str, object] | None = None,
    ) -> Path:
        metadata_dir = run_dir / "99_Metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        if run_snapshot is not None:
            _write_json_fixture(metadata_dir / "run_snapshot.json", run_snapshot)
        if output_preflight is not None:
            _write_json_fixture(metadata_dir / "output_preflight.json", output_preflight)
        if matlab_status is not None:
            _write_json_fixture(metadata_dir / "matlab_status.json", matlab_status)
        if manifest_content is not None:
            (metadata_dir / "run_manifest.md").write_text(manifest_content, encoding="utf-8")
        return metadata_dir

    return _builder
