"""Tests for the rerun-python command in the parity experiment runner."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from dev.tests.support.payload_builders import (
    build_edges_payload,
    build_energy_result,
    build_network_payload,
    build_vertices_payload,
)
from dev.tests.support.run_state_builders import (
    materialize_checkpoint_surface,
    materialize_run_snapshot,
)

from .support import (
    _build_experiment_root,
    _exact_validated_params,
    _materialize_exact_matlab_batch,
    _write_json,
)
from source.analysis.parity.constants import (
    CHECKPOINTS_DIR,
    EXPERIMENT_INDEX_PATH,
    PYTHON_DERIVED_PARAMS_PATH,
    RUN_MANIFEST_PATH,
    SHARED_PARAMS_PATH,
    SUMMARY_JSON_PATH,
)

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")


@pytest.mark.integration
def test_rerun_python_creates_fresh_dest_root_and_writes_summary(tmp_path, monkeypatch):
    experiment_root = _build_experiment_root(tmp_path)
    source_run_root = experiment_root / "runs" / "source-run"
    dest_run_root = experiment_root / "runs" / "dest-run"
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"placeholder-tiff")

    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices", "edges", "network"),
        payloads={
            "energy": build_energy_result(),
            "vertices": build_vertices_payload(
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                ]
            ),
            "edges": build_edges_payload(
                traces=[
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                ],
                connections=[[0, 1], [1, 2]],
            ),
            "network": build_network_payload(strands=[[0, 1, 2]]),
        },
    )
    _write_json(
        source_run_root / "03_Analysis" / "comparison_report.json",
        {
            "matlab": {
                "vertices_count": 4,
                "edges_count": 5,
                "strand_count": 3,
            },
            "python": {
                "vertices_count": 4,
                "edges_count": 2,
                "network_strands_count": 1,
            },
            "vertices": {
                "matlab_count": 4,
                "python_count": 4,
            },
            "edges": {
                "matlab_count": 5,
                "python_count": 2,
            },
            "network": {
                "matlab_strand_count": 3,
                "python_strand_count": 1,
            },
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        {"number_of_edges_per_vertex": 4},
    )
    materialize_run_snapshot(
        source_run_root,
        {
            "run_id": "run-1",
            "provenance": {
                "input_file": str(input_file),
            },
        },
    )
    source_edges_checkpoint = (
        source_run_root / CHECKPOINTS_DIR / "checkpoint_edges.pkl"
    )
    source_edges_bytes = source_edges_checkpoint.read_bytes()

    calls: list[dict[str, object]] = []

    class FakeProcessor:
        def process_image(
            self,
            image,
            parameters,
            *,
            run_dir=None,
            force_rerun_from=None,
            **_kwargs,
        ):
            calls.append(
                {
                    "shape": tuple(image.shape),
                    "parameters": dict(parameters),
                    "run_dir": run_dir,
                    "force_rerun_from": force_rerun_from,
                }
            )
            checkpoint_dir = Path(run_dir) / CHECKPOINTS_DIR
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            from joblib import dump

            dump(
                build_edges_payload(
                    traces=[
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    ],
                    connections=[[0, 1], [1, 2], [2, 3]],
                ),
                checkpoint_dir / "checkpoint_edges.pkl",
            )
            dump(
                build_network_payload(strands=[[0, 1, 2], [2, 3]]),
                checkpoint_dir / "checkpoint_network.pkl",
            )
            return {}

    monkeypatch.setattr("source.io.tiff.load_tiff_volume", lambda _path: np.ones((2, 2, 2)))
    monkeypatch.setattr("source.core.pipeline.SLAVVProcessor", FakeProcessor)

    parity_experiment.main(
        [
            "rerun-python",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
            "--rerun-from",
            "edges",
        ]
    )

    assert source_edges_checkpoint.read_bytes() == source_edges_bytes
    assert calls == [
        {
            "shape": (2, 2, 2),
            "parameters": {"number_of_edges_per_vertex": 4},
            "run_dir": str(dest_run_root.resolve()),
            "force_rerun_from": "edges",
        }
    ]

    summary_payload = json.loads(
        (dest_run_root / SUMMARY_JSON_PATH).read_text(encoding="utf-8")
    )
    assert summary_payload["matlab_counts"] == {"vertices": 4, "edges": 5, "strands": 3}
    assert summary_payload["source_python_counts"] == {"vertices": 4, "edges": 2, "strands": 1}
    assert summary_payload["new_python_counts"] == {"vertices": 4, "edges": 3, "strands": 2}
    assert summary_payload["diff_vs_matlab"] == {"vertices": 0, "edges": -2, "strands": -1}
    assert summary_payload["diff_vs_source_python"] == {"vertices": 0, "edges": 1, "strands": 1}
    assert (dest_run_root / "00_Refs" / "source_comparison_report.json").is_file()
    assert (dest_run_root / "00_Refs" / "source_validated_params.json").is_file()
    assert (dest_run_root / SHARED_PARAMS_PATH).is_file()
    assert (dest_run_root / PYTHON_DERIVED_PARAMS_PATH).is_file()
    run_manifest = json.loads(
        (dest_run_root / RUN_MANIFEST_PATH).read_text(encoding="utf-8")
    )
    assert run_manifest["dataset_hash"] == parity_experiment.fingerprint_file(input_file)
    index_lines = (
        (experiment_root / EXPERIMENT_INDEX_PATH)
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert any('"command":"rerun-python"' in line for line in index_lines)


@pytest.mark.integration
def test_rerun_python_syncs_exact_vertex_checkpoint_from_matlab(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
    input_file = tmp_path / "input.tif"
    input_file.write_bytes(b"placeholder-tiff")
    _materialize_exact_matlab_batch(source_run_root)

    materialize_checkpoint_surface(
        source_run_root,
        stages=("energy", "vertices", "edges", "network"),
        payloads={
            "energy": {"energy_origin": "python_native_hessian"},
            "vertices": {
                "positions": np.zeros((2, 3), dtype=np.float32),
                "scales": np.zeros((2,), dtype=np.int16),
                "energies": np.zeros((2,), dtype=np.float32),
                "radii_microns": np.array([1.0, 1.0], dtype=np.float32),
                "count": 2,
            },
            "edges": build_edges_payload(
                traces=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
                connections=[[0, 1]],
            ),
            "network": build_network_payload(strands=[[0, 1]]),
        },
    )
    _write_json(
        source_run_root / "03_Analysis" / "comparison_report.json",
        {
            "matlab": {"vertices_count": 2, "edges_count": 1, "strand_count": 1},
            "python": {"vertices_count": 2, "edges_count": 1, "network_strands_count": 1},
            "vertices": {"matlab_count": 2, "python_count": 2},
            "edges": {"matlab_count": 1, "python_count": 1},
            "network": {"matlab_strand_count": 1, "python_strand_count": 1},
        },
    )
    _write_json(
        source_run_root / "99_Metadata" / "validated_params.json",
        _exact_validated_params(),
    )
    materialize_run_snapshot(
        source_run_root,
        {"run_id": "run-1", "provenance": {"input_file": str(input_file)}},
    )

    class FakeProcessor:
        def process_image(
            self, image, parameters, *, run_dir=None, force_rerun_from=None, **_kwargs
        ):
            from joblib import load

            checkpoint_vertices = load(
                Path(run_dir) / CHECKPOINTS_DIR / "checkpoint_vertices.pkl"
            )
            np.testing.assert_array_equal(
                checkpoint_vertices["positions"],
                np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                checkpoint_vertices["scales"],
                np.array([1, 2], dtype=np.int16),
            )
            np.testing.assert_array_equal(
                checkpoint_vertices["energies"],
                np.array([-2.0, -1.0], dtype=np.float32),
            )
            from joblib import dump

            checkpoint_dir = Path(run_dir) / CHECKPOINTS_DIR
            dump(
                build_edges_payload(
                    traces=[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]], connections=[[0, 1]]
                ),
                checkpoint_dir / "checkpoint_edges.pkl",
            )
            dump(build_network_payload(strands=[[0, 1]]), checkpoint_dir / "checkpoint_network.pkl")
            return {}

    monkeypatch.setattr("source.io.tiff.load_tiff_volume", lambda _path: np.ones((2, 2, 2)))
    monkeypatch.setattr("source.core.pipeline.SLAVVProcessor", FakeProcessor)

    parity_experiment.main(
        [
            "rerun-python",
            "--source-run-root",
            str(source_run_root),
            "--dest-run-root",
            str(dest_run_root),
            "--rerun-from",
            "edges",
        ]
    )

    provenance = json.loads(
        (dest_run_root / "99_Metadata" / "experiment_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["exact_vertex_checkpoint_sync"] is True
