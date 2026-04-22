"""Integration coverage for the developer parity experiment runner."""

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

parity_experiment = importlib.import_module("dev.scripts.cli.parity_experiment")


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.mark.integration
def test_rerun_python_creates_fresh_dest_root_and_writes_summary(tmp_path, monkeypatch):
    source_run_root = tmp_path / "source-run"
    dest_run_root = tmp_path / "dest-run"
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
        source_run_root / parity_experiment.CHECKPOINTS_DIR / "checkpoint_edges.pkl"
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
            checkpoint_dir = Path(run_dir) / parity_experiment.CHECKPOINTS_DIR
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

    monkeypatch.setattr(parity_experiment, "load_tiff_volume", lambda _path: np.ones((2, 2, 2)))
    monkeypatch.setattr(parity_experiment, "SLAVVProcessor", FakeProcessor)

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
        (dest_run_root / parity_experiment.SUMMARY_JSON_PATH).read_text(encoding="utf-8")
    )
    assert summary_payload["matlab_counts"] == {"vertices": 4, "edges": 5, "strands": 3}
    assert summary_payload["source_python_counts"] == {"vertices": 4, "edges": 2, "strands": 1}
    assert summary_payload["new_python_counts"] == {"vertices": 4, "edges": 3, "strands": 2}
    assert summary_payload["diff_vs_matlab"] == {"vertices": 0, "edges": -2, "strands": -1}
    assert summary_payload["diff_vs_source_python"] == {"vertices": 0, "edges": 1, "strands": 1}
    assert (
        dest_run_root / "99_Metadata" / "source_comparison_report.json"
    ).is_file()
    assert (
        dest_run_root / "99_Metadata" / "source_validated_params.json"
    ).is_file()
