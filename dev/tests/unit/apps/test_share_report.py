from pathlib import Path

import pytest
from dev.tests.support.payload_builders import (
    build_edges_payload,
    build_network_payload,
    build_processing_results,
    build_vertices_payload,
)

from slavv.apps.share_report import (
    build_share_report_html,
    make_share_report_filename,
    record_share_event,
)

np = pytest.importorskip("numpy")


@pytest.fixture
def sample_processing_results():
    vertices = build_vertices_payload(
        positions=[[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
        energies=np.array([-1.2, -1.0, -0.9], dtype=float),
        radii_microns=np.array([2.0, 2.5, 3.0], dtype=float),
    )
    edges = build_edges_payload(
        traces=[
            np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float),
            np.array([[0.0, 4.0, 0.0], [3.0, 4.0, 0.0]], dtype=float),
        ],
        connections=np.array([[0, 1], [1, 2]], dtype=int),
        energies=np.array([-1.0, -0.8], dtype=float),
    )
    network = build_network_payload(
        strands=[[0, 1, 2]],
        bifurcations=np.array([], dtype=int),
        edges=edges,
    )

    return build_processing_results(
        vertices=vertices,
        edges=edges,
        network=network,
        parameters={
            "microns_per_voxel": [1.0, 1.0, 1.0],
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 3.0,
            "scales_per_octave": 1.0,
            "approximating_PSF": False,
            "number_of_edges_per_vertex": 2,
        },
    )


def test_build_share_report_html_contains_core_sections(sample_processing_results):
    report = build_share_report_html(
        sample_processing_results,
        dataset_name="sample_volume.tif",
        image_shape=(8, 8, 8),
    )

    assert report["file_name"] == "sample_volume_share_report.html"
    assert report["signature"]
    assert "sample_volume.tif" in report["html"]
    assert "Headline Metrics" in report["html"]
    assert "Interactive Network" in report["html"]
    assert "Distribution Snapshot" in report["html"]
    assert "Plotly.newPlot" in report["html"]
    assert "Total Length" in report["html"]


def test_record_share_event_updates_state_and_appends_jsonl(tmp_path: Path):
    state = {"share_report_event_log_path": str(tmp_path / "events.jsonl")}

    log_path = record_share_event(
        state,
        "share_report_downloaded",
        "sample_volume.tif",
        "abc123",
        extra={"report_file_name": make_share_report_filename("sample_volume.tif")},
    )

    assert log_path == tmp_path / "events.jsonl"
    assert state["share_report_metrics"]["share_report_downloaded"] == 1

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"event": "share_report_downloaded"' in lines[0]
    assert '"report_signature": "abc123"' in lines[0]
