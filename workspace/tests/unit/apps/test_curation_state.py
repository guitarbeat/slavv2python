import numpy as np
from slavv_python.apps.state.curation import (
    apply_curated_session_results,
    build_curation_stats_rows,
    summarize_processing_counts,
    sync_curated_processing_results,
)
from workspace.tests.support.payload_builders import (
    build_edges_payload,
    build_network_payload,
    build_processing_results,
    build_vertices_payload,
)


def _sample_processing_results():
    vertices = build_vertices_payload(
        positions=[[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
        radii_microns=[2.0, 2.5, 3.0],
    )
    edges = build_edges_payload(
        traces=[
            np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float),
            np.array([[0.0, 4.0, 0.0], [3.0, 4.0, 0.0]], dtype=float),
        ],
        connections=np.array([[0, 1], [1, 2]], dtype=int),
        energies=np.array([-1.0, -0.8], dtype=float),
    )
    network = build_network_payload(strands=[[0, 1, 2]], bifurcations=np.array([], dtype=int))
    return build_processing_results(
        vertices=vertices,
        edges=edges,
        network=network,
        parameters={"microns_per_voxel": [1.0, 1.0, 1.0], "remove_cycles": True},
    )


def test_sync_curated_processing_results_rebuilds_network_and_preserves_baseline():
    processing_results = _sample_processing_results()
    curated_vertices = build_vertices_payload(
        positions=np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float),
        radii_microns=np.array([2.0, 2.5], dtype=float),
    )
    curated_edges = build_edges_payload(
        traces=[np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float)],
        connections=np.array([[0, 1]], dtype=int),
        energies=np.array([-1.0], dtype=float),
    )

    updated_results, baseline_counts, current_counts = sync_curated_processing_results(
        processing_results,
        curated_vertices,
        curated_edges,
    )

    assert baseline_counts == {
        "Vertices": 3,
        "Edges": 2,
        "Strands": 1,
        "Bifurcations": 0,
    }
    assert current_counts == {
        "Vertices": 2,
        "Edges": 1,
        "Strands": 1,
        "Bifurcations": 0,
    }
    assert len(updated_results["network"]["strands"]) == 1
    assert processing_results["vertices"]["positions"].shape[0] == 3
    assert processing_results["edges"]["connections"].shape == (2, 2)


def test_sync_curated_processing_results_keeps_existing_baseline_counts():
    processing_results = _sample_processing_results()
    baseline_counts = {
        "Vertices": 10,
        "Edges": 8,
        "Strands": 4,
        "Bifurcations": 1,
    }

    updated_results, preserved_baseline, current_counts = sync_curated_processing_results(
        processing_results,
        processing_results["vertices"],
        processing_results["edges"],
        baseline_counts=baseline_counts,
    )

    assert preserved_baseline == baseline_counts
    assert current_counts == summarize_processing_counts(updated_results)


def test_apply_curated_session_results_updates_session_state_and_clears_share_signature():
    processing_results = _sample_processing_results()
    session_state = {
        "processing_results": processing_results,
        "curation_baseline_counts": {
            "Vertices": 10,
            "Edges": 8,
            "Strands": 4,
            "Bifurcations": 1,
        },
        "share_report_prepared_signature": "signature-123",
    }

    baseline_counts, current_counts = apply_curated_session_results(
        session_state,
        processing_results["vertices"],
        processing_results["edges"],
        curation_mode="Automatic (Rule-based)",
    )

    assert baseline_counts == {
        "Vertices": 10,
        "Edges": 8,
        "Strands": 4,
        "Bifurcations": 1,
    }
    assert current_counts == summarize_processing_counts(session_state["processing_results"])
    assert session_state["last_curation_mode"] == "Automatic (Rule-based)"
    assert "share_report_prepared_signature" not in session_state


def test_build_curation_stats_rows_reports_signed_change_percentages():
    rows = build_curation_stats_rows(
        {"Vertices": 10, "Edges": 5, "Strands": 2, "Bifurcations": 0},
        {"Vertices": 8, "Edges": 6, "Strands": 3, "Bifurcations": 1},
    )

    assert rows == [
        {
            "Component": "Vertices",
            "Original": 10,
            "Current": 8,
            "Delta": -2,
            "Change (%)": "-20.00",
        },
        {
            "Component": "Edges",
            "Original": 5,
            "Current": 6,
            "Delta": 1,
            "Change (%)": "+20.00",
        },
        {
            "Component": "Strands",
            "Original": 2,
            "Current": 3,
            "Delta": 1,
            "Change (%)": "+50.00",
        },
        {
            "Component": "Bifurcations",
            "Original": 0,
            "Current": 1,
            "Delta": 1,
            "Change (%)": "n/a",
        },
    ]
