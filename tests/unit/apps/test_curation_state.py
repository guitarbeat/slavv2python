import numpy as np

from slavv.apps.curation_state import (
    build_curation_stats_rows,
    summarize_processing_counts,
    sync_curated_processing_results,
)


def _sample_processing_results():
    return {
        "vertices": {
            "positions": np.array(
                [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
                dtype=float,
            ),
            "radii_microns": np.array([2.0, 2.5, 3.0], dtype=float),
        },
        "edges": {
            "traces": [
                np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float),
                np.array([[0.0, 4.0, 0.0], [3.0, 4.0, 0.0]], dtype=float),
            ],
            "connections": np.array([[0, 1], [1, 2]], dtype=int),
            "energies": np.array([-1.0, -0.8], dtype=float),
        },
        "network": {
            "strands": [[0, 1, 2]],
            "bifurcations": np.array([], dtype=int),
        },
        "parameters": {
            "microns_per_voxel": [1.0, 1.0, 1.0],
            "remove_cycles": True,
        },
    }


def test_sync_curated_processing_results_rebuilds_network_and_preserves_baseline():
    processing_results = _sample_processing_results()
    curated_vertices = {
        "positions": np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float),
        "radii_microns": np.array([2.0, 2.5], dtype=float),
    }
    curated_edges = {
        "traces": [np.array([[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], dtype=float)],
        "connections": np.array([[0, 1]], dtype=int),
        "energies": np.array([-1.0], dtype=float),
    }

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
