from __future__ import annotations

from dev.tests.support.payload_builders import build_processing_results

from slavv.apps.dashboard_state import (
    load_dashboard_context,
    normalize_dashboard_results,
    resolve_dashboard_stats,
)
from slavv.runtime.run_state import RunSnapshot


def test_normalize_dashboard_results_returns_plain_dict_payload():
    processing_results = build_processing_results(overrides={"metadata": {"source": "dashboard"}})

    normalized = normalize_dashboard_results(processing_results)

    assert {"vertices", "edges", "network", "parameters", "energy_data"}.issubset(normalized)
    assert normalized["metadata"] == {"source": "dashboard"}


def test_resolve_dashboard_stats_requires_full_network_results():
    assert resolve_dashboard_stats(None, image_shape=(10, 10, 10)) is None
    assert resolve_dashboard_stats({"parameters": {}}, image_shape=(10, 10, 10)) is None


def test_resolve_dashboard_stats_uses_supplied_stats_builder():
    processing_results = build_processing_results()

    stats = resolve_dashboard_stats(
        processing_results,
        image_shape=(10, 10, 10),
        stats_builder=lambda results, *, image_shape: {
            "num_strands": len(results["network"]["strands"]),
            "image_shape": image_shape,
        },
    )

    assert stats == {"num_strands": 1, "image_shape": (10, 10, 10)}


def test_load_dashboard_context_normalizes_results_and_stats():
    processing_results = build_processing_results()
    seen_run_dirs: list[str] = []

    context = load_dashboard_context(
        {
            "current_run_dir": "run-dir",
            "processing_results": processing_results,
            "share_report_metrics": {"share_report_requested": 2},
            "dataset_name": "Sample dataset",
            "image_shape": (6, 5, 4),
        },
        snapshot_loader=lambda run_dir: (
            seen_run_dirs.append(run_dir) or RunSnapshot(run_id="run-123", stages={})
        ),
        stats_builder=lambda results, *, image_shape: {
            "num_strands": len(results["network"]["strands"]),
            "shape": image_shape,
        },
    )

    assert seen_run_dirs == ["run-dir"]
    assert context["run_dir"] == "run-dir"
    assert context["snapshot"].run_id == "run-123"
    assert context["dataset_name"] == "Sample dataset"
    assert context["share_metrics"] == {"share_report_requested": 2}
    assert context["results"]["vertices"]["positions"].shape == (3, 3)
    assert context["stats"] == {"num_strands": 1, "shape": (6, 5, 4)}
