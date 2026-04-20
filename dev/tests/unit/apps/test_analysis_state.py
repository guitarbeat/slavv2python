from __future__ import annotations

from dev.tests.support.payload_builders import build_energy_result, build_processing_results

from slavv.apps.analysis_state import (
    has_analysis_network,
    normalize_analysis_results,
    resolve_analysis_stats,
)


def test_normalize_analysis_results_returns_plain_dict_payload():
    processing_results = build_processing_results(overrides={"metadata": {"source": "analysis"}})

    normalized = normalize_analysis_results(processing_results)

    assert {"vertices", "edges", "network", "parameters", "energy_data"}.issubset(normalized)
    assert normalized["metadata"] == {"source": "analysis"}


def test_has_analysis_network_requires_network_payload():
    assert has_analysis_network(build_processing_results()) is True
    assert has_analysis_network({"energy_data": build_energy_result()}) is False


def test_resolve_analysis_stats_prefers_existing_stats():
    stats = resolve_analysis_stats(build_processing_results(), {"total_length": 5.0})

    assert stats == {"total_length": 5.0}


def test_resolve_analysis_stats_falls_back_to_processing_counts():
    stats = resolve_analysis_stats(build_processing_results(), None)

    assert stats["Vertices"] == 3
    assert stats["Edges"] == 2
