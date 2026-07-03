"""Property-based tests for network graph serialization round-trip.

# Feature: matlab-python-parity, Property 15: Network Graph Serialization Round-Trip

Verifies that any valid NetworkResult, when serialized to network.json via the
authoritative build_network_json_payload exporter and then deserialized back via
load_network_json_payload, preserves:
  - The strand endpoint-pair multiset (order-independent)
  - The bifurcation multiset (order-independent)
  - The vertex-degree array (element-wise)

Validates: Requirements 11.4
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slavv_python.storage.exporters.json_v1 import (
    build_network_json_payload,
    load_network_json_payload,
)

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _strand_endpoint_pair(strand: list[int]) -> tuple[int, int]:
    """Return the canonical (sorted) endpoint pair for a strand."""
    first = strand[0]
    last = strand[-1]
    return (min(first, last), max(first, last))


def _endpoint_pair_multiset(strands: list[list[int]]) -> Counter[tuple[int, int]]:
    """Build a multiset of endpoint pairs from a list of strands."""
    return Counter(_strand_endpoint_pair(s) for s in strands if len(s) >= 2)


def _bifurcation_multiset(bifurcations: np.ndarray) -> Counter[int]:
    """Build a multiset of bifurcation vertex indices."""
    return Counter(int(v) for v in bifurcations)


# ---------------------------------------------------------------------------
# Property 15: Network Graph Serialization Round-Trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    strand_count=st.integers(min_value=0, max_value=20),
    bifurcation_count=st.integers(min_value=0, max_value=10),
    vertex_count=st.integers(min_value=2, max_value=50),
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_network_serialization_roundtrip(
    strand_count: int,
    bifurcation_count: int,
    vertex_count: int,
) -> None:
    """Serializing a NetworkResult to JSON and loading back preserves topology.

    Checks that:
    1. The strand endpoint-pair multiset matches (order-independent).
    2. The bifurcation multiset matches (order-independent).
    3. The vertex-degree array is element-wise equal.
    """
    # Build strands: lists of vertex indices, each with ≥2 entries
    rng = np.random.default_rng(seed=(strand_count * 1000 + bifurcation_count * 100 + vertex_count))
    strands: list[list[int]] = []
    for _ in range(strand_count):
        length = rng.integers(2, min(7, vertex_count + 1))
        strand = rng.integers(0, vertex_count, size=int(length)).tolist()
        strands.append([int(v) for v in strand])

    # Build bifurcations: integer vertex indices in [0, vertex_count)
    bifurcations = np.array(
        [int(rng.integers(0, vertex_count)) for _ in range(bifurcation_count)],
        dtype=np.int32,
    )

    # Build vertex_degrees: one degree value per vertex
    vertex_degrees = rng.integers(0, 5, size=vertex_count).astype(np.int32)

    # Construct the processing results payload that build_network_json_payload expects
    processing_results: dict[str, Any] = {
        "parameters": {},
        "network": {
            "strands": strands,
            "bifurcations": bifurcations,
            "vertex_degrees": vertex_degrees,
        },
    }

    payload = build_network_json_payload(processing_results)

    # Serialize to a temp file and load back
    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = Path(tmp_dir) / "network.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        loaded = load_network_json_payload(json_path)

    loaded_network = loaded["network"]
    loaded_strands: list[list[int]] = loaded_network.get("strands", [])
    loaded_bifurcations: np.ndarray = loaded_network.get(
        "bifurcations", np.array([], dtype=np.int32)
    )
    loaded_vertex_degrees: np.ndarray = loaded_network.get(
        "vertex_degrees", np.array([], dtype=np.int32)
    )

    # 1. Strand endpoint-pair multiset (order-independent comparison)
    original_ep_multiset = _endpoint_pair_multiset(strands)
    loaded_ep_multiset = _endpoint_pair_multiset(loaded_strands)
    assert original_ep_multiset == loaded_ep_multiset, (
        f"Strand endpoint-pair multiset mismatch after round-trip.\n"
        f"  Original: {dict(original_ep_multiset)}\n"
        f"  Loaded:   {dict(loaded_ep_multiset)}\n"
        f"  strands={strands}"
    )

    # 2. Bifurcation multiset (order-independent comparison)
    original_bif_multiset = _bifurcation_multiset(bifurcations)
    loaded_bif_multiset = _bifurcation_multiset(loaded_bifurcations)
    assert original_bif_multiset == loaded_bif_multiset, (
        f"Bifurcation multiset mismatch after round-trip.\n"
        f"  Original: {dict(original_bif_multiset)}\n"
        f"  Loaded:   {dict(loaded_bif_multiset)}\n"
        f"  bifurcations={bifurcations.tolist()}"
    )

    # 3. Vertex-degree array (element-wise equality)
    assert np.array_equal(vertex_degrees, loaded_vertex_degrees), (
        f"Vertex-degree array mismatch after round-trip.\n"
        f"  Original: {vertex_degrees.tolist()}\n"
        f"  Loaded:   {loaded_vertex_degrees.tolist()}"
    )
