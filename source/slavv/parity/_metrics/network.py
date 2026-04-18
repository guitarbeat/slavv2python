from __future__ import annotations

from collections import Counter
from typing import Any

from .counts import _count_items, _infer_strand_count, _resolve_count
from .signatures import _sample_counter_diff, _strand_signatures


def compare_networks(
    matlab_network: dict[str, Any],
    python_network: dict[str, Any],
    matlab_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare network-level statistics."""
    if (
        matlab_stats is None
        and "strands_to_vertices" not in matlab_network
        and "strands" not in matlab_network
    ):
        matlab_stats = matlab_network
        matlab_network = {}

    comparison = {
        "matlab_strand_count": _resolve_count(
            (matlab_stats or {}).get("strand_count"),
            _resolve_count(
                (matlab_network or {}).get("strand_count"),
                _count_items(
                    (matlab_network or {}).get(
                        "strands_to_vertices",
                        (matlab_network or {}).get("strands"),
                    )
                ),
            ),
        ),
        "python_strand_count": _infer_strand_count(python_network),
        "exact_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
    }

    matlab_count = comparison["matlab_strand_count"]
    python_count = comparison["python_strand_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["strand_count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["strand_count_percent_difference"] = (
                comparison["strand_count_difference"] / avg_count
            ) * 100.0

    matlab_counter = Counter(_strand_signatures(matlab_network or {}))
    python_counter = Counter(_strand_signatures(python_network))
    comparison["exact_match"] = matlab_counter == python_counter
    comparison["matlab_only_samples"] = _sample_counter_diff(matlab_counter, python_counter)
    comparison["python_only_samples"] = _sample_counter_diff(python_counter, matlab_counter)
    return comparison
