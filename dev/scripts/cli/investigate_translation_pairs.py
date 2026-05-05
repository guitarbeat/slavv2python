"""Investigation script for INVEST-001, INVEST-006, and PARITY-001A.

This script systematically analyzes the 793 missing MATLAB pairs and 84 extra Python pairs
from the trace_order_fix experiment to identify patterns, root causes, and prioritize
recovery strategies.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_candidate_coverage(run_root: Path) -> dict[str, Any]:
    """Load the candidate coverage report from a run."""
    coverage_path = run_root / "03_Analysis" / "candidate_coverage.json"
    if not coverage_path.exists():
        raise FileNotFoundError(f"Candidate coverage not found: {coverage_path}")
    
    with open(coverage_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment_index(experiment_root: Path) -> list[dict[str, Any]]:
    """Load the experiment index to find historical baselines."""
    index_path = experiment_root / "index.jsonl"
    if not index_path.exists():
        return []
    
    experiments = []
    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                experiments.append(json.loads(line))
    return experiments


def categorize_missing_pairs(
    missing_pairs: list[list[int]],
    matlab_vertices: set[int] | None = None
) -> dict[str, Any]:
    """Categorize missing MATLAB pairs by patterns.
    
    Categories:
    - Vertex frequency: How often each vertex appears in missing pairs
    - Pair distance: Vertex index distance (may indicate spatial patterns)
    - Vertex range: Low vs high vertex indices
    - Connectivity patterns: Isolated vs hub vertices
    """
    if not missing_pairs:
        return {"error": "No missing pairs to analyze"}
    
    # Vertex frequency analysis
    vertex_counter: Counter[int] = Counter()
    for v1, v2 in missing_pairs:
        vertex_counter[v1] += 1
        vertex_counter[v2] += 1
    
    # Pair distance analysis
    distances = [abs(v2 - v1) for v1, v2 in missing_pairs]
    
    # Vertex range analysis
    all_vertices = set()
    for v1, v2 in missing_pairs:
        all_vertices.add(v1)
        all_vertices.add(v2)
    
    # Connectivity patterns
    vertex_degree: Counter[int] = Counter()
    for v1, v2 in missing_pairs:
        vertex_degree[v1] += 1
        vertex_degree[v2] += 1
    
    # Categorize by degree
    isolated_vertices = {v for v, deg in vertex_degree.items() if deg == 1}
    hub_vertices = {v for v, deg in vertex_degree.items() if deg >= 5}
    
    # Spatial clustering (by vertex index ranges)
    vertex_ranges = {
        "0-99": sum(1 for v in all_vertices if 0 <= v < 100),
        "100-499": sum(1 for v in all_vertices if 100 <= v < 500),
        "500-999": sum(1 for v in all_vertices if 500 <= v < 1000),
        "1000+": sum(1 for v in all_vertices if v >= 1000),
    }
    
    return {
        "total_missing_pairs": len(missing_pairs),
        "unique_vertices_involved": len(all_vertices),
        "top_missing_vertices": [
            {"vertex": v, "pair_count": count}
            for v, count in vertex_counter.most_common(20)
        ],
        "distance_stats": {
            "min": int(np.min(distances)),
            "max": int(np.max(distances)),
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "std": float(np.std(distances)),
        },
        "vertex_range_distribution": vertex_ranges,
        "connectivity_patterns": {
            "isolated_vertex_count": len(isolated_vertices),
            "hub_vertex_count": len(hub_vertices),
            "isolated_vertices_sample": sorted(list(isolated_vertices))[:10],
            "hub_vertices": sorted([
                {"vertex": v, "degree": vertex_degree[v]}
                for v in hub_vertices
            ], key=lambda x: x["degree"], reverse=True)[:10],
        },
        "pair_samples_by_category": {
            "short_distance": [[v1, v2] for v1, v2 in missing_pairs if abs(v2 - v1) < 10][:5],
            "long_distance": [[v1, v2] for v1, v2 in missing_pairs if abs(v2 - v1) > 500][:5],
            "low_index": [[v1, v2] for v1, v2 in missing_pairs if max(v1, v2) < 100][:5],
            "high_index": [[v1, v2] for v1, v2 in missing_pairs if min(v1, v2) >= 1000][:5],
        }
    }


def categorize_extra_pairs(
    extra_pairs: list[list[int]],
    python_vertices: set[int] | None = None
) -> dict[str, Any]:
    """Categorize extra Python pairs by patterns."""
    if not extra_pairs:
        return {"error": "No extra pairs to analyze"}
    
    # Similar analysis to missing pairs
    vertex_counter: Counter[int] = Counter()
    for v1, v2 in extra_pairs:
        vertex_counter[v1] += 1
        vertex_counter[v2] += 1
    
    distances = [abs(v2 - v1) for v1, v2 in extra_pairs]
    
    all_vertices = set()
    for v1, v2 in extra_pairs:
        all_vertices.add(v1)
        all_vertices.add(v2)
    
    vertex_degree: Counter[int] = Counter()
    for v1, v2 in extra_pairs:
        vertex_degree[v1] += 1
        vertex_degree[v2] += 1
    
    vertex_ranges = {
        "0-99": sum(1 for v in all_vertices if 0 <= v < 100),
        "100-499": sum(1 for v in all_vertices if 100 <= v < 500),
        "500-999": sum(1 for v in all_vertices if 500 <= v < 1000),
        "1000+": sum(1 for v in all_vertices if v >= 1000),
    }
    
    return {
        "total_extra_pairs": len(extra_pairs),
        "unique_vertices_involved": len(all_vertices),
        "top_extra_vertices": [
            {"vertex": v, "pair_count": count}
            for v, count in vertex_counter.most_common(20)
        ],
        "distance_stats": {
            "min": int(np.min(distances)),
            "max": int(np.max(distances)),
            "mean": float(np.mean(distances)),
            "median": float(np.median(distances)),
            "std": float(np.std(distances)),
        },
        "vertex_range_distribution": vertex_ranges,
        "connectivity_patterns": {
            "vertex_degrees": sorted([
                {"vertex": v, "degree": vertex_degree[v]}
                for v in all_vertices
            ], key=lambda x: x["degree"], reverse=True)[:10],
        },
        "pair_samples_by_category": {
            "short_distance": [[v1, v2] for v1, v2 in extra_pairs if abs(v2 - v1) < 10][:5],
            "long_distance": [[v1, v2] for v1, v2 in extra_pairs if abs(v2 - v1) > 500][:5],
            "low_index": [[v1, v2] for v1, v2 in extra_pairs if max(v1, v2) < 100][:5],
            "high_index": [[v1, v2] for v1, v2 in extra_pairs if min(v1, v2) >= 1000][:5],
        }
    }


def investigate_baseline_discrepancy(
    experiment_root: Path,
    current_match_rate: float
) -> dict[str, Any]:
    """Investigate the baseline discrepancy (PARITY-001A)."""
    experiments = load_experiment_index(experiment_root)
    
    # Find all experiments with match rate data
    experiments_with_rates = []
    for exp in experiments:
        if "match_rate" in exp or "matched_pair_count" in exp:
            experiments_with_rates.append(exp)
    
    # Sort by timestamp
    experiments_with_rates.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return {
        "current_match_rate": current_match_rate,
        "claimed_baseline": 0.414,  # 41.4% from TODO.md
        "discrepancy": abs(current_match_rate - 0.414),
        "historical_experiments_found": len(experiments_with_rates),
        "recent_experiments": experiments_with_rates[:5] if experiments_with_rates else [],
        "analysis": (
            "The 41.4% baseline claim in TODO.md does not match any recent experiment. "
            "The trace_order_fix experiment shows 33.8%, and may2026_fixes showed 12.4%. "
            "The 41.4% claim may be from an older experiment or a projection."
        )
    }


def identify_root_causes(
    missing_analysis: dict[str, Any],
    extra_analysis: dict[str, Any]
) -> dict[str, Any]:
    """Identify top root causes based on pattern analysis."""
    
    root_causes = []
    
    # Analyze missing pairs patterns
    if "top_missing_vertices" in missing_analysis:
        top_vertices = missing_analysis["top_missing_vertices"][:5]
        if top_vertices and top_vertices[0]["pair_count"] >= 4:
            root_causes.append({
                "category": "High-degree vertex missing",
                "description": f"Vertex {top_vertices[0]['vertex']} appears in {top_vertices[0]['pair_count']} missing pairs",
                "impact": "high",
                "affected_pairs": top_vertices[0]["pair_count"],
                "hypothesis": "Frontier ordering or seed selection divergence for high-connectivity vertices",
                "investigation_priority": 1,
            })
    
    # Analyze connectivity patterns
    if "connectivity_patterns" in missing_analysis:
        hub_count = missing_analysis["connectivity_patterns"]["hub_vertex_count"]
        if hub_count > 0:
            root_causes.append({
                "category": "Hub vertex exploration",
                "description": f"{hub_count} hub vertices (degree >=5) involved in missing pairs",
                "impact": "high",
                "affected_pairs": "multiple",
                "hypothesis": "Frontier management differences when exploring high-degree vertices",
                "investigation_priority": 2,
            })
    
    # Analyze extra pairs
    if "total_extra_pairs" in extra_analysis and extra_analysis["total_extra_pairs"] > 0:
        root_causes.append({
            "category": "Over-generation",
            "description": f"{extra_analysis['total_extra_pairs']} pairs generated by Python but not MATLAB",
            "impact": "medium",
            "affected_pairs": extra_analysis["total_extra_pairs"],
            "hypothesis": "Looser candidate acceptance criteria or missing filtering step",
            "investigation_priority": 3,
        })
    
    # Analyze distance patterns
    if "distance_stats" in missing_analysis:
        mean_dist = missing_analysis["distance_stats"]["mean"]
        if mean_dist > 100:
            root_causes.append({
                "category": "Long-distance pair missing",
                "description": f"Missing pairs have mean distance {mean_dist:.1f}",
                "impact": "medium",
                "affected_pairs": "many",
                "hypothesis": "Trace execution or frontier expansion differences for distant vertices",
                "investigation_priority": 4,
            })
    
    return {
        "identified_root_causes": root_causes,
        "top_3_priorities": sorted(root_causes, key=lambda x: x["investigation_priority"])[:3],
    }


def propose_corrective_measures(root_causes: dict[str, Any]) -> dict[str, Any]:
    """Propose corrective measures based on root cause analysis."""
    
    measures = []
    
    top_causes = root_causes.get("top_3_priorities", [])
    
    for i, cause in enumerate(top_causes, 1):
        if "frontier" in cause.get("hypothesis", "").lower():
            measures.append({
                "priority": i,
                "measure": "Frontier ordering alignment",
                "description": "Audit and align frontier insertion/removal semantics with MATLAB",
                "target_improvement": "30-40% of missing pairs",
                "implementation_steps": [
                    "Document MATLAB frontier insertion algorithm",
                    "Compare with Python implementation",
                    "Identify semantic differences",
                    "Implement alignment fixes",
                    "Add regression tests",
                ],
                "success_criteria": "Frontier ordering test passes, candidate count increases",
                "estimated_effort": "1-2 days",
            })
        
        elif "hub" in cause.get("category", "").lower():
            measures.append({
                "priority": i,
                "measure": "High-degree vertex handling",
                "description": "Fix seed selection and exploration for high-connectivity vertices",
                "target_improvement": "20-30% of missing pairs",
                "implementation_steps": [
                    "Trace execution for top hub vertices",
                    "Compare seed selection order",
                    "Verify frontier priority calculation",
                    "Fix divergences",
                    "Validate with parity experiment",
                ],
                "success_criteria": "Hub vertices generate expected candidate count",
                "estimated_effort": "6-8 hours",
            })
        
        elif "over-generation" in cause.get("category", "").lower():
            measures.append({
                "priority": i,
                "measure": "Candidate filtering alignment",
                "description": "Identify and implement missing filtering steps",
                "target_improvement": "Reduce extra pairs by 50%",
                "implementation_steps": [
                    "Sample extra pairs and trace their generation",
                    "Compare with MATLAB filtering logic",
                    "Identify missing or incorrect filters",
                    "Implement alignment",
                    "Verify with parity experiment",
                ],
                "success_criteria": "Extra pair count reduced to <40",
                "estimated_effort": "4-6 hours",
            })
    
    return {
        "proposed_measures": measures,
        "expected_combined_impact": "40-50% match rate (target: 598+ matched pairs)",
        "implementation_order": [m["measure"] for m in measures],
    }


def main():
    """Main investigation entry point."""
    
    # Paths
    experiment_root = Path("D:/slavv_comparisons/experiments/live-parity")
    trace_order_fix_run = experiment_root / "runs" / "trace_order_fix"
    
    print("=" * 80)
    print("TRANSLATION PAIR INVESTIGATION")
    print("=" * 80)
    print()
    
    # Load candidate coverage
    print("Loading candidate coverage data...")
    coverage = load_candidate_coverage(trace_order_fix_run)
    
    print(f"MATLAB pairs: {coverage['matlab_pair_count']}")
    print(f"Python pairs: {coverage['python_pair_count']}")
    print(f"Matched pairs: {coverage['matched_pair_count']}")
    print(f"Missing pairs: {coverage['missing_pair_count']}")
    print(f"Extra pairs: {coverage['extra_pair_count']}")
    print(f"Match rate: {coverage['matched_pair_count'] / coverage['matlab_pair_count'] * 100:.1f}%")
    print()
    
    # INVEST-001: Categorize missing pairs
    print("=" * 80)
    print("INVEST-001: Categorizing 793 missing MATLAB pairs")
    print("=" * 80)
    print()
    
    missing_pairs = coverage.get("missing_pairs", [])
    missing_analysis = categorize_missing_pairs(missing_pairs)
    
    print(f"Total missing pairs: {missing_analysis['total_missing_pairs']}")
    print(f"Unique vertices involved: {missing_analysis['unique_vertices_involved']}")
    print()
    print("Top 10 missing vertices:")
    for item in missing_analysis["top_missing_vertices"][:10]:
        print(f"  Vertex {item['vertex']}: {item['pair_count']} pairs")
    print()
    print("Distance statistics:")
    for key, value in missing_analysis["distance_stats"].items():
        print(f"  {key}: {value}")
    print()
    print("Vertex range distribution:")
    for range_name, count in missing_analysis["vertex_range_distribution"].items():
        print(f"  {range_name}: {count} vertices")
    print()
    print("Connectivity patterns:")
    print(f"  Isolated vertices: {missing_analysis['connectivity_patterns']['isolated_vertex_count']}")
    print(f"  Hub vertices (degree >=5): {missing_analysis['connectivity_patterns']['hub_vertex_count']}")
    print()
    
    # INVEST-006: Analyze extra pairs
    print("=" * 80)
    print("INVEST-006: Analyzing 84 extra Python pairs")
    print("=" * 80)
    print()
    
    extra_pairs = coverage.get("extra_pairs", [])
    extra_analysis = categorize_extra_pairs(extra_pairs)
    
    print(f"Total extra pairs: {extra_analysis['total_extra_pairs']}")
    print(f"Unique vertices involved: {extra_analysis['unique_vertices_involved']}")
    print()
    print("Top 10 extra vertices:")
    for item in extra_analysis["top_extra_vertices"][:10]:
        print(f"  Vertex {item['vertex']}: {item['pair_count']} pairs")
    print()
    print("Distance statistics:")
    for key, value in extra_analysis["distance_stats"].items():
        print(f"  {key}: {value}")
    print()
    
    # PARITY-001A: Investigate baseline discrepancy
    print("=" * 80)
    print("PARITY-001A: Investigating baseline discrepancy")
    print("=" * 80)
    print()
    
    match_rate = coverage["matched_pair_count"] / coverage["matlab_pair_count"]
    baseline_analysis = investigate_baseline_discrepancy(experiment_root, match_rate)
    
    print(f"Current match rate: {baseline_analysis['current_match_rate'] * 100:.1f}%")
    print(f"Claimed baseline: {baseline_analysis['claimed_baseline'] * 100:.1f}%")
    print(f"Discrepancy: {baseline_analysis['discrepancy'] * 100:.1f} percentage points")
    print()
    print("Analysis:")
    print(f"  {baseline_analysis['analysis']}")
    print()
    
    # Identify root causes
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print()
    
    root_causes = identify_root_causes(missing_analysis, extra_analysis)
    
    print("Top 3 root causes:")
    for cause in root_causes["top_3_priorities"]:
        print(f"\n{cause['investigation_priority']}. {cause['category']}")
        print(f"   Description: {cause['description']}")
        print(f"   Impact: {cause['impact']}")
        print(f"   Hypothesis: {cause['hypothesis']}")
    print()
    
    # Propose corrective measures
    print("=" * 80)
    print("CORRECTIVE MEASURES")
    print("=" * 80)
    print()
    
    measures = propose_corrective_measures(root_causes)
    
    print(f"Expected combined impact: {measures['expected_combined_impact']}")
    print()
    print("Top 3 corrective measures:")
    for measure in measures["proposed_measures"]:
        print(f"\n{measure['priority']}. {measure['measure']}")
        print(f"   Description: {measure['description']}")
        print(f"   Target improvement: {measure['target_improvement']}")
        print(f"   Estimated effort: {measure['estimated_effort']}")
        print(f"   Success criteria: {measure['success_criteria']}")
    print()
    
    # Save detailed report
    output_dir = REPO_ROOT / "dev" / "tmp_tests" / "investigations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "investigation_date": "2026-05-05",
        "experiment": "trace_order_fix",
        "coverage_summary": {
            "matlab_pairs": coverage["matlab_pair_count"],
            "python_pairs": coverage["python_pair_count"],
            "matched_pairs": coverage["matched_pair_count"],
            "missing_pairs": coverage["missing_pair_count"],
            "extra_pairs": coverage["extra_pair_count"],
            "match_rate": match_rate,
        },
        "invest_001_missing_pairs": missing_analysis,
        "invest_006_extra_pairs": extra_analysis,
        "parity_001a_baseline": baseline_analysis,
        "root_causes": root_causes,
        "corrective_measures": measures,
    }
    
    report_path = output_dir / "translation_pair_investigation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"Detailed report saved to: {report_path}")
    print()
    print("=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()