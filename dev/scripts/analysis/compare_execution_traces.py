#!/usr/bin/env python3
"""Automated comparison tool for SLAVV execution traces (JSONL)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Divergence:
    iteration: int
    event_type: str
    key: str
    val1: Any
    val2: Any
    message: str


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            events.append(json.loads(line))
    return events


def compare_traces(path1: Path, path2: Path, energy_tol: float = 1e-5) -> list[Divergence]:
    trace1 = load_jsonl(path1)
    trace2 = load_jsonl(path2)
    
    divergences = []
    
    # Simple alignment by index for now (assuming events are emitted in same order)
    max_len = min(len(trace1), len(trace2))
    
    current_iteration = 0
    
    for i in range(max_len):
        e1 = trace1[i]
        e2 = trace2[i]
        
        if e1["event"] != e2["event"]:
            divergences.append(Divergence(
                current_iteration, e1["event"], "event_type", e1["event"], e2["event"],
                f"Event mismatch at index {i}"
            ))
            return divergences # Fatal alignment error
            
        if e1["event"] == "iteration_start":
            current_iteration = e1["iteration"]
            if e1["iteration"] != e2["iteration"]:
                divergences.append(Divergence(
                    current_iteration, "iteration_start", "iteration", e1["iteration"], e2["iteration"],
                    "Iteration number mismatch"
                ))
            if e1["current_linear"] != e2["current_linear"]:
                divergences.append(Divergence(
                    current_iteration, "iteration_start", "current_linear", e1["current_linear"], e2["current_linear"],
                    "Frontier pop mismatch (different linear index)"
                ))
                return divergences
                
        elif e1["event"] == "seed_selected":
            if e1["seed_idx"] != e2["seed_idx"]:
                 divergences.append(Divergence(
                    current_iteration, "seed_selected", "seed_idx", e1["seed_idx"], e2["seed_idx"],
                    "Seed index mismatch"
                ))
            if e1["selected_linear"] != e2["selected_linear"]:
                divergences.append(Divergence(
                    current_iteration, "seed_selected", "selected_linear", e1["selected_linear"], e2["selected_linear"],
                    "Selected seed location mismatch"
                ))
                return divergences
            
            # Numerical energy check
            if not math.isclose(e1["selected_energy"], e2["selected_energy"], rel_tol=energy_tol, abs_tol=energy_tol):
                 divergences.append(Divergence(
                    current_iteration, "seed_selected", "selected_energy", e1["selected_energy"], e2["selected_energy"],
                    f"Selected seed energy mismatch (delta={abs(e1['selected_energy'] - e2['selected_energy']):.2e})"
                ))
                 
        elif e1["event"] == "join":
             if {e1["start_vertex"], e1["end_vertex"]} != {e2["start_vertex"], e2["end_vertex"]}:
                 divergences.append(Divergence(
                    current_iteration, "join", "vertices", 
                    (e1["start_vertex"], e1["end_vertex"]), 
                    (e2["start_vertex"], e2["end_vertex"]),
                    "Join vertex mismatch"
                ))
                 return divergences

    if len(trace1) != len(trace2):
        print(f"WARNING: Traces have different lengths ({len(trace1)} vs {len(trace2)})")
        
    return divergences


def main():
    parser = argparse.ArgumentParser(description="Compare two SLAVV execution traces.")
    parser.add_argument("trace1", type=Path, help="First trace file (JSONL)")
    parser.add_argument("trace2", type=Path, help="Second trace file (JSONL)")
    parser.add_argument("--energy-tol", type=float, default=1e-5, help="Energy tolerance")
    
    args = parser.parse_args()
    
    if not args.trace1.is_file():
        print(f"Error: {args.trace1} not found")
        sys.exit(1)
    if not args.trace2.is_file():
        print(f"Error: {args.trace2} not found")
        sys.exit(1)
        
    print(f"Comparing traces:\n  1: {args.trace1}\n  2: {args.trace2}\n")
    
    divergences = compare_traces(args.trace1, args.trace2, energy_tol=args.energy_tol)
    
    if not divergences:
        print("✅ No divergences found. Traces match perfectly.")
    else:
        print(f"❌ Found {len(divergences)} divergence(s).\n")
        
        # Divergence summary
        summary = {}
        for d in divergences:
            key = f"{d.event_type}.{d.key}"
            summary[key] = summary.get(key, 0) + 1
            
        print("Divergence Summary:")
        for key, count in sorted(summary.items(), key=lambda item: item[1], reverse=True):
            print(f"  - {key}: {count}")
        print()
        
        # Show first divergence in detail
        first = divergences[0]
        print(f"FIRST DIVERGENCE at Iteration {first.iteration}:")
        print(f"  Event: {first.event_type}")
        print(f"  Key:   {first.key}")
        print(f"  Trace 1: {first.val1}")
        print(f"  Trace 2: {first.val2}")
        print(f"  Message: {first.message}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()
