#!/usr/bin/env python3
"""
Lightweight diagnostic CLI tool to compare two global watershed discovery traces.
Supports comparing .json and .jsonl trace files step-by-step to find the first divergence point.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Dict, Union


def load_trace(path: Path) -> List[Dict[str, Any]]:
    """Load watershed trace events from a JSON or JSONL file."""
    if not path.is_file():
        print(f"Error: Trace file not found: {path}", file=sys.stderr)
        sys.exit(1)

    events: List[Dict[str, Any]] = []
    
    # Try parsing as JSONL first, then fallback to single JSON array
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # Line is not valid JSON, might be a single JSON file instead of JSONL
                    break
    except Exception:
        pass

    # If JSONL load yielded no events, try loading as a single JSON array
    if not events:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    events = data
                elif isinstance(data, dict):
                    events = [data]
        except Exception as e:
            print(f"Error: Failed to parse trace file {path}: {e}", file=sys.stderr)
            sys.exit(1)

    return events


def format_event(event: Dict[str, Any]) -> str:
    """Format an event dict for user-friendly console display."""
    event_type = event.get("event", "unknown")
    if event_type == "iteration_start":
        return f"Iteration {event.get('iteration')}: Start at linear index {event.get('current_linear')} (energy={event.get('current_energy')})"
    elif event_type == "seed_selected":
        return f"  Seed {event.get('seed_idx')}: Selected linear index {event.get('selected_linear')} (energy={event.get('selected_energy')})"
    elif event_type == "join":
        return f"  JOIN Event: Connected vertex {event.get('vertex_a')} and {event.get('vertex_b')}"
    return str(event)


def compare_traces(events_a: List[Dict[str, Any]], events_b: List[Dict[str, Any]]) -> bool:
    """Compare two traces and print detailed divergence analysis."""
    print(f"Trace A: {len(events_a)} events loaded.")
    print(f"Trace B: {len(events_b)} events loaded.")
    print("-" * 60)

    divergence_found = False
    max_len = max(len(events_a), len(events_b))
    match_count = 0

    for i in range(max_len):
        if i >= len(events_a):
            print(f"\n[DIVERGENCE AT EVENT {i + 1}]")
            print(f"Trace A ended prematurely.")
            print(f"Trace B event: {format_event(events_b[i])}")
            divergence_found = True
            break
        if i >= len(events_b):
            print(f"\n[DIVERGENCE AT EVENT {i + 1}]")
            print(f"Trace B ended prematurely.")
            print(f"Trace A event: {format_event(events_a[i])}")
            divergence_found = True
            break

        ev_a = events_a[i]
        ev_b = events_b[i]

        # Key fields to compare
        type_a = ev_a.get("event")
        type_b = ev_b.get("event")

        mismatch_reason = ""
        if type_a != type_b:
            mismatch_reason = f"Event type mismatch: '{type_a}' vs '{type_b}'"
        elif type_a == "iteration_start":
            if ev_a.get("iteration") != ev_b.get("iteration"):
                mismatch_reason = f"Iteration number mismatch: {ev_a.get('iteration')} vs {ev_b.get('iteration')}"
            elif ev_a.get("current_linear") != ev_b.get("current_linear"):
                mismatch_reason = f"Current linear index mismatch: {ev_a.get('current_linear')} vs {ev_b.get('current_linear')}"
        elif type_a == "seed_selected":
            if ev_a.get("seed_idx") != ev_b.get("seed_idx"):
                mismatch_reason = f"Seed index mismatch: {ev_a.get('seed_idx')} vs {ev_b.get('seed_idx')}"
            elif ev_a.get("selected_linear") != ev_b.get("selected_linear"):
                mismatch_reason = f"Selected linear index mismatch: {ev_a.get('selected_linear')} vs {ev_b.get('selected_linear')}"
        elif type_a == "join":
            v_a1, v_a2 = sorted([ev_a.get("vertex_a", 0), ev_a.get("vertex_b", 0)])
            v_b1, v_b2 = sorted([ev_b.get("vertex_a", 0), ev_b.get("vertex_b", 0)])
            if (v_a1, v_a2) != (v_b1, v_b2):
                mismatch_reason = f"Join vertices mismatch: ({v_a1}, {v_a2}) vs ({v_b1}, {v_b2})"

        if mismatch_reason:
            print(f"\n[DIVERGENCE AT EVENT {i + 1}]")
            print(f"Reason: {mismatch_reason}")
            print(f"Trace A: {format_event(ev_a)}")
            print(f"Trace B: {format_event(ev_b)}")
            divergence_found = True
            break
        else:
            match_count += 1

    if not divergence_found:
        print(f"\n[OK] SUCCESS: Traces match exactly! All {match_count} events are identical.")
        return True
    else:
        print(f"\n[FAIL] FAILED: Traces diverged. Matched {match_count} events before divergence.")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two global watershed execution traces to locate divergences."
    )
    parser.add_argument("file_a", type=str, help="Path to first trace file (JSON/JSONL)")
    parser.add_argument("file_b", type=str, help="Path to second trace file (JSON/JSONL)")
    args = parser.parse_args()

    events_a = load_trace(Path(args.file_a))
    events_b = load_trace(Path(args.file_b))

    success = compare_traces(events_a, events_b)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
