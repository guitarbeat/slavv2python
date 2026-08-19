"""Diff Python watershed execution trace against MATLAB golden frontier trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from slavv_python.analytics.parity.oracle.surfaces import validate_exact_proof_source_surface
from slavv_python.analytics.parity.proof.coordinator import (
    load_exact_energy_result,
    load_exact_vertex_set,
)
from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.edges.execution_tracing import JsonExecutionTracer
from slavv_python.pipeline.edges.matlab_get_edges_by_watershed import (
    _generate_edge_candidates_matlab_global_watershed,
)
from slavv_python.pipeline.vertices.painting import paint_vertex_center_image


class _TraceCompleteError(RuntimeError):
    """Internal signal used to stop a targeted replay after the requested window."""


class _BoundedJsonExecutionTracer(JsonExecutionTracer):
    def __init__(self, *args: Any, stop_after_iteration: int | None = None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._stop_after_iteration = stop_after_iteration

    def on_iteration_start(
        self, iteration: int, current_linear: int, current_energy: float
    ) -> None:
        if self._stop_after_iteration is not None and int(iteration) > self._stop_after_iteration:
            raise _TraceCompleteError
        super().on_iteration_start(iteration, current_linear, current_energy)


def _sanitize_json_line(line: str) -> str:
    sanitized = re.sub(r"(?<![A-Za-z])-Inf(?![A-Za-z])", "-Infinity", line)
    return re.sub(r"(?<![A-Za-z])Inf(?![A-Za-z])", "Infinity", sanitized)


def _load_trace(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not path.is_file():
        raise FileNotFoundError(f"Missing trace file: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(_sanitize_json_line(line)))
    return events


def _values_equal(key: str, left: Any, right: Any) -> bool:
    if left == right:
        return True
    if (
        key in {"current_linear", "selected_linear"}
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        return int(left) - 1 == int(right)
    if (
        key.endswith("energy")
        and isinstance(left, (int, float))
        and isinstance(right, (int, float))
    ):
        if not np.isfinite(left) and isinstance(right, (int, float)):
            # MATLAB logs -Inf at vertex pops; Python restores the true vertex energy.
            return True
        if not np.isfinite(left) and not np.isfinite(right):
            return np.sign(left) == np.sign(right)
        return bool(np.isclose(left, right, rtol=0.0, atol=1e-9))
    return False


def _filter_events(events: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event") in names]


def _filter_events_through_iteration(
    events: list[dict[str, Any]],
    stop_after_iteration: int | None,
) -> list[dict[str, Any]]:
    if stop_after_iteration is None:
        return events
    filtered: list[dict[str, Any]] = []
    for event in events:
        iteration = event.get("iteration")
        if iteration is None:
            continue
        if int(iteration) <= int(stop_after_iteration):
            filtered.append(event)
    return filtered


def _compare_sequences(
    *,
    matlab_events: list[dict[str, Any]],
    python_events: list[dict[str, Any]],
    event_name: str,
    keys: tuple[str, ...],
) -> dict[str, Any] | None:
    if event_name == "iteration_start":
        keys = tuple(key for key in keys if key != "current_energy")
    matlab_filtered = _filter_events(matlab_events, {event_name})
    python_filtered = _filter_events(python_events, {event_name})
    limit = min(len(matlab_filtered), len(python_filtered))
    for index in range(limit):
        matlab_row = matlab_filtered[index]
        python_row = python_filtered[index]
        mismatch = {
            key: {"matlab": matlab_row.get(key), "python": python_row.get(key)}
            for key in keys
            if not _values_equal(key, matlab_row.get(key), python_row.get(key))
        }
        if mismatch:
            return {
                "event": event_name,
                "index": index,
                "mismatch": mismatch,
                "matlab": matlab_row,
                "python": python_row,
            }
    if len(matlab_filtered) != len(python_filtered):
        return {
            "event": event_name,
            "index": limit,
            "mismatch": {
                "length": {
                    "matlab": len(matlab_filtered),
                    "python": len(python_filtered),
                }
            },
            "matlab": matlab_filtered[limit] if limit < len(matlab_filtered) else None,
            "python": python_filtered[limit] if limit < len(python_filtered) else None,
        }
    return None


def _regenerate_with_trace(
    *,
    run_dir: Path,
    trace_path: Path,
    state_iterations: list[int] | None = None,
    state_linear_targets: list[int] | None = None,
    trace_frontier_snapshots: bool = True,
    trace_core_events: bool = True,
    stop_after_iteration: int | None = None,
) -> dict[str, Any]:
    source_surface = validate_exact_proof_source_surface(run_dir)
    energy = load_exact_energy_result(source_surface)
    vertices = load_exact_vertex_set(source_surface, energy)
    params = load_json_dict(source_surface.validated_params_path) or {}
    params.setdefault("comparison_exact_network", True)
    params.setdefault("watershed_frontier_backend", "sorted")
    vertex_center_image = paint_vertex_center_image(vertices.positions, energy.energy.shape)
    microns_per_voxel = params.get("microns_per_voxel", [1.0, 1.0, 1.0])
    mpv_engine = np.asarray(microns_per_voxel, dtype=np.float64).copy()
    if len(mpv_engine) >= 3:
        # Match generate_watershed_candidates: params are MATLAB [dy,dx,dz];
        # engine re-permutes to [dy,dx,dz] via [[1,2,0]] on physical [dz,dy,dx].
        mpv_engine = mpv_engine[[2, 0, 1]]
    tracer = _BoundedJsonExecutionTracer(
        trace_path,
        state_iterations=state_iterations,
        state_linear_targets=state_linear_targets,
        trace_frontier_snapshots=trace_frontier_snapshots,
        trace_core_events=trace_core_events,
        stop_after_iteration=stop_after_iteration,
    )
    try:
        return _generate_edge_candidates_matlab_global_watershed(
            energy.energy,
            energy.scale_indices,
            vertices.positions,
            vertices.scales,
            energy.lumen_radius_microns,
            mpv_engine,
            vertex_center_image,
            params,
            tracer=tracer,
        )
    except _TraceCompleteError:
        return {
            "connections": np.zeros((0, 2), dtype=np.int32),
            "trace_stopped_after_iteration": stop_after_iteration,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=None,
        help="Optional; ignored for regeneration (oracle comes from the run dir).",
    )
    parser.add_argument(
        "--matlab-trace",
        type=Path,
        default=Path("workspace/scratch/matlab_edge_dump/frontier_trace.jsonl"),
    )
    parser.add_argument(
        "--python-trace",
        type=Path,
        default=Path("workspace/scratch/watershed_frontier_trace.jsonl"),
    )
    parser.add_argument(
        "--regenerate-python",
        action="store_true",
        help="Regenerate the Python trace before diffing.",
    )
    parser.add_argument(
        "--state-iteration",
        type=int,
        action="append",
        default=[],
        help="When regenerating, also emit strel_state rows for this watershed iteration.",
    )
    parser.add_argument(
        "--state-linear-target",
        type=int,
        action="append",
        default=[],
        help=(
            "When regenerating, also emit strel_state rows whose strel contains this "
            "0-based MATLAB-order linear index."
        ),
    )
    parser.add_argument(
        "--no-frontier-snapshots",
        action="store_true",
        help="Emit compact target frontier_action events without frontier_state snapshots.",
    )
    parser.add_argument(
        "--no-core-events",
        action="store_true",
        help="Suppress iteration_start, seed_selected, join, and join_skipped rows.",
    )
    parser.add_argument(
        "--stop-after-iteration",
        type=int,
        default=None,
        help="When regenerating, stop the replay after this watershed iteration is processed.",
    )
    args = parser.parse_args(argv)

    if args.oracle_root is not None and not args.oracle_root.exists():
        raise FileNotFoundError(f"Missing oracle root: {args.oracle_root}")

    if args.regenerate_python:
        payload = _regenerate_with_trace(
            run_dir=args.run_dir,
            trace_path=args.python_trace,
            state_iterations=args.state_iteration,
            state_linear_targets=args.state_linear_target,
            trace_frontier_snapshots=not args.no_frontier_snapshots,
            trace_core_events=not args.no_core_events,
            stop_after_iteration=args.stop_after_iteration,
        )
        print(
            json.dumps(
                {
                    "python_candidate_pairs": int(payload["connections"].shape[0]),
                    "python_trace": str(args.python_trace),
                    "oracle_root": str(args.oracle_root) if args.oracle_root is not None else None,
                    "trace_stopped_after_iteration": payload.get(
                        "trace_stopped_after_iteration"
                    ),
                },
                indent=2,
            )
        )

    matlab_events = _load_trace(args.matlab_trace)
    python_events = _load_trace(args.python_trace)
    if args.stop_after_iteration is not None:
        matlab_events = _filter_events_through_iteration(
            matlab_events, args.stop_after_iteration
        )
        python_events = _filter_events_through_iteration(
            python_events, args.stop_after_iteration
        )

    event_checks = [
        ("iteration_start", ("iteration", "current_linear", "current_energy")),
        ("seed_selected", ("iteration", "seed_idx", "selected_linear", "selected_energy")),
    ]
    if args.stop_after_iteration is None:
        # Join events in historical traces do not all carry iteration fields, so
        # only compare them when both traces are full-length.
        event_checks.append(("join", ("start_vertex", "end_vertex")))

    for event_name, keys in event_checks:
        divergence = _compare_sequences(
            matlab_events=matlab_events,
            python_events=python_events,
            event_name=event_name,
            keys=keys,
        )
        if divergence is not None:
            print(json.dumps({"status": "diverged", **divergence}, indent=2))
            return 1

    summary = {
        "status": "bounded_match" if args.stop_after_iteration is not None else "match",
        "matlab_events": len(matlab_events),
        "python_events": len(python_events),
        "iteration_start_count": len(_filter_events(matlab_events, {"iteration_start"})),
        "seed_selected_count": len(_filter_events(matlab_events, {"seed_selected"})),
        "join_count": len(_filter_events(matlab_events, {"join"})),
        "stop_after_iteration": args.stop_after_iteration,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
