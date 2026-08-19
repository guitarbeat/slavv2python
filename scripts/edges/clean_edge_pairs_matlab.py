"""Compare Python clean-edge-pair ordering against MATLAB ``clean_edge_pairs``.

This is a focused crop/funnel diagnostic. It exports the post-crop candidate
surface to a MATLAB ``.mat`` file, runs the reference MATLAB function, and
compares MATLAB's retained original row indices to Python's
``prepare_candidate_indices_for_cleanup`` output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from slavv_python.analytics.parity.oracle.surfaces import validate_exact_proof_source_surface
from slavv_python.analytics.parity.proof.coordinator import load_exact_energy_result
from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.edges.finalize import (
    _matlab_precrop_resample_from_maps,
    prefilter_edge_indices_for_cleanup_matlab_style,
)
from slavv_python.pipeline.edges.selection_payloads import (
    initialize_edge_selection_diagnostics,
    prepare_candidate_indices_for_cleanup,
)
from slavv_python.utils.safe_unpickle import safe_load


def _matlab_vectorization_source() -> Path:
    return Path("external") / "Vectorization-Public" / "source"


def _matlab_cell_column(arrays: list[np.ndarray]) -> np.ndarray:
    cell = np.empty((len(arrays), 1), dtype=object)
    for index, values in enumerate(arrays):
        cell[index, 0] = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return cell


def _run_matlab_compare(
    source_dir: Path,
    input_path: Path,
    output_path: Path,
    *,
    max_degree: int,
) -> None:
    command = (
        f"addpath('{source_dir.as_posix()}'); "
        f"load('{input_path.as_posix()}'); "
        "[~, original_edge_indices] = clean_edge_pairs(edges2vertices, edge_energies, false); "
        "edges2vertices_clean = edges2vertices(original_edge_indices, :); "
        "degree_indices = clean_edges_vertex_degree_excess("
        f"edges2vertices_clean, {int(max_degree)}); "
        "edges2vertices_degree = edges2vertices_clean(degree_indices, :); "
        "cycle_indices = clean_edges_cycles(edges2vertices_degree); "
        f"save('{output_path.as_posix()}', "
        "'original_edge_indices', 'degree_indices', 'cycle_indices');"
    )
    subprocess.run(["matlab", "-batch", command], check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("workspace/scratch/cleanup_compare"))
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args(argv)

    source = validate_exact_proof_source_surface(args.run_dir)
    energy = load_exact_energy_result(source)
    params = load_json_dict(source.validated_params_path) or {}
    candidates = safe_load(args.run_dir / "04_Edges" / "candidates.pkl")

    connections = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    traces = candidates["traces"]
    energy_traces = candidates["energy_traces"]
    scale_traces = candidates["scale_traces"]
    matlab_global = bool(candidates.get("matlab_global_watershed_exact", False))

    kept_after_crop, cropped = prefilter_edge_indices_for_cleanup_matlab_style(
        list(range(len(traces))),
        traces,
        scale_traces,
        energy_traces,
        lumen_radius_microns=np.asarray(energy.lumen_radius_microns, dtype=np.float32),
        microns_per_voxel=np.asarray(
            params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
        ),
        size_of_image=energy.energy.shape,
        energy_map=energy.energy,
        scale_indices=energy.scale_indices,
    )
    _, _, resampled_energy_traces = _matlab_precrop_resample_from_maps(
        [np.asarray(trace, dtype=np.float64) for trace in traces],
        [np.asarray(trace, dtype=np.float64) for trace in scale_traces],
        [np.asarray(trace, dtype=np.float64) for trace in energy_traces],
        energy_map=energy.energy,
        scale_indices=energy.scale_indices,
    )
    resampled_metrics = np.asarray(
        [
            np.max(energy_trace) if len(energy_trace) else -1000.0
            for energy_trace in resampled_energy_traces
        ],
        dtype=np.float64,
    )
    resampled_metrics[np.isnan(resampled_metrics)] = -1000.0

    diagnostics = initialize_edge_selection_diagnostics(candidates, connections, traces)
    python_indices = prepare_candidate_indices_for_cleanup(
        connections,
        resampled_metrics,
        resampled_energy_traces,
        diagnostics,
        subset_indices=kept_after_crop,
        reject_nonnegative_energy_edges=not matlab_global,
    )

    crop_row_by_candidate = {candidate_index: row for row, candidate_index in enumerate(kept_after_crop)}
    python_rows_one_based = np.asarray(
        [crop_row_by_candidate[index] + 1 for index in python_indices], dtype=np.int64
    )

    args.work_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.work_dir / "clean_edge_pairs_input.mat"
    output_path = args.work_dir / "clean_edge_pairs_output.mat"

    savemat(
        input_path,
        {
            "edges2vertices": np.asarray(connections[kept_after_crop] + 1, dtype=np.uint32),
            "edge_energies": _matlab_cell_column(
                [
                    np.asarray(resampled_energy_traces[index], dtype=np.float64)
                    for index in kept_after_crop
                ]
            ),
        },
        do_compression=True,
    )

    max_degree = max(1, int(params.get("number_of_edges_per_vertex", 4)))
    _run_matlab_compare(
        _matlab_vectorization_source(),
        input_path,
        output_path,
        max_degree=max_degree,
    )
    matlab_output = loadmat(output_path)
    matlab_rows_one_based = np.asarray(matlab_output["original_edge_indices"], dtype=np.int64).reshape(
        -1
    )

    min_len = min(len(python_rows_one_based), len(matlab_rows_one_based))
    mismatch_positions = np.flatnonzero(
        python_rows_one_based[:min_len] != matlab_rows_one_based[:min_len]
    )
    summary = {
        "cropped_edge_count": int(cropped),
        "python_count": len(python_rows_one_based),
        "matlab_count": len(matlab_rows_one_based),
        "same_length": bool(len(python_rows_one_based) == len(matlab_rows_one_based)),
        "mismatch_count_prefix": len(mismatch_positions),
        "first_mismatch_position": int(mismatch_positions[0]) if mismatch_positions.size else None,
        "python_only_count": len(set(python_rows_one_based.tolist()) - set(matlab_rows_one_based.tolist())),
        "matlab_only_count": len(set(matlab_rows_one_based.tolist()) - set(python_rows_one_based.tolist())),
    }
    print(json.dumps(summary, indent=2))

    if mismatch_positions.size:
        samples = []
        for pos in mismatch_positions[: max(0, args.sample_size)]:
            samples.append(
                {
                    "position": int(pos),
                    "python_crop_row_1based": int(python_rows_one_based[pos]),
                    "matlab_crop_row_1based": int(matlab_rows_one_based[pos]),
                    "python_candidate_index": int(kept_after_crop[python_rows_one_based[pos] - 1]),
                    "matlab_candidate_index": int(kept_after_crop[matlab_rows_one_based[pos] - 1]),
                }
            )
        print(json.dumps({"mismatch_samples": samples}, indent=2))

    keep_degree = __import__(
        "slavv_python.pipeline.edges.cleanup",
        fromlist=["remove_excess_vertex_degrees", "break_graph_cycles"],
    ).remove_excess_vertex_degrees(
        connections[python_indices],
        resampled_metrics[python_indices],
        max_degree,
    )
    python_degree_rows = np.flatnonzero(keep_degree).astype(np.int64) + 1
    matlab_degree_rows = np.asarray(matlab_output["degree_indices"], dtype=np.int64).reshape(-1)
    degree_mismatches = np.flatnonzero(
        python_degree_rows[: min(len(python_degree_rows), len(matlab_degree_rows))]
        != matlab_degree_rows[: min(len(python_degree_rows), len(matlab_degree_rows))]
    )
    print(
        json.dumps(
            {
                "degree_python_count": len(python_degree_rows),
                "degree_matlab_count": len(matlab_degree_rows),
                "degree_same_length": bool(len(python_degree_rows) == len(matlab_degree_rows)),
                "degree_mismatch_count_prefix": len(degree_mismatches),
                "degree_first_mismatch_position": int(degree_mismatches[0])
                if degree_mismatches.size
                else None,
            },
            indent=2,
        )
    )

    break_graph_cycles = __import__(
        "slavv_python.pipeline.edges.cleanup",
        fromlist=["break_graph_cycles"],
    ).break_graph_cycles
    python_after_degree_indices = [
        index for keep, index in zip(keep_degree.tolist(), python_indices) if keep
    ]
    keep_cycles = break_graph_cycles(connections[python_after_degree_indices])
    python_cycle_rows = np.flatnonzero(keep_cycles).astype(np.int64) + 1
    matlab_cycle_rows = np.asarray(matlab_output["cycle_indices"], dtype=np.int64).reshape(-1)
    cycle_mismatches = np.flatnonzero(
        python_cycle_rows[: min(len(python_cycle_rows), len(matlab_cycle_rows))]
        != matlab_cycle_rows[: min(len(python_cycle_rows), len(matlab_cycle_rows))]
    )
    print(
        json.dumps(
            {
                "cycle_python_count": len(python_cycle_rows),
                "cycle_matlab_count": len(matlab_cycle_rows),
                "cycle_same_length": bool(len(python_cycle_rows) == len(matlab_cycle_rows)),
                "cycle_mismatch_count_prefix": len(cycle_mismatches),
                "cycle_first_mismatch_position": int(cycle_mismatches[0])
                if cycle_mismatches.size
                else None,
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
