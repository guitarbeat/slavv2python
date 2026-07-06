"""CLI handlers for parity diagnostics and energy evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from slavv_python.analytics.parity.cli_handlers.cli_support import _build_exact_proof_source_surface
from slavv_python.analytics.parity.constants import (
    ENERGY_PROOF_EVIDENCE_JSON_PATH,
    SUMMARY_JSON_PATH,
    SUMMARY_TEXT_PATH,
)
from slavv_python.analytics.parity.oracle.gaps import (
    persist_gap_diagnosis_report,
    render_gap_diagnosis_report,
)
from slavv_python.analytics.parity.oracle.matlab_vector_loader import (
    load_normalized_matlab_vectors,
)
from slavv_python.analytics.parity.oracle.python_checkpoint_loader import (
    load_normalized_python_checkpoints,
)
from slavv_python.analytics.parity.probes.adaptive_probes import (
    build_energy_probe_payload,
    compare_probe_jsonl,
    persist_energy_probe_payload,
    record_hypothesis,
)
from slavv_python.analytics.parity.proof.energy_proof_evidence import (
    build_energy_proof_evidence,
    require_energy_proof_evidence,
)
from slavv_python.analytics.parity.proof.reports import (
    persist_recording_tables,
    render_experiment_summary,
)
from slavv_python.analytics.parity.utils import (
    payload_hash,
    write_json_with_hash,
    write_text_with_hash,
)
from slavv_python.engine.state import load_json_dict
from slavv_python.pipeline.edges.execution_tracing import JsonExecutionTracer
from slavv_python.pipeline.edges.matlab_get_edges_by_watershed import (
    _generate_edge_candidates_matlab_global_watershed,
)

if TYPE_CHECKING:
    import argparse


def handle_trace_vertex(args: argparse.Namespace) -> None:
    """Run discovery for a single vertex and capture execution trace."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    checkpoints_dir = run_root / "02_Output" / "python_results" / "checkpoints"

    checkpoints = load_normalized_python_checkpoints(checkpoints_dir, stages=("energy", "vertices"))
    energy_data = checkpoints["energy"]
    vertex_data = checkpoints["vertices"]

    params_path = run_root / "99_Metadata" / "validated_params.json"
    if not params_path.is_file():
        params_path = run_root / "01_Params" / "validated_params.json"
    params = load_json_dict(params_path) or {}

    vertex_idx = args.vertex_idx
    if vertex_idx < 0 or vertex_idx >= len(vertex_data["positions"]):
        raise ValueError(
            f"vertex index {vertex_idx} out of range [0, {len(vertex_data['positions']) - 1}]"
        )

    v_pos = np.asarray(vertex_data["positions"][vertex_idx : vertex_idx + 1], dtype=np.float32)
    v_scale = np.asarray(vertex_data["scales"][vertex_idx : vertex_idx + 1], dtype=np.int32)

    tracer = JsonExecutionTracer(args.output_trace)

    aligned_energy = np.transpose(
        np.asarray(energy_data["energy"], dtype=np.float32), (2, 1, 0)
    ).copy(order="F")

    scale_indices = energy_data.get("scale_indices")
    aligned_scale_indices = None
    if scale_indices is not None:
        aligned_scale_indices = np.transpose(
            np.asarray(scale_indices, dtype=np.int16), (2, 1, 0)
        ).copy(order="F")

    aligned_v_pos = v_pos.copy()
    tmp = aligned_v_pos[:, 0].copy()
    aligned_v_pos[:, 0] = aligned_v_pos[:, 2]
    aligned_v_pos[:, 2] = tmp

    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    )
    aligned_microns = microns_per_voxel.copy()
    if len(aligned_microns) >= 3:
        tmp_m = aligned_microns[0]
        aligned_microns[0] = aligned_microns[2]
        aligned_microns[2] = tmp_m

    aligned_vertex_center_image = np.zeros_like(aligned_energy)

    _generate_edge_candidates_matlab_global_watershed(
        aligned_energy,
        aligned_scale_indices,
        aligned_v_pos,
        v_scale,
        np.asarray(energy_data["lumen_radius_microns"], dtype=np.float32),
        aligned_microns,
        aligned_vertex_center_image,
        params,
        tracer=tracer,
    )

    print(f"Execution trace for vertex {vertex_idx} captured to {args.output_trace}")


def handle_summarize(args: argparse.Namespace) -> None:
    """Print the saved experiment summary for a destination run root."""
    run_root = Path(args.run_root).expanduser().resolve()
    summary_text_path = run_root / SUMMARY_TEXT_PATH
    if summary_text_path.is_file():
        print(summary_text_path.read_text(encoding="utf-8"))
        return

    summary_payload = load_json_dict(run_root / SUMMARY_JSON_PATH)
    if summary_payload is None:
        raise ValueError(
            f"no experiment summary found under {run_root / SUMMARY_TEXT_PATH} or {run_root / SUMMARY_JSON_PATH}"
        )
    print(render_experiment_summary(summary_payload))


def handle_normalize_recordings(args: argparse.Namespace) -> None:
    """Flatten recorded run artifacts into CSV/JSONL tables."""
    run_root = Path(args.run_root).expanduser().resolve()
    index_payload = persist_recording_tables(run_root)
    print(
        "\n".join(
            [
                f"Normalized recording tables written for {run_root}",
                f"Table count: {index_payload['table_count']}",
            ]
        )
    )


def handle_diagnose_gaps(args: argparse.Namespace) -> None:
    """Join candidate coverage with origin-level diagnostics to surface gap hotspots."""
    run_root = Path(args.run_root).expanduser().resolve()
    report_payload = persist_gap_diagnosis_report(run_root, limit=max(1, int(args.limit)))
    persist_recording_tables(run_root)
    print(render_gap_diagnosis_report(report_payload))


def handle_diagnose_energy(args: argparse.Namespace) -> None:
    """Create deterministic adaptive Energy probe requests from current checkpoints."""
    run_root = Path(args.run_root).expanduser().resolve()
    require_energy_proof_evidence(run_root)
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source = _build_exact_proof_source_surface(run_root, oracle_root)
    if source.matlab_batch_dir is None:
        raise RuntimeError("Exact proof source is missing MATLAB batch directory")
    matlab = load_normalized_matlab_vectors(source.matlab_batch_dir, ("energy",))["energy"]
    python = load_normalized_python_checkpoints(source.checkpoints_dir, ("energy",))["energy"]
    params = load_json_dict(source.validated_params_path) or {}
    payload = build_energy_probe_payload(
        np.asarray(matlab["energy"]),
        np.asarray(python["energy"]),
        np.asarray(matlab["scale_indices"]),
        np.asarray(python["scale_indices"]),
        provenance={
            "run_root": str(run_root),
            "oracle_id": source.oracle_surface.oracle_id,
            "params_fingerprint": payload_hash(params),
        },
    )
    path = persist_energy_probe_payload(run_root, payload)
    print(path)


def handle_inspect_energy_evidence(args: argparse.Namespace) -> None:
    """Persist a read-only freshness report for Energy proof evidence."""
    run_root = Path(args.run_root).expanduser().resolve()
    report = build_energy_proof_evidence(run_root)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_root / ENERGY_PROOF_EVIDENCE_JSON_PATH
    )
    write_json_with_hash(output, report)
    print(output)
    if not report["valid"]:
        print("Energy proof evidence is stale: " + ", ".join(report["failures"]), file=sys.stderr)
        sys.exit(1)


def handle_compare_energy_probes(args: argparse.Namespace) -> None:
    """Compare normalized MATLAB and Python adaptive probe JSONL records."""
    report = compare_probe_jsonl(Path(args.matlab_jsonl), Path(args.python_jsonl))
    output = Path(args.output).expanduser().resolve()
    write_json_with_hash(output, report)
    write_text_with_hash(output.with_suffix(".txt"), json.dumps(report, indent=2, sort_keys=True))
    print(output)
    if not report["passed"]:
        sys.exit(1)


def handle_record_parity_hypothesis(args: argparse.Namespace) -> None:
    """Record one isolated parity hypothesis and enforce the circuit breaker."""
    record = record_hypothesis(
        Path(args.run_root).expanduser().resolve(),
        proof_report=Path(args.proof_report).expanduser().resolve(),
        first_failing_field=args.first_failing_field,
        probe_request_id=args.probe_request_id,
        hypothesis=args.hypothesis,
        expected_field=args.expected_field,
        kind=args.kind,
        design_review=bool(args.design_review),
    )
    print(json.dumps(record, indent=2, sort_keys=True))
