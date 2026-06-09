"""CLI handlers for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from slavv_python.engine.state import load_json_dict

from .bootstrap import (
    _copy_exact_bootstrap_refs,
    _finalize_init_exact_run,
    _reorient_exact_input_volume,
    _resolve_existing_init_exact_run,
    derive_exact_params_from_oracle,
    maybe_sync_exact_vertex_checkpoint,
)
from .constants import (
    ANALYSIS_DIR,
    EXACT_PROOF_JSON_PATH,
    EXACT_STAGE_ORDER,
    EXPERIMENT_PROVENANCE_PATH,
    METADATA_DIR,
    RUN_MANIFEST_PATH,
    SUMMARY_JSON_PATH,
    SUMMARY_TEXT_PATH,
)
from .coordinator import ExactProofCoordinator
from .gaps import (
    persist_gap_diagnosis_report,
    render_gap_diagnosis_report,
)
from .models import ExactProofSourceSurface
from .params_audit import load_params_file, persist_param_storage
from .preflight import run_exact_preflight_for_surfaces
from .proofs import (
    run_edge_replay,
    run_exact_preflight,
    run_lut_proof,
)
from .reports import (
    build_experiment_summary,
    extract_matlab_counts,
    extract_source_python_counts,
    persist_experiment_summary,
    persist_recording_tables,
    read_python_counts_from_run,
    render_exact_preflight_report,
    render_experiment_summary,
)
from .resume import resume_exact_run
from .surfaces import (
    copy_source_surface,
    ensure_dest_run_layout,
    load_dataset_surface,
    load_oracle_surface,
    resolve_input_file,
    validate_source_run_surface,
    write_run_manifest,
)
from .utils import (
    fingerprint_file,
    now_iso,
    write_json_with_hash,
)

if TYPE_CHECKING:
    import argparse


def _build_exact_proof_source_surface(
    run_root: Path,
    oracle_root: Path | None,
) -> ExactProofSourceSurface:
    """Resolve oracle paths and return the exact-proof source surface."""
    if oracle_root is None and (run_root / "01_Input" / "matlab_results").is_dir():
        oracle_root = run_root
    oracle_surface = load_oracle_surface(oracle_root)
    return ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=run_root / "02_Output" / "python_results" / "checkpoints",
        validated_params_path=run_root / "99_Metadata" / "validated_params.json",
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
    )


def handle_rerun_python(args: argparse.Namespace) -> None:
    """Orchestrate a Python-only rerun from a slavv_python comparison root."""
    from slavv_python.engine import SlavvPipeline
    from slavv_python.storage import load_tiff_volume

    source_surface = validate_source_run_surface(Path(args.source_run_root))
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()

    # Resolve input and params
    repo_root = Path.cwd()  # Assume CWD is repo root as per docs/AGENTS.md
    input_file = resolve_input_file(source_surface, args.input, repo_root=repo_root)
    params = load_params_file(source_surface, args.params_file)

    # Setup destination
    copy_source_surface(source_surface, dest_run_root)
    persist_param_storage(dest_run_root, params)

    # Sync exact vertex if needed
    # (Simplified oracle resolution for now)

    exact_vertex_sync = maybe_sync_exact_vertex_checkpoint(
        source_surface.run_root,
        dest_run_root,
    )

    dataset_hash = fingerprint_file(input_file)

    # Record provenance
    write_json_with_hash(
        dest_run_root / METADATA_DIR / "experiment_provenance.json",
        {
            "source_run_root": str(source_surface.run_root),
            "input_file": str(input_file),
            "dataset_hash": dataset_hash,
            "rerun_from": args.rerun_from,
            "exact_vertex_checkpoint_sync": exact_vertex_sync,
        },
    )

    # Run processing
    image = load_tiff_volume(input_file)
    processor = SlavvPipeline()
    processor.run(
        image,
        params,
        run_dir=str(dest_run_root),
        force_rerun_from=args.rerun_from,
    )

    # Generate summary
    report_payload = load_json_dict(source_surface.comparison_report_path) or {}
    summary_payload = build_experiment_summary(
        source_run_root=source_surface.run_root,
        dest_run_root=dest_run_root,
        input_file=input_file,
        rerun_from=args.rerun_from,
        matlab_counts=extract_matlab_counts(report_payload),
        source_python_counts=extract_source_python_counts(report_payload),
        new_python_counts=read_python_counts_from_run(dest_run_root),
    )
    persist_experiment_summary(dest_run_root, summary_payload)

    # Write manifest
    write_run_manifest(
        dest_run_root,
        run_kind="parity_run",
        status="completed",
        command="rerun-python",
        dataset_hash=dataset_hash,
        oracle_surface=None,
        params_payload=params,
        extra={"rerun_from": args.rerun_from},
    )

    persist_recording_tables(dest_run_root)
    print(render_experiment_summary(summary_payload))


def handle_trace_vertex(args: argparse.Namespace) -> None:
    """Run discovery for a single vertex and capture execution trace."""
    import numpy as np

    from slavv_python.analytics.parity.python_checkpoint_loader import (
        load_normalized_python_checkpoints,
    )
    from slavv_python.processing.stages.edges.execution_tracing import JsonExecutionTracer
    from slavv_python.processing.stages.edges.global_watershed import (
        _generate_edge_candidates_matlab_global_watershed,
    )

    run_root = Path(args.source_run_root).expanduser().resolve()
    checkpoints_dir = run_root / "02_Output" / "python_results" / "checkpoints"

    # Load energy and vertices
    checkpoints = load_normalized_python_checkpoints(checkpoints_dir, stages=("energy", "vertices"))
    energy_data = checkpoints["energy"]
    vertex_data = checkpoints["vertices"]

    # Load params
    params_path = run_root / "99_Metadata" / "validated_params.json"
    if not params_path.is_file():
        params_path = run_root / "01_Params" / "validated_params.json"
    params = load_json_dict(params_path) or {}

    # Select vertex
    vertex_idx = args.vertex_idx
    if vertex_idx < 0 or vertex_idx >= len(vertex_data["positions"]):
        raise ValueError(
            f"vertex index {vertex_idx} out of range [0, {len(vertex_data['positions']) - 1}]"
        )

    v_pos = np.asarray(vertex_data["positions"][vertex_idx : vertex_idx + 1], dtype=np.float32)
    v_scale = np.asarray(vertex_data["scales"][vertex_idx : vertex_idx + 1], dtype=np.int32)

    # Setup tracer
    tracer = JsonExecutionTracer(args.output_trace)

    # Run discovery with universe realignment
    # We must use the same transpose/swap logic as _generate_edge_candidates_matlab_frontier
    # in generate.py to ensure the engine sees the correct spatial orientation.

    # 1. Align volumes to coherent system: [Y, X, Z] -> [Z, X, Y]
    aligned_energy = np.transpose(
        np.asarray(energy_data["energy"], dtype=np.float32), (2, 1, 0)
    ).copy(order="F")

    scale_indices = energy_data.get("scale_indices")
    aligned_scale_indices = None
    if scale_indices is not None:
        aligned_scale_indices = np.transpose(
            np.asarray(scale_indices, dtype=np.int16), (2, 1, 0)
        ).copy(order="F")

    # 2. Align vertex positions: swap Z and Y [Y, X, Z] -> [Z, X, Y]
    aligned_v_pos = v_pos.copy()
    tmp = aligned_v_pos[:, 0].copy()
    aligned_v_pos[:, 0] = aligned_v_pos[:, 2]
    aligned_v_pos[:, 2] = tmp

    # 3. Align physics microns: [dy, dx, dz] -> [dz, dx, dy]
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    )
    aligned_microns = microns_per_voxel.copy()
    if len(aligned_microns) >= 3:
        tmp_m = aligned_microns[0]
        aligned_microns[0] = aligned_microns[2]
        aligned_microns[2] = tmp_m

    # 4. Setup dummy vertex center image (not used by watershed engine but required by signature)
    aligned_vertex_center_image = np.zeros_like(aligned_energy)

    # Run discovery
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

    print(f"âœ… Execution trace for vertex {vertex_idx} captured to {args.output_trace}")


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


def handle_prove_exact(args: argparse.Namespace) -> None:
    """Orchestrate a full-artifact exact proof."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source_surface = _build_exact_proof_source_surface(run_root, oracle_root)
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    report, _, _ = ExactProofCoordinator(source_surface).prove(
        dest_run_root,
        stage_arg=getattr(args, "stage", "all"),
        report_path_arg=getattr(args, "report_path", None),
    )
    if not report.get("passed"):
        import sys

        sys.exit(1)


def handle_prove_exact_sequence(args: argparse.Namespace) -> None:
    """Run prove-exact for each stage in order; stop at the first failure."""
    from shutil import copy2

    from slavv_python.analytics.parity.proof_report import render_exact_proof_report

    run_root = Path(args.source_run_root).expanduser().resolve()
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source_surface = _build_exact_proof_source_surface(run_root, oracle_root)
    coordinator = ExactProofCoordinator(source_surface)

    stage_results: list[dict[str, Any]] = []
    for stage in EXACT_STAGE_ORDER:
        report, json_path, _text_path = coordinator.prove(dest_run_root, stage_arg=stage)
        if json_path is not None and json_path.is_file():
            stage_json = dest_run_root / ANALYSIS_DIR / f"exact_proof_{stage}.json"
            stage_json.parent.mkdir(parents=True, exist_ok=True)
            copy2(json_path, stage_json)
        passed = bool(report.get("passed"))
        stage_results.append({"stage": stage, "passed": passed})
        print(f"prove-exact --stage {stage}: {'PASS' if passed else 'FAIL'}")
        if report.get("stage_summaries"):
            print(render_exact_proof_report(report))
        if not passed:
            import sys

            sys.exit(1)

    summary = {
        "passed": True,
        "stages": stage_results,
        "source_run_root": str(run_root),
        "dest_run_root": str(dest_run_root),
    }
    from .utils import write_json_with_hash, write_text_with_hash

    summary_json = dest_run_root / EXACT_PROOF_JSON_PATH
    summary_text = dest_run_root / ANALYSIS_DIR / "exact_proof_sequence.txt"
    write_json_with_hash(summary_json, summary)
    write_text_with_hash(
        summary_text,
        "\n".join(
            [
                "Exact proof sequence (all stages passed)",
                *(f"  {row['stage']}: PASS" for row in stage_results),
            ]
        ),
    )
    print(str(summary_json))


def handle_preflight_exact(args: argparse.Namespace) -> None:
    """Verify that a destination run root is ready for an exact proof."""
    report, json_path, text_path = run_exact_preflight(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        dataset_root=Path(args.dataset_root) if getattr(args, "dataset_root", None) else None,
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
    )
    print(render_exact_preflight_report(report))
    if json_path is not None:
        print(str(json_path))
    if text_path is not None:
        print(str(text_path))
    if not report.get("passed"):
        import sys

        sys.exit(1)


def handle_resume_exact_run(args: argparse.Namespace) -> None:
    """Resume a stale or interrupted init-exact-run directory."""
    from slavv_python.analytics.parity.job_registry import JobRegistry
    from slavv_python.analytics.parity.process_utils import (
        ensure_monitor_daemon_running,
        is_process_alive,
        is_python_process,
        kill_process_tree,
    )

    dest_run_root = Path(args.dest_run_root)
    monitor = bool(getattr(args, "monitor", False))
    force_kill = bool(getattr(args, "force_kill", False))

    # Check for active writer if monitoring enabled
    if monitor:
        registry = JobRegistry()
        active_job = registry.get_job_by_run_dir(dest_run_root)
        if active_job and is_process_alive(active_job.pid) and is_python_process(active_job.pid):
            if not force_kill:
                raise RuntimeError(
                    f"Run directory has active writer (PID {active_job.pid}).\n"
                    f"Job started: {active_job.started_at}\n"
                    f"Use --force-kill to terminate, or wait for completion.\n"
                    f"Check status: slavv jobs list"
                )
            print(f"Terminating active writer PID {active_job.pid}...")
            kill_process_tree(active_job.pid)
            registry.update_job(active_job.job_id, status="killed")

    dest_run_root = resume_exact_run(
        dest_run_root,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        stop_after=args.stop_after,
        force_rerun_from=getattr(args, "force_rerun_from", None),
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
        skip_preflight=bool(getattr(args, "skip_preflight", False)),
        n_jobs=int(args.n_jobs) if getattr(args, "n_jobs", None) is not None else None,
    )

    # Register job if monitoring enabled
    if monitor:
        import os

        registry = JobRegistry()
        oracle_root_str = str(Path(args.oracle_root)) if args.oracle_root else "unknown"
        stage = getattr(args, "force_rerun_from", None) or "all"

        job_id = registry.register_job(
            pid=os.getpid(),
            run_dir=dest_run_root,
            oracle_root=Path(oracle_root_str),
            stage=stage,
            command=" ".join(sys.argv),
            metadata={
                "stop_after": args.stop_after,
                "force_rerun_from": getattr(args, "force_rerun_from", None),
            },
        )
        ensure_monitor_daemon_running()
        print(f"Job registered for monitoring (ID: {job_id})")

    print(str(dest_run_root))


def handle_launch_exact_run(args: argparse.Namespace) -> None:
    """Launch an exact-route resume as a detached parity job."""
    from .jobs import launch_exact_run_job
    from slavv_python.analytics.parity.job_registry import JobRegistry
    from slavv_python.analytics.parity.process_utils import (
        ensure_monitor_daemon_running,
        is_process_alive,
        is_python_process,
        kill_process_tree,
    )

    dest_run_root = Path(args.dest_run_root)
    monitor = bool(getattr(args, "monitor", False))
    force_kill = bool(getattr(args, "force_kill", False))

    # Check for active writer if monitoring enabled
    if monitor:
        registry = JobRegistry()
        active_job = registry.get_job_by_run_dir(dest_run_root)
        if active_job and is_process_alive(active_job.pid) and is_python_process(active_job.pid):
            if not force_kill:
                raise RuntimeError(
                    f"Run directory has active writer (PID {active_job.pid}).\n"
                    f"Job started: {active_job.started_at}\n"
                    f"Use --force-kill to terminate, or wait for completion.\n"
                    f"Check status: slavv jobs list"
                )
            print(f"Terminating active writer PID {active_job.pid}...")
            kill_process_tree(active_job.pid)
            registry.update_job(active_job.job_id, status="killed")

    manifest = launch_exact_run_job(
        dest_run_root=dest_run_root,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        stop_after=args.stop_after,
        force_rerun_from=getattr(args, "force_rerun_from", None),
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
        skip_preflight=bool(getattr(args, "skip_preflight", False)),
        n_jobs=int(args.n_jobs) if getattr(args, "n_jobs", None) is not None else None,
    )

    # Register job if monitoring enabled
    if monitor:
        registry = JobRegistry()
        oracle_root_str = str(manifest.get("oracle_root", "unknown"))
        stage = getattr(args, "force_rerun_from", None) or "all"

        job_id = registry.register_job(
            pid=manifest["pid"],
            run_dir=dest_run_root,
            oracle_root=Path(oracle_root_str),
            stage=stage,
            command=" ".join(manifest["command"]),
            metadata={
                "stop_after": args.stop_after,
                "manifest_path": manifest.get("pid_file"),
            },
        )
        ensure_monitor_daemon_running()
        print(f"Job registered for monitoring (ID: {job_id})")

    print(manifest["pid"])
    print(manifest["stdout"])
    print(manifest["stderr"])


def handle_status_exact_run(args: argparse.Namespace) -> None:
    """Print a consolidated exact-route run status."""
    from slavv_python.interface.cli.monitor_service import (
        load_run_monitor_view,
        render_monitor_lines,
    )

    view = load_run_monitor_view(Path(args.run_dir))
    print("\n".join(render_monitor_lines(view)))


def handle_ensure_oracle_artifacts(args: argparse.Namespace) -> None:
    """Verify and optionally repair normalized Oracle Artifacts."""
    import json

    from .oracle_artifacts import ensure_oracle_artifacts

    statuses = ensure_oracle_artifacts(
        Path(args.oracle_root),
        stages=tuple(args.stage or ("all",)),
        matlab_batch_dir=Path(args.matlab_batch_dir) if args.matlab_batch_dir else None,
        repair=not args.no_repair,
    )
    payload = {
        "passed": all(status.ready for status in statuses.values()),
        "oracle_root": str(Path(args.oracle_root).expanduser().resolve()),
        "stages": {stage: status.to_dict() for stage, status in statuses.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        import sys

        sys.exit(1)


def handle_prove_luts(args: argparse.Namespace) -> None:
    """Verify exact parity for lookup tables."""
    report, _, _ = run_lut_proof(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
    )
    if not report.get("passed"):
        import sys

        sys.exit(1)


def handle_capture_candidates(args: argparse.Namespace) -> None:
    """Capture candidate pairs from a Python run for parity comparison."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    source_surface = _build_exact_proof_source_surface(run_root, oracle_root)
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    report, _, _ = ExactProofCoordinator(source_surface).capture_candidates(
        dest_run_root,
        include_debug_maps=bool(getattr(args, "debug_maps", False)),
    )
    if not report.get("passed"):
        import sys

        sys.exit(1)


def handle_replay_edges(args: argparse.Namespace) -> None:
    """Replay edge discovery from candidates for parity verification."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None

    oracle_surface = load_oracle_surface(oracle_root)
    source_surface = ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=run_root / "02_Output" / "python_results" / "checkpoints",
        validated_params_path=run_root / "99_Metadata" / "validated_params.json",
        oracle_surface=oracle_surface,
        matlab_batch_dir=oracle_surface.matlab_batch_dir,
        matlab_vector_paths=oracle_surface.matlab_vector_paths,
    )

    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    report, _, _ = run_edge_replay(source_surface, dest_run_root)
    if not report.get("passed"):
        import sys

        sys.exit(1)


def handle_fail_fast(args: argparse.Namespace) -> None:
    """Run cheap gates first and stop at the first failing gate."""
    # 1. Preflight
    handle_preflight_exact(args)

    # 2. LUT Proof
    handle_prove_luts(args)

    # 3. Candidate Capture
    handle_capture_candidates(args)

    # 4. Edge Replay
    handle_replay_edges(args)

    # 5. Final Exact Proof
    handle_prove_exact(args)


def handle_promote_oracle(args: argparse.Namespace) -> None:
    """Promote a MATLAB batch to a structured oracle root."""
    from .promotion import handle_promote_oracle as handler

    handler(args)


def handle_promote_dataset(args: argparse.Namespace) -> None:
    """Promote a raw file to a cataloged dataset."""
    from .promotion import handle_promote_dataset as handler

    handler(args)


def handle_promote_report(args: argparse.Namespace) -> None:
    """Promote a disposable run to a stable report."""
    from .promotion import handle_promote_report as handler

    handler(args)


def handle_init_exact_run(args: argparse.Namespace) -> None:
    """Initialize a fresh run root for an exact parity experiment."""
    from slavv_python.engine import SlavvPipeline
    from slavv_python.storage import load_tiff_volume

    dataset_surface = load_dataset_surface(Path(args.dataset_root))
    oracle_root = Path(args.oracle_root).expanduser().resolve() if args.oracle_root else None
    oracle_surface = load_oracle_surface(oracle_root)

    if oracle_surface.dataset_hash and oracle_surface.dataset_hash != dataset_surface.dataset_hash:
        raise ValueError(
            f"dataset and oracle hashes do not match: {dataset_surface.dataset_hash} != {oracle_surface.dataset_hash}"
        )

    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    params, selected_settings_paths, _selected_settings_payloads = derive_exact_params_from_oracle(
        oracle_surface
    )
    params["energy_storage_format"] = str(args.energy_storage_format).strip()

    init_resolution = _resolve_existing_init_exact_run(
        dest_run_root=dest_run_root,
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        stop_after=args.stop_after,
        allow_resume=bool(getattr(args, "resume", False)),
    )

    if init_resolution == "resume_pipeline":
        resume_exact_run(
            dest_run_root,
            dataset_root=Path(args.dataset_root) if getattr(args, "dataset_root", None) else None,
            oracle_root=oracle_root,
            stop_after=args.stop_after,
            memory_safety_fraction=float(
                getattr(args, "memory_safety_fraction", 0.8),
            ),
            force=bool(getattr(args, "force", False)),
            skip_preflight=bool(getattr(args, "skip_preflight", False)),
        )
        _finalize_init_exact_run(
            dest_run_root=dest_run_root,
            dataset_surface=dataset_surface,
            oracle_surface=oracle_surface,
            params=params,
            selected_settings_paths=selected_settings_paths,
            oracle_size_of_image=None,
            input_axis_permutation=None,
            stop_after=args.stop_after,
        )
        print(str(dest_run_root / RUN_MANIFEST_PATH))
        return

    oracle_size_of_image = None
    input_axis_permutation = None

    if init_resolution == "finalize_only":
        prov = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH) or {}
        raw_size = prov.get("oracle_size_of_image")
        if isinstance(raw_size, list) and len(raw_size) == 3:
            oracle_size_of_image = cast("tuple[int, int, int]", tuple(raw_size))
        raw_perm = prov.get("input_axis_permutation")
        if isinstance(raw_perm, list) and len(raw_perm) == 3:
            input_axis_permutation = cast("tuple[int, int, int]", tuple(raw_perm))
    elif init_resolution == "fresh":
        if not bool(getattr(args, "skip_preflight", False)):
            preflight_report, _, _ = run_exact_preflight_for_surfaces(
                dest_run_root,
                dataset_surface=dataset_surface,
                oracle_surface=oracle_surface,
                params=params,
                memory_safety_fraction=float(
                    getattr(args, "memory_safety_fraction", 0.8),
                ),
                force=bool(getattr(args, "force", False)),
                persist=True,
            )
            if not preflight_report.get("passed"):
                import sys

                sys.exit(
                    "preflight failed: " + ", ".join(preflight_report.get("errors", [])),
                )

        image = load_tiff_volume(dataset_surface.input_file)
        image, oracle_size_of_image, input_axis_permutation = _reorient_exact_input_volume(
            image, oracle_surface
        )
        if input_axis_permutation is not None:
            # Volume axes were reordered to the oracle layout; the energy stage must
            # apply the same permutation to per-axis physical metadata (microns, PSF).
            params["energy_axis_permutation"] = list(input_axis_permutation)
        ensure_dest_run_layout(dest_run_root)
        persist_param_storage(dest_run_root, params)
        _copy_exact_bootstrap_refs(
            dest_run_root,
            dataset_surface=dataset_surface,
            oracle_surface=oracle_surface,
        )

        write_json_with_hash(
            dest_run_root / EXPERIMENT_PROVENANCE_PATH,
            {
                "bootstrap_kind": "init-exact-run",
                "dataset_hash": dataset_surface.dataset_hash,
                "oracle_id": oracle_surface.oracle_id,
                "selected_settings_paths": selected_settings_paths,
                "oracle_size_of_image": list(oracle_size_of_image)
                if oracle_size_of_image
                else None,
                "input_axis_permutation": list(input_axis_permutation)
                if input_axis_permutation
                else None,
                "stop_after": args.stop_after,
                "created_at": now_iso(),
            },
        )

        processor = SlavvPipeline()
        processor.run(
            image,
            params,
            run_dir=str(dest_run_root),
            stop_after=args.stop_after,
        )

    _finalize_init_exact_run(
        dest_run_root=dest_run_root,
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        params=params,
        selected_settings_paths=selected_settings_paths,
        oracle_size_of_image=oracle_size_of_image,
        input_axis_permutation=input_axis_permutation,
        stop_after=args.stop_after,
    )
    print(str(dest_run_root / RUN_MANIFEST_PATH))


def handle_dedupe(args: argparse.Namespace) -> None:
    """Clean up and deduplicate index.jsonl in the experiment workspace root."""
    from .index import deduplicate_index_records, resolve_experiment_root

    repo_root = Path.cwd()
    exp_root = resolve_experiment_root(repo_root / "workspace") or resolve_experiment_root(
        repo_root
    )
    if not exp_root:
        raise RuntimeError("Could not find experiment root directory from CWD.")

    print(f"Auditing and deduplicating SLAVV index under {exp_root}...")
    dry_run = getattr(args, "dry_run", False)
    if dry_run:
        print("[Dry Run Mode] No files will be modified on disk.")

    removed = deduplicate_index_records(exp_root, dry_run=dry_run)

    if removed:
        print(f"\nFound {len(removed)} stale or duplicate records to remove:")
        for r in removed:
            print(f"  - ID: {r.get('id') or r.get('run_id')} (Kind: {r.get('kind')})")
            print(f"    Path: {r.get('path') or r.get('run_root')}")
        if not dry_run:
            print("\nSuccessfully cleaned up and updated index.jsonl!")
        else:
            print("\nDry run completed. Run without --dry-run to apply these changes.")
    else:
        print("\nWorkspace index is already perfectly canonical and clean!")
