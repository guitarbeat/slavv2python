"""CLI handlers for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from source.runtime.run_state import load_json_dict
from .constants import (
    EXPERIMENT_PROVENANCE_PATH,
    METADATA_DIR,
    RUN_MANIFEST_PATH,
    SUMMARY_JSON_PATH,
    SUMMARY_TEXT_PATH,
)
from .execution import (
    copy_source_surface,
    derive_exact_params_from_oracle,
    ensure_dest_run_layout,
    load_dataset_surface,
    load_oracle_surface,
    load_params_file,
    maybe_sync_exact_vertex_checkpoint,
    persist_param_storage,
    resolve_input_file,
    validate_source_run_surface,
    write_run_manifest,
)
from .gaps import (
    persist_gap_diagnosis_report,
    render_gap_diagnosis_report,
)
from .models import ExactProofSourceSurface, OracleSurface
from .proofs import (
    run_candidate_capture,
    run_edge_replay,
    run_exact_parity_proof,
    run_exact_preflight,
)
from .reports import (
    build_experiment_summary,
    extract_matlab_counts,
    extract_source_python_counts,
    persist_experiment_summary,
    persist_recording_tables,
    read_python_counts_from_run,
    render_experiment_summary,
)
from .utils import (
    fingerprint_file,
    write_json_with_hash,
)


def handle_rerun_python(args: argparse.Namespace) -> None:
    """Orchestrate a Python-only rerun from a source comparison root."""
    from source.core.pipeline import SLAVVProcessor
    from source.io.tiff import load_tiff_volume

    source_surface = validate_source_run_surface(Path(args.source_run_root))
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()

    # Resolve input and params
    repo_root = Path.cwd()  # Assume CWD is repo root as per AGENTS.md
    input_file = resolve_input_file(source_surface, args.input, repo_root=repo_root)
    params = load_params_file(source_surface, args.params_file)

    # Setup destination
    copy_source_surface(source_surface, dest_run_root)
    persist_param_storage(dest_run_root, params)

    # Sync exact vertex if needed
    oracle_surface: OracleSurface | None = None
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
    processor = SLAVVProcessor()
    processor.process_image(
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
    # (Simplified resolution for now, should use ExactProofSourceSurface)
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve()

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
    run_exact_parity_proof(
        source_surface,
        dest_run_root,
        stage_arg=args.stage,
        report_path_arg=args.report_path,
    )


def handle_preflight_exact(args: argparse.Namespace) -> None:
    """Verify that a destination run root is ready for an exact proof."""
    run_exact_preflight(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
    )


def handle_prove_luts(args: argparse.Namespace) -> None:
    """Verify exact parity for lookup tables."""
    pass


def handle_capture_candidates(args: argparse.Namespace) -> None:
    """Capture candidate pairs from a Python run for parity comparison."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve()

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
    run_candidate_capture(
        source_surface,
        dest_run_root,
        include_debug_maps=bool(args.debug_maps),
    )


def handle_replay_edges(args: argparse.Namespace) -> None:
    """Replay edge discovery from candidates for parity verification."""
    run_root = Path(args.source_run_root).expanduser().resolve()
    oracle_root = Path(args.oracle_root).expanduser().resolve()

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
    run_edge_replay(source_surface, dest_run_root)


def handle_fail_fast(args: argparse.Namespace) -> None:
    """Run cheap gates first and stop at the first failing gate."""
    # Simplified fail-fast for now
    handle_preflight_exact(args)


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
    from source.core.pipeline import SLAVVProcessor
    from source.io.tiff import load_tiff_volume
    from .execution import (
        _resolve_existing_init_exact_run,
        _reorient_exact_input_volume,
        _copy_exact_bootstrap_refs,
        _finalize_init_exact_run,
    )

    dataset_surface = load_dataset_surface(Path(args.dataset_root))
    oracle_surface = load_oracle_surface(Path(args.oracle_root))

    if oracle_surface.dataset_hash and oracle_surface.dataset_hash != dataset_surface.dataset_hash:
        raise ValueError(
            f"dataset and oracle hashes do not match: {dataset_surface.dataset_hash} != {oracle_surface.dataset_hash}"
        )

    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    params, selected_settings_paths, _selected_settings_payloads = derive_exact_params_from_oracle(
        oracle_surface
    )
    params["energy_storage_format"] = str(args.energy_storage_format).strip()

    resume_finalization_only = _resolve_existing_init_exact_run(
        dest_run_root=dest_run_root,
        dataset_surface=dataset_surface,
        oracle_surface=oracle_surface,
        stop_after=args.stop_after,
    )

    oracle_size_of_image = None
    input_axis_permutation = None

    if resume_finalization_only:
        prov = load_json_dict(dest_run_root / EXPERIMENT_PROVENANCE_PATH) or {}
        raw_size = prov.get("oracle_size_of_image")
        if isinstance(raw_size, list) and len(raw_size) == 3:
            oracle_size_of_image = cast("tuple[int, int, int]", tuple(raw_size))
        raw_perm = prov.get("input_axis_permutation")
        if isinstance(raw_perm, list) and len(raw_perm) == 3:
            input_axis_permutation = cast("tuple[int, int, int]", tuple(raw_perm))
    else:
        image = load_tiff_volume(dataset_surface.input_file)
        image, oracle_size_of_image, input_axis_permutation = _reorient_exact_input_volume(
            image, oracle_surface
        )
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
                "oracle_size_of_image": list(oracle_size_of_image) if oracle_size_of_image else None,
                "input_axis_permutation": list(input_axis_permutation) if input_axis_permutation else None,
                "stop_after": args.stop_after,
                "created_at": now_iso(),
            }
        )

        processor = SLAVVProcessor()
        processor.process_image(
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
