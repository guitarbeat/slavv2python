"""CLI handlers for native-first MATLAB-oracle parity experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from .execution import (
    copy_source_surface,
    load_params_file,
    maybe_sync_exact_vertex_checkpoint,
    persist_param_storage,
    resolve_input_file,
    validate_source_run_surface,
    write_run_manifest,
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
    load_json_dict,
    write_json_with_hash,
)
from .constants import METADATA_DIR

def handle_rerun_python(args: argparse.Namespace) -> None:
    """Orchestrate a Python-only rerun from a source comparison root."""
    from source.core.pipeline import SLAVVProcessor
    from source.io.tiff import load_tiff_volume

    source_surface = validate_source_run_surface(Path(args.source_run_root))
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    
    # Resolve input and params
    repo_root = Path.cwd() # Assume CWD is repo root as per AGENTS.md
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

def handle_prove_exact(args: argparse.Namespace) -> None:
    """Orchestrate a full-artifact exact proof."""
    pass

# ... (and so on)
