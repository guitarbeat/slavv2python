"""CLI handlers for exact-run lifecycle (init/resume/launch/promote)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

from slavv_python.analytics.parity.constants import (
    EXPERIMENT_PROVENANCE_PATH,
    METADATA_DIR,
    RUN_MANIFEST_PATH,
)
from slavv_python.analytics.parity.oracle.oracle_artifacts import ensure_oracle_artifacts
from slavv_python.analytics.parity.oracle.params_audit import (
    load_params_file,
    persist_param_storage,
)
from slavv_python.analytics.parity.oracle.promotion import (
    handle_promote_dataset as promote_dataset_handler,
)
from slavv_python.analytics.parity.oracle.promotion import (
    handle_promote_oracle as promote_oracle_handler,
)
from slavv_python.analytics.parity.oracle.promotion import (
    handle_promote_report as promote_report_handler,
)
from slavv_python.analytics.parity.oracle.surfaces import (
    copy_source_surface,
    ensure_dest_run_layout,
    load_dataset_surface,
    load_oracle_surface,
    resolve_input_file,
    validate_source_run_surface,
    write_run_manifest,
)
from slavv_python.analytics.parity.probes.adaptive_probes import ensure_rerun_allowed
from slavv_python.analytics.parity.proof.reports import (
    build_experiment_summary,
    extract_matlab_counts,
    extract_source_python_counts,
    persist_experiment_summary,
    persist_recording_tables,
    read_python_counts_from_run,
    render_exact_preflight_report,
    render_experiment_summary,
)
from slavv_python.analytics.parity.runs.bootstrap import (
    _copy_exact_bootstrap_refs,
    _finalize_init_exact_run,
    _reorient_exact_input_volume,
    _resolve_existing_init_exact_run,
    derive_exact_params_from_oracle,
    maybe_sync_exact_vertex_checkpoint,
)
from slavv_python.analytics.parity.runs.jobs import launch_exact_run_job
from slavv_python.analytics.parity.runs.launch_prepare import (
    LaunchPreparationError,
    assert_no_conflicting_registry_writer,
    prepare_detached_exact_run_launch,
)
from slavv_python.analytics.parity.runs.preflight import (
    run_exact_preflight,
    run_exact_preflight_for_surfaces,
)
from slavv_python.analytics.parity.runs.resume import resume_exact_run
from slavv_python.analytics.parity.runs.writer_session import (
    register_monitor_job,
    resume_writer_session,
)
from slavv_python.analytics.parity.utils import (
    fingerprint_file,
    now_iso,
    write_json_with_hash,
)
from slavv_python.engine import SlavvPipeline
from slavv_python.engine.state import load_json_dict
from slavv_python.interface.cli.monitor_service import (
    load_run_monitor_view,
    render_monitor_lines,
)
from slavv_python.storage import load_tiff_volume

if TYPE_CHECKING:
    import argparse


def handle_rerun_python(args: argparse.Namespace) -> None:
    """Orchestrate a Python-only rerun from a slavv_python comparison root."""
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
        sys.exit(1)


def handle_resume_exact_run(args: argparse.Namespace) -> None:
    """Resume a stale or interrupted init-exact-run directory."""
    dest_run_root = Path(args.dest_run_root)
    monitor = bool(getattr(args, "monitor", False))
    force_kill = bool(getattr(args, "force_kill", False))
    rerun_stage = getattr(args, "force_rerun_from", None)
    if rerun_stage:
        ensure_rerun_allowed(dest_run_root, stage=rerun_stage)

    stage = rerun_stage or "all"
    with resume_writer_session(
        dest_run_root,
        command=" ".join(sys.argv),
        stage=stage,
        monitor=monitor,
        force_kill=force_kill,
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        stop_after=args.stop_after,
        registry_metadata={
            "stop_after": args.stop_after,
            "force_rerun_from": rerun_stage,
        },
    ):
        dest_run_root = resume_exact_run(
            dest_run_root,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            oracle_root=Path(args.oracle_root) if args.oracle_root else None,
            stop_after=args.stop_after,
            force_rerun_from=rerun_stage,
            memory_safety_fraction=float(args.memory_safety_fraction),
            force=bool(args.force),
            skip_preflight=bool(getattr(args, "skip_preflight", False)),
            n_jobs=int(args.n_jobs) if getattr(args, "n_jobs", None) is not None else None,
        )

    print(str(dest_run_root))


def handle_launch_exact_run(args: argparse.Namespace) -> None:
    """Launch an exact-route resume as a detached parity job."""
    dest_run_root = Path(args.dest_run_root)
    monitor = bool(getattr(args, "monitor", False))
    force_kill = bool(getattr(args, "force_kill", False))

    try:
        assert_no_conflicting_registry_writer(dest_run_root, force_kill=force_kill)
        detached_command, foreground_command = prepare_detached_exact_run_launch(
            dest_run_root=dest_run_root,
            oracle_root=Path(args.oracle_root) if args.oracle_root else None,
            dataset_root=Path(args.dataset_root) if args.dataset_root else None,
            stop_after=args.stop_after,
            force_rerun_from=getattr(args, "force_rerun_from", None),
            memory_safety_fraction=float(args.memory_safety_fraction),
            force=bool(args.force),
            force_kill=force_kill,
            skip_preflight=bool(getattr(args, "skip_preflight", False)),
            skip_foreground_probe=bool(getattr(args, "skip_foreground_probe", False)),
            n_jobs=int(args.n_jobs) if getattr(args, "n_jobs", None) is not None else None,
        )
    except LaunchPreparationError as exc:
        raise SystemExit(str(exc)) from exc

    print("Foreground probe command:")
    print(" ".join(foreground_command))
    print("Detached writer command:")
    print(" ".join(detached_command))

    manifest = launch_exact_run_job(
        dest_run_root=dest_run_root,
        dataset_root=Path(args.dataset_root) if args.dataset_root else None,
        oracle_root=Path(args.oracle_root) if args.oracle_root else None,
        stop_after=args.stop_after,
        force_rerun_from=getattr(args, "force_rerun_from", None),
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
        skip_preflight=True,
        n_jobs=int(args.n_jobs) if getattr(args, "n_jobs", None) is not None else None,
        command_override=detached_command,
    )

    # Register job if monitoring enabled
    if monitor:
        oracle_root_path = Path(manifest["oracle_root"]) if manifest.get("oracle_root") else None
        stage = getattr(args, "force_rerun_from", None) or "all"
        register_monitor_job(
            run_dir=dest_run_root,
            oracle_root=oracle_root_path,
            stage=stage,
            command=" ".join(manifest["command"]),
            pid=int(manifest["pid"]),
            metadata={
                "stop_after": args.stop_after,
                "manifest_path": manifest.get("pid_file"),
            },
        )

    print(manifest["pid"])
    print(manifest["stdout"])
    print(manifest["stderr"])


def handle_status_exact_run(args: argparse.Namespace) -> None:
    """Print a consolidated exact-route run status."""
    view = load_run_monitor_view(Path(args.run_dir))
    print("\n".join(render_monitor_lines(view)))


def handle_ensure_oracle_artifacts(args: argparse.Namespace) -> None:
    """Verify and optionally repair normalized Oracle Artifacts."""
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
        sys.exit(1)


def handle_promote_oracle(args: argparse.Namespace) -> None:
    """Promote a MATLAB batch to a structured oracle root."""
    promote_oracle_handler(args)


def handle_promote_dataset(args: argparse.Namespace) -> None:
    """Promote a raw file to a cataloged dataset."""
    promote_dataset_handler(args)


def handle_promote_report(args: argparse.Namespace) -> None:
    """Promote a disposable run to a stable report."""
    promote_report_handler(args)


def handle_init_exact_run(args: argparse.Namespace) -> None:
    """Initialize a fresh run root for an exact parity experiment."""
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
