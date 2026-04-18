"""Python comparison execution helpers."""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Any


def run_python_vectorization(
    input_file: str,
    output_dir: str,
    params: dict[str, Any],
    *,
    run_dir: str | None = None,
    force_rerun_from: str | None = None,
    minimal_exports: bool = False,
    get_system_info_fn,
    load_tiff_volume_fn,
    processor_factory,
    format_progress_event_message_fn,
    resolve_python_energy_source_fn,
    export_pipeline_results_fn,
    visualizer_factory,
    load_python_candidate_edges_fn,
    load_python_candidate_audit_fn,
    load_python_candidate_lifecycle_fn,
) -> dict[str, Any]:
    """Run Python vectorization."""
    print("\n" + "=" * 60)
    print("Running Python Implementation")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")

    system_info = get_system_info_fn()

    print("Loading image...")
    image = load_tiff_volume_fn(input_file)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    processor = processor_factory()
    params_for_run = copy.deepcopy(params)
    params_for_run["comparison_exact_network"] = True

    print("Running pipeline...")
    start_time = time.time()

    def progress_callback(frac, stage):
        stage_descriptions = {
            "start": "Initializing pipeline...",
            "preprocess": "Preprocessing finished. Calculating multi-scale energy field (this step takes 20-30 minutes)...",
            "energy": "Energy calculation complete. Extracting vertices...",
            "vertices": "Vertices extracted. Tracing edges...",
            "edges": "Edges traced. Building final network...",
            "network": "Network assembly complete!",
        }
        msg = stage_descriptions.get(stage, f"completed {stage}")
        print(f"  Progress: {frac * 100:.1f}% - {msg}")

    last_message = ""

    def event_callback(event):
        nonlocal last_message
        message = format_progress_event_message_fn(event)
        if message and message != last_message:
            print(f"      -> {message:<120}", end="\r", flush=True)
            last_message = message

    try:
        results = processor.process_image(
            image,
            params_for_run,
            progress_callback=progress_callback,
            event_callback=event_callback,
            run_dir=run_dir,
            checkpoint_dir=os.path.join(output_dir, "checkpoints") if run_dir is None else None,
            force_rerun_from=force_rerun_from,
        )
        print()

        elapsed_time = time.time() - start_time
        print(f"\nPython execution completed in {elapsed_time:.2f} seconds")

        python_results = {
            "success": True,
            "elapsed_time": elapsed_time,
            "output_dir": output_dir,
            "vertices_count": len(results["vertices"]["positions"]) if "vertices" in results else 0,
            "edges_count": len(results["edges"]["traces"]) if "edges" in results else 0,
            "network_strands_count": len(results["network"]["strands"])
            if "network" in results
            else 0,
            "results": results,
            "system_info": system_info,
            "comparison_mode": {
                "network_cleanup": "bypass_python_specific_cleanup",
                "energy_source": resolve_python_energy_source_fn(results.get("energy_data")),
            },
        }

        print("Exporting results...")
        export_pipeline_results_fn(results, output_dir, base_name="python_comparison")

        if minimal_exports:
            print("Minimal export mode enabled; skipping VMV/CASX/CSV/JSON extra exports.")
            python_results["exports"] = {"profile": "minimal"}
        else:
            print("Exporting VMV and CASX formats...")
            try:
                visualizer = visualizer_factory()
                vmv_path = os.path.join(output_dir, "network.vmv")
                casx_path = os.path.join(output_dir, "network.casx")
                csv_base = os.path.join(output_dir, "network")
                json_path = os.path.join(output_dir, "network.json")

                visualizer.export_network_data(results, vmv_path, format="vmv")
                print(f"  VMV export: {vmv_path}")
                visualizer.export_network_data(results, casx_path, format="casx")
                print(f"  CASX export: {casx_path}")
                visualizer.export_network_data(results, csv_base, format="csv")
                print(f"  CSV export: {csv_base}_vertices.csv, {csv_base}_edges.csv")
                visualizer.export_network_data(results, json_path, format="json")
                print(f"  JSON export: {json_path}")

                python_results["exports"] = {
                    "profile": "full",
                    "vmv": vmv_path,
                    "casx": casx_path,
                    "csv": csv_base,
                    "json": json_path,
                }
            except Exception as exc:
                print(f"  Warning: Export failed: {exc}")
                import traceback

                traceback.print_exc()

        candidate_edges = load_python_candidate_edges_fn(Path(output_dir))
        if candidate_edges is not None:
            results["candidate_edges"] = candidate_edges
            python_results["candidate_edges"] = candidate_edges
        candidate_audit = load_python_candidate_audit_fn(Path(output_dir))
        if candidate_audit is not None:
            results["candidate_audit"] = candidate_audit
            python_results["candidate_audit"] = candidate_audit
        candidate_lifecycle = load_python_candidate_lifecycle_fn(Path(output_dir))
        if candidate_lifecycle is not None:
            results["candidate_lifecycle"] = candidate_lifecycle
            python_results["candidate_lifecycle"] = candidate_lifecycle

        return python_results
    except Exception as exc:
        elapsed_time = time.time() - start_time
        print(f"\nPython execution failed after {elapsed_time:.2f} seconds")
        import traceback

        traceback.print_exc()
        return {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": str(exc),
            "system_info": system_info,
        }
