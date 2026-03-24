"""
Comparison execution module.

This module handles the execution of MATLAB and Python vectorization pipelines
for comparison purposes.
"""

from __future__ import annotations

import copy
import glob
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from slavv.core import SLAVVProcessor
from slavv.io import export_pipeline_results, load_tiff_volume
from slavv.io.matlab_parser import load_matlab_batch_results
from slavv.utils import get_matlab_info, get_system_info
from slavv.visualization import NetworkVisualizer

from .management import generate_manifest, resolve_run_layout
from .metrics import compare_results
from .reporting import generate_summary


def load_parameters(params_file: str | None = None) -> dict[str, Any]:
    """Load parameters from JSON file or use defaults."""
    if params_file and os.path.exists(params_file):
        with open(params_file) as f:
            params = json.load(f)
    else:
        # Use Python defaults
        params = {
            "microns_per_voxel": [1.0, 1.0, 1.0],
            "radius_of_smallest_vessel_in_microns": 1.5,
            "radius_of_largest_vessel_in_microns": 50.0,
            "approximating_PSF": True,
            "excitation_wavelength_in_microns": 1.3,
            "numerical_aperture": 0.95,
            "sample_index_of_refraction": 1.33,
            "scales_per_octave": 1.5,
            "gaussian_to_ideal_ratio": 1.0,
            "spherical_to_annular_ratio": 1.0,
            "max_voxels_per_node_energy": 1e5,
        }

    # Convert lists to numpy arrays where needed
    if "microns_per_voxel" in params:
        params["microns_per_voxel"] = np.array(params["microns_per_voxel"])

    return params


def discover_matlab_artifacts(output_dir: str | Path) -> dict[str, Any]:
    """Discover the newest MATLAB batch folder and key output artifacts."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return {}

    batch_folders = sorted(
        path for path in output_path.iterdir() if path.is_dir() and path.name.startswith("batch_")
    )
    if not batch_folders:
        return {}

    batch_folder = batch_folders[-1]
    artifacts: dict[str, Any] = {"batch_folder": str(batch_folder)}

    vectors_dir = batch_folder / "vectors"
    if vectors_dir.exists():
        artifacts["vectors_dir"] = str(vectors_dir)
        network_files = sorted(vectors_dir.glob("network_*.mat"))
        if network_files:
            artifacts["network_mat"] = str(network_files[-1])

    return artifacts


def _run_command_with_timeout(
    cmd: list[str], cwd: Path, timeout_seconds: int
) -> tuple[int, str, str, bool]:
    """Run a command and tear down the full process tree on timeout."""
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
        return process.returncode or 0, stdout, stderr, False
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/f", "/t", "/pid", str(process.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            process.kill()

        stdout, stderr = process.communicate()
        returncode = process.returncode if process.returncode is not None else -9
        return returncode, stdout, stderr, True


def run_matlab_vectorization(
    input_file: str,
    output_dir: str,
    matlab_path: str,
    project_root: Path,
    batch_script: str | None = None,
) -> dict[str, Any]:
    """Run MATLAB vectorization via CLI."""
    print("\n" + "=" * 60)
    print("Running MATLAB Implementation")
    print("=" * 60)

    # Validate input file path
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    input_file = os.path.abspath(input_file)
    output_dir = os.path.abspath(output_dir)

    # Define script paths with absolute resolution.
    # Prefer workspace/scripts (current layout), fallback to scripts (legacy layout).
    if batch_script is None:
        script_name = "run_matlab_cli.bat" if os.name == "nt" else "run_matlab_cli.sh"
        candidates = [
            (project_root / "workspace" / "scripts" / "cli" / script_name).resolve(),
            (project_root / "scripts" / "cli" / script_name).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                batch_script = str(candidate)
                break
        if batch_script is None:
            raise FileNotFoundError(
                f"Could not find MATLAB CLI script '{script_name}' in any known location: {candidates}"
            )
    else:
        # Use provided script directly, but validate existence first
        if not os.path.exists(batch_script):
            raise FileNotFoundError(f"Custom batch script not found: {batch_script}")
        batch_script = os.path.abspath(batch_script)

    # Ensure executable permissions on POSIX
    if os.name != "nt" and not os.access(batch_script, os.X_OK):
        try:
            current_permissions = os.stat(batch_script).st_mode
            os.chmod(batch_script, current_permissions | 0o111)
        except Exception as e:
            print(f"Warning: Could not set executable permission on {batch_script}: {e}")

    # Build command list
    cmd = [batch_script, input_file, output_dir, matlab_path]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    print(f"Command: {' '.join(cmd)}")
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Capture system info
    system_info = get_system_info()
    matlab_info = get_matlab_info(matlab_path)
    system_info["matlab"] = matlab_info

    start_time = time.time()
    timeout_seconds = 3600

    returncode, stdout, stderr, timed_out = _run_command_with_timeout(
        cmd, project_root, timeout_seconds
    )
    elapsed_time = time.time() - start_time
    artifacts = discover_matlab_artifacts(output_dir)

    if timed_out:
        print(f"\nMATLAB execution timed out after {elapsed_time:.2f} seconds")
        print("STDOUT:")
        print(stdout)
        print("STDERR:")
        print(stderr)

        matlab_results = {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": f"TimeoutExpired after {timeout_seconds} seconds",
            "stdout": stdout,
            "stderr": stderr,
            "system_info": system_info,
            "output_dir": output_dir,
        }
        matlab_results.update(artifacts)
        return matlab_results

    if returncode != 0:
        print(f"\nMATLAB execution failed after {elapsed_time:.2f} seconds")
        print(f"Exit code: {returncode}")
        print("STDOUT:")
        print(stdout)
        print("STDERR:")
        print(stderr)

        matlab_results = {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": f"MATLAB exited with code {returncode}",
            "stdout": stdout,
            "stderr": stderr,
            "system_info": system_info,
            "output_dir": output_dir,
        }
        matlab_results.update(artifacts)
        return matlab_results

    print(f"\nMATLAB execution completed in {elapsed_time:.2f} seconds")
    print("STDOUT:")
    print(stdout)

    if stderr:
        print("STDERR:")
        print(stderr)

    matlab_results = {
        "success": True,
        "elapsed_time": elapsed_time,
        "output_dir": output_dir,
        "stdout": stdout,
        "stderr": stderr,
        "system_info": system_info,
    }
    matlab_results.update(artifacts)
    return matlab_results


def run_python_vectorization(
    input_file: str, output_dir: str, params: dict[str, Any]
) -> dict[str, Any]:
    """Run Python vectorization."""
    print("\n" + "=" * 60)
    print("Running Python Implementation")
    print("=" * 60)

    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")

    # Capture system info
    system_info = get_system_info()

    # Load image
    print("Loading image...")
    image = load_tiff_volume(input_file)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")

    # Initialize processor
    processor = SLAVVProcessor()

    # Run pipeline
    print("Running pipeline...")
    start_time = time.time()

    def progress_callback(frac, stage):
        print(f"  Progress: {frac * 100:.1f}% - {stage}")

    try:
        results = processor.process_image(
            image,
            params,
            progress_callback=progress_callback,
            checkpoint_dir=os.path.join(output_dir, "checkpoints"),
        )

        elapsed_time = time.time() - start_time

        print(f"\nPython execution completed in {elapsed_time:.2f} seconds")

        # Extract statistics
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
        }

        # Export results
        print("Exporting results...")
        export_pipeline_results(results, output_dir, base_name="python_comparison")

        # Export VMV and CASX formats for visualization
        print("Exporting VMV and CASX formats...")
        try:
            visualizer = NetworkVisualizer()
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
                "vmv": vmv_path,
                "casx": casx_path,
                "csv": csv_base,
                "json": json_path,
            }
        except Exception as e:
            print(f"  Warning: Export failed: {e}")
            import traceback

            traceback.print_exc()

        return python_results

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\nPython execution failed after {elapsed_time:.2f} seconds")
        import traceback

        traceback.print_exc()

        return {
            "success": False,
            "elapsed_time": elapsed_time,
            "error": str(e),
            "system_info": system_info,
        }


def orchestrate_comparison(
    input_file: str,
    output_dir: Path,
    matlab_path: str,
    project_root: Path,
    params: dict[str, Any],
    skip_matlab: bool = False,
    skip_python: bool = False,
) -> int:
    """Run full comparison workflow."""
    layout = resolve_run_layout(output_dir)
    matlab_output = layout["matlab_dir"]
    python_output = layout["python_dir"]
    analysis_dir = layout["analysis_dir"]
    metadata_dir = layout["metadata_dir"]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Run MATLAB
    matlab_results = None
    if not skip_matlab:
        os.makedirs(matlab_output, exist_ok=True)
        matlab_results = run_matlab_vectorization(
            input_file, str(matlab_output), matlab_path, project_root
        )
    else:
        print("\nSkipping MATLAB execution (--skip-matlab)")

    # Run Python
    python_results = None
    if not skip_python:
        os.makedirs(python_output, exist_ok=True)
        python_results = run_python_vectorization(input_file, str(python_output), params)
    else:
        print("\nSkipping Python execution (--skip-python)")

    # Compare results
    if matlab_results and python_results:
        # Try to load parsed MATLAB data
        matlab_parsed = None
        if matlab_results.get("success") and matlab_results.get("batch_folder"):
            print("\nLoading MATLAB output data...")
            try:
                matlab_parsed = load_matlab_batch_results(matlab_results["batch_folder"])
                print(f"Successfully loaded MATLAB data from {matlab_results['batch_folder']}")
            except Exception as e:
                print(f"Warning: Could not load MATLAB output data: {e}")
                print("Comparison will proceed with basic metrics only.")

        comparison = compare_results(matlab_results, python_results, matlab_parsed)

        # Save comparison report
        report_file = analysis_dir / "comparison_report.json"
        with open(report_file, "w") as f:
            # Remove non-serializable items for JSON
            report = {
                "matlab": copy.deepcopy(comparison["matlab"]),
                "python": copy.deepcopy(comparison["python"]),
                "performance": comparison["performance"],
                "vertices": comparison.get("vertices", {}),
                "edges": comparison.get("edges", {}),
                "network": comparison.get("network", {}),
            }
            json.dump(
                report,
                f,
                indent=2,
                default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
            )

        print(f"\nComparison report saved to: {report_file}")

    # Generate summary.txt automatically
    try:
        summary_file = analysis_dir / "summary.txt"
        generate_summary(output_dir, summary_file)
    except Exception as e:
        print(f"Note: Could not auto-generate summary: {e}")

    # Generate manifest automatically
    try:
        manifest_file = metadata_dir / "run_manifest.md"
        generate_manifest(output_dir, manifest_file)
        print(f"Manifest generated: {manifest_file}")
    except Exception as e:
        print(f"Note: Could not auto-generate manifest: {e}")

    # Print final summary status
    if matlab_results and python_results:
        success = matlab_results.get("success") and python_results.get("success")
        return 0 if success else 1
    return 0


def run_standalone_comparison(
    matlab_dir: Path, python_dir: Path, output_dir: Path, project_root: Path
) -> int:
    """
    Run comparison on existing result directories.

    Args:
        matlab_dir: Directory containing MATLAB results (e.g., .../TIMESTAMP_matlab_run)
        python_dir: Directory containing Python results (e.g., .../TIMESTAMP_python_run)
        output_dir: Directory to save comparison results
        project_root: Root of the repository

    Returns:
        0 if successful, 1 otherwise
    """
    print("\n" + "=" * 60)
    print("Running Standalone Comparison")
    print("=" * 60)
    print(f"MATLAB Dir: {matlab_dir}")
    print(f"Python Dir: {python_dir}")
    print(f"Output Dir: {output_dir}")

    layout = resolve_run_layout(output_dir)
    analysis_dir = layout["analysis_dir"]
    metadata_dir = layout["metadata_dir"]
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # 1. Reconstruct MATLAB results dict
    matlab_results = {
        "success": True,  # Assume success if we are selecting it
        "output_dir": str(matlab_dir),
        "elapsed_time": 0.0,  # Unknown
    }

    # Find batch folder
    matlab_layout = resolve_run_layout(matlab_dir)
    matlab_root = matlab_layout["matlab_dir"]
    batch_folders = [
        d for d in os.listdir(matlab_root) if (matlab_root / d).is_dir() and d.startswith("batch_")
    ]

    if batch_folders:
        batch_folder = str(matlab_root / sorted(batch_folders)[-1])
        matlab_results["batch_folder"] = batch_folder
        print(f"Found MATLAB batch folder: {batch_folder}")
    else:
        print("Warning: No MATLAB batch_* folder found.")

    # 2. Reconstruct Python results dict
    python_results = {"success": True, "output_dir": str(python_dir), "elapsed_time": 0.0}

    # Load python results
    # Try finding python_comparison_*.json
    # Check root of python dir
    python_layout = resolve_run_layout(python_dir)
    python_root = python_layout["python_dir"]
    json_files = glob.glob(str(python_root / "python_comparison_*.json"))

    if json_files:
        # Load the latest one
        latest_json = sorted(json_files)[-1]
        print(f"Loading Python results from: {latest_json}")
        try:
            with open(latest_json) as f:
                loaded_data = json.load(f)

            # Need to convert lists back to numpy arrays for metric comparison
            # This is a simplified reconstruction. For full fidelity we'd need convert_lists_to_arrays
            # But compare_results should handle basic lists or we can do a quick pass

            # Specifically restore vertices/positions and edges/traces
            if "vertices" in loaded_data and "positions" in loaded_data["vertices"]:
                loaded_data["vertices"]["positions"] = np.array(
                    loaded_data["vertices"]["positions"]
                )
                if "radii" in loaded_data["vertices"]:
                    loaded_data["vertices"]["radii"] = np.array(loaded_data["vertices"]["radii"])

            # Edges traces are list of arrays
            if "edges" in loaded_data and "traces" in loaded_data["edges"]:
                loaded_data["edges"]["traces"] = [
                    np.array(t) for t in loaded_data["edges"]["traces"]
                ]

            python_results["results"] = loaded_data
            python_results["vertices_count"] = len(
                loaded_data.get("vertices", {}).get("positions", [])
            )
            python_results["edges_count"] = len(loaded_data.get("edges", {}).get("traces", []))

        except Exception as e:
            print(f"Error loading Python JSON: {e}")
            python_results["success"] = False
    else:
        # Fallback: Check for network.json
        print("Warning: No python_comparison_*.json found. Checking for network.json...")
        network_json_paths = glob.glob(str(python_root / "network.json"))

        if network_json_paths:
            network_json = network_json_paths[0]
            print(f"Loading Python results from fallback: {network_json}")
            try:
                with open(network_json) as f:
                    net_data = json.load(f)

                # network.json structure: {vertices: {positions: ...}, edges: {connections: ...}}
                # Adapt to structure expected by compare_results
                loaded_data = {
                    "vertices": net_data.get("vertices", {}),
                    "edges": net_data.get("edges", {}),
                    "network": net_data.get("network", {}),
                }

                # Restore arrays
                if "positions" in loaded_data["vertices"]:
                    loaded_data["vertices"]["positions"] = np.array(
                        loaded_data["vertices"]["positions"]
                    )

                # Note: network.json usually has 'connections' (N,2) ints, not 'traces'.
                # We map connections to traces as best effort if traces missing
                # Count from connections if traces missing
                if "connections" in loaded_data["edges"]:
                    python_results["edges_count"] = len(loaded_data["edges"]["connections"])

                python_results["results"] = loaded_data
                python_results["vertices_count"] = len(
                    loaded_data.get("vertices", {}).get("positions", [])
                )

                # Count from connections if traces missing
                if "connections" in loaded_data.get("edges", {}):
                    python_results["edges_count"] = len(loaded_data["edges"]["connections"])
                else:
                    python_results["edges_count"] = len(
                        loaded_data.get("edges", {}).get("traces", [])
                    )

            except Exception as e:
                print(f"Error loading network.json: {e}")
                python_results["success"] = False
        else:
            print("Error: No result files found.")
            python_results["success"] = False

    # 3. Compare
    # Try to load parsed MATLAB data
    matlab_parsed = None
    if matlab_results.get("batch_folder"):
        print("\nLoading MATLAB output data...")
        try:
            matlab_parsed = load_matlab_batch_results(matlab_results["batch_folder"])
            print("Successfully loaded MATLAB data")
        except Exception as e:
            print(f"Warning: Could not load MATLAB output data: {e}")

    comparison = compare_results(matlab_results, python_results, matlab_parsed)

    # 4. Save
    report_file = analysis_dir / "comparison_report.json"
    with open(report_file, "w") as f:
        # Remove non-serializable items for JSON
        report = {
            "matlab": copy.deepcopy(comparison["matlab"]),
            "python": copy.deepcopy(comparison["python"]),
            "performance": comparison["performance"],
            "vertices": comparison.get("vertices", {}),
            "edges": comparison.get("edges", {}),
            "network": comparison.get("network", {}),
        }
        json.dump(
            report,
            f,
            indent=2,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o),
        )
    print(f"\nComparison report saved to: {report_file}")

    # Generate summary
    try:
        summary_file = analysis_dir / "summary.txt"
        generate_summary(output_dir, summary_file)
    except Exception as e:
        print(f"Note: Could not auto-generate summary: {e}")

    # Generate manifest
    try:
        manifest_file = metadata_dir / "run_manifest.md"
        generate_manifest(output_dir, manifest_file)
    except Exception as e:
        print(f"Note: Could not auto-generate manifest: {e}")

    return 0
