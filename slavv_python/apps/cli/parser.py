"""Argument parser construction for the SLAVV CLI."""

from __future__ import annotations

import argparse

from ...utils import PIPELINE_PROFILE_CHOICES

_EXPORT_FILE_NAMES = {
    "csv": "network.csv",
    "json": "network.json",
    "casx": "network.casx",
    "vmv": "network.vmv",
    "mat": "network.mat",
}


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="slavv_python",
        description="SLAVV - Segmentation-Less, Automated, Vascular Vectorization",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    parser.add_argument("--tui", action="store_true", help="Launch interactive TUI wizard/monitor")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the SLAVV pipeline on a TIFF volume")
    run_parser.add_argument("-i", "--input", required=True, help="Path to input TIFF file")
    run_parser.add_argument(
        "-o",
        "--output",
        default="./slavv_output",
        help="Output directory (default: ./slavv_output)",
    )
    run_parser.add_argument(
        "--run-dir",
        default=None,
        help="Structured run directory for resumable status tracking",
    )
    run_parser.add_argument(
        "--profile",
        dest="pipeline_profile",
        choices=list(PIPELINE_PROFILE_CHOICES),
        default="paper",
        help=(
            "Pipeline profile preset (default: paper). "
            "'paper' is the primary native Python workflow; "
            "'matlab_compat' preserves the legacy MATLAB-shaped defaults."
        ),
    )
    run_parser.add_argument(
        "--energy-storage-format",
        choices=["auto", "npy", "zarr"],
        default=None,
        help=(
            "Storage format for resumable energy arrays. "
            "When omitted, the selected profile default is used."
        ),
    )
    run_parser.add_argument(
        "--energy-method",
        choices=["hessian", "frangi", "sato", "simpleitk_objectness", "cupy_hessian"],
        default=None,
        help=(
            "Energy computation method. "
            "'simpleitk_objectness' is experimental and spacing-aware; "
            "'cupy_hessian' is experimental and requires an NVIDIA GPU."
        ),
    )
    run_parser.add_argument(
        "--energy-projection-mode",
        choices=["matlab", "paper"],
        default=None,
        help=(
            "Projection mode for the default Hessian energy stack. "
            "'paper' uses the published blended scale-estimate projection."
        ),
    )
    run_parser.add_argument(
        "--edge-method",
        choices=["tracing", "watershed"],
        default=None,
        help="Edge extraction method. When omitted, the selected profile default is used.",
    )
    run_parser.add_argument(
        "--vessel-radius",
        type=float,
        default=None,
        help="Smallest vessel radius in microns. When omitted, the profile default is used.",
    )
    run_parser.add_argument(
        "--largest-vessel-radius",
        type=float,
        default=None,
        help="Largest vessel radius in microns. When omitted, the profile default is used.",
    )
    run_parser.add_argument(
        "--microns-per-voxel",
        type=float,
        nargs=3,
        default=None,
        metavar=("Y", "X", "Z"),
        help="Voxel size in microns. When omitted, the profile default is used.",
    )
    run_parser.add_argument(
        "--scales-per-octave",
        type=float,
        default=None,
        help="Scale density for multi-scale filtering. When omitted, the profile default is used.",
    )
    run_parser.add_argument(
        "--gaussian-to-ideal-ratio",
        type=float,
        default=None,
        help="Gaussian-vs-ideal matched-filter blend. When omitted, the profile default is used.",
    )
    run_parser.add_argument(
        "--spherical-to-annular-ratio",
        type=float,
        default=None,
        help="Spherical-vs-annular weighting ratio. When omitted, the profile default is used.",
    )
    run_parser.add_argument(
        "--energy-upper-bound",
        type=float,
        default=None,
        help="Upper energy bound for vertex and edge candidate acceptance.",
    )
    run_parser.add_argument(
        "--space-strel-apothem",
        type=int,
        default=None,
        help="Vertex suppression spacing in voxels.",
    )
    run_parser.add_argument(
        "--space-strel-apothem-edges",
        type=int,
        default=None,
        help="Edge exclusion spacing in voxels.",
    )
    run_parser.add_argument(
        "--length-dilation-ratio",
        type=float,
        default=None,
        help="Rendering-to-detection dilation ratio for exclusion volumes.",
    )
    run_parser.add_argument(
        "--number-of-edges-per-vertex",
        type=int,
        default=None,
        help="Maximum number of edge traces launched per seed vertex.",
    )
    run_parser.add_argument(
        "--step-size-per-origin-radius",
        type=float,
        default=None,
        help="Tracing step size relative to the origin vertex radius.",
    )
    run_parser.add_argument(
        "--max-edge-length-per-origin-radius",
        type=float,
        default=None,
        help="Maximum tracing length relative to the origin vertex radius.",
    )
    run_parser.add_argument(
        "--max-edge-energy",
        type=float,
        default=None,
        help="Maximum allowable energy along traced edges.",
    )
    run_parser.add_argument(
        "--min-hair-length-in-microns",
        type=float,
        default=None,
        help="Minimum terminal hair length to preserve during network cleanup.",
    )
    run_parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for processing (default: 1). Use -1 for all cores.",
    )
    run_parser.add_argument(
        "--export",
        nargs="+",
        choices=["csv", "json", "mat", "casx", "vmv", "all"],
        default=[],
        help="Export formats (can specify multiple)",
    )
    run_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    run_parser.add_argument(
        "--stop-after",
        choices=["energy", "vertices", "edges", "network"],
        default=None,
        help="Stop the pipeline early after completing this stage.",
    )
    run_parser.add_argument(
        "--force-rerun-from",
        choices=["energy", "vertices", "edges", "network"],
        default=None,
        help="Ignore checkpoints and force recalculation starting from this stage.",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Analyze an exported network JSON file")
    analyze_parser.add_argument("-i", "--input", required=True, help="Path to input network.json")
    analyze_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    plot_parser = subparsers.add_parser(
        "plot", help="Generate plots from an exported network JSON file"
    )
    plot_parser.add_argument("-i", "--input", required=True, help="Path to input network.json")
    plot_parser.add_argument(
        "-o", "--output", default="plots.html", help="Output HTML file path (default: plots.html)"
    )
    plot_parser.add_argument(
        "--number-of-bins", type=int, default=50, help="Number of bins for histograms"
    )
    plot_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    subparsers.add_parser("info", help="Print version and system information")

    status_parser = subparsers.add_parser("status", help="Inspect the status of a resumable run")
    status_parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing run metadata",
    )
    status_parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    return parser


__all__ = ["_EXPORT_FILE_NAMES", "_build_cli_parser"]
