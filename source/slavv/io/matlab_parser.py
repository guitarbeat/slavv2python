#!/usr/bin/env python3
"""
MATLAB Output Parser for SLAVV Vectorization Results

This module loads and extracts data from MATLAB .mat files produced by vectorize_V200.
It handles the structure of MATLAB batch output folders and provides utilities to
extract vertices, edges, network statistics, and timing information.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


class MATLABParseError(Exception):
    """Exception raised when MATLAB output parsing fails."""


def _get_struct_value(obj: Any, name: str) -> Any:
    """Safely fetch a MATLAB struct field."""
    if obj is None or not hasattr(obj, name):
        return None

    value = getattr(obj, name)
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(()).item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _count_items(value: Any) -> int:
    """Count items for MATLAB arrays, including object arrays and row matrices."""
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0
        if value.ndim == 0:
            return 1
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return 1


def _normalize_space_subscripts(value: Any) -> np.ndarray:
    """Convert MATLAB 1-based `[y, x, z]` voxel subscripts into Python coordinates."""
    array = np.asarray(value)
    if array.size == 0:
        return np.array([])
    if array.ndim == 1:
        array = array.reshape(1, -1)
    try:
        normalized = array.astype(np.float32, copy=True)
    except (TypeError, ValueError):
        return array
    if normalized.shape[1] >= 3:
        normalized[:, :3] -= 1.0
        normalized[:, :3] = normalized[:, [2, 1, 0]]
    return normalized


def _normalize_index_array(value: Any) -> np.ndarray:
    """Convert MATLAB 1-based index arrays into 0-based integer arrays."""
    array = np.asarray(value)
    if array.size == 0:
        return np.array([])
    try:
        normalized = array.astype(np.int32, copy=True)
    except (TypeError, ValueError):
        return np.array([])
    if np.min(normalized) >= 1:
        normalized -= 1
    return normalized


def _normalize_trace_cells(space_subs: Any) -> list[np.ndarray]:
    """Normalize MATLAB cell-array edge/strand traces."""
    if isinstance(space_subs, np.ndarray) and space_subs.dtype == object:
        return [
            _normalize_space_subscripts(trace)
            for trace in space_subs
            if trace is not None and np.asarray(trace).size > 0
        ]
    if isinstance(space_subs, np.ndarray) and space_subs.size > 0:
        return [_normalize_space_subscripts(space_subs)]
    return []


def _infer_strand_count(
    mat_data: dict[str, Any], network_data: dict[str, Any], network_stats: Any | None
) -> int:
    """Infer strand count from MATLAB network outputs when num_strands is absent."""
    explicit_count = _get_struct_value(network_stats, "num_strands")
    if explicit_count:
        return int(explicit_count)

    strand_lengths = _get_struct_value(network_stats, "strand_lengths")
    strand_lengths_count = _count_items(strand_lengths)
    if strand_lengths_count > 0:
        return strand_lengths_count

    strands2vertices_count = _count_items(mat_data.get("strands2vertices"))
    if strands2vertices_count > 0:
        return strands2vertices_count

    strand_count = len(network_data.get("strands", []))
    if strand_count > 0:
        return strand_count

    return _count_items(mat_data.get("strand_subscripts"))


def find_batch_folder(output_dir: Union[str, Path]) -> Path | None:
    """Find the most recent MATLAB batch folder in the output directory.

    Parameters
    ----------
    output_dir : str | Path
        Directory to search for batch folders

    Returns
    -------
    Optional[Path]
        Path to the most recent batch folder, or None if not found
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return None

    # Find all batch folders (format: batch_YYMMDD-HHmmss)
    batch_folders = [d for d in output_path.iterdir() if d.is_dir() and d.name.startswith("batch_")]

    if not batch_folders:
        logger.warning(f"No batch folders found in {output_dir}")
        return None

    # Sort by name (chronological due to timestamp format) and return most recent
    batch_folders.sort()
    return batch_folders[-1]


def load_mat_file_safe(file_path: Path) -> dict[str, Any] | None:
    """Safely load a MATLAB .mat file with error handling.

    Parameters
    ----------
    file_path : Path
        Path to the .mat file

    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary containing MATLAB data, or None if loading fails
    """
    try:
        # Load with squeeze_me=True to remove singleton dimensions
        # struct_as_record=False to access struct fields as attributes
        data = loadmat(str(file_path), squeeze_me=True, struct_as_record=False, mat_dtype=True)
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def extract_vertices(mat_data: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract vertex information from MATLAB network data."""
    vertices_info = {"positions": np.array([]), "radii": np.array([]), "count": 0}

    # Check for root-level arrays (common in this dataset)
    if "vertex_space_subscripts" in mat_data:
        positions = _normalize_space_subscripts(mat_data["vertex_space_subscripts"])
        if positions.ndim == 1 and positions.size > 0:
            positions = positions.reshape(-1, 1)
        vertices_info["positions"] = positions
        vertices_info["count"] = positions.shape[0] if positions.size > 0 else 0

        # Handle scales/radii
        if "vertex_scale_subscripts" in mat_data:
            scale_indices = _normalize_index_array(mat_data["vertex_scale_subscripts"])
            vertices_info["scale_indices"] = scale_indices

            # Try to map to radii if range is available
            if "lumen_radius_in_microns_range" in mat_data:
                radii_range = np.array(mat_data["lumen_radius_in_microns_range"])
                if scale_indices.size > 0:
                    # Matlab indices are 1-based usually, check min
                    # If they are from Matlab, they might be 1-based.
                    # But scipy.io might load them as is.
                    # Let's assume 1-based if coming from Matlab.
                    # Actually, let's check min value.
                    vertices_info["radii"] = radii_range[scale_indices.astype(int)]

    # Fallback to struct-based extraction
    elif "vertex" in mat_data or "vertices" in mat_data:
        vertex_struct = mat_data.get("vertex", mat_data.get("vertices"))
        if hasattr(vertex_struct, "space_subscripts"):
            positions = _normalize_space_subscripts(vertex_struct.space_subscripts)
            vertices_info["positions"] = positions
            vertices_info["count"] = positions.shape[0] if positions.size > 0 else 0
        elif hasattr(vertex_struct, "positions"):
            positions = _normalize_space_subscripts(vertex_struct.positions)
            vertices_info["positions"] = positions
            vertices_info["count"] = positions.shape[0] if positions.size > 0 else 0

        if hasattr(vertex_struct, "scale_subscripts"):
            vertices_info["scale_indices"] = _normalize_index_array(vertex_struct.scale_subscripts)

        if hasattr(vertex_struct, "radii"):
            vertices_info["radii"] = np.array(vertex_struct.radii)

    logger.info(f"Extracted {vertices_info['count']} vertices")
    return vertices_info


def extract_edges(mat_data: dict[str, Any]) -> dict[str, Any]:
    """Extract edge information from MATLAB network data."""
    edges_info = {"connections": np.array([]), "traces": [], "count": 0, "total_length": 0.0}

    # Check for root-level arrays
    if "edges2vertices" in mat_data:
        indices = _normalize_index_array(mat_data["edges2vertices"])
        edges_info["connections"] = indices
        edges_info["count"] = indices.shape[0] if indices.size > 0 else 0

    if "edge_space_subscripts" in mat_data:
        edges_info["traces"] = _normalize_trace_cells(mat_data["edge_space_subscripts"])

    # Fallback to struct-based
    if edges_info["count"] == 0:
        edge_struct = mat_data.get("edge", mat_data.get("edges"))
        if edge_struct is not None:
            if hasattr(edge_struct, "vertices"):
                indices = _normalize_index_array(edge_struct.vertices)
                edges_info["connections"] = indices
                edges_info["count"] = indices.shape[0]

            if hasattr(edge_struct, "space_subscripts"):
                edges_info["traces"] = _normalize_trace_cells(edge_struct.space_subscripts)

    logger.info(f"Extracted {edges_info['count']} edges")
    return edges_info


def extract_network_data(mat_data: dict[str, Any]) -> dict[str, Any]:
    """Extract network topology and statistics."""
    network_data = {"strands": [], "stats": {}}

    # Extract strands (topology)
    if "strand_subscripts" in mat_data:
        network_data["strands"] = _normalize_trace_cells(mat_data["strand_subscripts"])
    elif "strand" in mat_data:
        # Alternative key used by some MATLAB outputs
        s = mat_data["strand"]
        if isinstance(s, np.ndarray) and s.size > 0:
            network_data["strands"] = _normalize_trace_cells(s)

    if "strands2vertices" in mat_data:
        strands_to_vertices = _normalize_index_array(mat_data["strands2vertices"])
        if strands_to_vertices.size > 0:
            if strands_to_vertices.ndim == 1:
                strands_to_vertices = strands_to_vertices.reshape(1, -1)
            network_data["strands_to_vertices"] = [
                [int(value) for value in row if int(value) >= 0] for row in strands_to_vertices
            ]

    # Extract statistics
    if "network_statistics" in mat_data:
        ns = mat_data["network_statistics"]
        network_data["stats"]["strand_count"] = _infer_strand_count(mat_data, network_data, ns)
        total_length = _get_struct_value(ns, "length")
        if total_length in (None, 0, 0.0):
            strand_lengths = _get_struct_value(ns, "strand_lengths")
            if isinstance(strand_lengths, np.ndarray) and strand_lengths.size > 0:
                total_length = float(np.sum(strand_lengths))
        network_data["stats"]["total_length_microns"] = total_length or 0.0
        network_data["stats"]["mean_radius_microns"] = _get_struct_value(ns, "strand_ave_radii")

        # Handle mean radius being an array
        mr = network_data["stats"]["mean_radius_microns"]
        if isinstance(mr, np.ndarray):
            network_data["stats"]["mean_radius_microns"] = (
                float(np.mean(mr)) if mr.size > 0 else 0.0
            )
    elif "strand_subscripts" in mat_data or "strands2vertices" in mat_data:
        network_data["stats"]["strand_count"] = _infer_strand_count(mat_data, network_data, None)

    return network_data


def extract_network_stats(mat_data: dict[str, Any]) -> dict[str, Any]:
    """Extract network statistics from MATLAB data. Returns a flat stats dict."""
    net = extract_network_data(mat_data)
    stats = net.get("stats", {})
    strand_count = stats.get("strand_count")
    if not strand_count:
        strand_count = len(net.get("strands", []))
    # Ensure expected keys with defaults
    return {
        "strand_count": strand_count,
        "total_length_microns": stats.get("total_length_microns", 0.0),
        "mean_radius_microns": stats.get("mean_radius_microns", 0.0),
    }


def load_matlab_batch_results(batch_folder: Union[str, Path]) -> dict[str, Any]:
    """Load and aggregate results from a MATLAB batch output folder."""
    batch_path = Path(batch_folder)
    if not batch_path.exists():
        raise MATLABParseError(f"Batch folder not found: {batch_folder}")
    if not batch_path.is_dir():
        raise MATLABParseError(f"Batch folder is not a directory: {batch_folder}")

    logger.info(f"Loading MATLAB results from: {batch_path}")

    results = {
        "vertices": {"count": 0, "positions": np.array([]), "radii": np.array([])},
        "edges": {"count": 0, "indices": np.array([]), "traces": [], "total_length": 0.0},
        "network": {"strands": []},
        "network_stats": {},
        "timings": {},
        "batch_folder": str(batch_path),
        "files": [],
    }

    vectors_dir = batch_path / "vectors"
    if not vectors_dir.exists():
        raise MATLABParseError(f"Vectors directory not found: {vectors_dir}")

    timings_file = batch_path / "timings.json"
    if timings_file.exists():
        try:
            with open(timings_file, encoding="utf-8") as handle:
                timings = json.load(handle)
            results["timings"] = {
                "total": timings.get("total_seconds", 0.0),
                "stage_seconds": timings.get("stage_seconds", {}),
                "wrapper_mode": timings.get("wrapper_mode", ""),
                "matlab_version": timings.get("matlab_version", ""),
            }
        except Exception as exc:
            logger.warning("Failed to read MATLAB timings from %s: %s", timings_file, exc)

    # Helper to merge dicts
    def merge_info(target, source):
        for k, v in source.items():
            if isinstance(v, np.ndarray):
                if v.size > 0:
                    target[k] = v
            elif isinstance(v, list):
                if v:
                    target[k] = v
            elif v:
                target[k] = v

    # 1. Load Vertices (prefer curated/final artifacts when available)
    v_files = list(vectors_dir.glob("curated_vertices_*.mat")) or list(
        vectors_dir.glob("vertices_*.mat")
    )
    if v_files:
        f = v_files[-1]
        logger.info(f"Loading vertices: {f.name}")
        data = load_mat_file_safe(f)
        if data:
            v_info = extract_vertices(data)
            merge_info(results["vertices"], v_info)
            results["files"].append(str(f))

    # 2. Load Edges (prefer curated/final artifacts when available)
    e_files = list(vectors_dir.glob("curated_edges_*.mat")) or list(vectors_dir.glob("edges_*.mat"))
    if e_files:
        f = e_files[-1]
        logger.info(f"Loading edges: {f.name}")
        data = load_mat_file_safe(f)
        if data:
            e_info = extract_edges(data)
            merge_info(results["edges"], e_info)
            results["files"].append(str(f))

    # 3. Load Network (for stats and topology)
    n_files = list(vectors_dir.glob("network_*.mat"))
    if n_files:
        f = n_files[-1]
        logger.info(f"Loading network: {f.name}")
        data = load_mat_file_safe(f)
        if data:
            net_data = extract_network_data(data)
            for key, value in net_data.items():
                if key != "stats" and value:
                    results["network"][key] = value
            results["network_stats"].update(net_data.get("stats", {}))
            results["files"].append(str(f))

    return results


def load_matlab_results_from_output_dir(output_dir: Union[str, Path]) -> dict[str, Any] | None:
    """Find and load MATLAB results from an output directory.

    Convenience function that finds the most recent batch folder and loads results.

    Parameters
    ----------
    output_dir : str | Path
        Directory containing MATLAB batch output folders

    Returns
    -------
    Optional[Dict[str, Any]]
        Results dictionary from load_matlab_batch_results, or None if no batch folder found
    """
    batch_folder = find_batch_folder(output_dir)
    if batch_folder is None:
        return None

    try:
        return load_matlab_batch_results(batch_folder)
    except MATLABParseError as e:
        logger.error(f"Failed to load MATLAB results: {e}")
        return None


if __name__ == "__main__":
    # Test the parser with command-line arguments
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Parse MATLAB vectorization output")
    parser.add_argument("batch_folder", help="Path to MATLAB batch folder")
    parser.add_argument("--output", help="Output JSON file for parsed results")

    args = parser.parse_args()

    try:
        results = load_matlab_batch_results(args.batch_folder)

        print("\n" + "=" * 60)
        print("MATLAB Results Summary")
        print("=" * 60)
        print(f"Vertices: {results['vertices'].get('count', 0)}")
        print(f"Edges: {results['edges'].get('count', 0)}")
        print(f"Strands: {results['network_stats'].get('strand_count', 0)}")
        print(
            f"Total length: {results['network_stats'].get('total_length_microns', 0):.2f} microns"
        )
        print(f"Mean radius: {results['network_stats'].get('mean_radius_microns', 0):.2f} microns")
        print(f"\nTiming: {results['timings'].get('total', 0):.2f} seconds")
        print(f"\nFiles loaded: {len(results['files'])}")
        for file_path in results["files"]:
            print(f"  - {file_path}")
        print("=" * 60)

        if args.output:
            # Save to JSON (with numpy array conversion)
            output_data = {
                "vertices_count": results["vertices"].get("count", 0),
                "edges_count": results["edges"].get("count", 0),
                "network_stats": results["network_stats"],
                "timings": results["timings"],
                "batch_folder": results["batch_folder"],
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")

    except MATLABParseError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
