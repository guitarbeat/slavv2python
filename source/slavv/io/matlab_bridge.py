"""
MATLAB to Python bridge for SLAVV checkpoint import.

This module converts MATLAB ``batch_*`` output folders into Python checkpoint
pickles so the Python pipeline can resume from MATLAB-produced stages.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import h5py
import joblib
import numpy as np

from slavv.io.matlab_parser import (
    extract_edges,
    extract_network_data,
    extract_vertices,
    find_batch_folder,
    load_mat_file_safe,
)

logger = logging.getLogger(__name__)


def _settings_file(batch_path: Path, stem: str) -> Path | None:
    """Return the latest settings MAT file for a given stage stem."""
    settings_dir = batch_path / "settings"
    if not settings_dir.exists():
        return None
    candidates = sorted(settings_dir.glob(f"{stem}_*.mat"))
    return candidates[-1] if candidates else None


def _load_stage_settings(batch_path: Path, stem: str) -> dict[str, Any]:
    """Load a MATLAB settings MAT file with safe fallbacks."""
    settings_path = _settings_file(batch_path, stem)
    if settings_path is None:
        return {}
    data = load_mat_file_safe(settings_path)
    return data or {}


def _normalize_setting_value(value: Any) -> Any:
    """Convert MATLAB-loaded settings into Python-native scalars and lists."""
    array = np.asarray(value)
    if array.size == 0:
        return None
    if array.ndim == 0 or array.size == 1:
        item = array.reshape(-1)[0]
        return item.item() if isinstance(item, np.generic) else item
    return array.tolist()


def _coerce_radius_axes(radius_range: Any) -> np.ndarray:
    """Normalize MATLAB radius settings into a ``(num_scales, 3)`` array."""
    axes = np.asarray(radius_range, dtype=np.float32)
    if axes.size == 0:
        return np.ones((1, 3), dtype=np.float32)
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)
    if axes.shape[1] == 1:
        axes = np.repeat(axes, 3, axis=1)
    return axes.astype(np.float32, copy=False)


def _scalar_radius_range(radius_axes: np.ndarray) -> np.ndarray:
    """Collapse anisotropic radii into the scalar convention used by tracing."""
    if radius_axes.size == 0:
        return np.ones((1,), dtype=np.float32)
    if radius_axes.shape[1] == 1:
        return radius_axes[:, 0].astype(np.float32, copy=False)
    return np.cbrt(np.prod(radius_axes, axis=1)).astype(np.float32, copy=False)


def _read_matlab_energy_hdf5(batch_path: Path) -> tuple[np.ndarray, Path]:
    """Read the MATLAB HDF5 sidecar that stores scale and energy volumes."""
    data_dir = batch_path / "data"
    candidates = sorted(
        path
        for path in data_dir.glob("energy_*")
        if path.is_file() and path.suffix.lower() != ".mat"
    )
    if not candidates:
        raise FileNotFoundError(f"No MATLAB HDF5 energy sidecar found in {data_dir}")

    energy_path = candidates[-1]
    with h5py.File(energy_path, "r") as handle:
        if "d" not in handle:
            raise KeyError(f"MATLAB energy sidecar {energy_path} does not contain dataset 'd'")
        data = handle["d"][()]

    array = np.asarray(data)
    if array.ndim != 4:
        raise ValueError(
            f"Expected MATLAB energy sidecar dataset 'd' to be 4D, got shape {array.shape}"
        )
    if array.shape[0] != 2 and array.shape[-1] == 2:
        array = np.moveaxis(array, -1, 0)
    if array.shape[0] < 2:
        raise ValueError(
            f"Expected MATLAB energy sidecar dataset 'd' to expose 2 channels, got {array.shape}"
        )

    return array, energy_path


def _load_matlab_energy_checkpoint(batch_path: Path) -> dict[str, Any]:
    """Build a pipeline-compatible MATLAB energy checkpoint payload."""
    raw_energy, energy_path = _read_matlab_energy_hdf5(batch_path)
    settings = _load_stage_settings(batch_path, "energy")

    scale_channel = np.asarray(raw_energy[0], dtype=np.float32)
    energy_channel = np.asarray(raw_energy[1], dtype=np.float32)
    scale_indices = np.rint(scale_channel).astype(np.int16, copy=False) - 1
    scale_indices = np.maximum(scale_indices, 0).astype(np.int16, copy=False)

    lumen_radius_pixels_axes = _coerce_radius_axes(settings.get("lumen_radius_in_pixels_range", []))
    lumen_radius_pixels = _scalar_radius_range(lumen_radius_pixels_axes)
    lumen_radius_microns = np.asarray(
        settings.get("lumen_radius_in_microns_range", lumen_radius_pixels),
        dtype=np.float32,
    ).reshape(-1)
    if lumen_radius_microns.size == 0:
        lumen_radius_microns = lumen_radius_pixels.astype(np.float32, copy=False)

    microns_per_voxel = np.asarray(
        settings.get("microns_per_voxel", np.ones(3, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if microns_per_voxel.size == 1:
        microns_per_voxel = np.repeat(microns_per_voxel, 3)
    pixels_per_sigma_psf = np.asarray(
        settings.get("pixels_per_sigma_PSF", np.zeros(3, dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    if pixels_per_sigma_psf.size == 1:
        pixels_per_sigma_psf = np.repeat(pixels_per_sigma_psf, 3)
    microns_per_sigma_psf = (pixels_per_sigma_psf * microns_per_voxel).astype(
        np.float32, copy=False
    )

    return {
        "energy": energy_channel.astype(np.float32, copy=False),
        "scale_indices": scale_indices,
        "lumen_radius_pixels": lumen_radius_pixels.astype(np.float32, copy=False),
        "lumen_radius_pixels_axes": lumen_radius_pixels_axes.astype(np.float32, copy=False),
        "lumen_radius_microns": lumen_radius_microns.astype(np.float32, copy=False),
        "pixels_per_sigma_PSF": pixels_per_sigma_psf.astype(np.float32, copy=False),
        "microns_per_sigma_PSF": microns_per_sigma_psf,
        "energy_sign": -1.0,
        "image_shape": tuple(int(value) for value in energy_channel.shape),
        "energy_origin": "matlab_batch_hdf5",
        "energy_source": "matlab_batch_hdf5",
        "matlab_batch_folder": str(batch_path),
        "matlab_energy_path": str(energy_path),
    }


def _mat_vertices_to_python(
    mat_vertices: dict[str, Any],
    energy_checkpoint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert MATLAB parser vertex payload into a pipeline-compatible checkpoint."""
    positions = np.asarray(mat_vertices.get("positions", np.empty((0, 3))), dtype=np.float32)
    if positions.size == 0:
        positions = np.empty((0, 3), dtype=np.float32)
    elif positions.ndim == 1:
        positions = positions.reshape(1, -1).astype(np.float32, copy=False)
    if positions.shape[1] > 3:
        positions = positions[:, :3]

    count = int(positions.shape[0])
    scales = np.asarray(
        mat_vertices.get("scale_indices", np.zeros((count,), dtype=np.int16))
    ).reshape(-1)
    if scales.size == 0:
        scales = np.zeros((count,), dtype=np.int16)
    elif scales.size != count:
        scales = np.resize(scales, count)
    scales = np.rint(scales).astype(np.int16, copy=False)
    scales = np.maximum(scales, 0).astype(np.int16, copy=False)

    radii_microns = np.asarray(mat_vertices.get("radii", np.array([])), dtype=np.float32).reshape(
        -1
    )
    radii_pixels = np.zeros((count,), dtype=np.float32)

    if energy_checkpoint is not None:
        microns_range = np.asarray(energy_checkpoint["lumen_radius_microns"], dtype=np.float32)
        pixels_range = np.asarray(energy_checkpoint["lumen_radius_pixels"], dtype=np.float32)
        if count:
            safe_scales = np.clip(scales.astype(np.int64), 0, len(microns_range) - 1)
            radii_microns = microns_range[safe_scales].astype(np.float32, copy=False)
            radii_pixels = pixels_range[safe_scales].astype(np.float32, copy=False)
    elif radii_microns.size == count:
        radii_pixels = radii_microns.astype(np.float32, copy=True)

    if radii_microns.size != count:
        radii_microns = np.zeros((count,), dtype=np.float32)

    return {
        "positions": positions.astype(np.float32, copy=False),
        "scales": scales,
        "energies": np.zeros((count,), dtype=np.float32),
        "radii_pixels": radii_pixels.astype(np.float32, copy=False),
        "radii_microns": radii_microns.astype(np.float32, copy=False),
        "radii": radii_microns.astype(np.float32, copy=False),
        "count": count,
    }


def _mat_edges_to_python(mat_edges: dict[str, Any]) -> dict[str, Any]:
    """Convert matlab_parser edge dict to a pipeline-compatible edge checkpoint."""
    connections = np.asarray(mat_edges.get("connections", np.array([])), dtype=np.int32)
    if connections.size == 0:
        connections = np.zeros((0, 2), dtype=np.int32)
    elif connections.ndim == 1:
        connections = connections.reshape(1, -1)

    traces_raw = mat_edges.get("traces", [])
    traces: list[np.ndarray] = []
    for trace in traces_raw:
        arr = np.asarray(trace, dtype=np.float32)
        if arr.size == 0:
            continue
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        traces.append(arr.astype(np.float32, copy=False))

    energies = [np.zeros((len(trace),), dtype=np.float32) for trace in traces]
    scales = [np.zeros((len(trace),), dtype=np.int16) for trace in traces]

    return {
        "traces": traces,
        "connections": connections.astype(np.int32, copy=False),
        "energies": np.zeros((len(traces),), dtype=np.float32),
        "energy_traces": energies,
        "scale_traces": scales,
        "origin_indices": connections[:, 0].astype(np.int32, copy=False)
        if len(connections)
        else np.zeros((0,), dtype=np.int32),
        "count": int(max(len(traces), len(connections))),
        "total_length": float(mat_edges.get("total_length", 0.0)),
    }


def import_matlab_batch(
    batch_folder: str | Path,
    checkpoint_dir: str | Path,
    *,
    stages: list | None = None,
) -> dict[str, str]:
    """
    Import MATLAB batch output and write Python-compatible checkpoint files.

    Parameters
    ----------
    batch_folder
        Path to a MATLAB ``batch_*`` folder or a parent directory containing one.
    checkpoint_dir
        Directory where Python checkpoint pickles will be written.
    stages
        Stages to import. Defaults to ``["energy", "vertices", "edges", "network"]``.
    """
    batch_path = Path(batch_folder)

    if not (batch_path / "vectors").exists():
        found = find_batch_folder(batch_path)
        if found is None:
            raise FileNotFoundError(f"No MATLAB batch_* folder found in {batch_folder}")
        batch_path = found

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    vectors_dir = batch_path / "vectors"
    data_dir = batch_path / "data"

    if stages is None:
        stages = ["energy", "vertices", "edges", "network"]

    written: dict[str, str] = {}
    energy_checkpoint: dict[str, Any] | None = None

    if "energy" in stages:
        energy_checkpoint = _load_matlab_energy_checkpoint(batch_path)
        output_path = checkpoint_path / "checkpoint_energy.pkl"
        joblib.dump(energy_checkpoint, output_path)
        written["energy"] = str(output_path)
        logger.info("Wrote MATLAB energy checkpoint: %s", output_path)

    if "vertices" in stages:
        vertex_files = sorted(vectors_dir.glob("curated_vertices_*.mat")) or sorted(
            vectors_dir.glob("vertices_*.mat")
        )
        if vertex_files:
            mat = load_mat_file_safe(vertex_files[-1])
            if mat:
                if energy_checkpoint is None:
                    try:
                        energy_checkpoint = _load_matlab_energy_checkpoint(batch_path)
                    except Exception as exc:  # pragma: no cover - fallback path only
                        logger.warning(
                            "Unable to preload MATLAB energy metadata for vertices: %s", exc
                        )
                raw_vertices = extract_vertices(mat)
                vertices_checkpoint = _mat_vertices_to_python(raw_vertices, energy_checkpoint)
                output_path = checkpoint_path / "checkpoint_vertices.pkl"
                joblib.dump(vertices_checkpoint, output_path)
                written["vertices"] = str(output_path)
                logger.info(
                    "Wrote vertices checkpoint (%d vertices): %s",
                    vertices_checkpoint["count"],
                    output_path,
                )

    if "edges" in stages:
        edge_files = sorted(vectors_dir.glob("curated_edges_*.mat")) or sorted(
            vectors_dir.glob("edges_*.mat")
        )
        if edge_files:
            mat = load_mat_file_safe(edge_files[-1])
            if mat:
                edges_checkpoint = _mat_edges_to_python(extract_edges(mat))
                output_path = checkpoint_path / "checkpoint_edges.pkl"
                joblib.dump(edges_checkpoint, output_path)
                written["edges"] = str(output_path)
                logger.info(
                    "Wrote edges checkpoint (%d edges): %s",
                    edges_checkpoint["count"],
                    output_path,
                )

    if "network" in stages:
        network_files = sorted(vectors_dir.glob("network_*.mat"))
        if network_files:
            mat = load_mat_file_safe(network_files[-1])
            if mat:
                network_checkpoint = extract_network_data(mat)
                output_path = checkpoint_path / "checkpoint_network.pkl"
                joblib.dump(network_checkpoint, output_path)
                written["network"] = str(output_path)
                logger.info("Wrote network checkpoint: %s", output_path)

    if not written:
        logger.warning("No MATLAB data files found in %s", batch_path)
    else:
        logger.info(
            "Imported %d MATLAB stage(s) into %s: %s",
            len(written),
            checkpoint_path,
            list(written.keys()),
        )

    if "energy" in stages and not data_dir.exists():
        logger.warning("Expected MATLAB data directory missing during import: %s", data_dir)

    return written


def load_matlab_batch_params(batch_folder: str | Path) -> dict[str, Any]:
    """Extract Python-compatible processing parameters from a MATLAB batch folder."""
    batch_path = Path(batch_folder)

    if not (batch_path / "settings").exists():
        found = find_batch_folder(batch_path)
        if found is None:
            raise FileNotFoundError(f"No MATLAB batch_* folder found in {batch_folder}")
        batch_path = found

    params: dict[str, Any] = {}

    energy_settings = _load_stage_settings(batch_path, "energy")
    for key in (
        "microns_per_voxel",
        "radius_of_smallest_vessel_in_microns",
        "radius_of_largest_vessel_in_microns",
        "approximating_PSF",
        "excitation_wavelength_in_microns",
        "numerical_aperture",
        "sample_index_of_refraction",
        "scales_per_octave",
        "gaussian_to_ideal_ratio",
        "spherical_to_annular_ratio",
        "max_voxels_per_node_energy",
    ):
        if key in energy_settings:
            normalized = _normalize_setting_value(energy_settings[key])
            if normalized is not None:
                params[key] = normalized

    vertex_settings = _load_stage_settings(batch_path, "vertices")
    for key in ("energy_upper_bound", "max_voxels_per_node", "space_strel_apothem"):
        if key in vertex_settings:
            normalized = _normalize_setting_value(vertex_settings[key])
            if normalized is not None:
                params[key] = normalized
    if "length_dilation_ratio" in vertex_settings:
        normalized = _normalize_setting_value(vertex_settings["length_dilation_ratio"])
        if normalized is not None:
            params["length_dilation_ratio"] = normalized

    edge_settings = _load_stage_settings(batch_path, "edges")
    for key in (
        "number_of_edges_per_vertex",
        "space_strel_apothem_edges",
        "max_edge_length_per_origin_radius",
        "sigma_edge_smoothing",
    ):
        if key in edge_settings:
            normalized = _normalize_setting_value(edge_settings[key])
            if normalized is not None:
                params[key] = normalized
    if "length_dilation_ratio_vertices" in edge_settings:
        normalized = _normalize_setting_value(edge_settings["length_dilation_ratio_vertices"])
        if normalized is not None:
            params["sigma_per_influence_vertices"] = normalized
    if "length_dilation_ratio_edges" in edge_settings:
        normalized = _normalize_setting_value(edge_settings["length_dilation_ratio_edges"])
        if normalized is not None:
            params["sigma_per_influence_edges"] = normalized

    network_settings = _load_stage_settings(batch_path, "network")
    for key in ("sigma_strand_smoothing", "is_combining_strands"):
        if key in network_settings:
            normalized = _normalize_setting_value(network_settings[key])
            if normalized is not None:
                params[key] = normalized

    return params
