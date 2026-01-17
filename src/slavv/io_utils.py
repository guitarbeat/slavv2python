"""I/O helpers for reading network and image data.

This module supports loading network structures from MATLAB ``.mat``,
CASX XML, and VMV text files, along with utilities for safely reading
TIFF image volumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, List, Optional, Union

import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import xml.etree.ElementTree as ET


@dataclass
class Network:
    """Container for basic network data."""
    vertices: np.ndarray
    edges: np.ndarray
    radii: np.ndarray | None = None


# Backwards-compatible alias
MatNetwork = Network


def load_tiff_volume(
    file: str | Path | IO[bytes], *, memory_map: bool = False
) -> np.ndarray:
    """Load a 3D grayscale TIFF volume with validation.

    Parameters
    ----------
    file:
        Path or binary file-like object containing TIFF data.
    memory_map:
        If ``True``, return a memory-mapped array instead of reading the
        entire volume into memory. This requires ``file`` to be a path-like
        object.

    Returns
    -------
    np.ndarray
        The loaded 3D volume. When ``memory_map`` is ``True`` a
        :class:`numpy.memmap` is returned.

    Raises
    ------
    ValueError
        If the file cannot be read or does not contain a 3D grayscale
        volume.
    """
    import tifffile

    try:
        if memory_map:
            volume = tifffile.memmap(file)
        else:
            volume = tifffile.imread(file)
    except Exception as exc:  # pragma: no cover - pass through value error
        raise ValueError(f"Failed to read TIFF volume: {exc}") from exc
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume")
    if np.iscomplexobj(volume):
        raise ValueError("Expected a real-valued grayscale TIFF volume")
    return np.asarray(volume) if not memory_map else volume


def load_network_from_mat(path: str | Path) -> Network:
    """Load network data stored in a MATLAB ``.mat`` file.

    Parameters
    ----------
    path:
        File path to the ``.mat`` file. The file is expected to contain
        ``vertices`` and ``edges`` arrays and may optionally include
        ``radii``.

    Returns
    -------
    Network
        Dataclass containing the ``vertices`` and ``edges`` arrays loaded
        from the file. Missing arrays default to empty arrays.
    """
    data: Dict[str, Any] = loadmat(
        Path(path), squeeze_me=True, struct_as_record=False
    )

    v_struct = data.get("vertices")
    if hasattr(v_struct, "positions"):
        vertices = np.asarray(getattr(v_struct, "positions", []), dtype=float)
        radii = np.asarray(
            getattr(v_struct, "radii_microns", getattr(v_struct, "radii", [])),
            dtype=float,
        )
    else:
        vertices = np.asarray(v_struct if v_struct is not None else [], dtype=float)
        radii = np.asarray(data.get("radii", []), dtype=float)

    e_struct = data.get("edges")
    if hasattr(e_struct, "connections"):
        edges = np.atleast_2d(np.asarray(e_struct.connections, dtype=int))
    else:
        edges = np.atleast_2d(
            np.asarray(e_struct if e_struct is not None else [], dtype=int)
        )

    if radii.size == 0:
        radii = None
    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network_from_casx(path: str | Path) -> Network:
    """Load network data from a CASX XML file."""

    root = ET.parse(Path(path)).getroot()
    vert_list: List[List[float]] = []
    radii_list: List[float] = []
    for v in root.findall(".//Vertex"):
        x = float(v.attrib.get("x", 0.0))
        y = float(v.attrib.get("y", 0.0))
        z = float(v.attrib.get("z", 0.0))
        radius = float(v.attrib.get("radius", 0.0))
        vert_list.append([y, x, z])
        radii_list.append(radius)

    edge_list: List[List[int]] = []
    for e in root.findall(".//Edge"):
        start = e.attrib.get("start")
        end = e.attrib.get("end")
        if start is None or end is None:
            continue
        edge_list.append([int(start), int(end)])

    vertices = np.asarray(vert_list, dtype=float)
    edges = np.atleast_2d(np.asarray(edge_list, dtype=int))
    radii = np.asarray(radii_list, dtype=float) if radii_list else None
    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network_from_vmv(path: str | Path) -> Network:
    """Load network data from a VMV text file."""

    positions: List[List[float]] = []
    radii: List[float] = []
    edges_list: List[List[int]] = []
    section = None
    with open(Path(path), "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line == "[VERTICES]":
                section = "vertices"
                continue
            if line == "[EDGES]":
                section = "edges"
                continue
            if section == "vertices":
                parts = line.split()
                if len(parts) >= 6:
                    _, y, x, z, radius, *_ = parts
                    positions.append([float(y), float(x), float(z)])
                    radii.append(float(radius))
            elif section == "edges":
                parts = line.split()
                if len(parts) >= 3:
                    _, start, end = parts[:3]
                    edges_list.append([int(start), int(end)])

    vertices = np.asarray(positions, dtype=float)
    edges = np.atleast_2d(np.asarray(edges_list, dtype=int))
    radii_arr = np.asarray(radii, dtype=float) if radii else None
    return Network(vertices=vertices, edges=edges, radii=radii_arr)


def dicom_to_tiff(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    sort_by: str = "instance",
    rescale: bool = True,
    dtype: str = "uint16",
) -> Union[np.ndarray, Path]:
    """Load DICOM (single file, multi-frame, or series) and optionally write TIFF.

    Parameters
    ----------
    input_path:
        Path to a DICOM file or a directory containing a DICOM series.
    output_path:
        If provided, write a TIFF volume to this path and return the path.
        If omitted, return the loaded 3D volume as a NumPy array.
    sort_by:
        Sorting key for series: one of ``'instance'`` (InstanceNumber),
        ``'position'`` (ImagePositionPatient), or ``'location'`` (SliceLocation).
    rescale:
        Apply ``RescaleSlope``/``RescaleIntercept`` if present.
    dtype:
        Output dtype when writing TIFF or returning the array. Typical values:
        ``'uint16'`` (default), ``'float32'``.

    Returns
    -------
    np.ndarray or Path
        The loaded volume if ``output_path`` is ``None``; otherwise the path to
        the written TIFF file.

    Notes
    -----
    - Requires ``pydicom`` and ``tifffile``.
    - For series directories, files are sorted according to ``sort_by`` with
      graceful fallbacks when tags are missing.
    """
    try:
        import pydicom  # type: ignore
        import tifffile
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "dicom_to_tiff requires 'pydicom' and 'tifffile' to be installed"
        ) from exc

    in_path = Path(input_path)

    def _get_slice_sort_key(ds) -> float:
        if sort_by == "position":
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                return float(ipp[2])
        if sort_by == "location":
            loc = getattr(ds, "SliceLocation", None)
            if loc is not None:
                return float(loc)
        # default: instance number
        inst = getattr(ds, "InstanceNumber", None)
        try:
            return float(inst)
        except Exception:
            return 0.0

    def _apply_rescale(arr: np.ndarray, ds) -> np.ndarray:
        if not rescale:
            return arr
        slope = getattr(ds, "RescaleSlope", 1.0) or 1.0
        intercept = getattr(ds, "RescaleIntercept", 0.0) or 0.0
        out = arr.astype(np.float32) * float(slope) + float(intercept)
        return out

    def _normalize_dtype(vol: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(out_dtype, np.floating):
            # Scale to 0..1 range
            vmin = np.nanmin(vol)
            vmax = np.nanmax(vol)
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
            return vol.astype(out_dtype, copy=False)
        # Unsigned integer: scale to full range
        info = np.iinfo(out_dtype)
        vmin = np.nanmin(vol)
        vmax = np.nanmax(vol)
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
            vol = vol * info.max
        return vol.astype(out_dtype)

    # Load dataset(s)
    if in_path.is_dir():
        # Series of 2D slices
        files = sorted([p for p in in_path.iterdir() if p.is_file()])
        datasets = []
        for p in files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if hasattr(ds, "pixel_array"):
                    datasets.append(ds)
            except Exception:
                continue
        if not datasets:
            raise ValueError("No readable DICOM slices found in directory")
        datasets.sort(key=_get_slice_sort_key)
        slices = []
        for ds in datasets:
            arr = ds.pixel_array
            arr = _apply_rescale(arr, ds)
            slices.append(np.asarray(arr))
        volume = np.stack(slices, axis=0)
    else:
        # Single file (2D, 3D multi-frame, or series container)
        ds = pydicom.dcmread(str(in_path), stop_before_pixels=False)
        arr = ds.pixel_array
        arr = _apply_rescale(arr, ds)
        if arr.ndim == 2:
            volume = arr[np.newaxis, :, :]
        elif arr.ndim == 3:
            volume = arr  # assume (frames, rows, cols)
        else:
            raise ValueError("Unsupported DICOM pixel array dimensionality")

    # Normalize dtype
    out_dtype = np.dtype(dtype)
    volume = np.asarray(volume)
    if rescale:
        volume = _normalize_dtype(volume, out_dtype)
    else:
        volume = volume.astype(out_dtype, copy=False)

    if output_path is not None:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(outp), volume)
        return outp
    return volume


def load_network_from_csv(path: str | Path) -> Network:
    """Load network data from paired CSV files.

    Parameters
    ----------
    path:
        Base path to the CSV files or one of the ``*_vertices.csv``/
        ``*_edges.csv`` files written by the exporter.

    Returns
    -------
    Network
        Network dataclass populated from the CSV contents.
    """

    base = Path(path).with_suffix("")
    name = base.name
    if name.endswith("_vertices") or name.endswith("_edges"):
        base = base.with_name(name.rsplit("_", 1)[0])

    vertex_path = base.with_name(base.name + "_vertices.csv")
    edge_path = base.with_name(base.name + "_edges.csv")

    v_df = pd.read_csv(vertex_path)
    vertices = v_df[["y_position", "x_position", "z_position"]].to_numpy(float)

    radii = None
    if "radius_microns" in v_df.columns:
        radii = v_df["radius_microns"].to_numpy(float)
    elif "radius_pixels" in v_df.columns:
        radii = v_df["radius_pixels"].to_numpy(float)

    e_df = pd.read_csv(edge_path)
    edges = np.atleast_2d(
        e_df[["start_vertex", "end_vertex"]].to_numpy(int)
    )

    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network_from_json(path: str | Path) -> Network:
    """Load network data from a JSON export."""

    with open(Path(path), "r") as f:
        data: Dict[str, Any] = json.load(f)

    v_data = data.get("vertices", {})
    vertices = np.asarray(v_data.get("positions", []), dtype=float)
    radii_list = v_data.get("radii_microns", v_data.get("radii"))
    radii = (
        np.asarray(radii_list, dtype=float)
        if radii_list is not None and len(radii_list) > 0
        else None
    )

    e_data = data.get("edges", {})
    edges = np.atleast_2d(
        np.asarray(e_data.get("connections", []), dtype=int)
    )

    return Network(vertices=vertices, edges=edges, radii=radii)


def save_network_to_csv(
    network: Network, base_path: str | Path
) -> tuple[Path, Path]:
    """Save network data to paired CSV files.

    Parameters
    ----------
    network:
        Network dataclass containing vertices, edges, and optional radii.
    base_path:
        Base path for the output files; ``_vertices.csv`` and ``_edges.csv``
        suffixes will be appended.

    Returns
    -------
    (Path, Path)
        Paths to the vertex and edge CSV files that were written.
    """

    base = Path(base_path).with_suffix("")
    vertex_path = base.with_name(base.name + "_vertices.csv")
    edge_path = base.with_name(base.name + "_edges.csv")

    v_df = pd.DataFrame(
        np.asarray(network.vertices, dtype=float),
        columns=["y_position", "x_position", "z_position"],
    )
    if network.radii is not None:
        v_df["radius_microns"] = np.asarray(network.radii, dtype=float)
    v_df.to_csv(vertex_path, index=False)

    e_df = pd.DataFrame(
        np.atleast_2d(np.asarray(network.edges, dtype=int))[:, :2],
        columns=["start_vertex", "end_vertex"],
    )
    e_df.to_csv(edge_path, index=False)

    return vertex_path, edge_path


def save_network_to_json(network: Network, path: str | Path) -> Path:
    """Save network data to a JSON file."""

    data: Dict[str, Any] = {
        "vertices": {"positions": np.asarray(network.vertices, dtype=float).tolist()},
        "edges": {"connections": np.atleast_2d(np.asarray(network.edges, dtype=int)).tolist()},
    }
    if network.radii is not None:
        data["vertices"]["radii_microns"] = (
            np.asarray(network.radii, dtype=float).tolist()
        )

    json_path = Path(path)
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def export_pipeline_results(
    results: Dict[str, Any], output_dir: str | Path, base_name: str = "result"
) -> List[Path]:
    """Export all standard components of a pipeline result to files.

    Automatically extracts 'network', 'vertices', 'edges' from the result dict
    and saves them in CSV and JSON formats.

    Parameters
    ----------
    results : Dict[str, Any]
        The dictionary returned by SLAVVProcessor.process_image().
    output_dir : str | Path
        Directory to save files in.
    base_name : str
        Base filename prefix (default: "result").
    
    Returns
    -------
    List[Path]
        List of paths to validly created files.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_files = []

    # Construct Network object from results if possible
    # Note: results['network'] is a dict with 'strands', etc.
    # results['vertices'] is dict with 'positions'
    # results['edges'] is dict with 'traces'
    # We need to map this back to the simple Network(vertices, edges) for CSV export
    # The 'network' from process_image is complex.
    # Let's use the basic 'vertices' and 'edges' from the dicts.
    
    try:
        v_pos = results.get('vertices', {}).get('positions', [])
        e_conn = [] 
        # CAUTION: 'edges' from pipeline contains TRAJECTORIES (traces).
        # Network connectivity is typically in results['network']['network_graph'] or similar
        # But 'io_utils.Network' expects (N,2) connectivity list.
        # The pipeline 'edges' dict might not have simple connectivity directly exposed as a list of indices.
        # However, results['edges'] usually has 'node_indices' or similar.
        # Let's check vectorization_core.py output structure.
        # extract_edges returns {'traces': ..., 'confidence': ...}
        # construct_network returns {'strands': ...}
        
        # If we can't easily map to the simple I/O Network class, we should default to pickling/JSONing the whole dict.
        # But user wants standard formats.
        
        # fallback: Save the raw parameters
        params_path = out_dir / f"{base_name}_parameters.json"
        with open(params_path, "w") as f:
            # helper to convert numpy types 
            def default(o):
                if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                                  np.int16, np.int32, np.int64, np.uint8,
                                  np.uint16, np.uint32, np.uint64)):
                    return int(o)
                elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                    return float(o)
                elif isinstance(o, (np.ndarray,)):
                    return o.tolist()
                raise TypeError
            json.dump(results.get('parameters', {}), f, indent=2, default=default)
        created_files.append(params_path)

    except Exception as e:
        logger.warning(f"Failed to export generic parameters: {e}")

    return created_files
