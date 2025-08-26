"""I/O helpers for reading network and image data.

This module supports loading network structures from MATLAB ``.mat``,
CASX XML, and VMV text files, along with utilities for safely reading
TIFF image volumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, IO, List

import numpy as np
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


def load_tiff_volume(file: str | Path | IO[bytes]) -> np.ndarray:
    """Load a 3D grayscale TIFF volume with validation.

    Parameters
    ----------
    file:
        Path or binary file-like object containing TIFF data.

    Returns
    -------
    np.ndarray
        The loaded 3D volume.

    Raises
    ------
    ValueError
        If the file cannot be read or does not contain a 3D grayscale
        volume.
    """
    import tifffile

    try:
        volume = tifffile.imread(file)
    except Exception as exc:  # pragma: no cover - pass through value error
        raise ValueError(f"Failed to read TIFF volume: {exc}") from exc
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume")
    if np.iscomplexobj(volume):
        raise ValueError("Expected a real-valued grayscale TIFF volume")
    return np.asarray(volume)


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
    data: Dict[str, Any] = loadmat(Path(path), squeeze_me=True)
    vertices = np.asarray(data.get("vertices", []), dtype=float)
    edges = np.atleast_2d(np.asarray(data.get("edges", []), dtype=int))
    radii = np.asarray(data.get("radii", []), dtype=float)
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
