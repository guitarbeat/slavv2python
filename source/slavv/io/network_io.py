"""Network file I/O — load and save vascular network data.

Supports: MATLAB .mat, CASX XML, VMV text, CSV, JSON.
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)


@dataclass
class Network:
    """Container for basic network data."""
    vertices: np.ndarray
    edges: np.ndarray
    radii: Optional[np.ndarray] = None


# Backwards-compatible alias
MatNetwork = Network


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_network_from_mat(path: Union[str, Path]) -> Network:
    """Load network data stored in a MATLAB ``.mat`` file."""
    matlab_data_dict: Dict[str, Any] = loadmat(
        Path(path), squeeze_me=True, struct_as_record=False
    )
    v_struct = matlab_data_dict.get("vertices")
    if hasattr(v_struct, "positions"):
        vertices = np.asarray(getattr(v_struct, "positions", []), dtype=float)
        radii = np.asarray(
            getattr(v_struct, "radii_microns", getattr(v_struct, "radii", [])),
            dtype=float,
        )
    else:
        vertices = np.asarray(v_struct if v_struct is not None else [], dtype=float)
        radii = np.asarray(matlab_data_dict.get("radii", []), dtype=float)

    e_struct = matlab_data_dict.get("edges")
    if hasattr(e_struct, "connections"):
        edges = np.atleast_2d(np.asarray(e_struct.connections, dtype=int))
    else:
        edges = np.atleast_2d(
            np.asarray(e_struct if e_struct is not None else [], dtype=int)
        )

    return Network(vertices=vertices, edges=edges, radii=radii if radii.size else None)


def load_network_from_casx(path: Union[str, Path]) -> Network:
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


def load_network_from_vmv(path: Union[str, Path]) -> Network:
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


def load_network_from_csv(path: Union[str, Path]) -> Network:
    """Load network data from paired CSV files."""
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
    edges = np.atleast_2d(e_df[["start_vertex", "end_vertex"]].to_numpy(int))
    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network_from_json(path: Union[str, Path]) -> Network:
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
    edges = np.atleast_2d(np.asarray(e_data.get("connections", []), dtype=int))
    return Network(vertices=vertices, edges=edges, radii=radii)


# ──────────────────────────────────────────────────────────────────────────────
# Savers
# ──────────────────────────────────────────────────────────────────────────────

def save_network_to_csv(
    network: Network, base_path: Union[str, Path]
) -> Tuple[Path, Path]:
    """Save network data to paired CSV files."""
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


def save_network_to_json(network: Network, path: Union[str, Path]) -> Path:
    """Save network data to a JSON file."""
    data: Dict[str, Any] = {
        "vertices": {"positions": np.asarray(network.vertices, dtype=float).tolist()},
        "edges": {"connections": np.atleast_2d(np.asarray(network.edges, dtype=int)).tolist()},
    }
    if network.radii is not None:
        data["vertices"]["radii_microns"] = np.asarray(network.radii, dtype=float).tolist()
    json_path = Path(path)
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path
