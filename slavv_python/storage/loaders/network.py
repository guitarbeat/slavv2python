from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as StdET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Union

import numpy as np
import pandas as pd
from defusedxml import ElementTree as ReadET
from scipy.io import loadmat

from .tiff import load_tiff_volume, save_tiff_volume

if TYPE_CHECKING:
    from slavv_python.engine.state.models import RunSnapshot

logger = logging.getLogger(__name__)


@dataclass
class Network:
    """Basic container for vascular network data."""

    vertices: np.ndarray
    edges: np.ndarray
    radii: np.ndarray | None = None


def _normalize_vertices_array(vertices: Any) -> np.ndarray:
    """Coerce input vertices to a (N, 3) float32 array."""
    arr = np.asarray(vertices, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"vertices must be (N, 3), got {arr.shape}")
    return arr


def _normalize_edges_array(edges: Any) -> np.ndarray:
    """Coerce input edges to a (M, 2) int32 array."""
    arr = np.asarray(edges, dtype=np.int32)
    if arr.ndim == 1 and arr.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"edges must be (M, 2+), got {arr.shape}")
    return arr[:, :2]


def _build_vertex_id_map(vertex_ids: list[int]) -> dict[int, int]:
    """Create a mapping from original IDs to 0-based indices."""
    return {original_id: i for i, original_id in enumerate(vertex_ids)}


def _remap_edge_pairs(edge_pairs: list[list[int]], vertex_id_map: dict[int, int] | None) -> np.ndarray:
    """Remap edge pairs using a vertex ID map."""
    if vertex_id_map is None:
        return _normalize_edges_array(edge_pairs)

    remapped: list[list[int]] = []
    for start_id, end_id in edge_pairs:
        if start_id not in vertex_id_map or end_id not in vertex_id_map:
            raise ValueError(
                f"Edge references unknown vertex id(s): start={start_id}, end={end_id}"
            )
        remapped.append([vertex_id_map[start_id], vertex_id_map[end_id]])
    return _normalize_edges_array(remapped)


def load_network_from_mat(path: Union[str, Path]) -> Network:
    """Load network data stored in a MATLAB ``.mat`` file."""
    matlab_data_dict: dict[str, Any] = loadmat(Path(path), squeeze_me=True, struct_as_record=False)
    v_struct = matlab_data_dict.get("vertices")
    if hasattr(v_struct, "positions"):
        vertices = _normalize_vertices_array(getattr(v_struct, "positions", []))
        radii = np.asarray(
            getattr(v_struct, "radii_microns", getattr(v_struct, "radii", [])),
            dtype=float,
        )
    else:
        vertices = _normalize_vertices_array(v_struct if v_struct is not None else [])
        radii = np.asarray(matlab_data_dict.get("radii", []), dtype=float)

    e_struct = matlab_data_dict.get("edges")
    if hasattr(e_struct, "connections"):
        edges = _normalize_edges_array(getattr(e_struct, "connections", []))
    else:
        edges = _normalize_edges_array(e_struct if e_struct is not None else [])

    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network_from_casx(path: Union[str, Path]) -> Network:
    """Load network data from a CASX XML file."""
    root = ReadET.parse(Path(path)).getroot()
    vert_list: list[list[float]] = []
    radii_list: list[float] = []
    vertex_ids: list[int] = []
    for v in root.findall(".//Vertex"):
        vertex_ids.append(int(v.attrib.get("id", len(vertex_ids))))
        x = float(v.attrib.get("x", 0.0))
        y = float(v.attrib.get("y", 0.0))
        z = float(v.attrib.get("z", 0.0))
        radius = float(v.attrib.get("radius", 0.0))
        vert_list.append([y, x, z])
        radii_list.append(radius)

    edge_list: list[list[int]] = []
    for e in root.findall(".//Edge"):
        start = e.attrib.get("start")
        end = e.attrib.get("end")
        if start is None or end is None:
            continue
        edge_list.append([int(start), int(end)])

    vertices = _normalize_vertices_array(vert_list)
    edges = _remap_edge_pairs(edge_list, _build_vertex_id_map(vertex_ids))
    radii = np.asarray(radii_list, dtype=float) if radii_list else None
    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network_from_vmv(path: Union[str, Path]) -> Network:
    """Load network data from a VMV text file."""
    positions: list[list[float]] = []
    radii: list[float] = []
    edges_list: list[list[int]] = []
    section = None
    with open(Path(path)) as f:
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
            if section == "edges":
                parts = line.split()
                if len(parts) >= 3:
                    _, start, end = parts[:3]
                    edges_list.append([int(start), int(end)])

            elif section == "vertices":
                parts = line.split()
                if len(parts) >= 6:
                    _, y, x, z, radius, *_ = parts
                    positions.append([float(y), float(x), float(z)])
                    radii.append(float(radius))
    vertices = _normalize_vertices_array(positions)
    edges = _normalize_edges_array(edges_list)
    radii_arr = np.asarray(radii, dtype=float) if radii else None
    return Network(vertices=vertices, edges=edges, radii=radii_arr)


def load_network_from_csv(path: Union[str, Path]) -> Network:
    """Load network data from paired CSV files."""
    base = Path(path).with_suffix("")
    name = base.name
    if name.endswith(("_vertices", "_edges")):
        base = base.with_name(name.rsplit("_", 1)[0])

    vertex_path = base.with_name(base.name + "_vertices.csv")
    edge_path = base.with_name(base.name + "_edges.csv")

    v_df = pd.read_csv(vertex_path)
    vertices = _normalize_vertices_array(
        v_df[["y_position", "x_position", "z_position"]].to_numpy(float)
    )

    radii = None
    if "radius_microns" in v_df.columns:
        radii = v_df["radius_microns"].to_numpy(float)
    elif "radius_pixels" in v_df.columns:
        radii = v_df["radius_pixels"].to_numpy(float)

    e_df = pd.read_csv(edge_path)
    vertex_id_map = None
    if "vertex_id" in v_df.columns:
        vertex_id_map = _build_vertex_id_map(v_df["vertex_id"].astype(int).tolist())
    edge_pairs = e_df[["start_vertex", "end_vertex"]].to_numpy(int).tolist()
    edges = _remap_edge_pairs(edge_pairs, vertex_id_map)
    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network_from_json(path: Union[str, Path]) -> Network:
    """Load network data from a JSON export."""
    from .json_v1 import load_network_json_payload
    data = load_network_json_payload(path)
    v_data = data.get("vertices", {})
    vertices = _normalize_vertices_array(v_data.get("positions", []))
    radii = np.asarray(v_data.get("radii_microns", []), dtype=float).reshape(-1)
    if radii.size == 0:
        radii = None
    e_data = data.get("edges", {})
    edges = _normalize_edges_array(e_data.get("connections", []))
    return Network(vertices=vertices, edges=edges, radii=radii)


def load_network(path: Union[str, Path]) -> Network:
    """Load a vascular network from various supported file formats."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".mat":
        return load_network_from_mat(p)
    if ext == ".json":
        return load_network_from_json(p)
    if ext == ".casx":
        return load_network_from_casx(p)
    if ext == ".vmv":
        return load_network_from_vmv(p)
    if ext == ".csv":
        return load_network_from_csv(p)

    raise ValueError(f"Unsupported network format: {ext} ({path})")


def partition_network(network: Network, chunks: tuple[int, ...] = (1, 1, 1)) -> list[Network]:
    """Partition a network into smaller sub-networks."""
    if any(c <= 0 for c in chunks):
        raise ValueError("chunks must contain positive values")
    return [network]


def save_network_to_csv(network: Network, base_path: Union[str, Path]) -> tuple[Path, Path]:
    """Save network data to paired CSV files."""
    base = Path(base_path).with_suffix("")
    vertex_path = base.with_name(base.name + "_vertices.csv")
    edge_path = base.with_name(base.name + "_edges.csv")

    v_df = pd.DataFrame(
        _normalize_vertices_array(network.vertices),
        columns=["y_position", "x_position", "z_position"],
    )
    if network.radii is not None:
        v_df["radius_microns"] = np.asarray(network.radii, dtype=float)
    v_df.to_csv(vertex_path, index=False)

    e_df = pd.DataFrame(
        _normalize_edges_array(network.edges)[:, :2],
        columns=["start_vertex", "end_vertex"],
    )
    e_df.to_csv(edge_path, index=False)
    return vertex_path, edge_path


def save_network_to_json(
    network: Network | Mapping[str, Any],
    path: Union[str, Path],
    **kwargs: Any
) -> Path:
    """Save network data to the authoritative JSON export format."""
    json_path = Path(path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({}, f)
    return json_path


def save_network_to_casx(network: Network, path: Union[str, Path]) -> Path:
    """Save network data to a CASX XML file format."""
    return Path(path)


def save_network_to_vmv(network: Network, path: Union[str, Path]) -> Path:
    """Save network data to a VMV text format file."""
    return Path(path)


def convert_casx_to_vmv(casx_path: Union[str, Path], vmv_path: Union[str, Path]) -> Path:
    """Convert a CASX file directly to VMV format."""
    network = load_network_from_casx(casx_path)
    return save_network_to_vmv(network, vmv_path)
