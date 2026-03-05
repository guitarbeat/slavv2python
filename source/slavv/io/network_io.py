"""Network file I/O — load and save vascular network data.

Supports: MATLAB .mat, CASX XML, VMV text, CSV, JSON.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
from scipy.io import loadmat

logger = logging.getLogger(__name__)


@dataclass
class Network:
    """Container for basic network data."""

    vertices: np.ndarray
    edges: np.ndarray
    radii: np.ndarray | None = None


# Backwards-compatible alias
MatNetwork = Network


# ──────────────────────────────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────────────────────────────


def load_network_from_mat(path: Union[str, Path]) -> Network:
    """Load network data stored in a MATLAB ``.mat`` file."""
    matlab_data_dict: dict[str, Any] = loadmat(Path(path), squeeze_me=True, struct_as_record=False)
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
        edges = np.atleast_2d(np.asarray(e_struct if e_struct is not None else [], dtype=int))

    return Network(vertices=vertices, edges=edges, radii=radii if radii.size else None)


def _convert_edges_to_strands(edges: np.ndarray) -> list[list[int]]:
    """Helper method to construct strands from an edge list.

    A VMV strand is essentially an array of connected nodes in sequence.
    This performs a simple connected components-like traversal, although
    a robust graph might have branches. For basic VMV writing without explicit
    strands, we'll treat each edge as a short strand, or trace continuous paths
    with degree <= 2. To avoid full NetworkX dependency here, we trace simple paths.
    """
    import networkx as nx

    g = nx.Graph()
    g.add_edges_from(edges)

    # Very simple strand logic: each edge is a strand if no robust pathing is needed,
    # but let's try to extract paths that don't pass through bifurcations (degree > 2).
    # Since this is a basic converter, we'll extract simply connected components
    # as strands, or just use edges. Let's trace linear segments.

    strands = []
    visited_edges = set()

    for u, v in edges:
        edge = tuple(sorted((u, v)))
        if edge in visited_edges:
            continue

        # Trace forward from v
        strand = [u, v]
        visited_edges.add(edge)

        current = v
        while g.degree(current) == 2:
            neighbors = list(g.neighbors(current))
            next_node = neighbors[0] if neighbors[1] == strand[-2] else neighbors[1]
            next_edge = tuple(sorted((current, next_node)))
            if next_edge in visited_edges:
                break
            strand.append(next_node)
            visited_edges.add(next_edge)
            current = next_node

        # Trace backward from u
        current = u
        while g.degree(current) == 2:
            neighbors = list(g.neighbors(current))
            next_node = neighbors[0] if neighbors[1] == strand[1] else neighbors[1]
            next_edge = tuple(sorted((current, next_node)))
            if next_edge in visited_edges:
                break
            strand.insert(0, next_node)
            visited_edges.add(next_edge)
            current = next_node

        strands.append(strand)

    return strands


def load_network_from_casx(path: Union[str, Path]) -> Network:
    """Load network data from a CASX XML file."""
    root = ET.parse(Path(path)).getroot()
    vert_list: list[list[float]] = []
    radii_list: list[float] = []
    for v in root.findall(".//Vertex"):
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

    vertices = np.asarray(vert_list, dtype=float)
    edges = np.atleast_2d(np.asarray(edge_list, dtype=int))
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
    if name.endswith(("_vertices", "_edges")):
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
    with open(Path(path)) as f:
        data: dict[str, Any] = json.load(f)

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


def save_network_to_csv(network: Network, base_path: Union[str, Path]) -> tuple[Path, Path]:
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
    data: dict[str, Any] = {
        "vertices": {"positions": np.asarray(network.vertices, dtype=float).tolist()},
        "edges": {"connections": np.atleast_2d(np.asarray(network.edges, dtype=int)).tolist()},
    }
    if network.radii is not None:
        data["vertices"]["radii_microns"] = np.asarray(network.radii, dtype=float).tolist()
    json_path = Path(path)
    with open(json_path, "w") as f:
        json.dump(data, f)
    return json_path


def save_network_to_casx(network: Network, path: Union[str, Path]) -> Path:
    """Save network data to a CASX XML file format."""
    casx_path = Path(path)

    root = ET.Element("CasX")
    network_elem = ET.SubElement(root, "Network")

    # Write vertices
    vertices_elem = ET.SubElement(network_elem, "Vertices")
    for i, pt in enumerate(network.vertices):
        y, x, z = pt
        radius = network.radii[i] if network.radii is not None else 0.0
        # Use attributes for Vertex
        v_elem = ET.SubElement(vertices_elem, "Vertex")
        v_elem.set("id", str(i))
        # Important: CASX original uses specific x, y, z mappings.
        # The loader reads y, x, z into x, y, z labels, so we reverse it here.
        v_elem.set("x", str(x))
        v_elem.set("y", str(y))
        v_elem.set("z", str(z))
        v_elem.set("radius", str(radius))

    # Write edges
    edges_elem = ET.SubElement(network_elem, "Edges")
    for i, edge in enumerate(network.edges):
        start, end = edge
        e_elem = ET.SubElement(edges_elem, "Edge")
        e_elem.set("id", str(i))
        e_elem.set("start", str(int(start)))
        e_elem.set("end", str(int(end)))

    tree = ET.ElementTree(root)
    # Python 3.9+ feature for pretty printing if desired, but we'll use base write string formatting
    ET.indent(tree, space="  ", level=0)
    tree.write(casx_path, encoding="UTF-8", xml_declaration=True)

    return casx_path


def save_network_to_vmv(network: Network, path: Union[str, Path]) -> Path:
    """Save network data to a VMV text format file."""
    vmv_path = Path(path)

    strands = _convert_edges_to_strands(network.edges)

    with open(vmv_path, "w") as f:
        f.write("# VMV Format Export\n")

        # Write vertices block
        f.write("[VERTICES]\n")
        # Format: <id> <x> <y> <z> <radius> <extra>
        for i, pt in enumerate(network.vertices):
            y, x, z = pt
            radius = network.radii[i] if network.radii is not None else 0.0
            # VMV expects y, x, z to map back to x, y, z logically, but matching load order:
            f.write(f"{i} {y} {x} {z} {radius} 0.0\n")

        f.write("\n[EDGES]\n")
        # Write strands block / edges block
        for i, strand in enumerate(strands):
            # Sequence: <strand_id> <num_points> <pt1> <pt2> ...
            # Or edge mode: <id> <node1> <node2>
            # Based on the test, it parses '[EDGES]' with `_, start, end = parts[:3]`
            # We'll just write simple edges for now to match the loader syntax.
            if len(strand) == 2:
                f.write(f"{i} {strand[0]} {strand[1]}\n")
            else:
                # If a strand has multiple segments, save as multiple edges to match test expectations
                for j in range(len(strand) - 1):
                    f.write(f"{i}_{j} {strand[j]} {strand[j + 1]}\n")

    return vmv_path


def convert_casx_to_vmv(casx_path: Union[str, Path], vmv_path: Union[str, Path]) -> Path:
    """Convert a CASX file directly to VMV format."""
    network = load_network_from_casx(casx_path)
    return save_network_to_vmv(network, vmv_path)
