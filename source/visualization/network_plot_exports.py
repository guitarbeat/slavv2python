"""Export helpers for SLAVV network visualizations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..utils import calculate_path_length

logger = logging.getLogger(__name__)


def export_csv(
    vertices: dict[str, Any],
    edges: dict[str, Any],
    network: dict[str, Any],
    parameters: dict[str, Any],
    output_path: str,
) -> str:
    """Export data as CSV files."""
    base_path = Path(output_path).with_suffix("")
    n_vertices = len(vertices["positions"])

    def _ensure_len(arr: Any) -> np.ndarray:
        if arr is None or len(arr) != n_vertices:
            return np.full(n_vertices, np.nan)
        return np.asarray(arr)

    vertex_df = pd.DataFrame(
        {
            "vertex_id": range(n_vertices),
            "y_position": vertices["positions"][:, 0],
            "x_position": vertices["positions"][:, 1],
            "z_position": vertices["positions"][:, 2],
            "energy": _ensure_len(vertices.get("energies")),
            "radius_microns": _ensure_len(vertices.get("radii_microns", vertices.get("radii", []))),
            "radius_pixels": _ensure_len(vertices.get("radii_pixels", vertices.get("radii", []))),
            "scale": _ensure_len(vertices.get("scales")),
        }
    )
    vertex_path = f"{base_path}_vertices.csv"
    vertex_df.to_csv(vertex_path, index=False)

    edge_data = []
    for i, (trace, connection) in enumerate(zip(edges["traces"], edges["connections"])):
        start_vertex = connection[0] if len(connection) > 0 else None
        end_vertex = connection[1] if len(connection) > 1 else None
        trace_arr = np.array(trace)
        length = calculate_path_length(
            trace_arr * parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])
        )

        edge_data.append(
            {
                "edge_id": i,
                "start_vertex": start_vertex,
                "end_vertex": end_vertex,
                "length": length,
                "n_points": len(trace),
            }
        )

    edge_df = pd.DataFrame(edge_data)
    edge_path = f"{base_path}_edges.csv"
    edge_df.to_csv(edge_path, index=False)

    logger.info(f"CSV export complete: {vertex_path}, {edge_path}")
    return vertex_path


def _convert_numpy(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return _convert_numpy(obj.tolist())
    if isinstance(obj, np.generic):
        return _convert_numpy(obj.item())
    if isinstance(obj, set):
        return [_convert_numpy(item) for item in obj]
    if isinstance(obj, tuple):
        return [_convert_numpy(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): _convert_numpy(value) for key, value in obj.items()}
    return [_convert_numpy(item) for item in obj] if isinstance(obj, list) else obj


def export_json(processing_results: dict[str, Any], output_path: str) -> str:
    """Export complete results as JSON."""
    from source.io import save_network_to_json

    save_network_to_json(processing_results, output_path)
    logger.info(f"JSON export complete: {output_path}")
    return output_path


def export_vmv(
    vertices: dict[str, Any],
    edges: dict[str, Any],
    network: dict[str, Any],
    parameters: dict[str, Any],
    output_path: str,
) -> str:
    """Export in VMV format."""
    microns_per_voxel = parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])
    vertex_positions = vertices["positions"]
    vertex_radii = vertices.get("radii_microns", vertices.get("radii", []))
    if len(vertex_radii) == 0 and len(vertex_positions) > 0:
        vertex_radii = np.ones(len(vertex_positions))

    connection_to_edge_idx = {}
    for idx, conn in enumerate(edges["connections"]):
        if conn is not None and len(conn) >= 2:
            u, v = conn[0], conn[1]
            if u is not None and v is not None:
                connection_to_edge_idx[(int(u), int(v))] = idx
                connection_to_edge_idx[(int(v), int(u))] = idx

    vmv_points = []
    point_to_idx = {}

    def get_or_add_point(pos_um, radius_um):
        key = tuple(np.round(pos_um, 5))
        if key in point_to_idx:
            return point_to_idx[key]
        idx = len(vmv_points) + 1
        vmv_points.append([*list(pos_um), radius_um])
        point_to_idx[key] = idx
        return idx

    vmv_strands = []
    for strand in network["strands"]:
        strand_point_indices = []
        if len(strand) < 2:
            continue

        is_coord = False
        if isinstance(strand, np.ndarray):
            if strand.ndim == 2 and strand.shape[1] >= 3:
                is_coord = True
            elif strand.ndim == 1 and strand.dtype.kind == "f" and len(strand) >= 3:
                is_coord = True
                strand = strand.reshape(1, -1)

        if is_coord:
            for k in range(len(strand)):
                pt = strand[k]
                if not isinstance(pt, (np.ndarray, list, tuple)):
                    continue
                pos_vox = pt[:3]
                radius = pt[3] if len(pt) > 3 else 1.0
                pos_um = np.array(
                    [
                        pos_vox[1] * microns_per_voxel[1],
                        -pos_vox[0] * microns_per_voxel[0],
                        -pos_vox[2] * microns_per_voxel[2],
                    ]
                )
                strand_point_indices.append(get_or_add_point(pos_um, radius))
        else:
            for i in range(len(strand) - 1):
                u, v = int(strand[i]), int(strand[i + 1])
                edge_idx = connection_to_edge_idx.get((u, v))
                if edge_idx is None:
                    continue
                trace = edges["traces"][edge_idx]
                if trace is None or len(trace) == 0:
                    continue

                trace_arr = np.array(trace)
                pos_u = vertex_positions[u]
                d_start = np.linalg.norm(trace_arr[0] - pos_u)
                d_end = np.linalg.norm(trace_arr[-1] - pos_u)
                if d_end < d_start:
                    trace_arr = trace_arr[::-1]

                r_u = vertex_radii[u] if u < len(vertex_radii) else 1.0
                r_v = vertex_radii[v] if v < len(vertex_radii) else 1.0
                diffs = np.diff(trace_arr, axis=0)
                diffs_phys = diffs * microns_per_voxel
                seg_lens = np.sqrt(np.sum(diffs_phys**2, axis=1))
                cum_lens = np.concatenate(([0], np.cumsum(seg_lens)))
                total_len = cum_lens[-1]
                if total_len > 1e-6:
                    r_interp = r_u + (r_v - r_u) * (cum_lens / total_len)
                else:
                    r_interp = np.full(len(trace_arr), r_u)

                start_k = 0 if i == 0 else 1
                for k in range(start_k, len(trace_arr)):
                    pos_vox = trace_arr[k]
                    pos_um = np.array(
                        [
                            pos_vox[1] * microns_per_voxel[1],
                            -pos_vox[0] * microns_per_voxel[0],
                            -pos_vox[2] * microns_per_voxel[2],
                        ]
                    )
                    strand_point_indices.append(get_or_add_point(pos_um, r_interp[k]))

        if len(strand_point_indices) > 1:
            vmv_strands.append(strand_point_indices)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("$PARAM_BEGIN\n")
        f.write(f"NUM_VERTS\t{len(vmv_points)}\n")
        f.write(f"NUM_STRANDS\t{len(vmv_strands)}\n")
        f.write("NUM_ATTRIB_PER_VERT\t4\n")
        f.write("$PARAM_END\n\n")
        f.write("$VERT_LIST_BEGIN\n")
        for i, pt in enumerate(vmv_points):
            f.write(f"{i + 1}\t{pt[0]:.6f}\t{pt[1]:.6f}\t{pt[2]:.6f}\t{pt[3]:.6f}\n")
        f.write("$VERT_LIST_END\n\n")
        f.write("$STRANDS_LIST_BEGIN\n")
        for i, s in enumerate(vmv_strands):
            f.write(f"{i + 1}\t" + "\t".join(map(str, s)) + "\n")
        f.write("$STRANDS_LIST_END")

    logger.info(f"VMV export complete: {output_path}")
    return output_path


def export_casx(
    vertices: dict[str, Any],
    edges: dict[str, Any],
    network: dict[str, Any],
    parameters: dict[str, Any],
    output_path: str,
) -> str:
    """Export in CASX format."""
    positions = np.asarray(vertices.get("positions", []), dtype=float)
    radii_array = _casx_radii_array(vertices, len(positions))
    energies = vertices.get("energies")
    scales = vertices.get("scales")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(
            _build_casx_xml(parameters, positions, radii_array, energies, scales, edges, network)
        )

    logger.info(f"CASX export complete: {output_path}")
    return output_path


def _casx_radii_array(vertices: dict[str, Any], n_positions: int) -> np.ndarray:
    radii = vertices.get("radii_microns", vertices.get("radii", []))
    radii_array = np.asarray(radii, dtype=float).reshape(-1)
    if len(radii_array) >= n_positions:
        return radii_array
    padded = np.zeros((n_positions,), dtype=float)
    padded[: len(radii_array)] = radii_array
    return padded


def _casx_parameter_lines(parameters: dict[str, Any]) -> list[str]:
    lines = ["  <Parameters>"]
    for key, value in parameters.items():
        value_str = (
            " ".join(map(str, value))
            if isinstance(value, (list, tuple, np.ndarray))
            else str(value)
        )
        lines.append(f'    <Parameter name="{key}" value="{value_str}"/>')
    if "microns_per_voxel" not in parameters:
        lines.append('    <Parameter name="microns_per_voxel" value="1.0 1.0 1.0"/>')
    lines.append("  </Parameters>")
    return lines


def _casx_vertex_line(
    index: int, pos: np.ndarray, radius: float, energies: Any, scales: Any
) -> str:
    line = (
        f'      <Vertex id="{index}" x="{pos[1]:.3f}" y="{pos[0]:.3f}" '
        f'z="{pos[2]:.3f}" radius="{radius:.3f}"'
    )
    if energies is not None and index < len(energies):
        line += f' energy="{energies[index]:.3f}"'
    if scales is not None and index < len(scales):
        line += f' scale="{scales[index]:.3f}"'
    return line + "/>"


def _casx_vertex_lines(
    positions: np.ndarray, radii_array: np.ndarray, energies: Any, scales: Any
) -> list[str]:
    lines = ["    <Vertices>"]
    for index, pos in enumerate(positions):
        radius = radii_array[index] if index < len(radii_array) else 0.0
        lines.append(_casx_vertex_line(index, pos, radius, energies, scales))
    lines.append("    </Vertices>")
    return lines


def _casx_edge_lines(edges: dict[str, Any]) -> list[str]:
    lines = ["    <Edges>"]
    for index, connection in enumerate(edges["connections"]):
        start_vertex, end_vertex = connection
        if start_vertex is not None and end_vertex is not None:
            lines.append(f'      <Edge id="{index}" start="{start_vertex}" end="{end_vertex}"/>')
    lines.append("    </Edges>")
    return lines


def _casx_strand_lines(network: dict[str, Any]) -> list[str]:
    strand_lines = [
        f'      <Strand id="{index}">' + " ".join(map(str, strand)) + "</Strand>"
        for index, strand in enumerate(network.get("strands", []))
        if len(strand) > 0
    ]
    return ["    <Strands>", *strand_lines, "    </Strands>"]


def _casx_bifurcation_lines(network: dict[str, Any]) -> list[str]:
    bifurcations = network.get("bifurcations", [])
    if len(bifurcations) == 0:
        return []
    bifurcation_lines = [
        f'      <Bifurcation id="{index}" vertex_id="{bifurcation}"/>'
        for index, bifurcation in enumerate(bifurcations)
    ]
    return ["    <Bifurcations>", *bifurcation_lines, "    </Bifurcations>"]


def _build_casx_xml(
    parameters: dict[str, Any],
    positions: np.ndarray,
    radii_array: np.ndarray,
    energies: Any,
    scales: Any,
    edges: dict[str, Any],
    network: dict[str, Any],
) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<CasX>"]
    lines.extend(_casx_parameter_lines(parameters))
    lines.append("  <Network>")
    lines.extend(_casx_vertex_lines(positions, radii_array, energies, scales))
    lines.extend(_casx_edge_lines(edges))
    lines.extend(_casx_strand_lines(network))
    lines.extend(_casx_bifurcation_lines(network))
    lines.extend(("  </Network>", "</CasX>"))
    return "\n".join(lines) + "\n"


def sanitize_for_matlab(data: Any) -> Any:
    """Sanitize data structures for MATLAB export."""
    if data is None:
        return []
    if isinstance(data, dict):
        return {str(k): sanitize_for_matlab(v) for k, v in data.items()}
    if isinstance(data, list):
        return [sanitize_for_matlab(v) for v in data]
    if isinstance(data, tuple):
        return tuple(sanitize_for_matlab(v) for v in data)
    return list(data) if isinstance(data, set) else data


def export_mat(
    vertices: dict[str, Any],
    edges: dict[str, Any],
    network: dict[str, Any],
    parameters: dict[str, Any],
    output_path: str,
) -> str:
    """Export data to MATLAB .mat format."""
    try:
        from scipy.io import savemat
    except ImportError as e:
        raise ImportError("scipy is required for MAT export. Please install scipy.") from e

    data = {
        "vertices": {
            "positions": np.asarray(vertices.get("positions", [])),
            "energies": np.asarray(vertices.get("energies", [])),
            "radii_microns": np.asarray(vertices.get("radii_microns", vertices.get("radii", []))),
            "radii_pixels": np.asarray(vertices.get("radii_pixels", vertices.get("radii", []))),
            "scales": np.asarray(vertices.get("scales", [])),
        },
        "edges": {
            "connections": np.asarray(edges.get("connections", []), dtype=object),
            "traces": np.array([np.asarray(t) for t in edges.get("traces", [])], dtype=object),
        },
        "network": {
            "strands": np.asarray(network.get("strands", []), dtype=object),
            "bifurcations": np.asarray(network.get("bifurcations", [])),
            "vertex_degrees": np.asarray(network.get("vertex_degrees", [])),
        },
        "parameters": sanitize_for_matlab(parameters),
    }

    savemat(output_path, data, do_compression=True)
    logger.info(f"MAT export complete: {output_path}")
    return output_path
