"""Pipeline result exporters and network partitioning utilities."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Union

import numpy as np

from .network_io import Network

logger = logging.getLogger(__name__)


def export_pipeline_results(
    results: dict[str, Any],
    output_dir: Union[str, Path],
    base_name: str = "result",
) -> list[Path]:
    """Export all standard components of a pipeline result to files.

    Saves pipeline parameters to JSON.  Returns list of written file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def _default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Cannot serialize {type(o)}")

    try:
        params_path = out_dir / f"{base_name}_parameters.json"
        with open(params_path, "w") as f:
            json.dump(results.get("parameters", {}), f, indent=2, default=_default)
        created.append(params_path)
    except Exception as exc:
        logger.warning("Failed to export parameters: %s", exc)

    return created


def partition_network(
    network: Network,
    chunks: tuple[int, int],
    overlap: float = 0.0,
    output_dir: Union[str, Path] | None = None,
) -> dict[tuple[int, int], Network]:
    """Partition a network into spatial (Y, X) bins.

    Corresponds to ``partition_casx_by_xy_bins.m``.
    """
    ny, nx = chunks
    if ny <= 0 or nx <= 0:
        raise ValueError("chunks must contain positive Y/X bin counts")
    vertices = network.vertices
    if len(vertices) == 0:
        return {}

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    extent = max_coords - min_coords
    y_step = extent[0] / ny
    x_step = extent[1] / nx

    partitions: dict[tuple[int, int], Network] = {}
    for y_i in range(ny):
        for x_i in range(nx):
            y_min = min_coords[0] + y_i * y_step - overlap
            y_max = min_coords[0] + (y_i + 1) * y_step + overlap
            x_min = min_coords[1] + x_i * x_step - overlap
            x_max = min_coords[1] + (x_i + 1) * x_step + overlap

            mask = (
                (vertices[:, 0] >= y_min)
                & (vertices[:, 0] <= y_max)
                & (vertices[:, 1] >= x_min)
                & (vertices[:, 1] <= x_max)
            )
            if not np.any(mask):
                continue

            orig_idx = np.where(mask)[0]
            remap = {old: new for new, old in enumerate(orig_idx)}
            sub_verts = vertices[mask]
            sub_radii = network.radii[mask] if network.radii is not None else None

            sub_edges = [
                [remap[u], remap[v]] for u, v in network.edges if u in remap and v in remap
            ]
            partitions[(y_i, x_i)] = Network(
                vertices=sub_verts,
                edges=np.array(sub_edges) if sub_edges else np.empty((0, 2), dtype=int),
                radii=sub_radii,
            )

    return partitions


def parse_registration_file(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """Parse a legacy SLAVV registration text file.

    Corresponds to ``registration_txt2mat.m``.
    Returns ``(starts, dims)`` each as (N, 3) arrays.
    """
    content = Path(path).read_text()
    num_match = re.search(r"num\s*=\s*(\d+)", content)
    if not num_match:
        raise ValueError("Could not find 'num =' in file")
    num_images = int(num_match[1])

    matches = re.findall(r"\(([^)]+)\)", content)
    coords = []
    for m in matches:
        parts = [float(x.strip()) for x in m.split(",")]
        if len(parts) == 3:
            coords.append(parts)

    if len(coords) < 2 * num_images:
        logger.warning("Expected %d coordinate triplets, found %d", 2 * num_images, len(coords))

    starts = np.array(coords[:num_images])
    dims = np.array(coords[num_images : 2 * num_images])
    return starts, dims
