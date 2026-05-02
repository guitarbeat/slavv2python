"""Vertex painting helpers."""

from __future__ import annotations

import logging

import numpy as np
from skimage.draw import ellipsoid
from typing_extensions import TypeAlias

Int64Array: TypeAlias = "np.ndarray"

logger = logging.getLogger(__name__)


def paint_vertex_image(
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Create a painted vertex-volume image (1-indexed, 0=background).

    This paints ellipsoidal occupancy regions around each vertex for overlap-aware edge logic,
    including maintained-path terminal detection that should treat entering a vertex body as a hit.
    """
    vertex_image = np.zeros(image_shape, dtype=np.uint16)

    for i, (pos, scale) in enumerate(zip(vertex_positions, vertex_scales)):
        radii = lumen_radius_pixels[scale]
        try:
            ellipsoid_mask = ellipsoid(radii[0], radii[1], radii[2], spacing=(1.0, 1.0, 1.0))
            coords = np.where(ellipsoid_mask)
            center: Int64Array = np.array(ellipsoid_mask.shape, dtype=np.int64) // 2
            rr = coords[0] - center[0]
            cc = coords[1] - center[1]
            dd = coords[2] - center[2]

            y_coords = rr + int(np.round(pos[0]))
            x_coords = cc + int(np.round(pos[1]))
            z_coords = dd + int(np.round(pos[2]))

            valid_mask = (
                (y_coords >= 0)
                & (y_coords < image_shape[0])
                & (x_coords >= 0)
                & (x_coords < image_shape[1])
                & (z_coords >= 0)
                & (z_coords < image_shape[2])
            )

            vertex_image[
                y_coords[valid_mask],
                x_coords[valid_mask],
                z_coords[valid_mask],
            ] = i + 1
        except Exception as exc:
            logger.warning(f"Failed to paint vertex {i} at {pos} with scale {scale}: {exc}")
            continue

    logger.info(f"Painted {len(vertex_positions)} vertices into volume image")
    return vertex_image


def paint_vertex_center_image(
    vertex_positions: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Create a sparse image containing only vertex center identities."""
    center_image: np.ndarray = np.zeros(image_shape, dtype=np.uint16)
    if len(vertex_positions) == 0:
        return center_image

    coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)[:, :3]).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, image_shape[2] - 1)
    center_image[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(
        1,
        len(coords) + 1,
        dtype=np.uint16,
    )
    return center_image
