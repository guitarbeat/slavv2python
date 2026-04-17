"""Shared watershed tracing and label helpers for parity supplementation."""

from __future__ import annotations

import numpy as np
from skimage.segmentation import watershed


def _rasterize_trace_segment(
    start: np.ndarray,
    end: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Rasterize a straight voxel segment between two points, preserving endpoints."""
    start_coord = np.rint(np.asarray(start, dtype=np.float32)[:3]).astype(np.int32, copy=False)
    end_coord = np.rint(np.asarray(end, dtype=np.float32)[:3]).astype(np.int32, copy=False)
    max_coord = np.asarray(image_shape, dtype=np.int32) - 1
    start_coord = np.clip(start_coord, 0, max_coord)
    end_coord = np.clip(end_coord, 0, max_coord)

    steps = int(np.max(np.abs(end_coord - start_coord)))
    if steps <= 0:
        return start_coord.reshape(1, 3).astype(np.float32, copy=False)

    coords = np.rint(np.linspace(start_coord, end_coord, num=steps + 1)).astype(np.int32)
    deduped = [coords[0]]
    for coord in coords[1:]:
        if not np.array_equal(coord, deduped[-1]):
            deduped.append(coord)
    return np.asarray(deduped, dtype=np.float32)


def _build_watershed_join_trace(
    start: np.ndarray,
    contact: np.ndarray,
    end: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Construct a simple ordered trace that joins two vertices through a watershed contact."""
    start_half = _rasterize_trace_segment(start, contact, image_shape)
    end_half = _rasterize_trace_segment(contact, end, image_shape)
    if len(end_half) > 0 and len(start_half) > 0 and np.array_equal(start_half[-1], end_half[0]):
        end_half = end_half[1:]
    if len(end_half) == 0:
        return start_half
    return np.vstack([start_half, end_half]).astype(np.float32, copy=False)


def _best_watershed_contact_coords(
    labels: np.ndarray,
    energy: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Return the lowest-energy face contact voxel for each touching watershed pair."""
    best_contacts: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    shifts = (
        np.array((1, 0, 0), dtype=np.int32),
        np.array((0, 1, 0), dtype=np.int32),
        np.array((0, 0, 1), dtype=np.int32),
    )

    for shift in shifts:
        source_slices = tuple(slice(None, -int(delta)) if delta else slice(None) for delta in shift)
        target_slices = tuple(slice(int(delta), None) if delta else slice(None) for delta in shift)
        source_labels = labels[source_slices]
        target_labels = labels[target_slices]
        is_touching = (source_labels != target_labels) & (source_labels > 0) & (target_labels > 0)
        if not np.any(is_touching):
            continue

        source_coords = np.argwhere(is_touching).astype(np.int32, copy=False)
        target_coords = source_coords + shift
        source_pairs = source_labels[is_touching].astype(np.int32, copy=False) - 1
        target_pairs = target_labels[is_touching].astype(np.int32, copy=False) - 1
        pair_indices = np.stack([source_pairs, target_pairs], axis=1)
        pair_indices.sort(axis=1)

        source_energy = energy[source_slices][is_touching]
        target_energy = energy[target_slices][is_touching]
        prefer_target = target_energy < source_energy
        contact_coords = source_coords.copy()
        contact_coords[prefer_target] = target_coords[prefer_target]
        contact_energy = np.where(prefer_target, target_energy, source_energy).astype(
            np.float32,
            copy=False,
        )

        order = np.lexsort((contact_energy, pair_indices[:, 1], pair_indices[:, 0]))
        pair_indices = pair_indices[order]
        contact_coords = contact_coords[order]
        contact_energy = contact_energy[order]
        keep: np.ndarray = np.ones((len(pair_indices),), dtype=bool)
        keep[1:] = np.any(pair_indices[1:] != pair_indices[:-1], axis=1)

        for pair_array, coord, pair_energy in zip(
            pair_indices[keep],
            contact_coords[keep],
            contact_energy[keep],
        ):
            pair = (int(pair_array[0]), int(pair_array[1]))
            best = best_contacts.get(pair)
            if best is None or float(pair_energy) < best[0]:
                best_contacts[pair] = (float(pair_energy), coord.astype(np.int32, copy=False))

    return {pair: coord for pair, (_, coord) in best_contacts.items()}


def _build_watershed_labels(
    energy: np.ndarray,
    vertex_positions: np.ndarray,
    energy_sign: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Build MATLAB-style watershed labels seeded at rounded vertex centers."""
    image_shape = energy.shape
    markers = np.zeros(image_shape, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(image_shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)
    labels = watershed(-energy_sign * energy, markers)
    return labels, image_shape
