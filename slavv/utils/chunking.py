"""
Chunking utilities for memory-efficient processing in SLAVV.
"""
from typing import List, Tuple


def get_chunking_lattice(
    shape: Tuple[int, int, int], max_voxels: int, margin: int
) -> List[Tuple[Tuple[slice, slice, slice], Tuple[slice, slice, slice], Tuple[slice, slice, slice]]]:
    """Generate overlapping z-axis chunks to limit voxel processing.

    Parameters
    ----------
    shape:
        Image shape as ``(y, x, z)``.
    max_voxels:
        Maximum voxels allowed per chunk including margins.
    margin:
        Overlap in voxels applied on both sides of each chunk along ``z``.

    Returns
    -------
    list of tuples
        ``(chunk_slice, output_slice, inner_slice)`` where ``chunk_slice``
        indexes the padded region in the source image, ``output_slice``
        corresponds to the destination region, and ``inner_slice`` selects the
        interior region of the chunk to copy into ``output_slice``.
    """

    y, x, z = shape
    plane_voxels = y * x
    max_depth = max_voxels // plane_voxels
    if max_depth <= 0 or max_depth >= z:
        return [
            (
                (slice(0, y), slice(0, x), slice(0, z)),
                (slice(0, y), slice(0, x), slice(0, z)),
                (slice(0, y), slice(0, x), slice(0, z)),
            )
        ]

    margin = min(margin, max_depth // 2)
    core_depth = max_depth - 2 * margin
    if core_depth <= 0:
        core_depth = 1

    slices = []
    start = 0
    while start < z:
        end = min(start + core_depth, z)
        pad_before = margin if start > 0 else 0
        pad_after = margin if end < z else 0
        chunk_slice = (
            slice(0, y),
            slice(0, x),
            slice(start - pad_before, end + pad_after),
        )
        output_slice = (slice(0, y), slice(0, x), slice(start, end))
        inner_slice = (
            slice(0, y),
            slice(0, x),
            slice(pad_before, pad_before + (end - start)),
        )
        slices.append((chunk_slice, output_slice, inner_slice))
        start = end

    return slices
