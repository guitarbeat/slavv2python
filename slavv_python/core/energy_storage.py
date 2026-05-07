from __future__ import annotations

from typing import Any, Callable

import numpy as np


def select_energy_storage_format(
    storage_format: str,
    *,
    total_voxels: int,
    max_voxels: int,
    require_zarr_backend: Callable[[], Any],
) -> str:
    """Choose the resumable energy storage backend."""
    if storage_format == "auto":
        return "zarr" if total_voxels > max_voxels else "npy"
    if storage_format == "zarr":
        require_zarr_backend()
    return storage_format


def remove_storage_path(path: Any) -> None:
    """Remove a file or directory-backed storage artifact."""
    if path.exists():
        if path.is_dir():
            for child in path.iterdir():
                if child.is_dir():
                    remove_storage_path(child)
                else:
                    child.unlink()
            path.rmdir()
        else:
            path.unlink()


def zarr_chunks_for_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return conservative chunk sizes for energy arrays."""
    if len(shape) == 3:
        return tuple(min(int(axis), 64) for axis in shape)
    if len(shape) == 4:
        return (
            min(int(shape[0]), 64),
            min(int(shape[1]), 64),
            min(int(shape[2]), 64),
            1,
        )
    return tuple(min(int(axis), 64) for axis in shape)


def open_energy_storage_array(
    path: Any,
    *,
    mode: str,
    dtype: Any,
    shape: tuple[int, ...],
    fill_value: float | int | None,
    storage_format: str,
    require_zarr_backend: Callable[[], Any],
) -> Any:
    """Open a resumable energy array in either NPY memmap or Zarr format."""
    if storage_format == "zarr":
        zarr_module = require_zarr_backend()
        if mode == "r+":
            return zarr_module.open(str(path), mode="r+")
        return zarr_module.open(
            str(path),
            mode="w",
            shape=shape,
            dtype=dtype,
            chunks=zarr_chunks_for_shape(shape),
            fill_value=fill_value,
        )

    if mode == "r+":
        return np.lib.format.open_memmap(path, mode="r+")
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    if fill_value is not None:
        array[...] = fill_value
    return array
