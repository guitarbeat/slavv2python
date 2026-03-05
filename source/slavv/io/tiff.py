"""Image file I/O — TIFF and DICOM loading.

This module handles loading 3-D grayscale microscopy volumes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import IO, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


def load_tiff_volume(
    file: Union[str, Path, IO[bytes]], *, memory_map: bool = False
) -> np.ndarray:
    """Load a 3D grayscale TIFF volume with validation.

    Parameters
    ----------
    file:
        Path or binary file-like object containing TIFF data.
    memory_map:
        If ``True``, return a memory-mapped array instead of reading the
        entire volume into memory.  Requires ``file`` to be a path-like object.

    Returns
    -------
    np.ndarray
        The loaded 3-D volume.

    Raises
    ------
    ValueError
        If the file cannot be read or does not contain a 3-D grayscale volume.
    """
    import tifffile

    tif_logger = logging.getLogger("tifffile")
    original_level = tif_logger.level
    tif_logger.setLevel(logging.ERROR)
    try:
        volume = tifffile.memmap(file) if memory_map else tifffile.imread(file)
    except Exception as exc:
        raise ValueError(f"Failed to read TIFF volume: {exc}") from exc
    finally:
        tif_logger.setLevel(original_level)

    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume")
    if np.iscomplexobj(volume):
        raise ValueError("Expected a real-valued grayscale TIFF volume")
    return volume if memory_map else np.asarray(volume)


def dicom_to_tiff(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    *,
    sort_by: str = "instance",
    rescale: bool = True,
    dtype: str = "uint16",
) -> Union[np.ndarray, Path]:
    """Load DICOM (single file, multi-frame, or series) and optionally write TIFF.

    Parameters
    ----------
    input_path:
        Path to a DICOM file or directory containing a DICOM series.
    output_path:
        If provided, write a TIFF volume to this path and return the path.
        If omitted, return the loaded 3-D volume as a NumPy array.
    sort_by:
        Sorting key: ``'instance'``, ``'position'``, or ``'location'``.
    rescale:
        Apply ``RescaleSlope``/``RescaleIntercept`` if present.
    dtype:
        Output dtype string (e.g. ``'uint16'``, ``'float32'``).

    Returns
    -------
    np.ndarray or Path
    """
    try:
        import pydicom  # type: ignore
        import tifffile
    except Exception as exc:
        raise RuntimeError(
            "dicom_to_tiff requires 'pydicom' and 'tifffile' to be installed"
        ) from exc

    in_path = Path(input_path)

    def _sort_key(ds) -> float:
        if sort_by == "position":
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) == 3:
                return float(ipp[2])
        if sort_by == "location":
            loc = getattr(ds, "SliceLocation", None)
            if loc is not None:
                return float(loc)
        inst = getattr(ds, "InstanceNumber", None)
        try:
            return float(inst)
        except Exception:
            return 0.0

    def _apply_rescale(arr: np.ndarray, ds) -> np.ndarray:
        if not rescale:
            return arr
        slope = getattr(ds, "RescaleSlope", 1.0) or 1.0
        intercept = getattr(ds, "RescaleIntercept", 0.0) or 0.0
        return arr.astype(np.float32) * float(slope) + float(intercept)

    def _normalize_dtype(vol: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(out_dtype, np.floating):
            vmin, vmax = np.nanmin(vol), np.nanmax(vol)
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
            return vol.astype(out_dtype, copy=False)
        info = np.iinfo(out_dtype)
        vmin, vmax = np.nanmin(vol), np.nanmax(vol)
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin) * info.max
        return vol.astype(out_dtype)

    if in_path.is_dir():
        files = sorted(p for p in in_path.iterdir() if p.is_file())
        datasets = []
        for p in files:
            try:
                ds = pydicom.dcmread(str(p), stop_before_pixels=False)
                if hasattr(ds, "pixel_array"):
                    datasets.append(ds)
            except Exception:
                continue
        if not datasets:
            raise ValueError("No readable DICOM slices found in directory")
        datasets.sort(key=_sort_key)
        volume = np.stack([np.asarray(_apply_rescale(ds.pixel_array, ds)) for ds in datasets], axis=0)
    else:
        ds = pydicom.dcmread(str(in_path), stop_before_pixels=False)
        arr = _apply_rescale(ds.pixel_array, ds)
        if arr.ndim == 2:
            volume = arr[np.newaxis]
        elif arr.ndim == 3:
            volume = arr
        else:
            raise ValueError("Unsupported DICOM pixel array dimensionality")

    out_dtype = np.dtype(dtype)
    volume = np.asarray(volume)
    volume = _normalize_dtype(volume, out_dtype) if rescale else volume.astype(out_dtype, copy=False)

    if output_path is not None:
        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(outp), volume)
        return outp
    return volume
