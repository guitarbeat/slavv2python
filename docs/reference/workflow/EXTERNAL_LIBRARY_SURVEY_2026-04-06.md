# External Library Survey For The Python Port

[Up: Documentation Index](../../README.md)

Date: 2026-04-06

## Status

The main ideas from this survey have already been implemented in the repo:

- `SimpleITK` is wired into the energy pipeline as an optional backend.
- `CuPy` is wired into the energy pipeline as an optional backend.
- `Zarr` is wired into resumable energy storage.
- `napari` is available as an optional curator backend.
- `numba` is already present as an optional acceleration path.

Relevant code now lives in [source/slavv/core/energy.py](../../../source/slavv/core/energy.py), [source/slavv/runtime/run_state.py](../../../source/slavv/runtime/run_state.py), and [source/slavv/visualization/napari_curator.py](../../../source/slavv/visualization/napari_curator.py).

## Remaining Ideas

These are still unimplemented and are the only items that remain worth tracking here:

- `cuCIM` for GPU image operations with a scikit-image-like surface.
- `Dask` for lazy chunked computation when volume size becomes the bottleneck.
- `connected-components-3d` for faster 3D label cleanup helpers.
- `MONAI` for any future learned segmentation or denoising branch.
- `OpenCV` only if a targeted 2D preprocessing step becomes necessary.

## Bottom Line

The survey is no longer a broad recommendation memo. The important additions are already in the codebase, and the only open question is whether any of the remaining libraries should be adopted later.