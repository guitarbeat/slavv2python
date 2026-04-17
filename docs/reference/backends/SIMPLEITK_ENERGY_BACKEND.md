# SimpleITK Energy Backend

[Up: Documentation Index](../../README.md)

This note describes the experimental `simpleitk_objectness` energy backend for
the Python SLAVV port.

## When To Use It

Use `simpleitk_objectness` when you want a spacing-aware 3D vesselness backend
without changing the default Hessian implementation for existing runs.

This backend is a good fit for:

- exploratory comparisons against the built-in `hessian` path
- anisotropic microscopy volumes where physical voxel spacing matters
- small integration spikes that stay in the deterministic image-processing
  workflow

It is not intended to replace the default backend yet.

## Installation

Install the optional extra:

```powershell
pip install -e ".[sitk]"
```

## CLI Usage

```powershell
slavv run -i volume.tif -o slavv_output --energy-method simpleitk_objectness
```

## Array Order And Spacing

The SLAVV codebase stores image volumes as `(y, x, z)` arrays.

The SimpleITK backend converts those arrays to `(z, y, x)` before creating a
SimpleITK image, then sets spacing in `(x, y, z)` order using:

- `microns_per_voxel[1]` for `x`
- `microns_per_voxel[0]` for `y`
- `microns_per_voxel[2]` for `z`

After the filter runs, the result is converted back to the repo-standard
`(y, x, z)` order before any scale aggregation or downstream processing.

## Parameters Honored In V1

The first implementation explicitly honors:

- `microns_per_voxel`
- `radius_of_smallest_vessel_in_microns`
- `radius_of_largest_vessel_in_microns`
- `scales_per_octave`
- `energy_sign`
- `return_all_scales`

## Parameters Not Emulated In V1

The backend does not emulate the MATLAB-style PSF and annular controls used by
the default Hessian path. If you choose `simpleitk_objectness`, these controls
are not applied:

- `approximating_PSF`
- `gaussian_to_ideal_ratio`
- `spherical_to_annular_ratio`

The code logs a warning when the selected parameters imply non-default values
for those controls.

## How It Differs From Other Energy Methods

- `hessian`
  This is the default backend and remains the parity-oriented baseline.
- `frangi`
  Uses scikit-image vesselness filtering with the current pipeline's scale
  selection logic.
- `sato`
  Uses scikit-image tubeness filtering with the same aggregation model.
- `simpleitk_objectness`
  Uses SimpleITK Hessian/objectness filtering with explicit physical spacing and
  optional installation.

For now, `simpleitk_objectness` should be treated as an experimental comparison
backend rather than a new default.