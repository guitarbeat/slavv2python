# Energy Computation Methods

[Up: Documentation Index](../../README.md)

This note explains the supported `energy_method` options in the maintained
Python SLAVV pipeline, how they interact with the energy parameter surface, and
where to extend the implementation when a new method is needed.

The active validation surface lives in `source/utils/validation.py`.

## Supported Methods

| Method | Where it lives | Best use | Notes |
| --- | --- | --- | --- |
| `hessian` | `source/core/energy.py` and `source/core/_energy/native_hessian.py` | Default production and parity-oriented raw-image runs | Native matched-filter implementation modeled on the released MATLAB `get_energy_V202` / `energy_filter_V200` path. |
| `frangi` | `skimage.filters.frangi` via `source/core/energy.py` | Quick vesselness experiments | Exploratory alternate backend. |
| `sato` | `skimage.filters.sato` via `source/core/energy.py` | Alternate vesselness experiments | Falls back to `hessian` if the installed `scikit-image` surface does not provide `sato`. |
| `simpleitk_objectness` | `source/core/_energy/backends.py` | Spacing-aware exploratory comparisons | Experimental, non-parity backend. |
| `cupy_hessian` | `source/core/_energy/backends.py` | NVIDIA GPU acceleration experiments | Experimental performance path built on the legacy Gaussian/Hessian approximation work. |

The CLI exposes the same options through:

```powershell
slavv run -i volume.tif -o slavv_output --energy-method hessian
slavv run -i volume.tif -o slavv_output --energy-method frangi
slavv run -i volume.tif -o slavv_output --energy-method sato
slavv run -i volume.tif -o slavv_output --energy-method simpleitk_objectness
slavv run -i volume.tif -o slavv_output --energy-method cupy_hessian
```

Programmatic usage uses the same parameter key:

```python
params = {
    "energy_method": "hessian",
    "radius_of_smallest_vessel_in_microns": 1.5,
    "radius_of_largest_vessel_in_microns": 50.0,
}
```

## Projection Modes

The default `hessian` backend also honors `energy_projection_mode`:

- `matlab`
  Uses the released MATLAB behavior: minimum-energy projection across the
  scale dimension and the corresponding best-scale index.
- `paper`
  Uses the projection described in the published paper: annular best-scale
  estimate, spherical weighted scale estimate over negative energies, blended
  by `spherical_to_annular_ratio`, then sampled back onto the nearest scale.

CLI example:

```powershell
slavv run -i volume.tif -o slavv_output --energy-method hessian --energy-projection-mode paper
```

## Shared Parameters

All energy methods still respect the shared energy configuration surface in
`source/utils/validation.py`, especially:

- `radius_of_smallest_vessel_in_microns`
- `radius_of_largest_vessel_in_microns`
- `scales_per_octave`
- `microns_per_voxel`
- `approximating_PSF`
- `gaussian_to_ideal_ratio`
- `spherical_to_annular_ratio`
- `energy_projection_mode`
- `energy_sign`

`hessian` uses the full native matched-filter path. `frangi` and `sato`
reuse the scale schedule and energy-sign conventions, but they defer the
per-scale vesselness computation to scikit-image. If `sato` is unavailable in
the installed `scikit-image`, the code falls back to `hessian` instead of
failing the run.

## Choosing A Method

- Use `hessian` when you want the default production behavior and the most
  faithful maintained raw-image implementation of the released MATLAB energy
  stage.
- Use `frangi` or `sato` when you are benchmarking alternate vesselness
  backends or doing exploratory native-Python runs.
- Use `simpleitk_objectness` when you specifically want SimpleITK spacing-aware
  objectness behavior.
- Use `cupy_hessian` when you want experimental GPU acceleration and accept
  that it is not the default parity-oriented backend.

## Extending With A New Method

When adding a new energy backend, update these surfaces together:

1. `source/core/energy.py`
   Add the implementation and wire it into both direct and resumable
   evaluation.
2. `source/utils/validation.py`
   Allow the new `energy_method` value in parameter validation.
3. `source/apps/cli_parser.py` and `source/apps/cli_shared.py`
   Expose the new choice through `slavv run --energy-method`.
4. `dev/tests/unit/core/test_energy_methods.py`
   Add focused correctness coverage for the new backend.
5. `dev/tests/unit/core/test_energy_field_storage.py`
   Add or extend regression coverage if the direct and resumable paths need to
   remain aligned.
6. Documentation
   Update this file and any user-facing workflow docs that recommend a default
   method.

## Contributor Notes

- Keep library code on `logging`; do not add `print()` calls in
  `source/core/energy.py`.
- Preserve `float32` outputs for persisted energy volumes unless there is a
  deliberate format change.
- If a new backend cannot support the resumable path cleanly, stop and document
  the limitation before exposing it through the CLI.
