# Energy Computation Methods

[Up: Documentation Index](../../README.md)

This note explains the supported `energy_method` options in the Python SLAVV
pipeline, how they interact with the existing parameter surface, and where to
extend the implementation when a new method is needed.

The active validation surface lives in `source/slavv/utils/validation.py`.

## Supported Methods

| Method | Where it lives | Best use | Notes |
| --- | --- | --- | --- |
| `hessian` | `source/slavv/core/energy.py` | Default production work | Uses the full Hessian response path and remains the safest choice for reproducible runs. |
| `frangi` | `skimage.filters.frangi` via `source/slavv/core/energy.py` | Quick vesselness experiments | Good for exploratory native-Python runs. |
| `sato` | `skimage.filters.sato` via `source/slavv/core/energy.py` | Alternate vesselness experiments | Falls back to `hessian` if the installed `scikit-image` surface does not provide `sato`. |

The CLI exposes the same options through:

```powershell
slavv run -i volume.tif -o slavv_output --energy-method hessian
slavv run -i volume.tif -o slavv_output --energy-method frangi
slavv run -i volume.tif -o slavv_output --energy-method sato
```

Programmatic usage uses the same parameter key:

```python
params = {
    "energy_method": "hessian",
    "radius_of_smallest_vessel_in_microns": 1.5,
    "radius_of_largest_vessel_in_microns": 50.0,
}
```

## Shared Parameters

All three methods still respect the shared energy configuration surface in
`source/slavv/utils/validation.py`, especially:

- `radius_of_smallest_vessel_in_microns`
- `radius_of_largest_vessel_in_microns`
- `scales_per_octave`
- `microns_per_voxel`
- `approximating_PSF`
- `gaussian_to_ideal_ratio`
- `spherical_to_annular_ratio`
- `energy_sign`

`hessian` uses the full matched-filter path. `frangi` and `sato`
reuse the scale schedule and energy-sign conventions, but they defer the
per-scale vesselness computation to scikit-image. If `sato` is unavailable in
the installed `scikit-image`, the code falls back to `hessian` instead of
failing the run.

## Choosing A Method

- Use `hessian` when you want the default production behavior and the most
  stable direct/resumable energy path.
- Use `frangi` or `sato` when you are benchmarking alternate vesselness
  backends or doing exploratory native-Python runs.
- Treat `frangi` and `sato` outputs as alternative analysis surfaces, not as
  replacements for the default `hessian` path.

## Extending With A New Method

When adding a new energy backend, update these surfaces together:

1. `source/slavv/core/energy.py`
   Add the implementation and wire it into both direct and resumable
   evaluation.
2. `source/slavv/utils/validation.py`
   Allow the new `energy_method` value in parameter validation.
3. `source/slavv/apps/cli.py`
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
  `source/slavv/core/energy.py`.
- Preserve `float32` outputs for persisted energy volumes unless there is a
  deliberate format change.
- If a new backend cannot support the resumable path cleanly, stop and document
  the limitation before exposing it through the CLI.
