# Zarr Energy Storage

[Up: Documentation Index](../../README.md)

This note explains the optional `zarr` storage path for resumable energy
artifacts in the Python SLAVV pipeline.

The energy stage can persist large intermediate arrays such as:

- `best_energy`
- `best_scale`
- `energy_4d` when `return_all_scales=True`

These arrays were historically stored as `.npy` memmaps. The pipeline now also
supports `.zarr` directories for the same resumable surfaces.

## When To Use It

- Use `--energy-storage-format zarr` when resumable energy volumes are large
  enough that chunked directory-backed storage is a better operational fit than
  single-file memmaps.
- Use `--energy-storage-format auto` to keep the default behavior and let the
  energy stage switch to Zarr automatically for larger runs.
- Keep `--energy-storage-format npy` if you want the legacy memmap layout for
  comparison, debugging, or compatibility with existing local tooling.

## CLI Surface

The `slavv run` surface exposes the storage mode directly:

```powershell
slavv run -i volume.tif -o slavv_output --energy-storage-format auto
slavv run -i volume.tif -o slavv_output --energy-storage-format npy
slavv run -i volume.tif -o slavv_output --energy-storage-format zarr
```

Programmatic usage uses the validated `energy_storage_format` parameter:

```python
params = {
    "energy_method": "hessian",
    "energy_storage_format": "zarr",
    "return_all_scales": True,
}
```

## What Gets Written

When the Zarr path is selected, the energy stage writes directory-backed
artifacts under the stage directory:

- `best_energy.zarr`
- `best_scale.zarr`
- `energy_4d.zarr` when all scales are persisted

The resumable manifest/reporting surface now treats both files and directories
as stage artifacts, so these stores remain visible through the existing
run-state metadata flow.

## Dependency And Installation

Zarr stays optional. Install it with the repo extra:

```powershell
pip install -e ".[zarr]"
```

If `energy_storage_format="zarr"` is selected without `zarr` installed, the
energy stage raises a runtime error with install guidance instead of silently
falling back.

## Behavior Notes

- The direct energy path is unchanged; this feature only affects persisted
  resumable storage.
- Direct and resumable energy aggregation still share the same computation
  logic, so changing storage format should not change `energy`,
  `scale_indices`, or `energy_4d` results.
- Existing `.npy` artifacts are still supported, and config changes continue to
  clear incompatible persisted energy storage before recomputing the stage.