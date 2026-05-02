# Paper Profile

[Up: Reference Docs](../README.md)

This document describes the maintained public workflow for running SLAVV as a
native Python implementation of the published method.

## Public Finish Line

The public product goal is now distinct from the exact MATLAB proof track:

- `paper`
  The primary user-facing profile. It runs the native Python TIFF-to-network
  pipeline and uses the paper-style Hessian projection.
- `matlab_compat`
  A legacy-oriented preset that keeps the MATLAB-shaped default projection and
  related public defaults, while still running the maintained Python pipeline.

Exact MATLAB parity remains a developer workflow owned by the parity docs and
the `parity_experiment.py` tooling. It is no longer the public entrypoint.

## CLI Workflow

Use `paper` unless you have a specific reason to compare against the older
MATLAB-shaped defaults.

```powershell
slavv run -i volume.tif -o slavv_output --export json
slavv run -i volume.tif -o slavv_output --profile matlab_compat --export json
slavv analyze -i slavv_output/network.json
slavv plot -i slavv_output/network.json -o plots.html
```

The `run` surface now resolves parameters in this order:

1. profile defaults
2. explicit CLI or app overrides
3. shared validation

That rule is the same across CLI, Streamlit, and direct library calls that pass
`pipeline_profile`.

## Export Contract

`network.json` is now a versioned authoritative export.

Every new JSON export writes:

- `schema`
  Name and version for the public JSON contract.
- `metadata`
  Profile, image shape, voxel size, run identifiers, and stage provenance when
  available.
- `parameters`
  The validated parameter surface that produced the result.
- `vertices`
  Positions, radii, energies, scales, and degree-facing fields.
- `edges`
  Connections plus stored traces and edge attributes needed for downstream
  analysis and plotting.
- `network`
  Strand topology, bifurcations, orphan/cycle cleanup outputs, and strand-level
  relationships.
- `summary`
  Precomputed analysis statistics when the payload is complete enough.

`slavv analyze` and `slavv plot` consume this schema directly. Legacy thin JSON
files are still readable during the compatibility window, but all new writes use
the authoritative schema.

## Streamlit Workflow

The processing page exposes the same profile contract as the CLI:

- profile selector
- paper-facing vessel scale controls
- energy method and projection controls
- edge tracing limits and cleanup thresholds

Switching profiles resets the profile-backed defaults so app runs stay aligned
with the selected workflow.
