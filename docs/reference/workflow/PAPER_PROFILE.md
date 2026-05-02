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

## Maintained Edge-Tracing Semantics

On the maintained public tracing path, terminal detection is not limited to
exact center-voxel hits.

- The tracer still prefers explicit vertex-center hits when they occur.
- The maintained fallback path must also treat entry into a painted target
  vertex body as a terminal hit.
- If a paper-workflow run still looks too sparse after that, the next thing to
  inspect is trace dynamics rather than export or center-only lookup logic.

The current maintained debugging read on the real crop
`180709_EL_center_crop_24x256x256` is:

- wiring painted vertex occupancy into terminal detection materially improved
  edge completion
- direct terminal hits increased from `5` to `15`
- chosen edges increased from `12` to `15`
- `energy_rise_step_halving` remained the dominant stop reason

That means current sparse-network debugging should start from continuous
tracing behavior, not from the older red herring that the maintained path was
only checking center voxels.

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
