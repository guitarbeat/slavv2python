# Streamlit Manual-Curation Workflow

## In short

The Streamlit application curates actual SLAVV pipeline results. Start with an uploaded TIFF, a built-in sample TIFF, or compatible stage results from an existing run. Built-in samples use the same processing code as uploaded data.

```powershell
slavv-app
```

## Process data

Open **Processing** and choose an input source:

- **Upload TIFF** reads the selected file with the normal SLAVV TIFF loader.
- **Built-in sample** creates a deterministic synthetic vessel volume, saves it as TIFF data, and sends those bytes through the same loader and full Energy → Vertices → Edges → Network pipeline.

The straight-vessel sample is the reliable default for a quick interactive session. Its reduced scale range is a sample-friendly processing preset, not a precomputed result. The junction samples are available for exploratory topology checks. You can download any generated TIFF and reproduce the same input from the CLI.

Select **Run processing** and wait for all four pipeline stages to complete. The resulting typed stage objects become the shared application run context used by every downstream page.

## Curate the result

Open **Curation** after an Edge Set is available. The default **Trust path:
MATLAB-familiar browser curator** (ADR 0014) follows the two-stage sequence:
Vertex Curation, then Edge Curation. It provides:

- calibrated X, Y, and Z projections with depth, logarithmic thickness, pan, zoom, and a physical scale bar;
- the original grayscale/cyan/red visual language, a field-of-view minimap, and intensity and Energy histograms;
- point, rectangle, polyline, and circular-complement toggling, plus local threshold, Sweep, Crop, Paint, Add Vertex, and Add Edge tools where their stage supports them;
- local undo/redo with keyboard shortcuts, and a versioned `.slavv-curation.json` session file for Save/Load;
- a final **Apply and rebuild Network** action after Edge Curation.

Pointer and slider interactions stay in the component, so they do not rerun the Streamlit page. Press **Escape** or choose another tool to end a continuous tool mode. Returning from Edge Curation to Vertex Curation preserves the local edit history, and rejecting a vertex also rejects its incident edges.

Applying review removes rejected vertices and edges, drops edges incident to rejected vertices, remaps connection indices, and rebuilds the Network from the retained objects. It also invalidates stale analysis and export/share signatures so downstream pages use the current network.

The original intensity volume is resolved from the current processing session or run manifest. When only Energy is available, the workspace explicitly enters degraded projection mode: intensity controls and the cranium Crop tool are disabled, and Trust claim chrome is suppressed (ADR 0014). Add Vertex is similarly disabled when scale/radius metadata is unavailable.

**Desktop manual review (experimental)** is a Python Qt/PyVista or napari interface that opens in a separate local window when optional GUI dependencies and a display are available. It is **not** the Trust MATLAB-familiar claim surface. Automatic and model-assisted workflows operate on the same active results.

## Compare with the original MATLAB curator

On Windows with MATLAB R2019a installed, launch the preserved `y_junction_32` sample from the repository root:

```matlab
run('scripts/curation/launch_matlab_curator_sample.m')
```

The script opens the unmodified `external/Vectorization-Public/source/vertex_curator.m` against preserved MATLAB artifacts. It normalizes radii to a column vector and voxel spacing to a three-element row before launch. Any saved curation is written under `workspace/scratch/manual_curation_showcase`; the preserved fixture remains unchanged.

## Continue downstream

Use **Visualization** to inspect the curated geometry and create exports. Use **Analysis** for topology and morphology metrics. The persistent sidebar reports available stages, the last curation method, run provenance, and the recommended next action.

## Reopen a persisted run

Open **Workspaces** to browse compatible temporary app runs and durable runs under `workspace/runs`. Select a row to inspect its status, volume shape, available stages, run ID, update time, and source path. Choose **Open read-only** to load its available stage results into the shared application session.

You can also select **Open existing run** in the sidebar and enter a structured run directory containing `99_Metadata/run_snapshot.json`, `99_Metadata/validated_params.json`, and at least one typed stage checkpoint under `02_Output/python_results/checkpoints/`.

Reopened checkpoints are read-only on disk. Curation may rebuild results in the browser session, but it does not overwrite the source metadata or checkpoints. Export the session result if it must be saved separately.

## Troubleshooting

- **Curation is empty:** finish Processing through Edges or open a run containing an Edges checkpoint.
- **The sample is slow:** use **Straight vessel (32 x 64 x 64)** and keep the sample processing defaults.
- **Desktop curation is unavailable:** use browser review, or install the optional GUI dependencies and run the app on a host with a display.
- **Browser Crop is disabled:** reload from Processing so the original intensity volume is present; an Energy-only reopened run cannot perform intensity-based cranium rejection.
- **A reopened run is partial:** use the readiness indicators to see which pages can consume its available checkpoints.
