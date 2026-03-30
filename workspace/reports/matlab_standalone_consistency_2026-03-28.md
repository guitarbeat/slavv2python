# MATLAB Standalone Consistency Check

Date: 2026-03-28

## Goal

Run the MATLAB workflow three times on its own, with no Python rerun or imported checkpoints, and check whether the outputs stay consistent.

## How I ran it

I used a fresh output directory for each trial so MATLAB resume logic could not reuse a previous batch:

- Session root: `D:\slavv_comparisons\20260328_matlab_consistency_023500`
- Input volume: `C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\data\slavv_test_volume.tif`
- Parameters: `C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\comparison_output_live_parity\99_Metadata\comparison_params.normalized.json`
- MATLAB executable: `C:\Program Files\MATLAB\R2019a\bin\matlab.exe`
- Windows launcher: `workspace/scripts/cli/run_matlab_cli.bat`
- MATLAB wrapper: `workspace/scripts/cli/run_matlab_vectorization.m`
- Upstream MATLAB code root: `external/Vectorization-Public`

Each trial effectively ran the standalone MATLAB entrypoint with the equivalent of:

```powershell
workspace\scripts\cli\run_matlab_cli.bat `
  "data\slavv_test_volume.tif" `
  "D:\slavv_comparisons\20260328_matlab_consistency_023500\run_0X" `
  "C:\Program Files\MATLAB\R2019a\bin\matlab.exe" `
  "comparison_output_live_parity\99_Metadata\comparison_params.normalized.json"
```

Inside MATLAB, `run_matlab_vectorization.m` ran the four workflow stages in order:

1. `energy`
2. `vertices`
3. `edges`
4. `network`

## Files involved

Each run root contained:

- `matlab_run.log`
- `matlab_resume_state.json`
- one `batch_YYMMDD-HHmmss/` folder

Each batch folder created the same top-level directories:

- `curations/`
- `data/`
- `settings/`
- `vectors/`
- `visual_data/`
- `visual_vectors/`

The files that mattered most for this check were:

- `timings.json`
- `data/energy_<timestamp>_slavv_test_volume.mat`
- `settings/batch.mat`
- `settings/energy_<timestamp>.mat`
- `settings/vertices_<timestamp>.mat`
- `settings/edges_<timestamp>.mat`
- `settings/network_<timestamp>.mat`
- `settings/workflow_<timestamp>.mat` (four workflow snapshots per run)
- `vectors/curated_vertices_<timestamp>_slavv_test_volume.mat`
- `vectors/curated_edges_<timestamp>_slavv_test_volume.mat`
- `vectors/network_<timestamp>_slavv_test_volume.mat`
- `vectors/vertices_<timestamp>_slavv_test_volume.mat`
- `vectors/edges_<timestamp>_slavv_test_volume.mat`
- `visual_data/*`

Per run there were 22 files under the run root and batch tree. After normalizing timestamped names, the three runs matched the same 19 canonical artifact patterns. The `curations/` and `visual_vectors/` directories existed but were empty in these runs.

## Run summary

| Run | Batch folder | Wall time (s) | MATLAB total (s) | Energy (s) | Vertices (s) | Edges (s) | Network (s) | Vertices | Edges | Strands |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `run_01\batch_260328-133924` | 909.11 | 848.09 | 806.67 | 11.61 | 28.27 | 1.26 | 1682 | 1379 | 682 |
| 2 | `run_02\batch_260328-135509` | 896.49 | 830.30 | 792.68 | 12.40 | 24.32 | 0.76 | 1682 | 1379 | 682 |
| 3 | `run_03\batch_260328-141011` | 743.12 | 693.96 | 660.57 | 9.84 | 22.71 | 0.68 | 1682 | 1379 | 682 |

Timing spread:

- Wall time ranged from `743.12s` to `909.11s`
- MATLAB internal total ranged from `693.96s` to `848.09s`
- The variance was almost entirely in `energy`
- `vertices`, `edges`, and `network` were comparatively stable

## Consistency checks

### Counts

All three runs matched exactly on the headline outputs:

- Vertices: `1682`
- Edges: `1379`
- Strands: `682`

### Parsed output digests

I parsed the MATLAB outputs and hashed the normalized data structures that matter for parity:

- vertex positions
- vertex scale indices
- vertex radii
- edge endpoint connections
- edge traces
- network strands

All six parsed digests matched across all three runs.

That means the normalized MATLAB outputs were consistent across runs, not just the counts.

### File inventory

The canonicalized batch inventory matched across all three runs.

Small raw file-size differences did appear in two timestamped files:

- `settings/workflow_<timestamp>.mat`
- `vectors/network_<timestamp>_slavv_test_volume.mat`

Those differences were tiny, and they did not change the parsed vertices, edges, or strand topology. The normalized digests for edges and strands were still identical.

## Verdict

The MATLAB standalone workflow was consistent across these three fresh runs.

What stayed the same:

- file layout
- parsed vertex data
- parsed edge connectivity and traces
- parsed network strands
- headline counts: `1682 / 1379 / 682`

What varied:

- runtime, especially the `energy` stage
- some timestamped raw artifact bytes and file sizes

## Practical takeaway

For parity work, the March 28, 2026 standalone MATLAB runs look stable enough to treat MATLAB output on this test volume as reproducible at the parsed-data level. If Python differs from MATLAB here, the evidence now points much more strongly to Python-side parity logic than to MATLAB run-to-run drift on this workload.
