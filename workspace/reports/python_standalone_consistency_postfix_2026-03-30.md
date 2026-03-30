# Python Standalone Consistency Check, Post-Fix

Date: 2026-03-30

## Goal

Run the Python workflow three times on its own after the deterministic edge-padding fix, and check whether the outputs stay consistent.

## How I ran it

I used a fresh structured run root for each trial so no checkpoints or edge units could be reused:

- Session root: `D:\slavv_comparisons\20260330_python_consistency_postfix`
- Input volume: `C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\data\slavv_test_volume.tif`
- Parameters source: `C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\comparison_output_live_parity\99_Metadata\comparison_params.normalized.json`
- Runner: `source/slavv/evaluation/comparison.py::run_python_vectorization`
- Pipeline entrypoint: `source/slavv/core/pipeline.py::SLAVVProcessor.process_image`

Important run mode notes:

- energy source was `native_python` on all three runs
- `comparison_exact_network` was `true` on all three runs because `run_python_vectorization` sets it by default for comparison-mode execution
- `run_03` was resumed from an interrupted first attempt, but it completed successfully and produced the same final outputs as the other two runs

## Files involved

Each run root used the structured layout:

- `99_Metadata/`
- `02_Output/python_results/`

The key files and directories were:

- `99_Metadata/run_snapshot.json`
- `99_Metadata/validated_params.json`
- `02_Output/python_results/python_comparison_parameters.json`
- `02_Output/python_results/checkpoints/checkpoint_energy.pkl`
- `02_Output/python_results/checkpoints/checkpoint_vertices.pkl`
- `02_Output/python_results/checkpoints/checkpoint_edges.pkl`
- `02_Output/python_results/checkpoints/checkpoint_network.pkl`
- `02_Output/python_results/stages/energy/best_energy.npy`
- `02_Output/python_results/stages/energy/best_scale.npy`
- `02_Output/python_results/stages/energy/resume_state.json`
- `02_Output/python_results/stages/energy/stage_manifest.json`
- `02_Output/python_results/stages/vertices/candidates.pkl`
- `02_Output/python_results/stages/vertices/cropped_candidates.pkl`
- `02_Output/python_results/stages/vertices/chosen_mask.pkl`
- `02_Output/python_results/stages/vertices/stage_manifest.json`
- `02_Output/python_results/stages/edges/candidates.pkl`
- `02_Output/python_results/stages/edges/chosen_edges.pkl`
- `02_Output/python_results/stages/edges/units/vertex_*.pkl`
- `02_Output/python_results/stages/edges/stage_manifest.json`
- `02_Output/python_results/stages/network/adjacency.pkl`
- `02_Output/python_results/stages/network/strands.pkl`
- `02_Output/python_results/stages/network/stage_manifest.json`
- `02_Output/python_results/network.casx`
- `02_Output/python_results/network.json`
- `02_Output/python_results/network.vmv`
- `02_Output/python_results/network_vertices.csv`
- `02_Output/python_results/network_edges.csv`

The canonical file inventory matched across all three runs, with `317` files in each run root.

## Run summary

| Run | Wall time (s) | Energy (s) | Vertices (s) | Edges (s) | Network (s) | Vertices | Edges | Strands | Candidate edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3301.46 | 3242.00 | 55.38 | 4.39 | 0.45 | 292 | 8 | 8 | 711 |
| 2 | 3281.29 | 3221.25 | 55.30 | 5.21 | 0.28 | 292 | 8 | 8 | 711 |
| 3 | 3385.70 | 3325.07 | 60.67 | 0.00 | 0.00 | 292 | 8 | 8 | 711 |

Timing observations:

- all three runs were dominated by the `energy` stage
- the final graph stages were tiny compared with energy
- `run_03` was resumed, so its final stage timestamps are coarse and the edge/network stage durations rounded down to zero in the run snapshot

## Consistency checks

### Counts

The output counts were stable:

- Vertices: `292` on all three runs
- Edges: `8` on all three runs
- Strands: `8` on all three runs

### Parsed output digests

I hashed the normalized saved outputs from checkpoints:

- vertex positions
- vertex scales
- edge connections
- edge traces
- network strands

Digest results:

- vertex positions: consistent
- vertex scales: consistent
- edge connections: consistent
- edge traces: consistent
- network strands: consistent

Concrete digest values were identical across all three runs:

- vertex digest: `27f14bd67c5573124a493f203bd9db499ee21fd301f284775166d8d0e849084c`
- edge digest: `4174d958303b559964434ce7ccedf9e066b7be182e9b91fdc2722fc9effbbe10`
- network digest: `8e438e82efee45453fdf7ba6b493a758044bc45a79701eff95755ebb40afefd3`

### Inventory

The run layout itself was consistent:

- same canonical file inventory
- same stage/checkpoint structure
- same number of produced files

## Verdict

The standalone Python workflow is now consistent across these three fresh runs on this parameter set.

What stayed the same:

- file layout
- vertex positions
- vertex scales
- vertex count (`292`)
- edge count (`8`)
- strand count (`8`)

What changed:

- only wall-clock runtime, mostly because energy dominates and the resumed run had different wall timing

## Practical takeaway

The deterministic edge-padding fix removed the previous Python-only graph drift. On this workload, the Python standalone path now behaves repeatably at the vertex, edge, and network levels, which puts it in much better shape for parity comparisons.
