# Python Standalone Consistency Check

Date: 2026-03-28

## Goal

Run the Python workflow three times on its own, with no imported MATLAB checkpoints, and check whether the outputs stay consistent.

## How I ran it

I used a fresh structured run root for each trial so no checkpoints or edge units could be reused:

- Session root: `D:\slavv_comparisons\20260328_142659_python_consistency`
- Input volume: `C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\data\slavv_test_volume.tif`
- Parameters source: `C:\Users\alw4834\OneDrive - The University of Texas at Austin\Documents 1\GitHub\slavv2python\comparison_output_live_parity\99_Metadata\comparison_params.normalized.json`
- Runner: `source/slavv/evaluation/comparison.py::run_python_vectorization`
- Pipeline entrypoint: `source/slavv/core/pipeline.py::SLAVVProcessor.process_image`

Important run mode notes:

- energy source was `native_python` on all three runs
- `comparison_exact_network` was `true` on all three runs because `run_python_vectorization` sets it by default for comparison-mode execution
- these were pure Python runs from the TIFF, not MATLAB-energy parity reruns

## Files involved

Each run root used the structured layout:

- `99_Metadata/`
- `02_Output/python_results/`

The key files and directories were:

- `99_Metadata/run_snapshot.json`
- `99_Metadata/validated_params.json`
- `02_Output/python_results/checkpoints/checkpoint_energy.pkl`
- `02_Output/python_results/checkpoints/checkpoint_vertices.pkl`
- `02_Output/python_results/checkpoints/checkpoint_edges.pkl`
- `02_Output/python_results/checkpoints/checkpoint_network.pkl`
- `02_Output/python_results/stages/energy/best_energy.npy`
- `02_Output/python_results/stages/energy/best_scale.npy`
- `02_Output/python_results/stages/energy/stage_manifest.json`
- `02_Output/python_results/stages/vertices/*`
- `02_Output/python_results/stages/edges/candidates.pkl`
- `02_Output/python_results/stages/edges/chosen_edges.pkl`
- `02_Output/python_results/stages/edges/units/vertex_*.pkl`
- `02_Output/python_results/stages/network/adjacency.pkl`
- `02_Output/python_results/stages/network/strands.pkl`
- `02_Output/python_results/network.casx`
- `02_Output/python_results/network.json`
- `02_Output/python_results/network.vmv`
- `02_Output/python_results/network_vertices.csv`
- `02_Output/python_results/network_edges.csv`

The canonical file inventory matched across all three runs, but many file sizes did not. The biggest differences were in:

- `checkpoint_edges.pkl`
- `checkpoint_network.pkl`
- `network.*` exports
- `stages/edges/candidates.pkl`
- many `stages/edges/units/vertex_*.pkl` files

That points to variability in edge tracing / candidate generation rather than only in final export formatting.

## Run summary

| Run | Wall time (s) | Energy (s) | Vertices (s) | Edges (s) | Network (s) | Vertices | Edges | Strands |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3725.33 | 3657.74 | 63.00 | 5.24 | 0.32 | 292 | 9 | 7 |
| 2 | 4125.74 | 4055.39 | 65.87 | 5.61 | 0.73 | 292 | 6 | 5 |
| 3 | 3612.93 | 3552.84 | 56.06 | 4.83 | 0.92 | 292 | 11 | 10 |

Timing observations:

- all three runs were dominated by the `energy` stage
- `energy` alone ranged from `3552.84s` to `4055.39s`
- `vertices`, `edges`, and `network` were small compared with `energy`

## Consistency checks

### Counts

The vertex count was stable:

- Vertices: `292` on all three runs

The graph outputs were not stable:

- Edges: `9`, `6`, `11`
- Strands: `7`, `5`, `10`

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
- edge connections: inconsistent
- edge traces: inconsistent
- network strands: inconsistent

So the saved Python outputs were not repeatable at the graph level across these three fresh runs.

### Inventory

The run layout itself was consistent:

- same canonical file inventory
- same stage/checkpoint structure

But there were size differences in 277 files across runs, especially under `stages/edges/units/` and in the final edge/network artifacts.

## Verdict

The standalone Python workflow was not consistent across these three fresh runs on this parameter set.

What stayed the same:

- file layout
- vertex positions
- vertex scales
- vertex count (`292`)

What changed:

- edge connectivity
- edge traces
- final edge count
- strand topology
- final strand count

## Practical takeaway

For native-Python runs on this workload, the instability shows up after vertices and before the final graph stabilizes. That makes the current Python standalone path materially less repeatable than the MATLAB standalone path we checked earlier. The strongest evidence points at nondeterminism somewhere in Python-side edge generation / edge selection, not at run-directory layout or export formatting.
