# MATLAB edge dumps (scratch)

Labeled names. Do not invent a second dump with “canonical” in the filename
unless the volume is full `180709_E`.

| File | Volume | What |
|---|---|---|
| `raw_full_candidates.mat` | full | Raw `get_edges_by_watershed` (84,650 pairs). Compare to Python `candidates.pkl`. |
| `global_presort_candidates.mat` | full | Same pairs after MATLAB `sort_edges`. |
| `raw_watershed_candidates.mat` | crop | Raw crop (19,225). |
| `raw_watershed_candidates_canonical.mat` | **crop** | Misnamed. Maps are `(64, 256, 256)`, not full. |
| `frontier_trace.jsonl` | crop era | Historical frontier traces. Not the current residual class. |

See `docs/solutions/parity/raw-vs-final-candidate-compare.md`.
