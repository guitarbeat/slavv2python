# Parity Findings

This note captures what the fresh March 27, 2026 MATLAB/Python reruns taught us
so the next parity phase can start from verified facts instead of chat history.

This file is intentionally the evidence log, not the workflow plan.

## Rapid Recall

- Still true now:
  - imported-MATLAB Python reruns are repeatable on the current machine
  - vertex parity is exact on the imported-MATLAB surface
  - stage-isolated `network` parity is exact when exact MATLAB `edges` are
    imported and Python reruns from `network`
  - the main remaining blocker is edge generation, not generic downstream
    network assembly
- Current failure shape:
  - Python still misses some MATLAB endpoint pairs before cleanup
  - many extra frontier edges cluster around the same shared vertices as the
    missing MATLAB pairs
- Best next code surface:
  - `source/slavv/core/edge_candidates.py`
- Best companion docs:
  - [PARITY_HUB.md](PARITY_HUB.md)
  - [edge_parity_plan.md](edge_parity_plan.md)
  - [MATLAB_PARITY_AUDIT_CHECKLIST.md](MATLAB_PARITY_AUDIT_CHECKLIST.md)

## Read This File When

- you want verified evidence rather than the current plan
- you need to remember which experiments helped and which ones failed
- you want the strongest artifact-level clues before changing code

Follow-up as of April 1, 2026:

- Fresh comparison runs now persist output-root preflight decisions to
  `99_Metadata/output_preflight.json`.
- MATLAB rerun semantics and failure summaries now persist to
  `99_Metadata/matlab_status.json` and `99_Metadata/matlab_failure_summary.json`.
- For current staged layout and recommended local output-root guidance, start
  with [COMPARISON_LAYOUT.md](../../reference/core/COMPARISON_LAYOUT.md).

Further follow-up as of April 6, 2026:

- Imported-MATLAB Python-only reruns are now repeatable on the current machine:
  three fresh reruns produced identical Python edge counts, strand counts,
  chosen endpoint-pair hashes, and chosen trace hashes.
- The current parity gap is therefore systematic rather than stochastic.
- Later skip-MATLAB threshold trials showed that global watershed metric
  thresholds are not a clean path to parity:
  - `parity_watershed_metric_threshold = -90.0` improved edge count but
    regressed strands.
  - `parity_watershed_metric_threshold = -50.0` improved candidate coverage but
    worsened final edge and strand parity.
- An analysis-only refresh now records provenance-aware conflict outcomes and
  chosen-edge source quality in the staged `summary.txt`, so future parity runs
  can answer whether frontier or watershed is carrying the weak long extras
  without ad hoc scripts.

Further follow-up as of April 7, 2026:

- A stage-isolated parity probe showed that Python `network` assembly already
  reaches exact parity when given exact MATLAB `edges`, but only when
  `comparison_exact_network=True` is enabled during the `network` rerun.
- The same stage-isolated rerun without that parity-mode flag produced a false
  negative (`364` Python strands vs `682` MATLAB strands), so replaying the
  network stage depends on more than the current normalized params snapshot.
- That materially changes the convergence plan: the remaining primary blocker
  is edge generation, not generic downstream network assembly.
- The comparison CLI now supports this probe directly through
  `--python-parity-rerun-from network`, and comparison-mode Python execution
  now forces the parity-specific network path instead of only setting it when
  the incoming params omit the key.

Concrete run roots cited below are retained evidence artifacts, not standing
source-of-truth paths for new work. For current staged layout expectations,
start from [COMPARISON_LAYOUT.md](../../reference/core/COMPARISON_LAYOUT.md).

## Fresh Baselines

### Canonical parity rerun

- Historical run root: `comparison_output_live_parity`
- Source run root: `D:\slavv_comparisons\20260327_150656_clean_parity`
- MATLAB batch: `batch_260327-150756`
- Python energy source: `matlab_batch_hdf5`
- Python start point: imported MATLAB `energy` and `vertices`, rerun from `edges`

Observed parity status from `03_Analysis/comparison_report.json`:

- `vertices_exact = true`
- `edges_exact = false`
- `strands_exact = false`
- `passed = false`

Observed counts:

- MATLAB vertices: `1682`
- Python vertices: `1682`
- MATLAB edges: `1379`
- Python edges: `1236`
- MATLAB strands: `682`
- Python strands: `592`

Observed edge diagnostics:

- Candidate traces: `2150`
- Candidate endpoint pairs: `1493`
- MATLAB endpoint pairs: `1379`
- Matched MATLAB endpoint pairs in raw Python candidates: `889`
- Missing MATLAB endpoint pairs before cleanup: `490`
- Extra Python candidate endpoint pairs: `604`
- Conflict rejections: `244`
- Degree/orphan/cycle pruning: `5 / 3 / 5`
- First missing candidate endpoint pair sample: `[2, 356]`

Interpretation:

- Exact vertex parity is real and stable under imported MATLAB energy/vertices.
- The dominant blocker is still upstream of final cleanup because Python never
  generates many MATLAB endpoint pairs in the raw candidate set.
- Downstream cleanup still matters, but it cannot recover the missing `490`
  MATLAB endpoint pairs that never appear as candidates.

### Independent full Python rerun

- Run root: `D:\slavv_comparisons\20260327_161610_clean_python_full`
- MATLAB comparison target: the same fresh MATLAB batch above
- Python start point: native Python run from `energy`

Observed counts:

- Python vertices: `292`
- Python edges: `5`
- Python strands: `5`

Interpretation:

- Native-Python upstream generation on this volume is still far from MATLAB.
- The imported-energy parity path remains the correct surface for exact
  edge/strand work.
- Full native parity should not be conflated with the current exact-network
  parity effort.

## Operational Lessons

- Fresh MATLAB reruns failed repeatedly on `C:` with HDF5 write errors during
  the `energy` stage.
- The same command succeeded immediately on `D:` with ample free space.
- The failure mode was environmental, not a regression in the orchestration
  code or parameter file.

Implication:

- Future live MATLAB reruns should prefer a high-free-space local path such as
  `D:\slavv_comparisons\...` and only copy the finished parity run back into the
  repo when promotion is needed.

## Code-Path Lessons

### What is working

- `source/slavv/io/matlab_bridge.py` correctly imports MATLAB HDF5 energy and
  curated vertices into pipeline-compatible checkpoints.
- `comparison_exact_network=True` correctly routes MATLAB-energy parity runs
  through the parity-specific edge cleanup and network assembly path.
- Shared parity-aware network assembly in `source/slavv/core/graph.py` removed
  earlier fresh-vs-resume divergence.
- A stage-isolated replay with imported MATLAB `edges` confirms that the
  parity-specific Python `network` stage can now match MATLAB exactly on the
  canonical imported-MATLAB surface.
- The stage-isolated replay is now a supported comparison workflow rather than
  an ad hoc probe, which makes it practical to keep `network` as a standing
  downstream gate while edge-generation work continues.
- Candidate endpoint coverage diagnostics are useful and should remain a primary
  regression signal.

### What is not the dominant blocker

- Vertex matching is no longer the main issue on the parity path.
- Generic downstream network assembly is not the main issue on the parity path
  when exact MATLAB `edges` are supplied and parity-mode `network` assembly is
  enabled.
- Simple short-range terminal attachment tweaks did not improve live parity in
  earlier experiments and sometimes made it worse.
- Pure cleanup-order tuning is not enough by itself because too many MATLAB edge
  endpoint pairs are absent before cleanup starts.
- The shorter-trace tie-break in Python is not an obvious mismatch by itself;
  MATLAB's `clean_edge_pairs.m` also pre-sorts equal-energy candidates by
  trajectory length before unique-pair cleanup.

### New April 6 code-level signal

Using the live rerun artifact `comparisons/20260406_live_parity_retest`:

- Raw frontier candidate pairs:
  - `892` matched MATLAB
  - `615` extra Python
- Raw watershed candidate pairs:
  - `81` matched MATLAB
  - `952` extra Python
- Final chosen frontier edges:
  - `847` matched MATLAB
  - `391` extra Python
- Final chosen watershed edges:
  - `47` matched MATLAB
  - `140` extra Python

Interpretation:

- Watershed supplementation is noisy, but it is not the whole explanation for
  the final parity gap.
- The majority of final extra Python edges are still frontier-sourced.
- Extra frontier candidates also tend to be weaker and longer than matched
  frontier candidates, which points back toward frontier discovery semantics or
  provenance-aware cleanup rather than another blunt watershed filter.

### April 6 conflict-provenance refresh

Using the analysis-only refresh artifact
`comparisons/20260406_conflict_provenance_refresh`:

- Conflict rejects by source:
  - `254` frontier
  - `741` watershed
- Conflict blockers by source:
  - `868` frontier
  - `326` watershed
- Conflict source pairs:
  - `236` frontier -> frontier
  - `24` frontier -> watershed
  - `632` watershed -> frontier
  - `302` watershed -> watershed
- Chosen frontier edges:
  - `847` matched MATLAB
  - `391` extra Python
  - median energy `-225.4` matched vs `-152.3` extra
  - median trace length `11` matched vs `16` extra
- Chosen watershed edges:
  - `47` matched MATLAB
  - `140` extra Python
  - median energy `-118.6` matched vs `-75.3` extra
  - median trace length `17` matched vs `21` extra
- Strongest extra frontier overlap:
  - `281/391` extra frontier edges share at least one vertex with a missing
    MATLAB endpoint pair
  - `18/20` of the strongest extra frontier edges share a vertex with at least
    one missing MATLAB endpoint pair
  - `41/50` of the strongest extra frontier edges do the same
- Shared-vertex missing-pair candidate hits:
  - vertex `359`: `0/4`
  - vertex `1283`: `0/4`
  - vertex `866`: `0/4`
- Negative tracer experiment:
  - a literal "claim the full recovered path, including the shared
    root/bifurcation voxel" change improved one local shared-vertex signal but
    worsened end-to-end parity on the imported-MATLAB rerun
  - baseline refresh: `1379` MATLAB edges vs `1425` Python, `682` MATLAB
    strands vs `681` Python, candidate endpoint pairs `2540/973/406`
    candidate/matched/missing
  - full-path-claim experiment: `1379` MATLAB edges vs `1306` Python,
    `682` MATLAB strands vs `620` Python, candidate endpoint pairs
    `2620/884/495`
  - conclusion: blanket full-path ownership is not the right next fix; it
    reduces extra frontier edges, but it also suppresses too many
    MATLAB-matching frontier survivors

Interpretation:

- Watershed candidates are frequently being rejected because frontier
  candidates painted the conflicting voxels first.
- That does not mean watershed is the main remaining bug. The final extra edge
  set is still dominated by frontier edges, and those extra frontier survivors
  remain weaker and longer than the matched frontier set.
- The strongest frontier extras are usually attached to the same vertices as
  missing MATLAB pairs, which points toward wrong local partner choice or local
  claim ordering rather than a purely global threshold problem.
- For the worst shared vertices examined so far, the missing MATLAB pairs never
  appear in the Python candidate set at all. That makes the current leading
  diagnosis an upstream frontier-discovery mismatch rather than a cleanup-only
  problem.
- Some extra frontier edges touching those vertices also come from neighboring
  origins, which suggests the remaining mismatch is not purely a per-origin
  budget problem; neighborhood-level frontier behavior matters too.
- The failed full-path-claim experiment reinforces that diagnosis. The next
  pass should aim for selective claim-ordering behavior around shared vertices,
  not a global ownership change after every terminal hit.
- The next parity pass should therefore prioritize frontier candidate semantics
  and frontier claim ordering before adding more global source-preference or
  threshold rules.
- Operator note: the skip-MATLAB reuse loop is sensitive to input provenance.
  Using the canonical absolute input path is the safest way to ensure the
  wrapper recognizes an otherwise reusable completed MATLAB batch.
- Replay note: stage-isolated `network` probes also need explicit parity-mode
  network semantics. Replaying from `99_Metadata/comparison_params.normalized.json`
  alone was not enough to reproduce the exact-network path in the April 7, 2026
  probe.

Artifact-level read on the worst shared vertices from the staged April 6, 2026
trial:

- The missing MATLAB pairs are not missing because those neighborhoods are
  inactive.
- At vertex `359`, the origin itself only produced `[359, 181]` from the
  frontier path, while the chosen extra frontier edges touching `359` came from
  neighboring origins `1180` and `1568`.
- At vertex `866`, the origin recorded `terminal_frontier_hit = 3` but only
  emitted `[866, 885]` into the candidate manifest.
- At vertex `1283`, the origin emitted `[1283, 1134]`, `[1283, 768]`, and
  watershed pair `[1283, 1659]`, while the missing MATLAB pairs still never
  entered the candidate manifest.
- The missing partner vertices themselves were active elsewhere:
  - `1023` had chosen pair `[394, 1023]`
  - `1203` had chosen pairs `[1309, 1203]` and `[1466, 1203]`
  - `1284` had chosen pairs `[1180, 1284]` and `[1284, 1272]`
  - `95` had chosen pairs `[95, 285]` and `[95, 1134]`
  - `542` had chosen pairs `[542, 768]` and `[1285, 542]`
- That points to local partner substitution around shared neighborhoods as the
  first concrete artifact-level divergence, which keeps frontier ownership,
  bifurcation handling, and claim ordering ahead of any new global threshold
  work.
- Geometry alone does not fully explain the substitutions:
  - around `866`, the chosen alternatives (`885`, `810`) are clearly closer
    than the missing MATLAB partners
  - around `1283`, missing partner `1319` is about as close as chosen partners
    `1659` and `1134`
- That makes a pure nearest-neighbor explanation too weak; the remaining gap is
  more likely to sit in local path formation, terminal ownership, or
  bifurcation-level claim ordering.

### Important bug found during the fresh rerun

The independent Python-from-`energy` rerun initially crashed because MATLAB
settings were loaded as integer-like floats such as
`number_of_edges_per_vertex = 4.0`, and the tracer later used that value as an
array shape.

The validation layer now normalizes integral MATLAB-style scalars before the
pipeline uses them.

Implication:

- When replaying MATLAB batch settings into Python, assume MATLAB numeric values
  may arrive as floats even when the Python pipeline expects integers.

## Highest-Leverage Next Phase

### Primary target

Focus on upstream candidate generation in `source/slavv/core/edge_candidates.py`,
especially where the parity tracer diverges from MATLAB before
`_choose_edges_matlab_style` runs.

The key question is:

- Why do only `889` of `1379` MATLAB endpoint pairs appear in the raw Python
  candidate set under imported MATLAB energy and exact vertices?

### Most promising direction

Port more of MATLAB's shared edge-discovery / watershed-join behavior rather
than continuing to tune local per-origin frontier termination heuristics.

Why:

- The current parity frontier is already good enough to hit exact vertices and a
  large subset of MATLAB edge pairs.
- The remaining failures look like missing cross-vertex/basin connectivity, not
  just local edge-choice mistakes.
- Earlier "near hit" or frontier-attachment experiments did not close the gap.

### Concrete next investigations

1. Compare MATLAB `get_edges_by_watershed.m` and the Python parity path for
   shared claim-map behavior, join-on-contact semantics, and how terminal
   ownership is resolved across origins.
2. Add a diagnostic artifact that records candidate endpoint pairs grouped by
   origin vertex so missing MATLAB pairs can be traced back to the exact origin
   vertices that never discover them.
3. For a few missing MATLAB endpoint pairs such as `[2, 356]`, inspect:
   - the MATLAB edge trace
   - the corresponding local energy neighborhood
   - whether Python launches from the same origin(s)
   - whether Python enters the same basin but fails to claim/merge it
4. Keep cleanup work gated to `comparison_exact_network=True` unless a change is
   clearly beneficial for default native-Python behavior too.
5. Use the landed provenance-aware conflict diagnostics to compare the strongest
   extra frontier edges against nearby missing MATLAB endpoint pairs and decide
   whether the next fix belongs in frontier discovery, claim ordering, or final
   conflict resolution.

## Guardrails For The Next Pass

- Treat `comparison_output_live_parity` as a retained historical evidence
  artifact, not as the canonical source of truth for new work.
- Do not use old failed scratch runs on `C:` as evidence.
- Prefer fresh MATLAB reruns on `D:` and promote only after the full run
  completes cleanly.
- Use `candidate_endpoint_coverage` as the first regression screen before
  chasing exact final edge/strand diffs.
- Expect native Python-from-`energy` to remain non-parity for now; do not read
  that gap as proof that the imported-energy parity work regressed.
