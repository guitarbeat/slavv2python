# MATLAB Parity Report

Date: April 9, 2026
Status: Chapter 1 closeout report

This report closes Chapter 1 of the MATLAB parity work.

For the active chapter, start with
[Shared Candidate Generation Alignment](../shared-candidate-generation/README.md)
and the maintained
[MATLAB Translation Guide](../../reference/MATLAB_TRANSLATION_GUIDE.md).

Concrete `comparison_output_*` paths below are retained Chapter 1 evidence
artifacts. They preserve the original run layout and path names used during
that work; for current staged layout guidance, use
[COMPARISON_LAYOUT.md](../../reference/COMPARISON_LAYOUT.md).

## Purpose

This report summarizes what was discovered during the recent MATLAB-vs-Python parity work in `slavv2python`, what experiments were run, where parity was achieved, where it was not achieved, and which local and external sources informed the conclusions.

The main question was:

- Why does the Python implementation still diverge from MATLAB on edge and strand outputs even when it is driven by imported MATLAB energy and exact MATLAB vertices?

## Executive Summary

- Exact vertex parity is stable on the synthetic saved-batch comparison surface.
- Edge parity improved materially during this round, but exact edge parity was not reached.
- The best retained result in this round is:
  - MATLAB/Python vertices: `110 / 110`
  - MATLAB/Python edges: `94 / 93`
  - MATLAB/Python strands: `49 / 54`
- The strongest evidence still points to a candidate-generation gap upstream of the final MATLAB-style chooser.
- Upstream MATLAB evidence indicates the modern public path is `get_edges_V300 -> get_edges_by_watershed`, not just the older `get_edges_V204/get_edges_for_vertex` frontier family.
- Two aggressive experiments were informative but were not retained because they made overall parity worse:
  - relaxing geodesic endpoint-degree gating
  - origin-owned candidate pruning

## Definitions

- `Vertex parity`: Python and MATLAB produce the same vertex count and matching vertex set under the comparison tolerances.
- `Edge parity`: Python and MATLAB produce the same undirected endpoint-pair set and, ideally, the same traces.
- `Strand parity`: Python and MATLAB produce the same network-level strand decomposition/count.
- `Candidate endpoint pair`: an undirected vertex pair that appears in the pre-choice candidate set.
- `Final endpoint pair`: an undirected vertex pair that survives the chooser and the later degree/orphan/cycle pruning.
- `Seed origin`: the vertex origin used to launch a frontier or parity candidate search.
- `Frontier candidate`: candidate discovered by the MATLAB-style frontier tracer.
- `Watershed candidate`: candidate discovered by watershed-contact logic.
- `Geodesic candidate`: candidate discovered by the parity-only local geodesic salvage stage.

## Scope And Test Surface

The main historical comparison surface used in this round was the saved MATLAB
batch rooted at:

- `comparison_output_synthetic_final_20260409_rerun`

This allowed repeated Python parity reruns without relaunching MATLAB, using imported MATLAB energy and exact MATLAB vertices.

Primary comparison input preserved from that run:

- `comparison_output_synthetic_final_20260409/00_InputFixtures/synthetic_branch_volume.tif`

## What Was Already True Before This Round

- Exact vertex parity had already been stabilized on this synthetic batch.
- The earlier frontier fix prevented invalid terminal hits from consuming origin ownership or budget too early.
- The remaining gap was already strongly suspected to be in edge candidate generation rather than in vertex extraction or MATLAB batch handling.

This aligned with the internal note in [PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md), especially the recommendation to investigate shared claim-map / watershed join behavior instead of continuing to only tune frontier termination heuristics.

## External Source Findings

External source review on April 9, 2026 suggested that the parity target had drifted toward the wrong MATLAB edge-generation surface.

Key finding:

- The public MATLAB driver uses `get_edges_V300`, and that function calls `get_edges_by_watershed`.
- This is materially different from treating parity as only a port of the older `get_edges_V204/get_edges_for_vertex` frontier family.

Evidence:

- In the local upstream checkout, `external/Vectorization-Public/vectorize_V200.m` calls `get_edges_V300(...)`.
- In the same checkout, `external/Vectorization-Public/source/get_edges_V300.m` calls `get_edges_by_watershed(...)`.
- `external/Vectorization-Public/source/get_edges_by_watershed.m` exposes shared maps such as `energy_map`, `vertex_index_map`, `pointer_map`, and branch/distance bookkeeping, which is consistent with a shared candidate-discovery and conflict-resolution surface rather than a purely local frontier launcher.

This upstream reading also fits the SLAVV paper’s framing that edge extraction operates with topological/connectivity constraints rather than only isolated local tracing.

## Web Searches Performed

Searches performed on April 9, 2026:

- `UTFOIL Vectorization-Public vectorize_V200 get_edges_V300 get_edges_by_watershed GitHub`
- `PLOS Computational Biology SLAVV vessel vectorization 2021`

Observed outcome:

- The paper source was easy to verify.
- General web search for individual MATLAB file pages was noisy, so the file-level confirmation was done from the known upstream repository paths and the local checkout.

## Experiment Log

### 1. Legacy Baseline Rerun

Run:

- `comparison_output_synthetic_final_20260409_rerun`

Summary:

- Vertices: `110 / 110`
- Edges: `94 / 85`
- Strands: `49 / 53`

Interpretation:

- Vertex parity was exact.
- Edge parity was short by `9`.
- The report pointed to candidate generation, with top missing seed origins including `9`, `49`, and `64`.

Primary evidence:

- `comparison_output_synthetic_final_20260409_rerun/03_Analysis/summary.txt`

### 2. Watershed Candidate Pivot (`all_contacts`)

Idea:

- Stop treating watershed only as a late supplement and instead admit valid watershed-contact candidates directly into the MATLAB-style chooser.

Comparison:

- `legacy_supplement`: `110 / 85 / 53`
- `all_contacts`: `110 / 84 / 48`

Interpretation:

- Edge count did not improve.
- Strand count improved meaningfully.
- Candidate coverage improved slightly.
- This was evidence that watershed-contact behavior mattered, but it was not enough by itself.

Primary evidence:

- `comparison_output_synthetic_final_20260409_allcontacts_manual/03_Analysis/manual_summary.txt`

### 3. Geodesic Salvage

Idea:

- Add a parity-only bounded geodesic stage after frontier plus watershed candidate generation to recover local missing connections in frontier-deficit areas.

Run:

- `comparison_output_synthetic_final_20260409_geodesic_salvage`

Summary:

- Vertices: `110 / 110`
- Edges: `94 / 92`
- Strands: `49 / 55`

Interpretation:

- This was a real edge improvement over the baseline rerun.
- Candidate MATLAB-covered endpoint pairs improved.
- Strand count got worse.
- Seed `9` improved materially.
- Seed `64` remained the top missing origin in the retained run.

Primary evidence:

- `comparison_output_synthetic_final_20260409_geodesic_salvage/summary.txt`

### 4. Relaxed Geodesic Endpoint-Degree Gate

Idea:

- Allow a seed origin to propose geodesic replacements even when the current candidate set had already saturated its per-vertex degree budget.

Run:

- `comparison_output_synthetic_final_20260409_geodesic_relaxed`

Summary:

- Vertices: `110 / 110`
- Edges: `94 / 99`
- Strands: `49 / 62`

Interpretation:

- This proved something important: the missing pairs around difficult origins could be generated.
- In this run, origin `49` gained geodesic candidates like `(49, 71)` and `(49, 94)`.
- Origin `64` gained geodesic candidates like `(64, 76)` and `(64, 96)`.
- But overall parity got worse because the final network overshot badly and strand count exploded.

Verdict:

- Informative, but not retained.

Primary evidence:

- `comparison_output_synthetic_final_20260409_geodesic_relaxed/summary.txt`

### 5. Origin-Owned Candidate Pruning

Idea:

- Prefer the seed's own frontier/geodesic candidates and trim away foreign-origin noise when a seed already had plausible owned candidates.

Run:

- `comparison_output_synthetic_final_20260409_origin_owned`

Summary:

- Vertices: `110 / 110`
- Edges: `94 / 87`
- Strands: `49 / 53`

Interpretation:

- This over-corrected.
- It reduced candidate coverage too aggressively and moved the result farther from MATLAB.

Verdict:

- Informative, but not retained.

Primary evidence:

- `comparison_output_synthetic_final_20260409_origin_owned/summary.txt`

### 6. Retained Narrow Improvement: Wider Geodesic Neighbor Search Only

Idea:

- Keep the geodesic salvage stage, but only retain the safer part of the experiment: widen the geodesic nearest-neighbor search from `6` to `10`.
- Back out the noisy endpoint-degree relaxation and back out the origin-owned pruning experiment.

Run:

- `comparison_output_synthetic_final_20260409_k10only`

Summary:

- Vertices: `110 / 110`
- Edges: `94 / 93`
- Strands: `49 / 54`

Interpretation:

- This is the best retained aggregate result from this round.
- It improves on the earlier retained geodesic-salvage baseline:
  - edges: `92 -> 93`
  - strands: `55 -> 54`
- Exact edge parity is still not reached.
- The remaining gap still points to candidate generation, with origin `64` still the top missing seed in the retained run.

Primary evidence:

- `comparison_output_synthetic_final_20260409_k10only/summary.txt`

## Where Parity Was Found

### Stable Parity

- Vertex count parity was exact in every serious retained run on the saved synthetic batch.
- Vertex parity remained exact through the watershed candidate pivot and geodesic salvage additions.
- The imported MATLAB-energy parity workflow is stable enough to support repeated diagnostic reruns.

### Partial Parity

- The final retained edge count is now off by only `1` on the saved synthetic batch.
- Several local candidate-generation changes clearly improved edge counts without breaking the broader test suite.

## Where Parity Was Not Found

- Exact edge endpoint-pair parity was not reached.
- Exact strand parity was not reached.
- Exact trace parity was not reached.
- Some difficult seed origins still remain under-covered in the retained candidate set.
- The retained runs still show a mismatch between candidate-generation improvements and final-network topology quality.

## Seed-Origin Findings

### Seed 9

- Geodesic salvage helped this origin the most in the informative runs.
- It showed that local missing-candidate recovery is possible.
- However, the recovered pairs still did not fully match MATLAB’s incident pairs.

### Seed 49

- In the relaxed geodesic experiment, this origin finally produced geodesic candidates such as `(49, 71)` and `(49, 94)`.
- That was strong evidence that the candidate-generation gap is not purely geometric impossibility.
- But the relaxed run overshot globally, so the mechanism still needs a chooser-compatible formulation.

### Seed 64

- This remains the hardest retained origin.
- In the retained `k10only` run it was still the top missing seed origin.
- The relaxed experiment showed that local geodesic candidates like `(64, 76)` and `(64, 96)` can be found, but the retained formulation does not keep them.
- This makes origin `64` the clearest next diagnostic target.

## What The Experiments Suggest Technically

### 1. The Current Gap Is Upstream Of The Final Chooser

The final chooser still matters, but the stronger evidence is that Python often never presents the same candidate universe that MATLAB eventually resolves.

Why this conclusion remains strong:

- Vertex parity is exact.
- The chooser already produces deterministic, test-covered behavior.
- When candidate-generation is widened aggressively, missing MATLAB-like pairs can appear.
- When those candidate-generation experiments are removed, parity falls back toward the earlier deficit pattern.

### 2. Watershed Behavior Is Necessary But Not Sufficient

The `all_contacts` pivot improved candidate coverage and strand behavior but did not fix edge parity alone.

### 3. Geodesic Recovery Can Help, But It Must Be Chooser-Compatible

Geodesic salvage clearly recovers useful local structure in some cases, but naive relaxation creates too many extras.

### 4. The Modern MATLAB Target Is More Shared-State / Watershed-Like Than The Older Frontier Mapping Suggested

This is the main conceptual correction from the source review.

## Retained Code State After This Round

The retained code changes from this parity pass include:

- parity watershed candidate mode support
- parity geodesic salvage support
- the retained geodesic nearest-neighbor default widened to `10`

Relevant files:

- `source/slavv/core/edge_candidates.py`
- `source/slavv/utils/validation.py`
- `tests/unit/core/test_candidate_diagnostics.py`
- `source/slavv/parity/metrics.py`
- `source/slavv/parity/reporting.py`

## Verification Performed

The retained final code state passed:

- targeted `ruff check` on touched files
- focused parity and edge pytest suites
- `python -m pytest -m "unit or integration"`

Latest broad suite result on the retained final state:

- `298 passed, 5 skipped, 15 deselected`

## Recommended Next Steps

### Highest-Leverage Technical Next Step

Investigate why origin `64` still fails in the retained candidate path even though the relaxed experiment proved nearby geodesic candidates can exist.

Concrete next questions:

- Why do `(64, 76)` and `(64, 96)` appear only in the relaxed experiment and not in the retained one?
- Is MATLAB effectively allowing a temporary over-budget candidate state before later pruning?
- Is the remaining gap actually in candidate admission, or in how shared maps / claim ownership are updated while candidates are being built?

### Secondary Next Step

Compare the retained Python parity candidate bookkeeping more directly against the shared-map behavior implied by MATLAB `get_edges_by_watershed`.

### What Not To Repeat Blindly

- Do not keep the relaxed endpoint-degree bypass as-is.
- Do not keep the origin-owned pruning as-is.

Both were useful experiments, but both worsened aggregate parity.

## Sources

The local evidence links below intentionally preserve the original Chapter 1
artifact paths.

### Local Evidence

- [comparison_output_synthetic_final_20260409_rerun/03_Analysis/summary.txt](../../../comparison_output_synthetic_final_20260409_rerun/03_Analysis/summary.txt)
- [comparison_output_synthetic_final_20260409_allcontacts_manual/03_Analysis/manual_summary.txt](../../../comparison_output_synthetic_final_20260409_allcontacts_manual/03_Analysis/manual_summary.txt)
- [comparison_output_synthetic_final_20260409_geodesic_salvage/summary.txt](../../../comparison_output_synthetic_final_20260409_geodesic_salvage/summary.txt)
- [comparison_output_synthetic_final_20260409_geodesic_relaxed/summary.txt](../../../comparison_output_synthetic_final_20260409_geodesic_relaxed/summary.txt)
- [comparison_output_synthetic_final_20260409_origin_owned/summary.txt](../../../comparison_output_synthetic_final_20260409_origin_owned/summary.txt)
- [comparison_output_synthetic_final_20260409_k10only/summary.txt](../../../comparison_output_synthetic_final_20260409_k10only/summary.txt)
- [PARITY_FINDINGS_2026-03-27.md](PARITY_FINDINGS_2026-03-27.md)
- `external/Vectorization-Public/vectorize_V200.m`
- `external/Vectorization-Public/source/get_edges_V300.m`
- `external/Vectorization-Public/source/get_edges_by_watershed.m`

### External Sources

- [Segmentation-Less, Automated, Vascular Vectorization (PLOS Computational Biology, October 8, 2021)](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1009451)
- [UTFOIL/Vectorization-Public](https://github.com/UTFOIL/Vectorization-Public)
- [raw `get_edges_by_watershed.m`](https://raw.githubusercontent.com/UTFOIL/Vectorization-Public/master_pullRQ/source/get_edges_by_watershed.m)

## Bottom Line

This round did not achieve full MATLAB network parity, but it did produce a clearer map of the problem:

- exact vertex parity is stable
- the best retained result is now `94 vs 93` edges
- the remaining gap is still candidate-generation driven
- the modern upstream MATLAB target is more watershed/shared-state oriented than the older frontier-only mapping implied
- origin `64` remains the most actionable next debugging surface
