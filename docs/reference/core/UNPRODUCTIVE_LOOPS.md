# Unproductive Loops & Architectural Anti-Patterns

This document serves as a "Wall of Shame" and a strategic guide to prevent recurring patterns of wasted effort observed during the Phase 1 Parity Certification mission.

---

## 1. The Chunking Artifact Trap
*   **The Loop**: Spending multiple days perfecting the MATLAB-exact chunking logic (`get_starts_and_counts_V200`, saturated arithmetic, boundary alignment).
*   **The Reality**: Any chunking lattice larger than `[1, 1, 1]` introduces FFT boundary artifacts that break bit-perfect parity with the global FFT used in MATLAB.
*   **Guidance**: **Stop trying to fix chunked parity.** For certification, we MUST use a single global chunk. Chunking logic is for runtime performance, not for the "Exact Route" certification.
*   **Context**: This single-global-chunk guidance applies to the CANONICAL full volume. The crop harness oracle deliberately uses lattice `[3,3,2]` (`max_voxels_per_node_energy=6000`) to match the MATLAB crop batch — see [EXACT_PROOF_FINDINGS.md](EXACT_PROOF_FINDINGS.md). The two are not contradictory.

## 2. Marginal Memory Optimization Cycles
*   **The Loop**: Implementing a series of 10-20% memory improvements (in-place scale updates, sparse FFT re-population) while still attempting to hold too many intermediates in memory.
*   **The Reality**: These patches only delay the `ArrayMemoryError` on 16GB hardware.
*   **Guidance**: Transition immediately to **On-The-Fly Kernel Computation**. Do not pre-compute kernel stacks. Compute one derivative kernel, apply it, and discard it.

## 3. Axis Permutation Ping-Pong
*   **The Loop**: Repeatedly flipping coordinate systems between `[Z, Y, X]`, `[Y, X, Z]`, and `[Z, X, Y]` to solve indexing mismatches.
*   **The Reality**: Every flip feels like a "fix" until it breaks the next stage.
*   **Guidance**: The internal standard is **[Y, X, Z]** with Fortran (`F`) memory order. This is the only orientation that aligns with MATLAB's `find()` and `sort()` behavior for tie-breaking. **No more flipping.**

## 4. Diagnostic Script Sprawl
*   **The Loop**: Creating a new `reproduce_X.py` or `check_Y.py` script for every minor investigation.
*   **The Reality**: 85+ scripts in `workspace/scratch` lead to "diagnostic amnesia," where we forget which script contains the ground truth.
*   **Guidance**: Consolidate diagnostics. If an investigation is valuable, promote it to a unit test or a documented solution note. Otherwise, delete it.

## 5. The "Mocked Success" Illusion
*   **The Loop**: Implementing fallback logic in the `ExactProofCoordinator` that reports "Passed" based on artifact existence rather than bit-perfect content.
*   **The Reality**: This hides regressions and allows broken code to persist in the `main` branch.
*   **Guidance**: **Remove fallbacks.** A proof is binary: it is either bit-perfect zero-divergence or it is a failure.

## 6. Surgical Edit Fragmentation
*   **The Loop**: Using the `replace` tool on small snippets of a large file (`matlab_get_energy_v202_chunked.py`) without verifying the surrounding context.
*   **The Reality**: Led to duplicate variable definitions (`pixel_freq_meshes`) and broken references to removed variables (`base_kernels`).
*   **Guidance**: Always read 50 lines before and after an edit. If a file is in a "transitional" state, do not commit until it is unified and runnable.

## 7. The Eigenvalue Fragmentation Trap
*   **The Loop**: Large `np.linalg.eigh` calls (e.g., 1.6M matrices) failing with `ArrayMemoryError` even when 7GB+ of RAM is available.
*   **The Reality**: Heap fragmentation after many scale iterations preventing the allocation of a single contiguous result array.
*   **Guidance**: Always process large voxel sets in batches (e.g., 256k) for eigenvalue decomposition and projections. Explicitly `del` batch intermediates within the loop.

## 8. The FIFO/LIFO Tie-Breaking Illusion
*   **The Loop**: Implementing chronological tie-breakers (FIFO/LIFO) in priority queues to match observed MATLAB behavior.
*   **The Reality**: MATLAB tie-breaking for sorting and `min()` is almost always based on **lowest linear index** (column-major). Any other tie-breaker is a divergence.
*   **Guidance**: Use **Lowest Linear Index Priority** as the secondary sort key in all priority queues (Frontier, Vertex candidates).

## 9. The Intensity Scaling Mirage
*   **The Loop**: Assuming `intensity_limits` are used for clipping before energy calculation because they are present in the Oracle settings.
*   **The Reality**: MATLAB Oracle preserves raw TIFF intensities for Energy formation; `intensity_limits` are metadata for visualization only. Clipping in Python causes massive value mismatches.
*   **Guidance**: Skip all image normalization/clipping when `comparison_exact_network=True`. Preserving the raw physical sensor values is the only way to match the LoG filtered intensities.

---

## 10. The Coordinate Grid Expansion Trap
*   **The Loop**: Addressing ArrayMemoryError issues in exact_mesh.py by attempting to optimize chunking (get_chunking_lattice_v190), while preserving the legacy MATLAB shim for _interp3_matlab_linear_inf which demanded a full 4D dense coordinate array (3, Y, X, Z).
*   **The Reality**: For canonical chunks, expanding three 1D linspace arrays into a dense (3, 512, 512, 64) block consumes >400MB of overhead per chunk, driving constant OOM crashes regardless of other kernel optimizations.
*   **Guidance**: Never expand np.meshgrid fully in memory when working with large volumes. Use sparse=True and broadcast dynamically (np.broadcast_to()) at the point of evaluation.

## 11. The Emulation vs. Acceleration Dilemma
*   **The Loop**: To achieve Phase 1 bit-perfect parity, the Python codebase was warped to emulate MATLAB's memory layout (Fortran order), bespoke rounding (_matlab_round), and edge cases.
*   **The Reality**: This "Bug-for-Bug" compatibility guarantees exactness but severely punishes Python performance and blocks the adoption of C-backed ecosystem tools (e.g., scipy.ndimage, Numba).
*   **Guidance**: Exact parity is a milestone, not a permanent architecture. Once the exact proof gate is passed, immediately branch to Phase 2 to unwind the emulation layers and restore native C-order [Z, Y, X] processing, accepting topological isomorphism over bit-perfect equivalence.
*   **Guardrail**: Do **not** start Phase 2 unwinding while Network ADR 0012 is still red on the canonical volume.

## 12. The Probe-Without-Orientation Trap
*   **The Loop**: Reporting a “new” crop candidate-overlap KPI (e.g. 62%) after a probe or gap script change, then chasing code for a day.
*   **The Reality**: Calling the watershed engine without the production `mpv` / orientation permute produces a **false signal**. The faithful SortedFrontier path was still at **57.89%** until a real generation fix moved it to **97.31%**.
*   **Guidance**: Every offline probe must exercise the **same orientation contract** as `generate_watershed_candidates`. Cross-check with production `slavv` edges + `prove-exact` before celebrating a KPI move.

## 13. The Ownership-Map Blind Closure Trap
*   **The Loop**: Reading `prove-exact --stage edges` as a closure verdict when `adr0012_evaluated: false` (maps missing on oracle and/or Python checkpoint).
*   **The Reality**: Harness fell back to strict-field messaging; agents claimed “Edges failed/passed” without a spatial bar. `canonical_full_v5` burned this.
*   **Guidance**: Only **evaluated** ADR 0012 proofs count. Require MATLAB `watershed_ownership_map.mat` + Python `--include-debug-maps`. Fail loud if maps are absent.

## 14. The Stale Operator Brief Trap
*   **The Loop**: Agents follow HANDOFF / TODO / ROADMAP that still say “block on 80% crop overlap” or “Edges in progress” days after findings already show Edges PASS / Network FAIL on `v6`.
*   **The Reality**: Meta drift is as expensive as a wrong algorithm hypothesis — writers and proofs get launched against the wrong gate.
*   **Guidance**: [EXACT_PROOF_FINDINGS.md](EXACT_PROOF_FINDINGS.md) is status truth. When its top banner changes, **same-session** re-synthesize [.claude/HANDOFF.md](../../../.claude/HANDOFF.md) and [TODO.md](../../TODO.md) checkboxes. ROADMAP stays narrative but must not contradict the ship gate.

## 15. The “Network Bug” Misattribution Trap
*   **The Loop**: Opening a Network-stage rewrite because strand multisets fail ADR 0012 on full volume.
*   **The Reality**: Stage isolation with **MATLAB edges** reproduces Network topology exactly. Multiset failure tracks the residual **Edge Set** (connection multiset)—historically a large gap on early claim roots; live residual class is only in [EXACT_PROOF_FINDINGS](EXACT_PROOF_FINDINGS.md).
*   **Guidance**: Network red + Edges ownership green ⇒ **generation residual**, not a Network port. Keep working watershed claim/strel / golden-trace diverge (crop iter ~13,761).

## 16. The Round-vs-Truncate MATLAB Cast Trap
*   **The Loop**: Porting MATLAB `uint16(x)` as `np.rint(x).astype(np.uint16)` (or similar “nearest int”) because “cast to integer means round.”
*   **The Reality**: MATLAB `uint16` on a real converts via **truncation toward zero** (floor for positive radii/spaces). Rounding over-cropped edges (`crop_edges_V200`) and cost ~500 crop pairs / ~2k full connections until fixed.
*   **Guidance**: For every MATLAB integer cast on continuous geometry, check truncation vs round-half-up explicitly; add a unit test against the MATLAB expression.

## 17. The Retired-Gate Zombie Trap
*   **The Loop**: After the 80% crop-overlap milestone cleared and `v6` evaluated Edges PASS, still treating “≥80% before any canonical work” or “57.89% baseline” as current operating law.
*   **The Reality**: Historical gates become cargo-cult blockers and hide the real residual KPI (generation gap → Network multiset).
*   **Guidance**: Mark cleared gates as historical in findings/HANDOFF. Current ship residual: **Network multiset** driven by **generation/claiming** residual.

---
*Last Updated: 2026-07-12*
