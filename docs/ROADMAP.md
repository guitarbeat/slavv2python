# SLAVV Roadmap

## In short

Python already matches MATLAB closely enough to ship (Phase 1 closed). Next work
is speed and the extra “identical last digits” bar — not reopening the ship
gate. About 90% exact on crop Energy is **not** 100%. Do not rerun the crop
Energy writer.

**Narrative milestones only.** This is the strategic, phase-level view of where
the project is headed. It does **not** track live status or tasks:

- **Active status / proofs / blockers** → [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk)
- **Concrete next actions (checkboxes)** → [TODO.md](TODO.md)
- **Operator brief (commands + decision point)** → [.claude/HANDOFF.md](../.claude/HANDOFF.md)
- **Requirements & plans** → [plans/](plans/) · **Decisions** → [adr/](adr/)

---

## North Star

A **certified MATLAB→Python port** of the SLAVV vessel-extraction pipeline
(Energy → Vertices → Edges → Network) on the canonical volume — and from that
trusted foundation, a **faster, maintainable production pipeline**.

“Certified” means each stage’s **defined parity bar** (ADR 0011 / ADR 0012):
close enough to ship, not identical last digits and not bit-identical watershed
queue order.

---

## Phase 0 — Port & exact-route foundation ✅

Complete. The full pipeline is ported, with an **exact route** built for
faithfulness and memory safety:

- `[Y, X, Z]` internal alignment + Fortran-order tie-breaking to reproduce
  MATLAB's column-major behavior; `float64` throughout.
- Incremental octave-chunked energy engine (no large 4D buffers).
- Certification policy: [ADR 0011](adr/0011-energy-float-certification-policy.md)
  (strict discrete + `np.allclose` continuous) and
  [ADR 0012](adr/0012-edge-watershed-parity-bar.md) (edge ownership-map + network
  strand/bifurcation multisets).
- Random Component Parity Suite for unit-level energy faithfulness.

---

## Phase 1 — Exact-route certification (closed)

**Goal (met):** evaluated per-stage `prove-exact` on full `180709_E` under ADR 0011
(Energy, Vertices) and ADR 0012 (Edges, Network). Live pass/fail is [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) only. Crop harness remains the regression guard.

### Achieved (definition; live claim surface in findings)

| Stage | Full `180709_E` | Notes |
|-------|-----------------|-------|
| Energy | ✅ CERTIFIED | ADR 0011 |
| Vertices | ✅ CERTIFIED | ADR 0011 |
| Edges | ✅ ADR 0012 PASS evaluated | Ownership-map bar on the claim root — **numbers in findings** |
| Network | ✅ ADR 0012 PASS evaluated | Multiset equality on the same claim root — **numbers in findings** |

Also cleared historically: crop generation / 80% gate, post-watershed finalization parity, crop final pair multiset on re-selection (regression guard), Edge Selection Ranking Residual (Claimed Trace Energy / ADR 0013). **Do not copy live pair/strand counts here.**

### After Phase 1

Frozen cert baseline: [phase1-baseline-freeze.json](reference/core/phase1-baseline-freeze.json) (`canonical_full_v18`, 2026-08-17). Phase 2 **profiling baseline** is recorded against that dest ([phase2-profiling-baseline.json](reference/core/phase2-profiling-baseline.json)). Broad Fortran-order unwind still needs an explicit Phase 2 ADR/gate before production code changes. The extra identical-last-digits program (`--strict-floats`, including Energy) is **separate**; crop leftover last-digit diffs are not 100%. Live stretch status is dest `stretch_status.json`, not ONE TRUTH. Do not relaunch the crop Energy writer. Strict-field `connections` / order remains optional and gated on Energy unlock.

Do **not** use `prove-exact-sequence` strict-field failure as a Phase 1 reopen.
Details: [EXACT_PROOF_FINDINGS](reference/core/EXACT_PROOF_FINDINGS.md), [TODO.md](TODO.md),
[HANDOFF](../.claude/HANDOFF.md), [Phase 1 spec](plans/phase-1-exact-route-spec.md).

---

## Phase 2 — Performance & scale (after Phase 1)

The exact route is correct-enough for certification but compute-heavy. Once
Phase 1 Network is green, optimize *without* silent parity regression:

- **Parity-Preserved Performance Innovations Baseline:** 9 verified algorithmic,
  mathematical, and memory optimizations have already been engineered into the Python
  engine, establishing the foundation for Phase 2 scaling without altering certified numerical
  or topological parity:
  1. **In-place octave accumulation:** 4D scale-stack elimination cutting peak RAM by **$30\times$** (300 MB $\to$ 10 MB/thread).
  2. **Batched $3 \times 3$ Hessian eigensolver:** vectorization via `np.linalg.eigh` + `np.einsum` achieving **$>20\times$ faster** tensor filtering.
  3. **Deterministic Joblib chunk parallelism:** fixed `chunk_idx` reduction order yielding **$\sim 5.2\times$ wall-clock throughput** on 6 cores.
  4. **Structuring element offset cache:** memoized relative offsets $(\Delta y, \Delta x, \Delta z)$ yielding **$>10\times$ faster** vertex body painting.
  5. **Sparse conjugate symmetry FFT:** sparse 1D broadcasting mask achieving **$50\%$ peak RAM reduction** in chunk IFFTs.
  6. **Claimed Trace Energy bake (ADR 0013):** candidate payload provenance enabling pure-function selection and resolving multiset strand parity (**100% ADR 0012 multiset pass**).
  7. **Indexed binary heap priority queue:** composite tie-breaking keys `(Energy, OriginSeedRank, FortranLinearIndex)` reducing queue complexity from **$O(N^2) \to O(N \log N)$**.
  8. **Sparse CSR graph decomposition:** `scipy.sparse.csgraph.connected_components` reducing network strand extraction to **$<7\text{ seconds}$**.
  9. **Continuous arc-length interpolation:** vectorized Euclidean arc-length centerline smoothing and resampling.
  
  *Full technical catalog:* [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](investigations/PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md)  
  *Publication manuscript:* [MATLAB_PYTHON_TRANSLATION_PAPER.md](investigations/MATLAB_PYTHON_TRANSLATION_PAPER.md) (JORS/SoftwareX/IEEE CiSE draft)

- **Profiling baseline (2026-08-17):** read-only timings measured against the frozen canonical destination (`canonical_full_v18`, documented in [phase2-profiling-baseline.json](reference/core/phase2-profiling-baseline.json)):
  - **Edges stage is the primary measured bottleneck:** Elapsed **5,534.0 s** (~92.2 min), Peak RAM: 1.57 GB.
  - **Network stage:** Elapsed **416.0 s** (~6.9 min), Peak RAM: 973.5 MB.
  - **Energy & Vertices:** Elapsed **0.0 s** on this dest (carried lineage / resumed cache; historical 6-worker chunk throughput documented in [exact-energy-chunk-parallelism.md](solutions/parity/exact-energy-chunk-parallelism.md)).
  - **Checkpoints:** Energy (167.8 MB), Network (106.4 MB), Edges (30.3 MB), Vertices (1.5 MB).

- **Parallelism & Next Optimizations:**
  - Auto-size `n_jobs` for energy and investigate parallel watershed frontier exploration.
  - Focused optimization on the 92.2-minute Edges bottleneck via compiled kernels / algorithmic pruning.
- **Optional unwind:** after a frozen cert baseline, Phase 2 may relax
  Fortran-order emulation toward idiomatic C-order — only under a new
  topological-tolerance gate (see [phase-2-optimization-spec.md](plans/phase-2-optimization-spec.md)).
- **Paper-profile certification:** same sequential bars on the public `paper`
  profile (phase-1-spec F2) — program ship confidence requires this after exact
  route.

> **Research & Publication input:**
> - [Post-parity optimization & the translation paper](research/post-parity-optimization-and-paper.md)
> - [Parity-Preserved Performance Innovations](investigations/PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md)
> - [Scientific Translation Paper Manuscript](investigations/MATLAB_PYTHON_TRANSLATION_PAPER.md)

**Do not start broad Phase 2 unwinding until a frozen cert baseline exists.** That freeze is recorded (2026-08-17). Phase 1 Network is green on the claim root in [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk). Profiling against the frozen dest is allowed; C-order unwind is not.

---

## Phase 3 — Breadth & productization (later)

- **More volumes:** `neurovasc-db` import and verify now that Phase 1 is closed.
- **Innovation path:** improvements beyond strict parity on the maintained
  Python route.
- **Productization:** packaging and broader CLI / Streamlit UX polish.

---

*Status lives in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md);
this roadmap is intentionally narrative. Last realigned: 2026-08-17.*
