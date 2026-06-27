# SLAVV Roadmap

**Narrative milestones only.** This is the strategic, phase-level view of where
the project is headed. It does **not** track live status or tasks:

- **Active status / proofs / blockers** → [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)
- **Concrete next actions (checkboxes)** → [TODO.md](TODO.md)
- **Requirements & plans** → [plans/](plans/) · **Decisions** → [adr/](adr/)

---

## North Star

A **bit-exact MATLAB→Python port** of the SLAVV vessel-extraction pipeline
(Energy → Vertices → Edges → Network), *certified* against the MATLAB oracle on
the canonical volume — and from that trusted foundation, a **faster, maintainable
production pipeline**.

---

## Phase 0 — Port & exact-route foundation ✅

Complete. The full pipeline is ported, with an **exact route** built for
faithfulness and memory safety:

- `[Y, X, Z]` internal alignment + Fortran-order tie-breaking to reproduce
  MATLAB's column-major behavior; `float64` throughout.
- Incremental octave-chunked energy engine (no large 4D buffers).
- Certification policy established: [ADR 0011](adr/0011-energy-float-certification-policy.md)
  (strict discrete + `np.allclose` continuous) and
  [ADR 0012](adr/0012-edge-watershed-parity-bar.md) (edge/network spatial bars).
- Random Component Parity Suite for unit-level energy faithfulness.

---

## Phase 1 — Exact-route certification (current)

**Goal:** strict sequential certification on full `180709_E` via
`prove-exact-sequence`, with the crop harness as the required pre-gate.

- **Crop pre-gate (tier-2):** Energy ✅ and Vertices ✅ certified; **Edges** in
  progress (closing to the ADR 0012 spatial ownership bar — residual is early
  trace termination on long paths, not ordering); **Network** pending upstream.
- **Canonical (tier-3):** the full-volume certification run is underway. A key
  enabler landed this phase — **bit-exact threaded chunk parallelism** for the
  exact-route energy stage (`--n-jobs`), cutting a multi-day energy run to
  hours and making canonical certification practical
  (see [solutions/parity/exact-energy-chunk-parallelism.md](solutions/parity/exact-energy-chunk-parallelism.md)).

**Path to done:** close Edges parity → certify Network → green
`prove-exact-sequence` on canonical `180709_E` → promote the summary to
`workspace/reports/`. Details in the
[Phase 1 spec](plans/phase-1-exact-route-spec.md) and [TODO.md](TODO.md).

---

## Phase 2 — Performance & scale (next)

The exact route is correct but compute-heavy. Once certification gives a trusted
baseline, optimize *without* regressing parity (every change validated bit-exact
against the oracle):

- **Parallelism (started):** bit-exact threaded chunk energy (`n_jobs`) and
  per-chunk garbage collection are in. Next: auto-size `n_jobs` from core count
  with a memory guard; evaluate process-level parallelism and further hot-loop
  reductions; profile the vertices/edges/network stages.
- **Throughput tooling:** `scripts/parity_run_throughput.py` for live run ETAs.
- **Memory headroom:** continue trimming peak per-chunk memory for larger volumes.

---

## Phase 3 — Breadth & productization (later)

- **More volumes:** import and verify additional datasets (`neurovasc-db`) once
  Phase 1 closes.
- **Innovation path:** improvements beyond strict parity (speed/quality) on the
  maintained Python route; reconcile or retire the demonstration shims in
  `slavv_python/pipeline/slavv_vectorize.py` so the facade can't be mistaken for
  the parity engine.
- **Productization:** packaging and broader CLI / Streamlit UX polish.

---

*Status lives in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md);
this roadmap is intentionally narrative.*
