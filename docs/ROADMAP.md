# SLAVV Roadmap

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

“Certified” means each stage’s **defined parity bar** (ADR 0011 / ADR 0012), not
bit-identical watershed queue order or strict-field edge-pair equality as the
ship metric.

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

## Phase 1 — Exact-route certification (current)

**Goal:** evaluated per-stage `prove-exact` on full `180709_E` under ADR 0011
(Energy, Vertices) and ADR 0012 (Edges, Network). Crop harness is the fast
iteration surface; the canonical volume is the claim surface.

### Achieved (definition; live claim surface in findings)

| Stage | Full `180709_E` | Notes |
|-------|-----------------|-------|
| Energy | ✅ CERTIFIED | ADR 0011 |
| Vertices | ✅ CERTIFIED | ADR 0011 |
| Edges | ✅ ADR 0012 PASS evaluated | Exact connection count + ownership on current claim root — **numbers in findings** |
| Network | ❌ OPEN | Multiset equality still red — **downstream of residual Edge Set**, not a Network rewrite |

Also cleared historically: crop generation / 80% gate, post-watershed finalization parity, crop final pair multiset on re-selection (regression guard). **Do not copy live pair/strand counts here.**

### Path to done

1. **Full Edge Set residual** — production fix for the displacing watershed join (localized; see findings banner + ablation). Crop Edge Selection stays green as guard.
2. Re-select or successor canonical Edges→Network; evaluated `prove-exact --stage edges` + `--stage network`.
3. **Phase 1 closed** when Network multiset is green on the claim root.

Do **not** use `prove-exact-sequence` strict-field failure as the ship gate.
Details: [EXACT_PROOF_FINDINGS](reference/core/EXACT_PROOF_FINDINGS.md), [TODO.md](TODO.md),
[HANDOFF](../.claude/HANDOFF.md), [Phase 1 spec](plans/phase-1-exact-route-spec.md).

---

## Phase 2 — Performance & scale (after Phase 1)

The exact route is correct-enough for certification but compute-heavy. Once
Phase 1 Network is green, optimize *without* silent parity regression:

- **Parallelism (started):** bit-exact threaded chunk energy (`n_jobs`). Next:
  auto-size `n_jobs`, vertices/edges/network profiling.
- **Optional unwind:** after a frozen cert baseline, Phase 2 may relax
  Fortran-order emulation toward idiomatic C-order — only under a new
  topological-tolerance gate (see [phase-2-optimization-spec.md](plans/phase-2-optimization-spec.md)).
- **Paper-profile certification:** same sequential bars on the public `paper`
  profile (phase-1-spec F2) — program ship confidence requires this after exact
  route.

> **Research input:** [Post-parity optimization & the translation paper](research/post-parity-optimization-and-paper.md)

**Do not start broad Phase 2 unwinding while Phase 1 Network is red.**

---

## Phase 3 — Breadth & productization (later)

- **More volumes:** `neurovasc-db` import and verify once Phase 1 closes.
- **Innovation path:** improvements beyond strict parity on the maintained
  Python route.
- **Productization:** packaging and broader CLI / Streamlit UX polish.

---

*Status lives in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md);
this roadmap is intentionally narrative. Last realigned: 2026-07-12.*
