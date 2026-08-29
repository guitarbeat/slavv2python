# SLAVV Developer Dashboard

**Single entry point** for what to do next. Checkboxes only here.

| Need | Home |
|------|------|
| **Live pass/fail / residual / claim root** | [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) |
| **Operator commands** | [.claude/HANDOFF.md](../.claude/HANDOFF.md) |
| **Phase 1 requirements** | [phase-1-exact-route-spec.md](plans/phase-1-exact-route-spec.md) |
| **Specs / ideas / solutions / ADRs** | [plans/](plans/) · [brainstorms/](brainstorms/) · [solutions/](solutions/) · [adr/](adr/) |
| **Performance innovations catalog** | [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](investigations/PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md) |
| **Translation paper manuscript** | [MATLAB_PYTHON_TRANSLATION_PAPER.md](investigations/MATLAB_PYTHON_TRANSLATION_PAPER.md) |
| **Authority map** | [docs/README.md](README.md#documentation-authority-map-one-concept--one-home) |

> Do **not** freeze run IDs, pair counts, or strand counts in this file. Those live only in ONE TRUTH.

---

## Closed — Phase 1 ship

**Phase 1 CLOSED** on claim root in [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) (`canonical_full_v18`).  
Former residual was Edge Selection Ranking Residual ([ADR 0013](adr/0013-claimed-energy-trace-provenance.md)). Crop remains a regression guard. Operator brief: [HANDOFF](../.claude/HANDOFF.md).

### Ship tasks

- [x] **1. Production fix — claimed `energy_map` traces + `sort_edges`** — verified on `canonical_full_v18` (2026-08-14)  
  Bake Claimed Trace Energy at watershed finalize; Selection keeps raw-max `sort_edges`. Cheap experiments + full Edge Set pair fix green.
- [x] **2. Successor full Edges → Network + evaluated proofs** — `canonical_full_v18`  
  Edges + Network both `adr0012_evaluated: true` and `passed: true`.
- [x] **3. Phase 1 closure**  
  ONE TRUTH + HANDOFF + figure KPI mirror updated same session (2026-08-14).

### Standing process (always)

- [ ] **Parity change verification** — Before/after any residual code change: focused unit tests + Ruff; no long writer until crop guards hold. Record proof outcomes with [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).
- [ ] **Doc freshness** — When ONE TRUTH moves: same-session HANDOFF + this file (open rows only) + `figures/claim/campaign_series.py` if paint KPIs change.

**Guardrails:** `preflight-exact` before recovery launch; never concurrent writers on one `--dest-run-root`; use `uv run slavv` after `uv sync`.

### Strategy (short)

1. Ship gate = **Network multiset** on full volume, not ownership % and not `prove-exact-sequence`.
2. Edge Set multiset drives Network; former residual class was **claimed-map `sort_edges` vs original-field traces** (closed; see ONE TRUTH).
3. Prefer funnel / cleanup comparator / `select_and_finalize_edge_set` over selection forks.
4. Anti-patterns: [UNPRODUCTIVE_LOOPS.md](reference/core/UNPRODUCTIVE_LOOPS.md).

---

## Next — after Phase 1 green

- [x] **Freeze Phase 1 baseline** — `canonical_full_v18` hash bridge [phase1-baseline-freeze.json](reference/core/phase1-baseline-freeze.json) (2026-08-17). Do not overwrite that dest.
- [x] **Phase 1 → Phase 2 handoff** — Network ADR 0012 green; freeze recorded. Fortran-unwind still needs an explicit Phase 2 ADR/gate before code changes.
- [x] **Phase 2 profiling baseline** — read-only timings from the frozen dest [phase2-profiling-baseline.json](reference/core/phase2-profiling-baseline.json). Energy/Vertices elapsed 0 = carried lineage. Measured bottleneck on dest = Edges. No unwind.
- [x] **Energy `--n-jobs auto`** — opt-in CPU/RAM guard on `resume-exact-run` / `launch-exact-run`; dest default stays serial `n_jobs=1`. Do not reimplement; do not raise the default; do not forward the token `auto` into a detached job. See [exact-energy-chunk-parallelism.md](solutions/parity/exact-energy-chunk-parallelism.md).
- [ ] **Translation paper manuscript** — draft in [MATLAB_PYTHON_TRANSLATION_PAPER.md](investigations/MATLAB_PYTHON_TRANSLATION_PAPER.md) now narrates all 9 catalog items; remaining work is journal packaging (citations, submission figures, cover letter), not a second catalog. Cross-reference: [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](investigations/PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md).
- [ ] **Phase 2 Edges/Network profiling** — dest measured bottleneck is Edges (5,534s / ~92.2m per [phase2-profiling-baseline.json](reference/core/phase2-profiling-baseline.json)). Split discovery vs selection only with an authorized writer. Cross-reference 9 innovations in [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](investigations/PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md). No Fortran unwind.
- [ ] **Phase 2 Fortran-order unwind** — needs an explicit Phase 2 ADR before production code changes.
- [ ] **Paper-profile certification** — phase-1-spec F2 / R7 (volume + oracle TBD).
- [ ] **neurovasc-db** — additional volumes after Phase 1 closed.
- [ ] **Stretch Energy (extra 100% bar)** — crop is ~90% bit-identical; leftover last-digit diffs are not 100%. Tiny photos matching does not unlock crop. Live status: [findings stretch subsection](reference/core/EXACT_PROOF_FINDINGS.md#true-zero-tolerance-stretch-separate-from-phase-1). Readable diagnosis: [crop-energy-stretch-float-isolation.md](solutions/parity/crop-energy-stretch-float-isolation.md). Do not relaunch v2.
- [ ] **Strict-field stretch (optional)** — exact connections / order-sensitive fields on crop after Energy unlock.

---

## Cleared Phase 1 work (archive — not open tasks)

All of the following are **done**. Do not re-open as status; evidence lives in [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk); historical trail in [findings diary](investigations/exact-proof-findings-diary/README.md).

| Theme | Outcome (summary) |
|-------|-------------------|
| Energy / Vertices full | Certified ADR 0011 (`v4` lineage seed) |
| Edges ADR 0012 full | Evaluated PASS on claim root (see ONE TRUTH) |
| Crop generation / frontier | Closed (match, gap 0); 80% gate retired |
| Crop Edge Selection re-selection | Pair multiset closed (regression guard) |
| Post-watershed finalization / cleanup | MATLAB-style path; cleanup comparator green |
| Full residual localization | Extra join displaces oracle pair; Claimed Trace Energy bake landed (#110); closed on `canonical_full_v18` |
| Infra | Policy, lattice F-order, SortedFrontier, fail-loud maps, energy `n_jobs`, float64, job lifecycle |

**Historical narrative** (superseded messaging): v10/76% match, >95% edge match rate, 57.89% crop overlap, 80% gate, edge 88.7% pair overlap — all non-current.

---

## Maintenance (docs / hub)

- [x] Contributor guide, parity evidence template, glossary/architecture, pre-gate & cert guides
- [x] Planning hub = this file; live status = ONE TRUTH only
- [x] 2026-07-12 / 07-15 / 07-16 docs consolidate + ONE TRUTH + clash deprecation
- [x] **2026-07-16 TODO lean rewrite** — open ship tasks only; historical gates collapsed to archive table
- [x] **Performance innovations & publication hub integration** — linked [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](investigations/PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md) and [MATLAB_PYTHON_TRANSLATION_PAPER.md](investigations/MATLAB_PYTHON_TRANSLATION_PAPER.md) in docs hub, roadmap, agents, and specs.
