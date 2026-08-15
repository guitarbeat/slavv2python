# SLAVV Developer Dashboard

**Single entry point** for what to do next. Checkboxes only here.

| Need | Home |
|------|------|
| **Live pass/fail / residual / claim root** | [ONE TRUTH](reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk) |
| **Operator commands** | [.claude/HANDOFF.md](../.claude/HANDOFF.md) |
| **Phase 1 requirements** | [phase-1-exact-route-spec.md](plans/phase-1-exact-route-spec.md) |
| **Specs / ideas / solutions / ADRs** | [plans/](plans/) · [brainstorms/](brainstorms/) · [solutions/](solutions/) · [adr/](adr/) |
| **Authority map** | [docs/README.md](README.md#documentation-authority-map-one-concept--one-home) |

> Do **not** freeze run IDs, pair counts, or strand counts in this file. Those live only in ONE TRUTH.

---

## Do now — Phase 1 ship

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
- [ ] **Doc freshness** — When ONE TRUTH moves: same-session HANDOFF + this file (open rows only) + `figures/parity_campaign_series.py` if paint KPIs change.

**Guardrails:** `preflight-exact` before recovery launch; never concurrent writers on one `--dest-run-root`; use `.venv\Scripts\slavv.exe` after `pip install -e .`.

### Strategy (short)

1. Ship gate = **Network multiset** on full volume, not ownership % and not `prove-exact-sequence`.
2. Edge Set multiset drives Network; residual class = **claimed-map `sort_edges` vs original-field traces** (see ONE TRUTH).
3. Prefer funnel / cleanup comparator / `select_and_finalize_edge_set` over selection forks.
4. Anti-patterns: [UNPRODUCTIVE_LOOPS.md](reference/core/UNPRODUCTIVE_LOOPS.md).

---

## Next — after Phase 1 green

- [ ] **Freeze Phase 1 baseline** — closure run root, proof hashes, release evidence, figure metrics ([transition spec](plans/phase-1-to-phase-2-transition-spec.md)).
- [ ] **Phase 1 → Phase 2 handoff** — only after Network ADR 0012 green; no early Fortran-unwind.
- [ ] **Paper-profile certification** — phase-1-spec F2 / R7 (volume + oracle TBD).
- [ ] **neurovasc-db** — additional volumes after Phase 1 closed.
- [ ] **Strict-field stretch (optional)** — exact connections / order-sensitive fields on crop after ship gate.

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
| Full residual localization | Extra join displaces oracle pair; ablation documented in ONE TRUTH — **fix still open** (ship task 1) |
| Infra | Policy, lattice F-order, SortedFrontier, fail-loud maps, energy `n_jobs`, float64, job lifecycle |

**Historical narrative** (superseded messaging): v10/76% match, >95% edge match rate, 57.89% crop overlap, 80% gate, edge 88.7% pair overlap — all non-current.

---

## Maintenance (docs / hub)

- [x] Contributor guide, parity evidence template, glossary/architecture, pre-gate & cert guides
- [x] Planning hub = this file; live status = ONE TRUTH only
- [x] 2026-07-12 / 07-15 / 07-16 docs consolidate + ONE TRUTH + clash deprecation
- [x] **2026-07-16 TODO lean rewrite** — open ship tasks only; historical gates collapsed to archive table
