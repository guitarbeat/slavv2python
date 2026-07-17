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

**Open gate:** Network ADR 0012 multiset on full claim root (downstream of residual Edge Set).  
**Not a Network rewrite.** Crop = regression guard. Details: [ONE TRUTH residual](reference/core/EXACT_PROOF_FINDINGS.md#active-residual-why-network-is-red) · [HANDOFF § A](../.claude/HANDOFF.md).

### Ship tasks (open)

- [ ] **1. Production fix — full residual join** — **BLOCKED (2026-07-17)**  
  Stop emitting the extra watershed join `cand 46698` `(26444→38584)` that degree-excess prefers over oracle `(34897,38584)` under equal post-resample max.  
  **2026-07-17:** ablation reconfirmed; both residual candidates terminate at same meeting voxel `[50,25,293]` (H2 confirmed). Three join-emission fix variants implemented + reverted — each regressed crop raw-candidate parity (19,225): emit-all-broad → 30,472; pop-only faithful map → bridge test fails; single-join highest-index → 23,264. **Localized selection tweaks cannot target the one hub without breaking the 19,225 crop balance.** Fix requires faithful MATLAB-semantics watershed rewrite (multi-join `while ~all` loop + pop-only labeling), scoped as its own effort. Production code unchanged.  
  **Do not:** cleanup secondary keys, endpoint-descending reorder, shared `energy_temp_flat` vertex-origin restore, or another guess at the join block.  
  **See:** [ONE TRUTH residual](reference/core/EXACT_PROOF_FINDINGS.md#join-emission-fix-attempts-2026-07-17--all-reverted-blocker-open).

- [ ] **2. Successor full Edges → Network + evaluated proofs**  
  New claim root (do not overwrite prior audits). Carry certified Energy/Vertices; rerun edges→network with ownership maps.  
  `prove-exact --stage edges` then `--stage network` (both evaluated ADR 0012).  
  Commands: [HANDOFF § B](../.claude/HANDOFF.md). Evidence: [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).

- [ ] **3. Phase 1 closure**  
  When Edges ✅ and Network ✅ on the same fresh claim root: update ONE TRUTH + HANDOFF + figure series same session; tick freeze-baseline below.

### Standing process (always)

- [ ] **Parity change verification** — Before/after any residual code change: focused unit tests + Ruff; no long writer until crop guards hold. Record proof outcomes with [PARITY_RUN_EVIDENCE.md](reference/workflow/PARITY_RUN_EVIDENCE.md).
- [ ] **Doc freshness** — When ONE TRUTH moves: same-session HANDOFF + this file (open rows only) + `figures/parity_campaign_series.py` if paint KPIs change.

**Guardrails:** `preflight-exact` before recovery launch; never concurrent writers on one `--dest-run-root`; use `.venv\Scripts\slavv.exe` after `pip install -e .`.

### Strategy (short)

1. Ship gate = **Network multiset** on full volume, not ownership % and not `prove-exact-sequence`.
2. Edge Set multiset drives Network; residual class = Candidate Set **join displacement** (see ONE TRUTH).
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

All of the following are **done**. Do not re-open as status; evidence and historical KPIs live in findings (ONE TRUTH + session diary / iteration log).

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
