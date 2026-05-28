# SLAVV Developer Dashboard

**Single entry point** for what to do next, where plans live, and where to put new thoughts so they do not scatter across chat, ad-hoc notes, and stale markdown.

> **Rule of thumb:** Checkboxes only here. **Status** → [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md). **Specs** → [plans/](plans/). **Fixes** → [solutions/](solutions/) (`/ce-compound`).

---

## Where things live

| Kind | Location | Put it here when… |
|------|----------|-------------------|
| **Active tasks** | **This file (`docs/TODO.md`)** | Concrete next actions with checkboxes |
| **Specs (requirements + plan)** | [plans/](plans/) | One `*-spec.md` per active initiative; index in [plans/README.md](plans/README.md) |
| **Ideas (pre-plan)** | [brainstorms/](brainstorms/) | Before a spec exists; promote into `plans/` when scoped |
| **Solved problems & runbooks** | [solutions/](solutions/) | `/ce-compound` writes here; parity index in [findings](reference/core/EXACT_PROOF_FINDINGS.md#-compound-learnings-parity-related) |
| **Architecture decisions** | [adr/](adr/) | Load-bearing design choice (do not re-litigate in TODO) |
| **Live exact-parity status** | [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md) | **Only place** for active run status, proof results, blockers (not TODO) |
| **Operator workflows** | [PARITY_PRE_GATE.md](reference/workflow/PARITY_PRE_GATE.md), [PARITY_CERTIFICATION_GUIDE.md](reference/workflow/PARITY_CERTIFICATION_GUIDE.md) | How to run pre-gate / certification |
| **Project narrative** | [ROADMAP.md](ROADMAP.md) | Milestones, codebase health, historical priorities (less tactical than this file) |
| **Investigation archives** | [investigations/](investigations/) | Deep dives that are context, not the task list |

**Do not duplicate:** Status tables and run state → [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md). This file = checkboxes + links only.

**Phase 1:** [phase-1-exact-route-spec.md](plans/phase-1-exact-route-spec.md) · **Status:** [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md#-active-phase-1-operations)

---

## Checklist — do now

### Phase 1 exact route (canonical + crop)

- [ ] **Crop energy mismatch** — Debug `prove-exact` failure on `crop_M_exact` (same shape `[64,256,256]`, different magnitudes). Check params vs MATLAB `settings/`, energy sign/scale, HDF5 plane assignment. Re-run `prove-exact-sequence` when fixed.
- [ ] **Canonical run** — Keep single process on `phase1_cert_network` until `stop-after network`; then `prove-exact-sequence` against `180709_E_batch_190910-103039`.
- [ ] **Crop tier-2 gate** — After crop energy passes, confirm all four stages zero missing/extra on `180709_E_crop_M` (harness only, not canonical claim).
- [ ] **Canonical tier-3 gate** — All four stages pass on full `180709_E`; promote summary to `workspace/reports/`; record milestone in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md).

### Harness & ops

- [ ] **Preflight before long runs** — `parity_experiment.py preflight-exact` on dest + dataset + oracle.
- [ ] **No duplicate writers** — Never concurrent `init-exact-run` / `resume-exact-run` on the same `--dest-run-root`.

---

## Checklist — next (after Phase 1 gates)

- [ ] **Document HDF5 in storage guide** — Optional: `/ce-compound-refresh` or edit [PARITY_EXPERIMENT_STORAGE.md](reference/workflow/PARITY_EXPERIMENT_STORAGE.md) for MATLAB energy artifact layout.
- [ ] **O(log N) frontier** — `heapq` / `SortedList` for `available_locations` in `global_watershed.py` (performance, not cert blocker).
- [ ] **API reference** — Public `SlavvPipeline` docstrings.
- [ ] **neurovasc-db** — Import and verify additional volumes when Phase 1 is closed.

---

## Historical context (superseded — do not treat as current tasks)

Older dashboard text referred to **v10 / 76% match** and **>95% edge match rate** as the certification bar. Phase 1 now uses **strict zero** per stage via `prove-exact-sequence` on the canonical volume. Edge **88.7%** (v29) remains a useful baseline narrative in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md), not the Phase 1 exit criterion.

---

## Maintenance

- [x] Contributor guide — `CONTRIBUTING.md`
- [x] Glossary / architecture — `GLOSSARY.md`, `TECHNICAL_ARCHITECTURE.md`
- [x] Parity pre-gate & certification guides — `PARITY_PRE_GATE.md`, `PARITY_CERTIFICATION_GUIDE.md`
- [x] Planning hub — this file; status + compound index in [EXACT_PROOF_FINDINGS.md](reference/core/EXACT_PROOF_FINDINGS.md)
