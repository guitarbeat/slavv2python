# Investigations & Technical Reports

> **History / Drafts, not live status.** Nothing in this folder is current pass/fail.

## In short

This folder houses active technical performance catalogs, publication drafts, and historical investigation write-ups. Phase 1 already shipped. If a historical file still says Network is open or Phase 1 is OPEN, that sentence is **out of date**.

**Live status:** [ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk)  
**Tasks:** [TODO.md](../TODO.md) · **Commands:** [HANDOFF](../../.claude/HANDOFF.md)

---

## Active Performance Innovations & Publication Manuscripts

| Document | Type | Description |
|---|---|---|
| [PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md](PARITY_PRESERVED_PERFORMANCE_INNOVATIONS.md) | Technical Reference | Formal catalog of 9 stage-by-stage mathematical, algorithmic, and memory innovations engineered during MATLAB-to-Python translation while preserving 1:1 parity (ADR 0011/0012). |
| [MATLAB_PYTHON_TRANSLATION_PAPER.md](MATLAB_PYTHON_TRANSLATION_PAPER.md) | Publication Manuscript | Draft scientific paper on exact MATLAB-to-Python parity methodology, numerical nuances, and scaling breakthroughs for JORS/SoftwareX/IEEE CiSE submission. |

---

## Product / UI

| Document | Type | Description |
|---|---|---|
| [2026-08-19-streamlit-gui-audit.md](2026-08-19-streamlit-gui-audit.md) | UI audit | Streamlit `slavv-app` does not start (relative import + unused `main()`); ranked page findings. Not parity status. |

---

## Deprecated / Historical Archives

| Entry | Note |
|-------|------|
| [kiro-matlab-python-parity/](kiro-matlab-python-parity/README.md) | ⛔ Frozen Kiro specs/tasks — **do not execute** |
| [2026-07-03-honesty-audit.md](2026-07-03-honesty-audit.md) | ⛔ Session audit; crop-era FAIL narrative |
| [exact-proof-findings-diary/](exact-proof-findings-diary/README.md) | ⛔ Pre-2026-08-13 findings dump (diary, join-emission essays). Not live status. |
| [v22 Pointer Corruption](v22-pointer-corruption/README.md) | Historical pointer-corruption investigation |

Retired parity investigations have been collapsed into the maintained parity
docs:

- **Translation pair analysis (2026-05-05):** archived a 33.8% trace-order
  experiment, disproved an older 41.4% baseline claim, and fed frontier-order
  work into the later v29 edge baseline. (These 33.8% / 41.4% figures are
  trace-order experiment metrics and are distinct from — not to be conflated
  with — the v29 ~88.7% pair-match baseline tracked in
  [EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md).
  **Note (2026-06-25):** all of these edge pair-match percentages are now
  **deprecated** — the edge bar is voxel ownership-map + trace tolerance per
  [ADR 0012](../adr/0012-edge-watershed-parity-bar.md), not pair-set overlap.)
- **Phase 3 edge closure notes (2026-05-23):** captured seed-aware
  tie-breaking, first-come-first-served voxel ownership, and static frontier
  priority lessons now reflected in the maintained exact-proof findings.

Live parity status, blockers, and current baselines belong **only** in
[ONE TRUTH](../reference/core/EXACT_PROOF_FINDINGS.md#one-truth--phase-1-parity-validated-from-disk).

When adding a new investigation entry:

- keep one stable entry document
- keep the scope narrow
- close or delete it once the work is absorbed into code or reference docs

Use a short, stable README or single Markdown file if a new inquiry genuinely needs its own archive entry.
