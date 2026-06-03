# Investigations

This folder is now archival-only.

Keep investigation docs here only when they explain historical decisions that still help maintain the current Python codebase. The large parity and MATLAB investigation docs were retired with the legacy tooling they supported.

Current archival bundle:

- [v22 Pointer Corruption](v22-pointer-corruption/README.md)

Retired parity investigations have been collapsed into the maintained parity
docs:

- **Translation pair analysis (2026-05-05):** archived a 33.8% trace-order
  experiment, disproved an older 41.4% baseline claim, and fed frontier-order
  work into the later v29 edge baseline.
- **Phase 3 edge closure notes (2026-05-23):** captured seed-aware
  tie-breaking, first-come-first-served voxel ownership, and static frontier
  priority lessons now reflected in the maintained exact-proof findings.

Live parity status, blockers, and current baselines belong in
[EXACT_PROOF_FINDINGS.md](../reference/core/EXACT_PROOF_FINDINGS.md).

When adding a new investigation entry:

- keep one stable entry document
- keep the scope narrow
- close or delete it once the work is absorbed into code or reference docs

Use a short, stable README or single Markdown file if a new inquiry genuinely needs its own archive entry.
