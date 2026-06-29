---
title: "Directory Restructure Plan (proposed — for review)"
type: spec
status: proposed
date: 2026-06-29
topic: directory-structure
---

# Directory Restructure Plan (proposed)

**Status: PROPOSED — no moves executed. Approve the target tree before execution.**

## Guiding principle
The package is already well-layered (stages / engine / interface / analytics /
support) and documented in the AGENTS.md repository map + ADRs. So this is a
**targeted** restructure, not a rewrite: fix the one real navigability smell
(the 48-file flat `analytics/parity/`) and a couple of minor items, and
explicitly **keep** everything that already reads clearly. Every move preserves
the public import surface via the package `__init__` so the **574 tests**, the
`analytics.parity.<name>` **test patches**, the `cli` re-export facade, and doc
links keep working.

## What stays (with rationale)
- `pipeline/{energy,vertices,edges,network}` — the four MATLAB stages; canonical and clear. *(edges/ is 37 files but cohesive; sub-grouping deferred — see Optional.)*
- `engine/` + `engine/state/` — orchestration + run ledger. Already deep-module clean.
- `interface/{cli,streamlit,shared_services,shared_state}` — already organized.
- `schema/`, `storage/`, `utils/`, `visualization/`, `workflows/` — keep.
- Top level (`docs/`, `tests/`, `scripts/`, `workspace/`, `external/`, configs) — keep.

## Core change — sub-group `analytics/parity/` (48 → 5 themed subpackages)

| New subpackage | Modules (current names) |
|---|---|
| `parity/proof/` | coordinator, proofs, artifact_comparator, array_normalization, energy_proof_evidence, energy_ulp_proof, mismatch_diagnostics, exact_proof_contract, proof_report, counts, reports, tables, index |
| `parity/runs/` | resume, jobs, job_registry, launch_prepare, monitor_daemon, parity_job_lifecycle, process_utils, writer_lease, preflight, execution, bootstrap |
| `parity/oracle/` | surfaces, oracle_artifacts, promotion, matlab_vector_loader, python_checkpoint_loader, params_audit, models, io, gaps |
| `parity/probes/` | adaptive_probes, trace_comparator, crop_export, edge_artifacts, matlab_fail_fast |
| `parity/cli/` | commands, cli_runs, cli_proofs, cli_diagnostics, cli_edges, cli_support |
| *(stays at `parity/` root)* | `__init__.py`, `constants.py`, `utils.py`, **`cli.py`** (the facade — see caveat) |

### Import-stability strategy
- **Internal imports** (within `slavv_python`) are updated to the new paths.
- **External/patched surfaces are preserved**: `parity/__init__.py` re-exports the
  handler names and `ExactProofCoordinator`; the test patch targets
  (`analytics.parity.cli.ExactProofCoordinator`, `..cli._build_exact_proof_source_surface`)
  stay valid because **`cli.py` does not move** (it remains the facade and is updated
  to import from `parity.cli.*` submodules and `parity.proof.coordinator`).
- ⚠️ **Naming-collision caveat:** a module `parity/cli.py` and a package `parity/cli/`
  **cannot coexist**. Resolution: name the CLI subpackage **`parity/cli_handlers/`**
  (not `cli/`), so the `cli.py` facade is untouched and the test-patch surface is
  100% preserved. *(This is the safe choice; flagged for your confirmation.)*

## Minor items
- **Duplicate `math.py`**: `utils/math.py` (generic) vs `analytics/math.py`. Rename
  `analytics/math.py` → `analytics/metrics_math.py` (or fold into `analytics/metrics/`)
  to remove the ambiguity. Update imports.
- **Loose `analytics/` files**: `cropping.py`, `trace_ops.py` sit at the analytics
  root. Leave unless they have a clear home (low value to move).

## Optional (only if you want, higher churn)
- Sub-group `pipeline/edges/` (37 files) into `edges/{tracing,watershed,selection,candidates}`.
  Cohesive today; defer unless navigation there is a pain point.
- Fold `image/` (2 files) into `utils/` or a `pipeline/preprocess/`. Marginal.

## Risks & mitigations
- **Breaks imports / test patches** → preserve via `__init__` re-exports; keep `cli.py`
  facade in place (use `cli_handlers/` for the subpackage). Run `mypy` + full suite after
  **each** subpackage move, not just at the end.
- **Doc-link drift** → repo-wide markdown link audit + the path references in AGENTS.md
  repo map / TECHNICAL_ARCHITECTURE / ADR 0008 updated in the same commit.
- **Mid-certification** → the energy investigation reads checkpoints, not parity module
  paths; low interaction, but do the restructure on its own branch and keep commits small.

## Execution plan (staged, test-gated)
1. Branch `refactor/parity-subpackages`.
2. Move one subpackage at a time (`oracle/` → `proof/` → `runs/` → `probes/` → `cli_handlers/`),
   updating internal imports + `parity/__init__` re-exports per step; run `ruff` + `mypy` +
   `pytest -m "unit or integration"` after each.
3. Minor items (`math.py` rename) as a separate commit.
4. Update docs (AGENTS.md repo map, TECHNICAL_ARCHITECTURE, ADR 0008 references) + link audit.
5. Final full-suite green + report before merge.

## Recommendation
Do the **parity sub-grouping with `cli_handlers/` (collision-safe) + the `math.py`
rename**; **skip** the optional edges/image moves for now (low value, higher churn).
This is the high-value, lowest-risk subset. Confirm the target tree (esp. the
`cli_handlers/` naming) and I'll execute stage-by-stage with tests gating each move.
