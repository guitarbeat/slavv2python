# README Removal And Rename Plan

Date: 2026-04-17

This plan turns the README utility audit into a concrete repository-shape
proposal. It covers every first-party `README.md` currently in the tree and
classifies each one as `keep`, `collapse`, or `remove`.

Vendored trees under `external/` are intentionally excluded.

## Executive Summary

The repository does not need a README in every subdirectory. The files that are
worth keeping are the ones that explain workflow state, documentation structure,
or contributor conventions that the folder names cannot express on their own.

The files that are easiest to remove are the thin index READMEs in tiny folders,
especially the chapter `working/` and `archive/` stubs and the small helper
script directories under `dev/scripts/`.

The strongest rename opportunities are the folder names that currently encode a
maintenance bucket instead of the thing the tools actually do.

## Decision Table

| Path | Decision | Priority | Why |
| --- | --- | --- | --- |
| [README.md](../README.md) | keep | P0 | Root project entry point; carries setup, workflows, and navigation. |
| [docs/README.md](README.md) | keep | P0 | Documentation hub and reading-order index. |
| [docs/chapters/README.md](chapters/README.md) | keep | P0 | Explains chapter lifecycle, active vs historical chapters, and how the chapter system works. |
| [docs/reference/README.md](reference/README.md) | keep | P0 | Stable shelf index for cross-cutting reference docs. |
| [dev/README.md](../dev/README.md) | keep | P0 | Explains the developer workspace layout and where tests and scripts live. |
| [dev/tests/README.md](../dev/tests/README.md) | keep | P0 | Contributor guide for test placement and ownership rules. |
| [docs/chapters/neighborhood-claim-alignment/README.md](chapters/neighborhood-claim-alignment/README.md) | keep | P0 | Active chapter entry point with live loop, scope, and working questions. |
| [docs/chapters/candidate-generation-handoff/README.md](chapters/candidate-generation-handoff/README.md) | keep | P1 | Historical handoff context still matters when reading the chapter lineage. |
| [docs/chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md](chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md) | keep | P1 | Completed spec archive with requirements, design, tasks, and outcome notes. |
| [docs/chapters/neighborhood-claim-alignment/working/README.md](chapters/neighborhood-claim-alignment/working/README.md) | remove | P2 | Folder is already self-describing and only contains the README itself. |
| [docs/chapters/neighborhood-claim-alignment/archive/README.md](chapters/neighborhood-claim-alignment/archive/README.md) | remove | P2 | Same pattern as `working/`; the folder name already signals archival status. |
| [docs/reference/core/README.md](reference/core/README.md) | collapse | P1 | Useful, but mostly duplicates the parent reference index and can be absorbed into it. |
| [docs/reference/backends/README.md](reference/backends/README.md) | collapse | P2 | Helpful grouping, but the backend filenames are already highly descriptive. |
| [docs/reference/workflow/README.md](reference/workflow/README.md) | collapse | P2 | Small enough to move into the parent reference index. |
| [dev/scripts/maintenance/README.md](../dev/scripts/maintenance/README.md) | collapse | P1 | Helper bucket index with low unique value; could become a better-named folder or a single `dev/scripts/README.md`. |
| [dev/scripts/benchmarks/README.md](../dev/scripts/benchmarks/README.md) | remove | P2 | Lowest-value helper index; the script name already explains the folder purpose. |

## Keep List

Keep these as durable entry points or convention documents:

- [README.md](../README.md)
- [docs/README.md](README.md)
- [docs/chapters/README.md](chapters/README.md)
- [docs/reference/README.md](reference/README.md)
- [dev/README.md](../dev/README.md)
- [dev/tests/README.md](../dev/tests/README.md)
- [docs/chapters/neighborhood-claim-alignment/README.md](chapters/neighborhood-claim-alignment/README.md)
- [docs/chapters/candidate-generation-handoff/README.md](chapters/candidate-generation-handoff/README.md)
- [docs/chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md](chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md)

## Collapse Targets

These files still have value, but they are the easiest to absorb into a parent
index if the goal is to reduce README count without losing much information.

1. [docs/reference/core/README.md](reference/core/README.md)
2. [dev/scripts/maintenance/README.md](../dev/scripts/maintenance/README.md)
3. [docs/reference/backends/README.md](reference/backends/README.md)
4. [docs/reference/workflow/README.md](reference/workflow/README.md)

Recommended collapse path:

- Expand [docs/reference/README.md](reference/README.md) so it can own the topic
  grouping now split across `core/`, `backends/`, and `workflow/`.
- If helper script grouping still needs a landing page, prefer a more descriptive
  parent folder name over a README-only index.

## Remove Targets

These are the easiest files to remove first because the parent folder names and
neighboring files already explain them well enough.

1. [docs/chapters/neighborhood-claim-alignment/working/README.md](chapters/neighborhood-claim-alignment/working/README.md)
2. [docs/chapters/neighborhood-claim-alignment/archive/README.md](chapters/neighborhood-claim-alignment/archive/README.md)
3. [dev/scripts/benchmarks/README.md](../dev/scripts/benchmarks/README.md)
4. [docs/reference/backends/README.md](reference/backends/README.md) if the parent index is expanded first

The most important sequencing rule is: do not delete a sub-index README until the
parent index is already capable of carrying its navigation load.

## Folder Rename Opportunities

If the goal is to make names do more work, these are the best rename candidates.

| Current folder | Better naming direction | Why |
| --- | --- | --- |
| `dev/scripts/maintenance/` | `dev/scripts/repo-maintenance/` or `dev/scripts/mapping-maintenance/` | The current name is generic; a purpose-based name reduces the need for an index README. |
| `dev/scripts/benchmarks/` | `dev/scripts/profiling/` or `dev/scripts/perf-probes/` | The folder contains one manual timing helper, so the function-based name can replace the README more cleanly. |
| `docs/chapters/neighborhood-claim-alignment/working/` | `docs/chapters/neighborhood-claim-alignment/live/` only if the folder must stay | `working/` is understandable but redundant; if retained, a clearer lifecycle label is preferable. |
| `docs/chapters/neighborhood-claim-alignment/archive/` | `docs/chapters/neighborhood-claim-alignment/history/` only if the folder must stay | `archive/` is already clear, but `history/` is slightly more direct if the folder remains. |

## Naming Principle

Use a README when the folder needs help describing:

- workflow state
- lifecycle or chronology
- contributor conventions
- a stable topic shelf

Use the filename and folder name alone when the folder contains:

- one script
- one tiny cluster of ordinary docs
- a single-purpose stub whose name already tells the story

## Bottom Line

If you want the tree to read faster at a glance, remove the `working/` and
`archive/` chapter stubs first, remove the benchmark helper README next, and
collapse the small reference sub-indexes into the parent reference index.

Keep the README files that explain live workflow, chapter structure, and test
placement; those are the ones that still add information the path names do not
carry.
