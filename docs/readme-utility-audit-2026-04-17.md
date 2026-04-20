# README Utility Audit

Date: 2026-04-17

## Scope

This audit covers the first-party repository tree and asks a narrow question:
which `README.md` files provide real orientation that the folder and file names
do not already provide, and which ones are mostly redundant indices that could
be collapsed or removed if the goal is to reduce README sprawl.

Vendored third-party trees under `external/` are excluded from the removal
recommendations. Those files belong to upstream projects and should be treated
as preserved mirrors unless the upstream layout changes.

## Main Finding

The repository already has a clear pattern:

- Root and top-level documentation READMEs do useful navigation work.
- Chapter READMEs are valuable when they describe live investigation state.
- Small helper directories often use READMEs as thin indexes where the folder
  name is already doing most of the explanatory work.

So the cleanest reduction path is not “delete all READMEs below the root.” The
better rule is: keep READMEs that explain workflow state, history, or entry
points; remove READMEs that only restate the obvious contents of a tiny folder.

## Evidence Snapshot

The tree shape matters here more than the file extension itself. These are the
most relevant first-party clusters:

| Folder | Shape | What the path already communicates |
| --- | --- | --- |
| [docs/](README.md) | small top-level documentation hub | The directory is a docs umbrella, but not which docs are stable entry points. |
| [docs/chapters/](chapters/README.md) | multi-chapter investigation system | The folder name alone does not tell you which chapter is active, historical, or archived. |
| [docs/reference/](reference/README.md) | stable cross-cutting reference shelf | The folder name says “reference,” but not how the topics are grouped. |
| [docs/reference/core/](reference/core/README.md) | 5 maintained topic docs | The filenames are descriptive; the README mostly acts as a local table of contents. |
| `docs/reference/backends/` | 4 backend notes | The filenames already carry most of the semantic load. |
| `docs/reference/workflow/` | 2 workflow notes | The folder is small enough that the parent index already does much of the work. |
| [docs/chapters/neighborhood-claim-alignment/](chapters/neighborhood-claim-alignment/README.md) | active chapter with sibling specs, archives, and working docs | The folder name is meaningful, but not enough to orient a reader to the live loop and current blockers. |
| `docs/chapters/neighborhood-claim-alignment/working/` | single-file helper folder | The folder name already says this is working material. |
| `docs/chapters/neighborhood-claim-alignment/archive/` | single-file helper folder | The folder name already says this is archival material. |
| [docs/chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/](chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md) | spec archive with 3 companion docs | The README helps distinguish a completed spec package from ordinary notes. |
| [dev/](../dev/README.md) | developer workspace | The directory name is generic; the README gives it structure. |
| [dev/tests/](../dev/tests/README.md) | test suite organization hub | The directory name tells you where tests live, not how to place them. |
| `dev/scripts/maintenance/` | 3 scripts plus cache noise | The directory already reads like an implementation bucket for maintenance helpers. |
| `dev/scripts/benchmarks/` | 1 benchmark helper | The filename of the script already explains most of the folder. |

## Keep

These files provide distinct value that the directory names alone do not carry.

| File | Why it is worth keeping |
| --- | --- |
| [README.md](../README.md) | Root-level project overview, setup, workflows, and entry points. |
| [docs/README.md](README.md) | Documents the repository documentation structure and reading order. |
| [docs/chapters/README.md](chapters/README.md) | Defines the chapter system, lifecycle, and navigation between active and historical chapters. |
| [docs/reference/README.md](reference/README.md) | Groups cross-cutting reference material by topic and keeps the shelf small. |
| [dev/README.md](../dev/README.md) | Explains the developer workspace layout and points contributors at tests and helper scripts. |
| [dev/tests/README.md](../dev/tests/README.md) | Test placement rules are not obvious from directory names alone; this is a real contributor guide. |
| [docs/chapters/neighborhood-claim-alignment/README.md](chapters/neighborhood-claim-alignment/README.md) | Active chapter state, working questions, scope, and current loop are all useful context that the folder name cannot express. |
| [docs/chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md](chapters/neighborhood-claim-alignment/comparison-layout-smoothing-spec/README.md) | Archive/spec container with requirements, design, tasks, and execution notes. |
| [docs/chapters/candidate-generation-handoff/README.md](chapters/candidate-generation-handoff/README.md) | Historical handoff context is still useful when tracing how the active chapter started. |

This is the part of the repository where a README does real work:

- [docs/chapters/README.md](chapters/README.md) resolves chapter lifecycle questions that the folder tree cannot answer.
- [docs/chapters/neighborhood-claim-alignment/README.md](chapters/neighborhood-claim-alignment/README.md) explains the active loop, the current hypotheses, and the next diagnostic surface.
- [docs/reference/README.md](reference/README.md) provides a stable shelf for durable cross-cutting docs.
- [dev/tests/README.md](../dev/tests/README.md) is a contributor guide, not a file index.

If these files disappeared, the repository would become less self-explanatory even
if every filename stayed the same.

## Collapse Or Replace

These files are useful, but their value is mostly structural. If reducing README count is a priority, these are the places where a single higher-level index or a more descriptive filename could replace them.

| File | Why it is a collapse candidate |
| --- | --- |
| [docs/reference/core/README.md](reference/core/README.md) | Good topic index, but it overlaps heavily with [docs/reference/README.md](reference/README.md). |
| `docs/reference/backends/README.md` | Helpful grouping, but the backend filenames already communicate most of the topic boundaries. |
| `docs/reference/workflow/README.md` | Same pattern as the other reference sub-indexes; useful, but not essential if the parent index becomes richer. |
| `dev/scripts/maintenance/README.md` | The directory only contains three scripts plus cache noise; a shorter `dev/scripts/README.md` or better script names could cover the same ground. |
| `dev/scripts/benchmarks/README.md` | This folder has one benchmark helper; the directory and script name already explain most of the purpose. |
| `docs/chapters/neighborhood-claim-alignment/working/README.md` | The directory currently contains only this README, so it behaves like a thin index rather than a distinct layer. |
| `docs/chapters/neighborhood-claim-alignment/archive/README.md` | Same as `working/`; the folder name already signals archival status. |

The strongest collapse argument is not that these READMEs are wrong. It is that
their parent folders already provide enough semantic context for the remaining
contents.

In particular:

- [docs/reference/core/README.md](reference/core/README.md) overlaps with the parent reference index and could become a parent-only topic list.
- `docs/reference/backends/README.md` and `docs/reference/workflow/README.md` are small enough that their topic grouping could move into [docs/reference/README.md](reference/README.md).
- `dev/scripts/maintenance/README.md` could be replaced by a better-named parent folder such as `repo-maintenance/` or a single `dev/scripts/README.md`.
- `dev/scripts/benchmarks/README.md` is the clearest case where the script name and folder name already carry nearly all the explanation.

## Easiest Removal Candidates

If the goal is to remove README files with the least loss of information, these look like the lowest-risk first cuts.

1. `dev/scripts/benchmarks/README.md`
2. `docs/chapters/neighborhood-claim-alignment/working/README.md`
3. `docs/chapters/neighborhood-claim-alignment/archive/README.md`
4. `dev/scripts/maintenance/README.md`

These are the easiest because they either describe a tiny directory, duplicate a folder name, or act as a local index with very little unique content.

The order here matters:

1. `docs/chapters/neighborhood-claim-alignment/working/README.md` is the most removable because the folder itself is a single-purpose stub.
2. `docs/chapters/neighborhood-claim-alignment/archive/README.md` is equally thin and similarly recoverable from context.
3. `dev/scripts/benchmarks/README.md` is low-risk because it indexes one script whose name already advertises its purpose.
4. `dev/scripts/maintenance/README.md` is still small, but slightly more defensible because it spans several helper scripts.

The chapter spec folders under [docs/chapters/neighborhood-claim-alignment/](chapters/neighborhood-claim-alignment/) also show that not every doc cluster needs a README. The adjacent folders [large-module-refactor-spec/](chapters/neighborhood-claim-alignment/large-module-refactor-spec/) and [parity-workflow-completion-spec/](chapters/neighborhood-claim-alignment/parity-workflow-completion-spec/) work without one because the directory names already announce a single well-defined artifact bundle.

## Higher-Value README Pattern

The READMEs that seem most justified share at least one of these traits:

- They explain live state or an active workflow.
- They define conventions that are not obvious from the directory tree.
- They serve as a stable entry point for a clustered set of topic files.
- They preserve historical handoff context that would otherwise be hard to recover.

That is why the root README, the documentation indexes, the active chapter README, and the test-organization README are high-value even in a stricter naming-first repository.

## Naming Levers That Reduce README Need

If the goal is not just to delete README files but to make the whole tree more
immediately understandable, the biggest leverage comes from folder names that
describe function instead of maintenance category.

The clearest examples are:

- `dev/scripts/maintenance/` could be renamed toward the purpose of the tools, such as `repo-maintenance/` or `mapping-maintenance/`.
- `dev/scripts/benchmarks/` could become `profiling/` or `perf-probes/` if the intent is manual timing rather than reusable benchmark infrastructure.
- `docs/chapters/neighborhood-claim-alignment/working/` and `archive/` are semantically clear but structurally redundant because their contents are already organized by the parent chapter.
- `docs/reference/core/`, `backends/`, and `workflow/` are fine names, but they become more self-contained if the parent index is expanded and the child READMEs are removed.

The important constraint is that name changes should pay back in scan speed. If a rename requires opening fewer files but creates a more ambiguous folder label, that is not an improvement.

## Bottom Line

If the repository is optimized for first-glance clarity, the strongest candidates for removal are the `working/` and `archive/` chapter stubs and the tiny helper indexes under `dev/scripts/`. The strongest candidates to keep are the root README, the docs index layer, the active chapter README, the chapter system index, and the test placement guide.

The general naming principle that falls out of this audit is:

- keep a README when it names a workflow or state transition that folder names cannot encode cleanly
- remove a README when it only repeats a folder label and a short file list
