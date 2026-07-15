# Skills Catalog

Reusable agent skills for `slavv2python` (30 total). Each skill lives in
`<skill-name>/SKILL.md`. Invoke a skill by name; see individual `SKILL.md` files
for arguments and detailed steps.

> Discovery helper: use **find-agent-skills** to locate or evaluate skills for a
> task (including the external `npx skills` registry) before creating a new one.

| Skill | Purpose |
|-------|---------|
| [`check-compiler-errors`](check-compiler-errors/SKILL.md) | Run compile and type-check commands and report failures |
| [`consolidate-concepts`](consolidate-concepts/SKILL.md) | Deduplicate docs terms and code constants into one canonical home |
| [`control-cli`](control-cli/SKILL.md) | Build or adapt a local harness to drive, inspect, and profile an interactive CLI or TUI without external services. |
| [`control-ui`](control-ui/SKILL.md) | Build or adapt a local browser/CDP harness to drive and inspect a web, IDE, or Electron UI. |
| [`deslop`](deslop/SKILL.md) | Remove AI-generated code slop and clean up code style |
| [`find-agent-skills`](find-agent-skills/SKILL.md) | Discover and evaluate reusable agent skills for a requested task (local first, then the external `npx skills` registry). |
| [`fix-ci`](fix-ci/SKILL.md) | Find failing PR checks, inspect logs or external check links, and apply focused fixes |
| [`fix-merge-conflicts`](fix-merge-conflicts/SKILL.md) | Resolve merge conflicts non-interactively, validate build and tests, and finalize conflict resolution |
| [`get-pr-comments`](get-pr-comments/SKILL.md) | Fetch and summarize review comments from the active pull request |
| [`grill-with-docs`](grill-with-docs/SKILL.md) | Grilling session that challenges your plan against the existing domain model, sharpens terminology, and updates documentation (docs/AGENTS.md, ADRs) inline a... |
| [`improve-codebase-architecture`](improve-codebase-architecture/SKILL.md) | Find deepening opportunities in a codebase, informed by the domain language in docs/AGENTS.md and the decisions in docs/adr/. |
| [`loop-on-ci`](loop-on-ci/SKILL.md) | Monitor PR checks and fix failures until green. |
| [`make-interfaces-feel-better`](make-interfaces-feel-better/SKILL.md) | Design engineering principles for making interfaces feel polished. |
| [`make-pr-easy-to-review`](make-pr-easy-to-review/SKILL.md) | Prepare PRs for review by cleaning noisy history, improving PR descriptions, and adding reviewer guidance without changing code behavior. |
| [`matlab-performance-optimizer`](matlab-performance-optimizer/SKILL.md) | Optimize MATLAB code for better performance through vectorization, memory management, and profiling. |
| [`new-branch-and-pr`](new-branch-and-pr/SKILL.md) | Create a fresh branch, complete work, and open a pull request |
| [`pr-review-canvas`](pr-review-canvas/SKILL.md) | Generate an interactive PR review walkthrough as an HTML page. |
| [`prove-parity`](prove-parity/SKILL.md) | Run a SLAVV exact-parity proof for one pipeline stage (energy/vertices/edges/network) against the MATLAB oracle and summarize per-field pass/fail. |
| [`review-and-ship`](review-and-ship/SKILL.md) | Review the current branch for bugs, intent fit, and test coverage; run or write tests; commit focused work; open or update a PR. |
| [`run-smoke-tests`](run-smoke-tests/SKILL.md) | Run Playwright smoke tests, debug failures, and verify fixes |
| [`self-improving-agent`](self-improving-agent/SKILL.md) | Capture durable corrections, repeated workflow lessons, and agent-process improvements for slavv2python. |
| [`solution-note-generator`](solution-note-generator/SKILL.md) | Promote verified fixes, parity discoveries, integration resolutions, and reusable runbooks into searchable docs/solutions notes. |
| [`systematic-debugging`](systematic-debugging/SKILL.md) | Use when encountering any bug, test failure, or unexpected behavior, before proposing fixes |
| [`thermo-nuclear-code-quality-review`](thermo-nuclear-code-quality-review/SKILL.md) | Run an extremely strict maintainability review for abstraction quality, giant files, and spaghetti-condition growth. |
| [`translation-paper-author`](translation-paper-author/SKILL.md) | Synthesize lessons learned during the MATLAB-to-Python parity translation to author a comprehensive technical paper or internal engineering guide. |
| [`verify-this`](verify-this/SKILL.md) | Verify a claim with fresh local evidence: restate it falsifiably, capture baseline and treatment, compare artifacts, and return VERIFIED, NOT VERIFIED, or IN... |
| [`weekly-review`](weekly-review/SKILL.md) | Produce a weekly synthesis of authored commits with highlights by bugfix, tech debt, and net-new work |
| [`what-did-i-get-done`](what-did-i-get-done/SKILL.md) | Summarize authored commits over a user-specified time period into a concise update |
| [`workflow-from-chats`](workflow-from-chats/SKILL.md) | Extract durable working preferences from recent Cursor chats and convert them into skills, rules, or workflow docs. |
| [`workspace-hygiene`](workspace-hygiene/SKILL.md) | Keep the slavv2python repository workspace clean and organized. |

---

## Adding a skill

1. Create `<skill-name>/SKILL.md` with `name` and `description` frontmatter.
2. Keep the `description` a single clear sentence with explicit triggers.
3. Add the skill to the table above (kept in sync with the directory).
4. See [.claude/INDEX.md](../INDEX.md) for the full repository agent catalog.
