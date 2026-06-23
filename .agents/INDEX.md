# .agents Directory Index

Quick reference catalog of all agents, instructions, rules, skills, and orchestrations.

**Last Updated**: 2026-06-23

## 📌 Current Handoff

| Brief | Location | Purpose |
|-------|----------|---------|
| **Phase 1 Parity Handoff** | [HANDOFF.md](HANDOFF.md) | Successor brief: decision point, operating sequence, canonical record links |

Ephemeral agent harness output belongs in `workspace/scratch/agent-tools/`, not the repo root (`agent-tools/`, `terminals/` are gitignored).

---

## 🌙 Active Orchestrations

No active orchestration artifacts are retained. The completed overnight
crop-Energy attempt was consolidated into [HANDOFF.md](HANDOFF.md),
[docs/TODO.md](../docs/TODO.md), and
[EXACT_PROOF_FINDINGS.md](../docs/reference/core/EXACT_PROOF_FINDINGS.md).

---

## 🤖 Agents

### Specialist Agents
| Agent | Keywords | Purpose |
|-------|----------|---------|
| **MATLAB Parity Specialist** | parity, MATLAB, edges, watershed, exact proof | Preserve and improve MATLAB-Python parity |
| **Docs Link Auditor** | docs audit, link check, documentation | Audit markdown docs for broken links and drift |
| **Python Refactor + Tests** | refactor, tests, code quality | Refactor Python code with test coverage |

### Utility Agents
| Agent | Keywords | Purpose |
|-------|----------|---------|
| **CI Watcher** | ci, github actions, build | Monitor CI pipeline health |
| **Thermo-Nuclear Code Quality Review** | code review, quality, standards | Deep code quality analysis |

**Location**: `agents/*.agent.md`  
**Guide**: `agents/README.md`

---

## 📋 Instructions

Scoped instructions that apply conditionally based on file patterns.

| Instruction | ApplyTo Pattern | Purpose |
|------------|-----------------|---------|
| **Parity-Safe Change** | `analytics/parity/**`, `engine/state/**` | Enforce staged layout compatibility for parity code |
| **Doc Path Hygiene** | `docs/**/*.md` | Maintain accurate paths in documentation |
| **Tests Placement** | `tests/**/*.py` | Ensure tests go in correct ownership folders |

**Location**: `instructions/*.instructions.md`  
**Guide**: `instructions/README.md`

---

## 📏 Rules

Global rules that always apply.

| Rule | Scene | Purpose |
|------|-------|---------|
| **Git Commit Message** | `git_message` | Conventional commit format with scopes |
| **No Inline Imports** | Always | Prevent import statements inside functions |
| **Python Quality Verification** | Always | Type hints, docstrings, error handling |
| **TypeScript Exhaustive Switch** | Always | Exhaustive switch statements in TypeScript |

**Location**: `rules/*.md`, `rules/*.mdc`  
**Guide**: `rules/README.md`

---

## 💬 Prompts

Reusable prompt templates for common workflows.

| Prompt | Use Case |
|--------|----------|
| **Implement Python Change** | Standard implementation workflow |
| **Parity Experiment** | Run parity comparison workflow |
| **Regression Gate** | Full quality gate before commits |

**Location**: `prompts/*.prompt.md`  
**Guide**: `prompts/README.md`

---

## 🛠️ Skills

Reusable skills library organized by category.

### Parity & MATLAB
- `matlab-performance-optimizer/` - Optimize MATLAB-to-Python translations

### Code Quality
- `check-compiler-errors/` - Run type checking and compilation
- `thermo-nuclear-code-quality-review/` - Deep quality analysis
- `systematic-debugging/` - Structured debugging approach
- `verify-this/` - Verification workflows

### CI/CD & Testing
- `fix-ci/` - Repair broken CI pipelines
- `loop-on-ci/` - Iterate on CI failures
- `run-smoke-tests/` - Quick smoke test execution

### Development Workflow
- `new-branch-and-pr/` - Create branch and PR
- `review-and-ship/` - Code review and merge workflow
- `make-pr-easy-to-review/` - PR preparation
- `fix-merge-conflicts/` - Resolve merge conflicts

### Documentation
- `grill-with-docs/` - Documentation generation (ADR, context)
- `solution-note-generator/` - Create solution documents
- `workflow-from-chats/` - Extract workflows from conversations

### Analysis & Improvement
- `improve-codebase-architecture/` - Architecture analysis
- `workspace-hygiene/` - Clean up workspace
- `weekly-review/` - Weekly progress review
- `what-did-i-get-done/` - Accomplishment summary

### UI/UX
- `control-cli/` - CLI interaction
- `control-ui/` - UI interaction
- `make-interfaces-feel-better/` - UI/UX improvements
- `pr-review-canvas/` - PR review visualization

### Utilities
- `deslop/` - Code cleanup
- `find-agent-skills/` - Discover available skills
- `get-pr-comments/` - Fetch PR feedback
- `self-improving-agent/` - Agent self-improvement
- `translation-paper-author/` - Technical writing

**Location**: `skills/*/SKILL.md`  
**Guide**: `skills/README.md`

---

## 🎭 Orchestrations

### Active Orchestrations

None. Use the repository-level planning and status documents for Phase 1
coordination rather than creating a second task ledger under `.agents/`.

**Location**: `orchestrations/active/`

### Completed Orchestrations
Archives in `archive/` and `orchestrations/completed/`:
- `self_vertices_parity/` - Initial vertices parity attempt (2026-06-08)
- Related sub-agents: explorers, workers, reviewers, challengers, auditors
- `sentinel/` - Monitoring orchestration

**Archive Location**: `archive/`  
**Guide**: `orchestrations/README.md`

---

## 📑 Specs

Project and task specifications.

| Spec | Location |
|------|----------|
| **Parity Job Monitoring** | `specs/parity-job-monitoring/` |

**Location**: `specs/`

---

## 📊 Statistics

- **Total Agents**: 5
- **Instructions**: 3
- **Rules**: 4
- **Prompts**: 3
- **Skills**: 28
- **Active Orchestrations**: 0
- **Archived Orchestrations**: 11

---

## 🔍 Quick Search

### By Task Type
- **MATLAB Parity**: MATLAB Parity Specialist agent, parity-safe-change instruction
- **Documentation**: Docs Link Auditor agent, doc-path-hygiene instruction, grill-with-docs skill
- **Code Quality**: Python Refactor agent, quality rules, thermo-nuclear-code-quality-review skill
- **Testing**: tests-placement instruction, run-smoke-tests skill, verify-this skill
- **CI/CD**: ci-watcher agent, fix-ci skill, loop-on-ci skill

### By Scope
- **Parity Work**: `agents/matlab-parity-*.agent.md`, `instructions/parity-*.instructions.md`, `skills/matlab-*/`
- **Documentation**: `agents/docs-*.agent.md`, `instructions/doc-*.instructions.md`, `skills/*-docs/`
- **Testing**: `instructions/tests-*.instructions.md`, `skills/*-test*/`, `skills/verify-*/`

---

## 📝 Maintenance

### Adding New Content
1. **Agents**: Add to `agents/` with `.agent.md` extension
2. **Instructions**: Add to `instructions/` with `.instructions.md` extension
3. **Rules**: Add to `rules/` with `.md` or `.mdc` extension
4. **Prompts**: Add to `prompts/` with `.prompt.md` extension
5. **Skills**: Add directory under `skills/` with `SKILL.md` file
6. **Orchestrations**: Create in `orchestrations/active/`

### Archiving Content
1. Move completed orchestrations to `orchestrations/completed/` or `archive/`
2. Extract reusable patterns to `skills/`
3. Update this INDEX.md
4. Clean up temporary files

### This Index
Update this file when:
- Adding/removing agents, instructions, rules, prompts, or skills
- Completing orchestrations
- Reorganizing structure
- Changing naming conventions
