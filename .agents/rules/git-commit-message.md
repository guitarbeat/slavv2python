---
alwaysApply: true
scene: git_message
---

## Commit Message Format

Use conventional commit style with a scope when applicable:

```
<type>(<scope>): <short summary>

<optional body explaining why, not what>
```

### Types
- `fix` — Bug fix
- `feat` — New feature or capability
- `refactor` — Code restructuring without behavior change
- `docs` — Documentation only
- `test` — Adding or updating tests only
- `chore` — Build config, tooling, dependency updates
- `parity` — MATLAB exact parity work (match rate changes, proof fixes)
- `perf` — Performance improvement

### Scopes (optional but preferred)
- `energy`, `vertices`, `edges`, `network` — Processing stage
- `engine`, `state` — Pipeline orchestration and run tracking
- `cli`, `streamlit` — Interface surfaces
- `analytics`, `parity` — Analysis and proof harness
- `storage` — I/O loaders and exporters
- `workflows` — Pipeline profiles and orchestration helpers

### Rules
- Keep the summary line under 72 characters.
- Use imperative mood: "fix alignment" not "fixed alignment".
- Reference match rate changes in parity commits: `parity(edges): align frontier ordering (80%→88.7%)`.
- Do not include file lists in commit messages — that's what `git diff` is for.
- Group related changes into a single logical commit when possible.

### Examples
```
fix(edges): correct frontier insertion priority for hub vertices
parity(edges): tighten candidate filtering to match MATLAB (88.7%→92%)
docs: align all path references after package reorganization
refactor(engine): extract snapshot logic into dedicated module
test(core): add regression coverage for watershed join logic
chore: move stale root files to workspace/scratch/
```
