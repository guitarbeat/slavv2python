---
name: workspace-hygiene
description: Keep the slavv2python repository workspace clean and organized. Use when the user asks to clean the worktree, remove dirty files, tidy the workspace, organize scratch artifacts, prevent root-level junk, or capture/apply workspace hygiene rules.
---

# Workspace Hygiene

Keep the repo root clean, preserve valuable experiment state, and verify the final Git state.

## Workflow

1. Start from the repository root and inspect:
   - `git status --short --branch`
   - `git diff --name-only`
   - `git diff --name-only --cached`
   - root-level untracked files and directories.
2. Classify each item before changing anything:
   - **Tracked edits**: code, docs, tests, config, or generated tracked files.
   - **Untracked scratch**: one-off diagnostics, logs, prompts, temporary reports, or agent output.
   - **Experiment state**: anything under `workspace/runs`, `workspace/oracles`, `workspace/datasets`, `workspace/reports`, or `workspace/scratch`.
   - **Build/cache output**: cache directories such as `.pytest_cache`, `.ruff_cache`, `.mypy_cache`, egg-info, or tool caches.
3. Preserve by default:
   - Never delete `workspace/runs`, `workspace/oracles`, `workspace/datasets`, or `workspace/reports` without explicit user approval.
   - Never revert tracked user edits unless the user asks for a clean worktree or explicitly names the files.
   - Never use `git reset --hard`; prefer `git restore` for tracked paths and targeted `git clean -fd -- <paths>` for known untracked junk.
4. Route clutter:
   - One-off scripts, logs, prompts, diffs, and diagnostic artifacts belong in `workspace/scratch`, not the repo root.
   - Test artifacts belong in `workspace/scratch/tmp_tests/` through the repo-local test fixtures.
   - Durable fixes or runbooks belong in `docs/solutions`; live parity status belongs in `docs/reference/core/EXACT_PROOF_FINDINGS.md`; active tasks belong in `docs/TODO.md`.
5. Clean deliberately:
   - For a user request like "clean it all up" or "I don't want anything dirty," revert all unstaged/staged tracked changes to `HEAD` and remove untracked files/directories that are not preserved experiment state.
   - If an untracked item may contain useful evidence, move it to `workspace/scratch` instead of deleting it, unless the user asked for a fully clean worktree.
6. Verify and report:
   - Re-run `git status --short --branch`.
   - Report what was reverted, removed, moved, or preserved.
   - The goal state for a full cleanup is exactly `## main...origin/main` or the equivalent clean branch line with no extra entries.

## Guardrails

- Do not clean broad ignored files with `git clean -fdx` unless the user explicitly asks to remove ignored caches and build outputs.
- Do not remove parity job logs, PID manifests, or proof artifacts from active run roots while a related process is alive.
- If `workspace/scratch/crop_energy_rerun_latest.pid` or a run-local `99_Metadata/parity_job.pid` is alive, report it before deleting or moving related artifacts.
