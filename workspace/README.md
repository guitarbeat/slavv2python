# Workspace Guide

This folder contains operational artifacts that support development and parity workflows.

## Layout

- `reports/`: point-in-time investigation notes, audits, and release run summaries.
- `scripts/`: helper scripts for parity CLI wrappers, maintenance tasks, and benchmarks.
- `tmp_tests/`: repo-local pytest temp root (managed by `tests/conftest.py`).
- `tmp_debug_cli_case*/`: ad-hoc debug fixtures and captured comparison metadata.

## Organization Rules

- Keep code under `source/` and tests under `tests/`; `workspace/` is for supporting artifacts only.
- Prefer date-stamped report filenames (`*_YYYY-MM-DD.md`) for new reports.
- Keep `workspace/scripts/` stable; other docs/tests reference these paths directly.
- Do not commit generated `__pycache__/` content.

## Cleanup Checklist

Run periodically from repo root:

```powershell
Get-ChildItem workspace -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
```

Before deleting temp folders, check for references in docs/reports:

```powershell
rg -n "workspace/tmp_|tmp_debug_cli_case" docs workspace tests README.md
```
