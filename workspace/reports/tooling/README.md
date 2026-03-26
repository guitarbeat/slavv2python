# Tooling Snapshots

This folder stores archived tool-output snapshots that were previously checked
into the repository root.

## Files

| File | Origin |
| --- | --- |
| `mypy_errors_20260323.txt` | Historical `python -m mypy` output from a local audit on March 23, 2026 |
| `pytest_errors_20260323.txt` | Historical `python -m pytest` collection failure output from a local audit on March 23, 2026 |
| `ruff_errors_20260323.txt` | Historical `python -m ruff check` output from a local audit on March 23, 2026 |

## Notes

- These are archival reference artifacts, not live status indicators.
- Prefer the canonical commands in `README.md` and `AGENTS.md` when you need
  current lint, type-check, or test results.
