---
description: "Run the repository regression gate (compileall, ruff format check, ruff check, mypy, pytest unit/integration) and summarize failures by severity and likely fix order."
argument-hint: "Optional scope or changed files to emphasize"
agent: "agent"
---
Run the standard regression gate for this repository and summarize results by priority.

## Commands

1. `python -m compileall source dev/scripts`
2. `python -m ruff format --check source dev/tests`
3. `python -m ruff check source dev/tests`
4. `python -m mypy`
5. `python -m pytest -m "unit or integration"`

## Reporting Requirements

- Execute commands in order and keep going after failures when possible.
- Summarize findings as:
  - Critical blockers (must fix before merge)
  - High-priority issues (likely to break CI or behavior soon)
  - Medium-priority issues (cleanup/quality)
- For each failure, include:
  - Command that failed
  - File and line references when available
  - Probable root cause
  - Smallest practical fix direction
- End with a concise pass/fail matrix for all five commands.

