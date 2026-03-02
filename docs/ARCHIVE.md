# Archive Summary

This file keeps a minimal record of legacy project history after consolidation.

## Historical Highlights

- 2026-01-27: MATLAB vs Python comparison framework completed and documented.
- 2026-01-27: Post-mortem captured debugging outcomes and migration lessons.
- 2026-01-17: Technical debt assessment snapshot was recorded.

## Current Policy

- Detailed historical debug/status reports were consolidated and removed from the flat docs set.
- Canonical, actively maintained docs are:
  - `DEVELOPMENT.md`
  - `README.md`

## Recovering Removed Detail

If you need the old deep-dive archive documents, use Git history:

```bash
git log -- docs/
git show <commit>:docs/<old-file-name>.md
```
