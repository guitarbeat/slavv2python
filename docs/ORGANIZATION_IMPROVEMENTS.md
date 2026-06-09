# Organization Improvements (June 2026)

## Problem
It was unclear why the repository has four top-level folders (`slavv_python/`, `tests/`, `scripts/`, `workspace/`) and when to use each one.

## Solution
Created comprehensive documentation to clarify folder purposes and boundaries.

### New Documentation

1. **[FOLDER_PURPOSE_GUIDE.md](reference/core/FOLDER_PURPOSE_GUIDE.md)** — Core guide explaining:
   - What each folder contains and why
   - Who uses each folder
   - What gets committed to Git vs gitignored
   - Decision tree for "where does my code go?"
   - Common confusion points and anti-patterns

2. **[scripts/README.md](../scripts/README.md)** — Developer scripts guide:
   - What scripts are maintained utilities
   - When to add code to scripts/ vs elsewhere
   - Import rules (scripts can import slavv_python, not vice versa)
   - Testing guidelines for scripts

3. **[workspace/README.md](../workspace/README.md)** — Experiment workspace guide:
   - Directory structure and purpose
   - Common workflows (oracle promotion, parity experiments)
   - File size considerations
   - Why it's gitignored

### Updated Documentation

- **[AGENTS.md § Repository Map](../AGENTS.md#repository-map)** — Added folder purpose summary and link to detailed guide
- **[AGENTS.md § Key Reference Documents](../AGENTS.md#key-reference-documents)** — Added Folder Purpose Guide to table
- **[README.md § What Is In This Repo](../README.md#what-is-in-this-repo)** — Added folder explanations and links
- **[docs/README.md § Core Maintained References](README.md#core-maintained-references)** — Added guide to index

## Key Insights

### Four Folders, Four Lifecycles

| Folder | Committed | Installed | Audience | Purpose |
|--------|-----------|-----------|----------|---------|
| `slavv_python/` | ✅ Yes | ✅ Yes | End users + devs | Production package |
| `tests/` | ✅ Yes | ❌ No | Devs + CI | Verification |
| `scripts/` | ✅ Yes | ❌ No | Devs only | Utilities |
| `workspace/` | ❌ No | ❌ No | Local dev | Experiments |

### Decision Tree

```
Is it part of the public product?
├─ YES → slavv_python/
└─ NO
   ├─ Does it test production code?
   │  ├─ YES → tests/
   │  └─ NO
   │     ├─ Is it a reusable developer tool?
   │     │  ├─ YES → scripts/
   │     │  └─ NO → workspace/scratch/
```

### Import Boundaries

**Allowed:**
- `tests/` → `slavv_python/`
- `scripts/` → `slavv_python/`
- `slavv_python/` → `slavv_python/` (internal)

**Forbidden:**
- `slavv_python/` → `scripts/` ❌
- `slavv_python/` → `tests/` ❌
- `slavv_python/` → `workspace/` ❌

## Common Confusion Resolved

### Q: Should `parity_experiment.py` be in `slavv_python/`?
**A:** No. It's a developer-only tool for MATLAB parity work, not part of the public product. It stays in `scripts/`.

### Q: Should I commit my `workspace/` folder?
**A:** No. It's gitignored because it contains large datasets, personal experiment runs, and temporary files.

### Q: Where do MATLAB parity tests go?
**A:** 
- Pre-gate integration tests → `tests/integration/parity/`
- Parity harness production code → `slavv_python/analytics/parity/`
- Parity experiment CLI tool → `scripts/parity_experiment.py`
- Oracle vectors and run outputs → `workspace/oracles/` and `workspace/runs/`

### Q: Where do one-off exploration scripts go?
**A:** `workspace/scratch/` — They're throwaway and not maintained.

## Impact

This documentation provides:
- Clear boundaries between production, test, utility, and experiment code
- Guidance for where new code should live
- Explanation of Git and pip install behavior for each folder
- Decision-making framework for contributors
- Reduced confusion about repository organization

## Related Documentation

- [FOLDER_PURPOSE_GUIDE.md](reference/core/FOLDER_PURPOSE_GUIDE.md) — Complete guide
- [scripts/README.md](../scripts/README.md) — Scripts directory guide
- [workspace/README.md](../workspace/README.md) — Workspace directory guide
- [tests/README.md](../tests/README.md) — Test organization guide
