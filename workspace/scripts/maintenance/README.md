# Maintenance Scripts

These scripts support repository upkeep and MATLAB audit work. Run them from
the repository root.

## Files

| File | Purpose |
| --- | --- |
| `check_mapped.py` | Refresh the generated appendix in `docs/MATLAB_MAPPING.md` with upstream `.m` files not called out in the maintained mapping tables |
| `find_matlab_scripts.py` | List MATLAB `.m` files under the upstream checkout and `workspace/` that behave like scripts instead of functions |
