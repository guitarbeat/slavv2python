# Maintenance Scripts

These scripts support repository upkeep and MATLAB audit work. Run them from
the repository root.

## Files

| File | Purpose |
| --- | --- |
| `refresh_matlab_mapping_appendix.py` | Refresh the generated appendix in `docs/reference/MATLAB_MAPPING.md` with upstream `.m` files not called out in the maintained mapping tables |
| `find_matlab_script_files.py` | List MATLAB `.m` files under the upstream checkout and `workspace/` that behave like scripts instead of functions |
