# Maintenance Scripts

These scripts support repository upkeep and MATLAB audit work. Run them from
the repository root.

## Files

| File | Purpose |
| --- | --- |
| `comparison_layout_smoothing.py` | Inventory legacy and grouped comparison runs, write machine-readable migration reports, and optionally apply experiment grouping, `status.json`, index, and pointer updates |
| `refresh_matlab_mapping_appendix.py` | Refresh the generated appendix in `docs/reference/MATLAB_MAPPING.md` with upstream `.m` files not called out in the maintained mapping tables |
| `find_matlab_script_files.py` | List MATLAB `.m` files under the upstream checkout and `dev/` that behave like scripts instead of functions |

