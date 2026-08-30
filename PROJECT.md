# Project: MATLAB-to-Python Transpilation & Differential Audit

## Architecture
This project executes an automated transpilation pipeline, structural/AST comparison engine, and synthetic differential validation suite to audit `external/Vectorization-Public/` against `slavv_python/`.

### Subsystems & Data Flow
1. **Transpilation Pipeline (`workspace/experiments/matlab2python_audit/tools/transpile_m2py.py`)**:
   - Parses MATLAB `.m` source trees using `miss_hit_core` AST visitor.
   - Generates raw and cleaned Python implementations organized into stage folders (`raw_transpiled/`, `cleaned_transpiled/`).
   - Emits `transpilation_manifest.json`.

2. **Structural & AST Differ (`workspace/experiments/matlab2python_audit/tools/ast_differ.py`)**:
   - Parses ASTs of transpiled Python and corresponding `slavv_python/` implementations using Python `ast`.
   - Extracts and compares control flow branches, loops, early returns, mathematical scaling, coordinate indexing (`[Y, X, Z]` Fortran order vs `[X, Y, Z]`), and constants.
   - Generates itemized comparison matrices (`ast_diffs/ast_comparison_matrix.json`).

3. **Synthetic Differential Execution Validator (`workspace/experiments/matlab2python_audit/tools/synthetic_validator.py`)**:
   - Runs in `production_probe` mode: exercises isolated synthetic fixtures through **`slavv_python` helpers only** (honest ProductionProbe / C2). Does **not** dual-run `cleaned_transpiled` modules as the behavioral surface.
   - Measures numerical tolerances, topological invariants, and records classifications in `synthetic_tests/validation_results.json`. Probe green is an audit aid, **not** Phase 1 Certification.

4. **Actionable Findings Compiler (`workspace/experiments/matlab2python_audit/tools/compile_audit_report.py`)**:
   - Aggregates transpilation inventory, AST comparison matrix, and synthetic behavioral findings into `workspace/experiments/matlab2python_audit/reports/AUDIT_REPORT.md` (and `AUDIT_REPORT.md` at repository root).

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Preprocessing Transpilation | Transpile `pre_processing.m`, `fix_intensity_bands.m`, `gaussian_blur.m`, `construct_structuring_element.m` | M1 | Survey 1 & 2 |
| F2 | Energy Stage Transpilation | Transpile `get_energy_V202.m`, `energy_filter_V200.m`, `get_filter_kernel.m`, `fourier_transform_V2.m`, `get_vessel_directions_V3.m` | M1 | Survey 1 & 2 |
| F3 | Vertices Stage Transpilation | Transpile `get_vertices_V200.m`, `choose_vertices_V200.m`, `paint_vertex_image.m`, `crop_vertices_V200.m` | M1 | Survey 1 & 2 |
| F4 | Edges Stage Transpilation | Transpile `get_edges_by_watershed.m`, `choose_edges_V200.m`, `add_vertices_to_edges.m`, `smooth_edges_V2.m`, `clean_edges*.m`, `sort_edges.m` | M1 | Survey 1 & 2 |
| F5 | Network Stage Transpilation | Transpile `get_network_V190.m`, `get_strand_objects.m`, `sort_network_V180.m`, `combine_strands.m` | M1 | Survey 1 & 2 |
| F6 | Transpilation Output Structuring | Organize raw and cleaned outputs into stage-based directories under `workspace/experiments/matlab2python_audit/` | M1 | Survey 3 |
| F7 | AST Feature Extractor | Parse Python ASTs to extract functions, loops, branches, early returns, math operators, slice/coordinate access | M2 | Survey 3 |
| F8 | AST Discrepancy Differ | Identify branch omissions, loop discrepancies, coordinate order deviations, and scaling differences | M2 | Survey 1, 2, 3 |
| F9 | AST Comparison Matrix | Output structured comparison matrix (`ast_comparison_matrix.json` and stage summaries) | M2 | Survey 3 |
| F10 | Synthetic Preprocessing & Energy Tests | Execute mock 3D volumes through `slavv_python/pipeline/energy` kernels under `production_probe` | M3 | Survey 2 & 3 |
| F11 | Synthetic Vertices Tests | Execute mock energy extrema fields through `slavv_python/pipeline/vertices` selection under `production_probe` | M3 | Survey 2 & 3 |
| F12 | Synthetic Edges & Watershed Tests | Execute mock seeds & catchment basins through `slavv_python/pipeline/edges` selection & cleanup under `production_probe` | M3 | Survey 2 & 3 |
| F13 | Synthetic Network & Strand Tests | Execute mock edge graphs through `slavv_python/pipeline/network` strand assembly & smoothing under `production_probe` | M3 | Survey 2 & 3 |
| F14 | Behavioral Divergence Classification | Classify AST differences into true logic divergences vs benign syntax/framework artifacts | M3 | Survey 3 |
| F15 | Complete Transpiled vs Python Inventory | Compile per-file and per-function mapping matrix in audit report | M4 | Survey 1 & 2 |
| F16 | Verified Code Defects / Deviations Report | Document file names, line numbers, root cause, and actionable recommendations for genuine bugs | M4 | ORIGINAL_REQUEST R4 |
| F17 | Filtered Artifacts Catalog | Document benign transpiler and architecture artifacts filtered out of defect findings | M4 | ORIGINAL_REQUEST R4 |
| F18 | Final AUDIT_REPORT.md Delivery | Publish finalized, comprehensive report at `workspace/experiments/matlab2python_audit/reports/AUDIT_REPORT.md` and repository root | M4 | ORIGINAL_REQUEST R4 |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Comprehensive MATLAB Transpilation | Implement `transpile_m2py.py` using `miss_hit_core`; convert all core files across 5 stages; organize `raw_transpiled/` and `cleaned_transpiled/` | none | DONE |
| 2 | M2: Structural & AST Comparison Engine | Implement `ast_differ.py`; extract control flow, math, coordinate mappings, constants; generate `ast_comparison_matrix.json` | M1 | DONE |
| 3 | M3: Synthetic Input Behavioral Validation | Implement `synthetic_validator.py` and test harnesses across all 5 stages; execute differential runs; classify discrepancies | M2 | DONE |
| 4 | M4: Actionable Findings Report Delivery | Implement `compile_audit_report.py`; produce full `AUDIT_REPORT.md` deliverable with inventory, verified defects, and filtered artifacts | M3 | DONE |

---

## Interface Contracts
### M1 (Transpiler) ↔ M2 (AST Differ)
- Directory Contract: `workspace/experiments/matlab2python_audit/cleaned_transpiled/<stage>/<module>.py`
- Manifest Contract: `workspace/experiments/matlab2python_audit/transpilation_manifest.json`
  - Format: `{"modules": [{"matlab_file": str, "stage": str, "transpiled_raw": str, "transpiled_cleaned": str, "python_counterpart": str, "status": str}]}`

### M2 (AST Differ) ↔ M3 (Synthetic Validator)
- Matrix Contract: `workspace/experiments/matlab2python_audit/ast_diffs/ast_comparison_matrix.json`
  - Format: `{"comparisons": [{"matlab_module": str, "python_module": str, "stage": str, "branch_diffs": list, "loop_diffs": list, "math_diffs": list, "coord_diffs": list, "severity": str}]}`

### M3 (Synthetic Validator) ↔ M4 (Audit Report Generator)
- Results Contract: `workspace/experiments/matlab2python_audit/synthetic_tests/validation_results.json`
  - Format: `{"test_results": [{"stage": str, "test_name": str, "target_modules": list, "passed": bool, "max_diff": float, "divergence_detected": bool, "details": dict}]}`

### M4 Deliverable Contract
- Deliverable Files:
  - `workspace/experiments/matlab2python_audit/reports/AUDIT_REPORT.md`
  - `AUDIT_REPORT.md` (root level)

---

## Code Layout
- Target Workspace: `d:/2P_Data/Aaron/slavv2python/workspace/experiments/matlab2python_audit/`
  - `tools/`:
    - `transpile_m2py.py` (M1: Transpiler AST visitor & converter, <= 1000 lines)
    - `ast_differ.py` (M2: Structural AST extractor & comparison engine, <= 1000 lines)
    - `synthetic_validator.py` (M3: Synthetic differential runner & test cases, <= 1000 lines)
    - `compile_audit_report.py` (M4: Audit report compiler & markdown builder, <= 1000 lines)
  - `raw_transpiled/`: Direct AST-generated Python files per stage
  - `cleaned_transpiled/`: Formatted, runnable Python modules per stage
  - `ast_diffs/`: AST feature trees, comparison matrices, diff JSONs
  - `synthetic_tests/`: Test cases, fixtures, and differential execution reports
  - `reports/`: `AUDIT_REPORT.md`
