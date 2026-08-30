# Original User Request

## Initial Request — 2026-08-14T19:15:07Z

Automate full-pipeline transpilation of MATLAB source code from `external/Vectorization-Public/` using `matlab2python` and execute an end-to-end structural, AST, and synthetic-input differential audit against `slavv_python/` to find overlooked logic, calculation discrepancies, and hidden edge cases.

Working directory: d:/2P_Data/Aaron/slavv2python/workspace/experiments/matlab2python_audit
Integrity mode: development

## Requirements

### R1. Comprehensive MATLAB Transpilation
Install and run `matlab2python` across all core MATLAB pipeline modules in `external/Vectorization-Public/` (Preprocessing, Energy, Vertices, Edges/Watershed/Selection, Network/Strands). Save the raw and cleaned transpiled outputs under the working directory.

### R2. Structural & AST Comparison Engine
Develop an automated analysis tool to compare each transpiled module against its corresponding `slavv_python/` implementation. Detect:
- Unimplemented MATLAB branches, loops, or early returns
- Discrepancies in mathematical operations, scaling, constants, or coordinate axis mappings
- Logic order differences (e.g., preprocessing before vs after sorting/resampling)

### R3. Synthetic Input Behavioral Validation
For core algorithmic discrepancies found during static analysis, write isolated synthetic test cases that exercise **`slavv_python` production helpers only** under `production_probe` mode (honest ProductionProbe / C2). Do **not** dual-run `cleaned_transpiled` modules as the behavioral surface — static AST flags still require a failing synthetic or oracle differential before any `GENUINE_BEHAVIORAL_DIVERGENCE` claim. Probe green is **not** Phase 1 Certification.

### R4. Actionable Findings Report
Deliver a structured findings document (`AUDIT_REPORT.md`) containing:
1. Complete inventory of transpiled modules vs `slavv_python` modules.
2. Verified genuine logic/code defects or deviations in `slavv_python`.
3. Filtered-out transpiler artifacts (syntax-only differences that have no numerical/topological impact).

## Acceptance Criteria

### Execution & Tooling
- [x] `matlab2python` successfully converts MATLAB files across all 5 pipeline stages.
- [x] Transpiled Python code is organized into stage-based directories.

### Verification & Test Suite
- [x] Automated AST / structural diff script runs cleanly across all paired modules and outputs an itemized comparison matrix.
- [x] Synthetic execution tests validate behavior for top suspected discrepancies.

### Final Deliverable
- [x] `AUDIT_REPORT.md` is produced with precise file names, line numbers, and actionable recommendations for any true divergences discovered.
