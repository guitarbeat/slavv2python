# SLAVV Parity Implementation Guide

[Up: Reference Docs](../README.md)

This guide provides technical instructions and best practices for aligning the SLAVV Python implementation with the released MATLAB ground truth. Use it when investigating match rate gaps or implementing new parity-proven features.

---

## 🚀 Parity Workflow

### 1. Identify the Gap
Use `dev/scripts/cli/parity_experiment.py` to measure current match rates against a canonical oracle.
```powershell
python dev/scripts/cli/parity_experiment.py prove-exact --stage edges ...
```

### 2. Categorize missing pairs
Extract a sample of MATLAB pairs that Python fails to generate. Group them by topological patterns:
- **Hub Vertices**: High-degree vertices missing many connections.
- **Filtering**: Pairs present in temporary maps but missing from final output.
- **Growth**: Traces that terminate early or follow different local geodesics.

### 3. Source Audit
Locate the relevant logic in `external/Vectorization-Public/source/get_edges_by_watershed.m`.
- **Note**: Be careful with commented-out code in MATLAB. Always verify if a line is active.
- **Note**: MATLAB uses Fortran-style (1-based, column-major) indexing. Python implementation uses `ravel(order="F")` to match this layout.

---

## 🛠️ Debugging Techniques

### Execution Tracing
When a specific edge is missing, use the `ExecutionTracer` to compare Python events against MATLAB:
1. Verify `current_strel_energies` matches MATLAB's local field.
2. Verify `adjusted` energies (after penalties) lead to the same `strel_idx`.
3. Check `available_locations` insertion to ensure the frontier priority is maintained.

### Unit Testing
Always add a regression test in `dev/tests/unit/core/test_global_watershed_comprehensive.py` before applying a fix.
- Use small 3x3x3 or 5x5x5 volumes to verify exact numerical parity.
- Mock the `available_locations` list to test specific insertion/removal edge cases.

---

## 📚 Key Reference Files

- `docs/reference/core/EXACT_PROOF_FINDINGS.md`: Live status of proven vs. diverging logic.
- `docs/reference/core/MATLAB_PARITY_MAPPING.md`: Structural map between MATLAB functions and Python modules.
- `docs/reference/core/WATERSHED_IMPLEMENTATION_NOTES.md`: Detailed technical notes on watershed parity and divergence patterns.
- `TODO.md`: Prioritized task list for remaining parity work.
