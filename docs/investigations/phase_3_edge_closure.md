# Investigation: Phase 3 Final Edge Closure

**Start Date:** 2026-05-23
**Status:** Active
**Goal:** Achieve 100% bit-accurate edge parity with MATLAB `vectorize_V200.m`.

---

## 📓 Journal of Findings

### 1. Seed-Aware Tie-Breaking (LIFO vs FIFO)
**Date:** 2026-05-23
**Observation:** MATLAB's `find(..., 1, 'last')` behavior for the first seed of a vertex creates a LIFO priority for identical energies. Subsequent seeds from the same origin use `find(..., 1, 'first')`, creating a FIFO priority.
**Impact:** Divergence at high-degree junctions where multiple voxels have symmetric penalized energies.
**Resolution:** Implemented `seed_idx`-aware branching in `insert_available_location`.

### 2. Path Fluidity (Competitive Claiming)
**Date:** 2026-05-23
**Hypothesis:** Python's "first-come-first-served" voxel claiming is too rigid. MATLAB's frontier logic allows a "better" path (lower energy or lower index tie-break) to overwrite an existing claim if it hasn't been finalized yet.
**Metric Correlation:** 452 extra pairs / 287 missing pairs in v10 suggest "path locking" is preventing some valid connections from forming.
**Planned Action:** Modify `reveal_unclaimed_strel` to support competitive claiming based on the same bit-accurate tie-breaking rules as the frontier.

### 3. Static Frontier Priority (The "Static Map" Breakthrough)
**Date:** 2026-05-23
**Observation:** MATLAB maintains two energy maps: `energy_map_temp` (read-only except for vertex resets) and `energy_map` (dynamic updates for penalties). 
**Impact:** Python's "Unified Map" was incorrectly allowing directional penalties to influence the expansion priority. In MATLAB, penalties only affect the *resulting trace values*, not the *order of voxel discovery*.
**Resolution:** Reverted energy map unification. Enforced `energy_map_temp` as the authoritative static source for frontier priority and binary search.

---

## 📈 Parity High-Water Marks

| Version | Match Rate | Matched | Missing | Extra | Key Change |
| :--- | :--- | :--- | :--- | :--- | :--- |
| v29 | 88.7% | 1062 | 135 | 371 | Parameter Alignment (Tolerance=4) |
| v10 | 76.0% | 910 | 287 | 452 | Precision Alignment (v1) |
| v11 | *TBD* | - | - | - | LIFO/FIFO + Competitive Claiming |
