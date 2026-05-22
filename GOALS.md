# SLAVV Curation and Parity Goals (GOALS.md)

This document defines the official engineering objectives, mathematical targets, and testing standards for the **SLAVV (Strand Localization and Vessel Vectorization)** Python codebase. It is aligned with the active priorities of **PAPER-001 (Public Paper Workflow Health)** and **PARITY-002/003 (Exact MATLAB Parity Milestone)**, established through joint developer alignment on May 21, 2026.

---

## 🗺️ Core Strategy: Balanced Parity & Stability

Our overall priority is a **balanced approach**:
1. **Maintain Public Workflow Stability (Critical):** The native-first, public-facing command surfaces (`slavv run`, `slavv analyze`, `slavv plot`) and the interactive user interface (`slavv-app`) must remain fully functional, stable, and regression-free at all times.
2. **Parallel Active Parity Track (High):** Actively develop mathematical alignments to bridge the remaining divergence, targeting a **95% matched edge pair threshold** against the MATLAB oracle vectors before finalizing the public release.

---

## 🛠️ Parity Integration Path: Native Integration

All mathematical and tie-breaking alignments against the legacy MATLAB reference are integrated **directly into the core Python code as the default behavior**. 
* Rather than maintaining isolated configuration toggles or profiles, the default out-of-the-box pipeline automatically runs with maximum precision and alignment, yielding 95%+ parity natively.
* The native code remains clean, documented, and modern while achieving mathematical bit-accuracy with the MATLAB reference.

---

## 🔬 Primary Algorithmic Focus: Frontier & Tie-Breaking

The primary focus for mathematical alignment is **Watershed Frontier Tie-Breaking**:
- **Problem:** When multiple voxels on the watershed expansion frontier have identical penalized energies, Python and MATLAB must select the exact same voxel. Disagreements here cascade into distinct tracing paths, causing the remaining 11.3% of edge discrepancies.
- **Solution:** Focus on integrating **Fortran-order linear index secondary sorting keys** directly into the frontier priority queue inside `slavv_python/processing/stages/edges/global_watershed.py`. This perfectly resolves exact energy ties identically to MATLAB's double-precision watershed.

---

## 🧪 Integration Test Scope: Lightweight & Self-Contained

The test suite is designed for high velocity, stability, and speed:
- **Repo-Local & In-Memory:** Use compact synthetic volumes generated in-memory or stored locally to verify pipeline integration.
- **Self-Contained Execution:** Avoid any external remote network downloads or complex git-annex dependencies.
- **Performance Budget:** Ensure the entire integration test suite runs and passes in **under 2 minutes** locally and in CI/CD, enabling rapid validation of changes.

---

## 🚀 Release & Lifecycle: Continuous Delivery

We adopt a strict **Continuous Delivery** workflow:
- **Automation is the Gatekeeper:** The automated GitHub Actions regression gate (`.github/workflows/regression-gate.yml`) runs linting (`ruff`), type checking (`mypy`), and the full test suite (`pytest`) on every commit to the `main` branch.
- **Green Means Ready:** We treat every green build passing the regression gate as a candidate for immediate production use, eliminating manual, multi-phase approval bottlenecks.

---

## 📋 Checklist & Definition of Done

### Phase 1: Zero-Failure Public Health (PAPER-001)
- [x] **End-to-End Pipeline Execution:** Verify `slavv run` completes end-to-end and writes structured run directories.
- [x] **Downstream Consumers:** Verify `slavv analyze` and `slavv plot` successfully consume output `network.json`.
- [x] **Interactive UI:** Verify `slavv-app` (Streamlit) launches and operates flawlessly on port 8501.
- [x] **Import Resolution:** Clear all circular and missing package imports across all submodules.
- [x] **Integration Testing:** Run paper-profile integration tests in < 95 seconds under Python 3.11 conda environment.

### Phase 2: Parallel Parity & Tie-Breaking (PARITY-002)
- [x] **Frontier Sorting Key:** Update `global_watershed.py` with Fortran-order secondary sort keys.
- [ ] **Bit-Accurate Precision:** Validate float64 math remains stable across all edge tracing stages.
- [ ] **Curation Alignments:** Verify ML feature extraction (`extract_uncurated_info`) and automated thresholds are perfectly aligned.
- [ ] **Parity Proof Gate:** Achieve a **95% or higher** matched edge pair rate on the exact-proof benchmark:
  ```powershell
  python scripts/cli/parity_experiment.py prove-exact --stage all
  ```

### Phase 3: Continuous Delivery Release
- [ ] **Green CI/CD Pipeline:** Merge aligned features to `main` with a clean pass across all OS environments.
- [ ] **Production Release:** Promote the stable vectorization engine to standard academic & research deployment.
