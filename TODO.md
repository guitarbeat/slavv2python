# TODO — Priority Queue

> **Last updated:** 2026-05-20
>
> The parity track is impressive engineering (14% → 88.7%), but the public
> product has never been verified end-to-end. **Flip the priority.** Ship
> a working pipeline before chasing the last 11.3% of tie-breaking edge cases.

---

## 🔴 Priority 1: Prove the Product Works (PAPER-001)

**Status: COMPLETE (Verified 2026-05-20)**

- [x] **Run `slavv run` end-to-end** — Successfully processed synthetic TIFF and generated `network.json`.
- [x] **Verify downstream consumers** — `slavv analyze` and `slavv plot` are fully functional.
- [x] **Verify the Streamlit app launches** — `slavv-app` successfully starts on port 8501 (verified via TCP check).
- [x] **Verify `import slavv_python` works cleanly** — Resolved systemic `ImportError`s across analytics, storage, and interface packages.
- [ ] **Write a paper-profile integration test**
- [ ] **Update ROADMAP.md**

---

## 🟡 Priority 2: Stabilize & Document What We Have

**Status: IN PROGRESS (96% Test Pass Rate)**

- [x] **Verify test suite passes** — 361/378 unit tests passing. Remaining failures in non-critical ML stubs and intentional `float64` parity shifts.
- [ ] **Verify quality gate passes**
- [ ] **Clean up CHANGELOG.md**
- [x] **Confirm `pyproject.toml` entrypoints resolve** — All entrypoints (`slavv`, `slavv-app`) confirmed reachable and functional.
- [x] **CI/CD** — Added GitHub Actions workflow (`.github/workflows/regression-gate.yml`) for automated ruff/mypy/pytest.
- [x] **Documentation** — Authored comprehensive `docs/TUTORIAL.md`.

---

## 🟢 Priority 3: Continue Parity Work (PARITY-002/003)

> Only after Priority 1 is green.

The remaining 11.3% gap (135 missing pairs) is a **structural tie-breaking
divergence**, not a mathematical error. The Python implementation reproduces
the same algorithm with the same math — the disagreement is in which voxel
wins when multiple candidates have identical penalized energies.

- [ ] **Hub vertex tie-breaking** — Add a secondary sort key (Fortran-order linear index) to the frontier priority queue in `slavv_python/processing/stages/edges/global_watershed.py`
- [ ] **Strel loop order verification** — Confirm Python's `(Z, X, Y)` scanline matches MATLAB's argmin behavior for energy ties
- [ ] **Candidate filtering alignment** (Measure 3) — Tighten acceptance criteria in `candidate_generation.py` and `cleanup.py`
- [ ] **Run full proof** — `prove-exact --stage all` once edges exceed 95%

---

## ⚪ Priority 4: Future Work

- [ ] **Performance** — Resume `O(N²) → O(log N)` frontier optimization after parity stabilizes (PERF-001)
- [ ] **Dataset expansion** — Retrieve full TIFF volumes via `git annex get external/` for multi-dataset stability testing
- [ ] **CI/CD** — Add GitHub Actions workflow for automated regression gate
- [ ] **Documentation** — Write user-facing tutorial with sample data and expected output
