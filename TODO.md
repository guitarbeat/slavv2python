# TODO — Priority Queue

> **Last updated:** 2026-05-20
>
> The parity track is impressive engineering (14% → 88.7%), but the public
> product has never been verified end-to-end. **Flip the priority.** Ship
> a working pipeline before chasing the last 11.3% of tie-breaking edge cases.

---

## 🔴 Priority 1: Prove the Product Works (PAPER-001)

**Goal:** Confirm that a user can install this package and run the full
TIFF → network → analysis → visualization pipeline without MATLAB.

- [ ] **Run `slavv run` end-to-end**
  ```powershell
  pip install -e ".[app,workspace]"
  slavv run -i data/slavv_test_volume.tif -o slavv_output --profile paper --export csv json
  ```
  - Does it complete without error?
  - Does it produce a `network.json` under the output directory?

- [ ] **Verify downstream consumers**
  ```powershell
  slavv analyze -i slavv_output\network.json
  slavv plot -i slavv_output\network.json -o plots.html
  ```
  - Does `analyze` print a summary?
  - Does `plot` produce a valid HTML file?

- [ ] **Verify the Streamlit app launches**
  ```powershell
  slavv-app
  ```
  - Does the web app open in a browser?
  - Can it load and display a sample dataset?

- [ ] **Verify `import slavv_python` works cleanly**
  ```python
  from slavv_python import SlavvPipeline, load_tiff_volume
  ```
  - No import errors, no CUDA warnings blocking startup

- [ ] **Write a paper-profile integration test**
  - Add `tests/integration/test_paper_pipeline_e2e.py`
  - Covers: load TIFF → run pipeline → assert network.json is valid
  - Must run in CI without MATLAB, GPU, or large datasets

- [ ] **Update ROADMAP.md** to track PAPER-001 as the active priority

---

## 🟡 Priority 2: Stabilize & Document What We Have

- [ ] **Verify test suite passes**
  ```powershell
  python -m pytest -m "unit or integration"
  ```
  - How many tests pass? How many fail? What's broken?

- [ ] **Verify quality gate passes**
  ```powershell
  python -m compileall slavv_python scripts
  python -m ruff check slavv_python tests
  python -m mypy
  ```

- [ ] **Clean up CHANGELOG.md** — Add a "Current (Unreleased)" section summarizing recent doc alignment and root cleanup work

- [ ] **Confirm `pyproject.toml` entrypoints resolve** — The entrypoints were recently fixed from stale `slavv_python.apps.*` paths to `slavv_python.interface.*`. Verify `slavv` and `slavv-app` commands actually work after `pip install -e .`

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
