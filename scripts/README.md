# Developer Scripts

**Purpose:** Maintained utility scripts used during development but not part of the public product.

These scripts are committed to Git because they're reusable tools for developers, but they're **not installed** when users run `pip install slavv`. They're for internal development workflows only.

---

## What Lives Here

### `cli/` — Command-Line Utilities
Developer-facing CLI tools for parity experiments, debugging, and investigation:

- **`parity_experiment.py`** — MATLAB oracle promotion and exact-proof runner
  - `promote-oracle` — Convert MATLAB batch output to reusable oracle
  - `preflight-exact` — Quick validation before long parity runs
  - `prove-exact` — Run exact parity comparison against oracle
  - `init-exact-run` — Initialize structured parity run directory

- **`compare_execution_traces.py`** — Compare execution traces for debugging
- **`investigate_translation_pairs.py`** — Analyze translation pair mismatches
- **`resume_pipeline_run.py`** — Resume interrupted pipeline runs
- **`export_180709_crop_m.py`** — Export specific crop dataset

### `matlab/` — MATLAB Drivers
Headless MATLAB scripts for oracle generation:

- Batch vectorization runners
- Oracle promotion workflows
- Crop extraction drivers

### `diagnostics/` — Inspection Tools
Tools for inspecting MATLAB artifacts and debugging parity issues.

---

## When to Add Code Here

**Add here when:**
- ✅ It's a reusable developer tool
- ✅ Multiple developers will use it
- ✅ It's part of a documented workflow (e.g., parity experiments)
- ✅ It needs to be maintained and versioned

**Don't add here when:**
- ❌ It's part of the public product → Put in `slavv_python/interface/cli/`
- ❌ It's a throwaway exploration → Put in `workspace/scratch/`
- ❌ It's test infrastructure → Put in `tests/support/`
- ❌ It's experiment data → Put in `workspace/`

---

## Import Rules

**Allowed:**
```python
# Scripts can import from slavv_python
from slavv_python.analytics.parity import ExactProofCoordinator
from slavv_python.engine import RunContext
```

**Forbidden:**
```python
# Production code should NEVER import from scripts
# This is backwards and will break for users
from scripts.cli.parity_experiment import something  # ❌ NO
```

---

## Testing Scripts

Tests for maintained scripts go in `tests/unit/scripts/`:

```
scripts/parity_experiment.py  →  tests/unit/scripts/test_parity_experiment.py
```

---

## Examples

### Example: Parity Workflow
```powershell
# Promote MATLAB output to reusable oracle
python scripts/parity_experiment.py promote-oracle \
  --matlab-batch-dir D:\incoming\batch_260421 \
  --oracle-root workspace\oracles\crop_M \
  --dataset-file D:\datasets\crop_M.tif \
  --oracle-id crop_M

# Run exact proof comparison
python scripts/parity_experiment.py prove-exact \
  --source-run-root workspace\runs\seed_run \
  --oracle-root workspace\oracles\crop_M \
  --dest-run-root workspace\runs\current_trial \
  --stage all
```

---

## Related Documentation

- [FOLDER_PURPOSE_GUIDE.md](../docs/reference/core/FOLDER_PURPOSE_GUIDE.md) — When to use each top-level folder
- [PARITY_PRE_GATE.md](../docs/reference/workflow/PARITY_PRE_GATE.md) — Parity experiment workflow
- [tests/README.md](../tests/README.md) — Test organization
