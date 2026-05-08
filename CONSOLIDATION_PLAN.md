# Test Consolidation Plan

## Status
- **Current Total Test Files**: ~100
- **Estimated Redundancy**: Significant duplication found in `tests/unit/analysis/` and likely `tests/unit/workspace_scripts/parity_experiment/`.

## Findings
### 1. ML Curator Tests
`tests/unit/analysis/test_ml_curator_comprehensive.py` appears to be a superset of several smaller test files.
- **Redundant Files**:
  - `tests/unit/analysis/test_ml_curator_features.py`
  - `tests/unit/analysis/test_ml_curator_improvements.py`
  - `tests/unit/analysis/test_ml_curator_security.py`
  - `tests/unit/analysis/test_ml_curator_training.py`
  - `tests/unit/analysis/test_ml_model_io.py`
  - `tests/unit/analysis/test_ml_training.py`
- **Action**: Verify `test_ml_curator_comprehensive.py` covers everything, then delete the redundant files.

### 2. Parity Experiment Tests
There are many small files in `tests/unit/workspace_scripts/parity_experiment/`.
- **Redundant Files**:
  - `test_dedupe.py`
  - `test_execution_and_reporting.py`
  - `test_rerun.py`
  - `test_initialization.py`
  - `test_validation_and_params.py`
  - `test_proof.py`
  - `test_promotion.py`
  - `test_parser.py`
- **Action**: Consolidate into a single `tests/unit/workspace_scripts/test_parity_experiment.py`.

### 3. App State Tests
`tests/unit/apps/` has many state tests that might share setup.
- **Candidates**:
  - `test_visualization_state.py`
  - `test_processing_state.py`
  - `test_analysis_state.py`
  - `test_dashboard_state.py`
  - `test_curation_state.py`
- **Action**: Evaluate if these can be merged into `tests/unit/apps/test_app_states.py`.

## Task List
- [x] Audit `test_ml_curator_comprehensive.py` against its counterparts.
- [x] Consolidate ML Curator tests and delete redundant files.
- [x] Merge `apps` state tests into a single file.
- [x] Merge runtime helper tests into a single file.
- [x] Merge `parity_experiment` tests into a single file.
- [ ] Run full test suite to ensure parity.
