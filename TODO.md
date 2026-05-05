# SLAVV Python Parity TODO

**Last Updated:** 2026-05-05
**Version:** 2.1 (Enhanced)
**Current Status:** 496/1197 Python candidates (41.4% match rate) - 58.6% gap remains

---

## 📊 Executive Summary

### Task Overview
| Category | Total | Critical | High | Medium | Low |
|----------|-------|----------|------|--------|-----|
| **Parity & Validation** | 8 | 2 | 3 | 2 | 1 |
| **Bug Fixes** | 2 | 1 | 1 | 0 | 0 |
| **Investigation** | 5 | 0 | 3 | 2 | 0 |
| **Testing** | 3 | 0 | 1 | 2 | 0 |
| **Documentation** | 4 | 0 | 0 | 2 | 2 |
| **Technical Debt** | 3 | 0 | 0 | 2 | 1 |
| **Performance** | 2 | 0 | 0 | 1 | 1 |
| **DevOps** | 1 | 0 | 0 | 1 | 0 |
| **TOTAL** | **28** | **3** | **8** | **12** | **5** |

### Current Parity Metrics
- **Python Candidates Generated:** 1,197
- **MATLAB Pairs Matched:** 496 (41.4%)
- **Missing MATLAB Pairs:** 701 (58.6%)
- **Test Suite Status:** 25/25 passing (100%)
- **Critical Blocker:** Trace order fix validation pending

### Sprint Goals (Next 2 Weeks)
1. ⏳ Validate trace order fix impact on match rate (PARITY-001 - Ready to execute)
2. ✅ Fix failing test (BUG-001 - Completed: numpy boolean comparison fix)
3. ⏳ Achieve >50% match rate (PARITY-002 - Blocked by PARITY-001)
4. ⏳ Identify root cause for 50% of missing pairs (INVEST-001 - Blocked by PARITY-001)

---

## 🔴 Critical Priority Tasks

### [CRITICAL] PARITY-001: Validate Trace Order Fix Impact
**Priority:** Critical | **Effort:** Medium (4-6 hours) | **Status:** 🔴 Blocked - Awaiting Execution

**Description:**  
Run comprehensive parity experiment to measure the impact of the trace order randomization fix implemented on 2026-05-05. This fix ensures deterministic candidate generation by using seeded RNG for trace order shuffling.

**Acceptance Criteria:**
- [ ] Parity experiment completed with trace order fix
- [ ] Match rate measured and compared to 41.4% baseline
- [ ] Results documented in `EXACT_PROOF_FINDINGS.md`
- [ ] Report promoted to permanent storage

**Implementation Steps:**
```powershell
# 1. Promote oracle (if not already done)
python dev/scripts/cli/parity_experiment.py promote-oracle `
    --matlab-batch-dir D:\incoming\batch_260421-151654 `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dataset-file D:\datasets\volume.tif `
    --oracle-id v22_a

# 2. Run fail-fast gates
python dev/scripts/cli/parity_experiment.py preflight-exact `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix

python dev/scripts/cli/parity_experiment.py prove-luts `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix

# 3. Capture and compare candidates
python dev/scripts/cli/parity_experiment.py capture-candidates `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix

# 4. Replay edge selection
python dev/scripts/cli/parity_experiment.py replay-edges `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix

# 5. Rerun Python pipeline
python dev/scripts/cli/parity_experiment.py rerun-python `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix `
    --rerun-from edges

# 6. Generate summary
python dev/scripts/cli/parity_experiment.py summarize `
    --run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix

# 7. Run full exact proof
python dev/scripts/cli/parity_experiment.py prove-exact `
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run `
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a `
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix `
    --stage all

# 8. Promote report
python dev/scripts/cli/parity_experiment.py promote-report `
    --run-root D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix
```

**Files Modified:**
- `source/core/edges_internal/edge_selection.py` (trace order fix)

**Dependencies:** None  
**Blocks:** PARITY-002, INVEST-001, INVEST-002  
**Labels:** `parity`, `validation`, `critical`, `determinism`

---

### [CRITICAL] BUG-001: Fix Failing Test - COMPLETED ✅
**Priority:** Critical | **Effort:** Small (30 minutes) | **Status:** ✅ Complete

**Description:**
The `test_reveal_unclaimed_only_claims_zero_vertex_voxels` test was failing due to incorrect use of `is` operator for numpy boolean comparison. The test was using `is False` which checks object identity, but numpy boolean values are not the same object as Python's `False`.

**Root Cause:**
- Test used `is False` and `is True` for numpy boolean array elements
- Should use `== False` and `== True` for value comparison

**Fix Applied:**
- Changed line 661: `assert result["is_without_vertex_in_strel"][0] is False` → `== False`
- Changed line 662: `assert result["is_without_vertex_in_strel"][1] is True` → `== True`

**Validation:**
- ✅ All 25/25 tests now passing (100%)
- ✅ No regressions in other tests
- ✅ Fix completed on 2026-05-05

**Files Modified:**
- `dev/tests/unit/core/test_global_watershed_comprehensive.py` (lines 661-662)

**Dependencies:** None
**Blocks:** None (was not actually blocking parity work)
**Labels:** `bug`, `critical`, `testing`, `numpy`, `completed`

---

### [CRITICAL] PARITY-002: Achieve 50% Match Rate Milestone
**Priority:** Critical | **Effort:** Large (2-3 days) | **Status:** 🟡 Planned

**Description:**  
Improve the MATLAB parity match rate from the current 41.4% baseline to at least 50%. This requires identifying and fixing the most impactful divergences between Python and MATLAB implementations.

**Acceptance Criteria:**
- [ ] Match rate ≥ 50% (598+ matched pairs out of 1,197)
- [ ] Top 3 root causes for missing pairs identified and documented
- [ ] Fixes implemented and validated with parity experiments
- [ ] Regression tests added for fixed divergences
- [ ] Updated metrics in `EXACT_PROOF_FINDINGS.md`

**Approach:**
1. Complete PARITY-001 to establish new baseline
2. Complete BUG-001 to fix frontier ordering
3. Run INVEST-001 to categorize missing pairs
4. Prioritize fixes by impact (number of affected pairs)
5. Implement top 3 fixes iteratively with validation

**Dependencies:** PARITY-001, BUG-001, INVEST-001  
**Blocks:** PARITY-003  
**Labels:** `parity`, `critical`, `milestone`, `match-rate`

---

## 🟠 High Priority Tasks

### [HIGH] INVEST-001: Categorize 701 Missing MATLAB Pairs
**Priority:** High | **Effort:** Large (1-2 days) | **Status:** 🟡 Planned

**Description:**  
Systematically analyze the 701 Python candidates that have no MATLAB match to identify patterns and root causes. This investigation will guide prioritization of parity fixes.

**Acceptance Criteria:**
- [ ] Sample of 50-100 missing pairs extracted and analyzed
- [ ] Root causes categorized (frontier ordering, join cleanup, sentinel lifecycle, etc.)
- [ ] Impact assessment for each category (number of affected pairs)
- [ ] Prioritized fix roadmap created
- [ ] Findings documented with specific examples

**Investigation Plan:**
1. Extract representative sample of missing Python candidates
2. For each candidate, trace execution path in Python vs MATLAB
3. Identify divergence point (seed selection, trace direction, frontier insertion, etc.)
4. Group by root cause pattern
5. Estimate fix effort and impact for each category
6. Create prioritized roadmap

**Hypotheses to Test:**
- Frontier ordering differences (FIFO vs LIFO vs priority)
- Join cleanup timing and conditions
- Vertex `-Inf` sentinel lifecycle
- Trace execution order subtleties
- Candidate filtering differences

**Dependencies:** PARITY-001 (for updated baseline)  
**Blocks:** PARITY-002, INVEST-002, INVEST-003  
**Labels:** `investigation`, `high`, `parity`, `analysis`

---

### [HIGH] INVEST-002: Deep-Dive Frontier Ordering Semantics
**Priority:** High | **Effort:** Medium (6-8 hours) | **Status:** 🟡 Planned

**Description:**  
Conduct detailed comparison of frontier insertion and processing logic between Python and MATLAB implementations. This builds on BUG-001 to ensure comprehensive understanding of frontier semantics.

**Acceptance Criteria:**
- [ ] MATLAB frontier insertion algorithm fully understood and documented
- [ ] Python implementation gaps identified and documented
- [ ] Semantic differences quantified (impact on candidate generation)
- [ ] Implementation plan for alignment created
- [ ] Test cases added for frontier ordering edge cases

**Research Questions:**
- How does MATLAB order frontier candidates for insertion?
- Does MATLAB use FIFO, LIFO, or priority-based insertion?
- How are frontier priorities calculated and compared?
- When are frontier candidates removed or invalidated?
- How does frontier ordering affect trace execution order?

**MATLAB Source Analysis:**
- `get_edges_by_watershed.m` (frontier management)
- Related helper functions for frontier operations
- Data structure usage patterns

**Dependencies:** BUG-001  
**Blocks:** PARITY-002  
**Labels:** `investigation`, `high`, `parity`, `frontier-ordering`, `semantics`

---

### [HIGH] INVEST-003: Join Cleanup Logic Verification
**Priority:** High | **Effort:** Medium (4-6 hours) | **Status:** 🟡 Planned

**Description:**  
Verify that Python's join cleanup logic matches MATLAB's implementation. Join cleanup affects candidate generation by removing or modifying candidates based on connectivity rules.

**Acceptance Criteria:**
- [ ] MATLAB join cleanup algorithm documented
- [ ] Python implementation compared and gaps identified
- [ ] Timing and trigger conditions verified
- [ ] Test cases added for join cleanup scenarios
- [ ] Implementation aligned if divergences found

**Research Questions:**
- When does MATLAB perform join cleanup?
- What conditions trigger join cleanup operations?
- How are join candidates identified and processed?
- What are the cleanup rules and priorities?
- How does cleanup affect subsequent candidate generation?

**MATLAB Source Analysis:**
- `get_edges_by_watershed.m` (join cleanup sections)
- Helper functions for join operations
- Cleanup trigger conditions

**Dependencies:** INVEST-001  
**Blocks:** PARITY-002  
**Labels:** `investigation`, `high`, `parity`, `join-cleanup`, `semantics`

---

### [HIGH] TEST-001: Expand Comprehensive Test Coverage
**Priority:** High | **Effort:** Medium (4-6 hours) | **Status:** 🟡 Planned

**Description:**  
Expand the comprehensive test suite beyond the current 25 tests to cover edge cases and scenarios identified through parity investigations. Focus on areas with known divergences.

**Acceptance Criteria:**
- [ ] 10+ new test cases added for identified edge cases
- [ ] Test coverage for frontier ordering edge cases
- [ ] Test coverage for join cleanup scenarios
- [ ] Test coverage for sentinel lifecycle edge cases
- [ ] All tests passing with clear expectations
- [ ] Test documentation updated

**Test Areas to Add:**
- Frontier insertion order variations
- Join cleanup trigger conditions
- Vertex sentinel lifecycle scenarios
- Trace execution order edge cases
- Candidate filtering boundary conditions

**Dependencies:** INVEST-001, INVEST-002, INVEST-003  
**Labels:** `testing`, `high`, `coverage`, `edge-cases`

---

## 🟡 Medium Priority Tasks

### [MEDIUM] PARITY-003: Achieve 80% Match Rate Target
**Priority:** Medium | **Effort:** XL (1-2 weeks) | **Status:** 🔵 Future

**Description:**  
Build on the 50% milestone to achieve 80% match rate through systematic fixing of remaining divergences. This represents the medium-term parity goal.

**Acceptance Criteria:**
- [ ] Match rate ≥ 80% (958+ matched pairs out of 1,197)
- [ ] All major semantic differences resolved
- [ ] Comprehensive test coverage for fixed areas
- [ ] Performance impact assessed and optimized
- [ ] Documentation updated with final implementation notes

**Dependencies:** PARITY-002, INVEST-001, INVEST-002, INVEST-003  
**Blocks:** PARITY-004  
**Labels:** `parity`, `medium`, `milestone`, `match-rate`

---

### [MEDIUM] INVEST-004: Vertex Sentinel Lifecycle Analysis
**Priority:** Medium | **Effort:** Medium (4-6 hours) | **Status:** 🔵 Future

**Description:**  
Investigate how MATLAB handles `-Inf` vertex sentinels throughout the watershed algorithm lifecycle. Sentinel handling affects candidate generation and validation.

**Acceptance Criteria:**
- [ ] MATLAB sentinel lifecycle fully documented
- [ ] Python implementation compared and aligned
- [ ] Sentinel timing and conditions verified
- [ ] Test cases added for sentinel scenarios
- [ ] Impact on candidate generation quantified

**Research Questions:**
- When does MATLAB set vertex values to `-Inf`?
- When does MATLAB check for `-Inf` sentinels?
- How do sentinels affect trace execution?
- What are the sentinel cleanup rules?

**Dependencies:** INVEST-001  
**Labels:** `investigation`, `medium`, `parity`, `sentinels`, `lifecycle`

---

### [MEDIUM] INVEST-005: Execution Trace Comparison Framework
**Priority:** Medium | **Effort:** Large (1-2 days) | **Status:** 🔵 Future

**Description:**  
Develop a framework for detailed execution trace comparison between Python and MATLAB for specific missing candidates. This will enable systematic debugging of divergences.

**Acceptance Criteria:**
- [ ] Trace logging framework implemented in Python
- [ ] MATLAB trace extraction capability developed
- [ ] Automated trace comparison tool created
- [ ] Sample traces for 10+ missing candidates captured
- [ ] Divergence points automatically identified

**Components:**
- Python execution tracer for watershed algorithm
- MATLAB trace extraction scripts
- Trace comparison and visualization tools
- Automated divergence detection

**Dependencies:** INVEST-001  
**Labels:** `investigation`, `medium`, `tooling`, `tracing`, `debugging`

---

### [MEDIUM] TEST-002: Performance Regression Test Suite
**Priority:** Medium | **Effort:** Medium (4-6 hours) | **Status:** 🔵 Future

**Description:**  
Create performance regression tests to ensure parity fixes don't significantly impact algorithm performance. Include benchmarks for key operations.

**Acceptance Criteria:**
- [ ] Baseline performance metrics established
- [ ] Automated performance regression tests created
- [ ] Performance impact of parity fixes measured
- [ ] Optimization opportunities identified
- [ ] Performance CI integration configured

**Benchmark Areas:**
- Candidate generation throughput
- Memory usage patterns
- Frontier operation performance
- Join cleanup performance

**Dependencies:** PARITY-002  
**Labels:** `testing`, `medium`, `performance`, `regression`, `benchmarks`

---

### [MEDIUM] DOC-001: Parity Implementation Guide
**Priority:** Medium | **Effort:** Medium (4-6 hours) | **Status:** 🔵 Future

**Description:**  
Create comprehensive documentation for future contributors working on MATLAB parity. Include common patterns, debugging techniques, and implementation guidelines.

**Acceptance Criteria:**
- [ ] Parity debugging guide created
- [ ] Common divergence patterns documented
- [ ] Implementation best practices documented
- [ ] Troubleshooting flowchart created
- [ ] Code examples for typical fixes provided

**Content Areas:**
- Parity experiment workflow
- Common divergence patterns and fixes
- Debugging techniques and tools
- MATLAB source reading guide
- Test development for parity work

**Dependencies:** PARITY-002, INVEST-001  
**Labels:** `documentation`, `medium`, `parity`, `guide`, `contributors`

---

### [MEDIUM] DOC-002: Algorithm Implementation Notes
**Priority:** Medium | **Effort:** Medium (3-5 hours) | **Status:** 🔵 Future

**Description:**  
Document the final implementation details of the watershed algorithm, including design decisions, MATLAB alignment choices, and performance considerations.

**Acceptance Criteria:**
- [ ] Algorithm overview with implementation details
- [ ] Design decision rationale documented
- [ ] MATLAB alignment choices explained
- [ ] Performance characteristics documented
- [ ] Future optimization opportunities noted

**Dependencies:** PARITY-003  
**Labels:** `documentation`, `medium`, `algorithm`, `implementation`, `design`

---

### [MEDIUM] TECH-DEBT-001: Refactor Global Watershed Module
**Priority:** Medium | **Effort:** Large (1-2 days) | **Status:** 🔵 Future

**Description:**  
Refactor the global watershed module to improve maintainability while preserving MATLAB parity. Focus on code organization and clarity without changing semantics.

**Acceptance Criteria:**
- [ ] Code organization improved (clear function separation)
- [ ] Documentation and comments enhanced
- [ ] Complex logic broken into smaller functions
- [ ] All existing tests still passing
- [ ] No performance regression
- [ ] MATLAB parity maintained

**Refactoring Areas:**
- Function decomposition for complex operations
- Improved variable naming and documentation
- Better error handling and validation
- Code duplication elimination

**Dependencies:** PARITY-003  
**Labels:** `technical-debt`, `medium`, `refactoring`, `maintainability`

---

### [MEDIUM] TECH-DEBT-002: Test Organization and Cleanup
**Priority:** Medium | **Effort:** Medium (4-6 hours) | **Status:** 🔵 Future

**Description:**  
Reorganize and clean up the test suite structure, removing redundant tests and improving test organization following the patterns in `dev/tests/README.md`.

**Acceptance Criteria:**
- [ ] Test file organization follows ownership patterns
- [ ] Redundant tests identified and removed
- [ ] Test naming conventions standardized
- [ ] Test documentation improved
- [ ] Marker usage optimized

**Dependencies:** TEST-001  
**Labels:** `technical-debt`, `medium`, `testing`, `organization`, `cleanup`

---

### [MEDIUM] DEVOPS-001: Parity CI Integration
**Priority:** Medium | **Effort:** Medium (6-8 hours) | **Status:** 🔵 Future

**Description:**  
Integrate parity validation into the CI pipeline to catch regressions early. Include automated parity experiments for critical changes.

**Acceptance Criteria:**
- [ ] Parity validation CI workflow created
- [ ] Automated parity experiments on PR changes
- [ ] Match rate regression detection
- [ ] Performance regression detection
- [ ] Failure notification and reporting

**Components:**
- GitHub Actions workflow for parity validation
- Automated oracle management
- Result comparison and reporting
- Failure notification system

**Dependencies:** PARITY-002  
**Labels:** `devops`, `medium`, `ci`, `automation`, `parity`

---

## 🟢 Low Priority Tasks

### [LOW] PERF-001: Algorithm Performance Optimization
**Priority:** Low | **Effort:** Large (1-2 days) | **Status:** 🔵 Future

**Description:**  
Optimize algorithm performance while maintaining MATLAB parity. Focus on memory usage and computational efficiency improvements.

**Acceptance Criteria:**
- [ ] Performance bottlenecks identified and profiled
- [ ] Optimization opportunities implemented
- [ ] MATLAB parity maintained after optimizations
- [ ] Performance improvements measured and documented
- [ ] Memory usage optimized

**Optimization Areas:**
- Memory allocation patterns
- Loop optimization
- Data structure efficiency
- Algorithmic improvements (where parity allows)

**Dependencies:** PARITY-003, TEST-002  
**Labels:** `performance`, `low`, `optimization`, `memory`, `efficiency`

---

### [LOW] PERF-002: Memory Usage Profiling
**Priority:** Low | **Effort:** Small (2-3 hours) | **Status:** 🔵 Future

**Description:**  
Profile memory usage patterns of the watershed algorithm to identify optimization opportunities and ensure scalability.

**Acceptance Criteria:**
- [ ] Memory usage profiling framework set up
- [ ] Memory usage patterns documented
- [ ] Peak memory usage identified
- [ ] Memory optimization opportunities noted
- [ ] Scalability limits documented

**Dependencies:** PERF-001  
**Labels:** `performance`, `low`, `profiling`, `memory`, `scalability`

---

### [LOW] DOC-003: API Documentation Enhancement
**Priority:** Low | **Effort:** Small (2-3 hours) | **Status:** 🔵 Future

**Description:**  
Enhance API documentation for the watershed algorithm components, including docstrings, type hints, and usage examples.

**Acceptance Criteria:**
- [ ] All public functions have comprehensive docstrings
- [ ] Type hints added throughout
- [ ] Usage examples provided
- [ ] API documentation generated
- [ ] Documentation style consistent

**Dependencies:** TECH-DEBT-001  
**Labels:** `documentation`, `low`, `api`, `docstrings`, `examples`

---

### [LOW] DOC-004: Performance Benchmarking Guide
**Priority:** Low | **Effort:** Small (1-2 hours) | **Status:** 🔵 Future

**Description:**  
Create guide for performance benchmarking and profiling of the watershed algorithm for future optimization work.

**Acceptance Criteria:**
- [ ] Benchmarking methodology documented
- [ ] Profiling tools and techniques explained
- [ ] Performance metrics defined
- [ ] Baseline measurements provided
- [ ] Optimization workflow documented

**Dependencies:** PERF-001, PERF-002  
**Labels:** `documentation`, `low`, `performance`, `benchmarking`, `guide`

---

### [LOW] TECH-DEBT-003: Code Style Standardization
**Priority:** Low | **Effort:** Small (1-2 hours) | **Status:** 🔵 Future

**Description:**  
Standardize code style across the watershed algorithm implementation, ensuring consistency with project conventions.

**Acceptance Criteria:**
- [ ] Code style issues identified and fixed
- [ ] Naming conventions standardized
- [ ] Import organization standardized
- [ ] Comment style consistent
- [ ] Linting rules satisfied

**Dependencies:** TECH-DEBT-001  
**Labels:** `technical-debt`, `low`, `style`, `consistency`, `linting`

---

## 📋 Completed Work (2026-05-05)

### ✅ Trace Order Randomization Fix
**File:** `source/core/edges_internal/edge_selection.py`
**Issue:** Trace order was randomized without seeded RNG, causing non-deterministic results
**Fix:** Always use seeded RNG (`np.random.default_rng(seed)`) for trace order shuffling
**Impact:** Ensures deterministic candidate generation for parity testing
**Status:** ✅ Implemented and ready for validation

### ✅ Directional Suppression Investigation
**File:** `source/core/_edge_candidates/global_watershed.py`
**Investigation:** Confirmed MATLAB applies directional suppression inside seed loop
**Action:** Reverted incorrect removal of directional suppression
**Status:** ✅ Python now matches MATLAB behavior

### ✅ Comprehensive Test Suite Creation
**File:** `dev/tests/unit/core/test_global_watershed_comprehensive.py`
**Coverage:** 25 tests covering all major global watershed components
**Status:** ✅ 25/25 tests passing (100% success rate)
**Note:** Test suite fully passing after fixing numpy boolean comparison issue

### ✅ Test Suite Bug Fix
**File:** `dev/tests/unit/core/test_global_watershed_comprehensive.py`
**Issue:** `test_reveal_unclaimed_only_claims_zero_vertex_voxels` failing due to `is` vs `==` for numpy booleans
**Fix:** Changed `is False`/`is True` to `== False`/`== True` for numpy array element comparison
**Status:** ✅ Complete - all tests now passing

### ✅ Documentation Updates
**Files:** `docs/reference/core/EXACT_PROOF_FINDINGS.md`, `COMMIT_MESSAGES.md`, `TODO.md`
**Content:** Added May 2026 investigation section, documented findings, created commit templates, corrected test status
**Status:** ✅ Complete

---

## 🔧 Quick Reference Commands

### Testing
```powershell
# Run all tests
python -m pytest dev/tests/

# Run unit and integration tests only
python -m pytest -m "unit or integration"

# Run specific test file with verbose output
python -m pytest dev/tests/unit/core/test_global_watershed_comprehensive.py -v

# Run failing test specifically
python -m pytest dev/tests/unit/core/test_global_watershed_comprehensive.py::test_global_watershed_frontier_ordering -vv
```

### Code Quality
```powershell
# Format code
python -m ruff format source dev/tests

# Lint and fix issues
python -m ruff check source dev/tests --fix

# Type checking
python -m mypy

# Full regression check
python -m compileall source dev/scripts
python -m ruff format --check source dev/tests
python -m ruff check source dev/tests
python -m mypy
python -m pytest -m "unit or integration"
```

### Parity Experiments
```powershell
# Quick parity check (fail-fast)
python dev/scripts/cli/parity_experiment.py fail-fast \
    --source-run-root D:\slavv_comparisons\experiments\live-parity\runs\seed_run \
    --oracle-root D:\slavv_comparisons\experiments\live-parity\oracles\v22_a \
    --dest-run-root D:\slavv_comparisons\experiments\live-parity\runs\my_test

# Full parity experiment (see PARITY-001 for complete workflow)
```

---

## 📚 Key References

### MATLAB Source Code
- **Primary:** `external/Vectorization-Public/source/get_edges_by_watershed.m`
- **Supporting:** Other files in `external/Vectorization-Public/source/`

### Documentation
- **Parity Findings:** `docs/reference/core/EXACT_PROOF_FINDINGS.md`
- **Parity Mapping:** `docs/reference/core/MATLAB_PARITY_MAPPING.md`
- **Implementation Plan:** `docs/reference/core/MATLAB_METHOD_IMPLEMENTATION_PLAN.md`
- **Test Guidelines:** `dev/tests/README.md`
- **Agent Rules:** `AGENTS.md`
- **Naming Guide:** `docs/reference/workflow/PYTHON_NAMING_GUIDE.md`

### Workflow Tools
- **Parity Experiments:** `dev/scripts/cli/parity_experiment.py`
- **Test Configuration:** `dev/tests/conftest.py`

---

## 🎯 Success Metrics & Milestones

### Sprint 1 (Next 2 Weeks) - Foundation
- [ ] **PARITY-001:** Trace order fix validated (Ready to execute)
- [x] **BUG-001:** Test suite fixed (numpy boolean comparison)
- [ ] **PARITY-002:** 50% match rate achieved (Blocked by PARITY-001)
- [ ] **INVEST-001:** Missing pairs categorized (Blocked by PARITY-001)

### Sprint 2 (Weeks 3-4) - Deep Investigation
- [ ] **INVEST-002:** Frontier semantics aligned
- [ ] **INVEST-003:** Join cleanup verified
- [ ] **TEST-001:** Test coverage expanded
- [ ] Match rate > 60%

### Month 2 - Implementation & Optimization
- [ ] **PARITY-003:** 80% match rate achieved
- [ ] **INVEST-004:** Sentinel lifecycle aligned
- [ ] **INVEST-005:** Trace comparison framework
- [ ] **TEST-002:** Performance regression tests

### Month 3 - Polish & Documentation
- [ ] **PARITY-004:** 95% match rate (stretch goal)
- [ ] **DOC-001:** Implementation guide complete
- [ ] **TECH-DEBT-001:** Code refactoring complete
- [ ] **DEVOPS-001:** CI integration complete

### Long-Term Goals
- [ ] Full MATLAB parity achieved (>95% match rate)
- [ ] Comprehensive test coverage (>90%)
- [ ] Performance optimized (no regression)
- [ ] Documentation complete and maintained
- [ ] CI/CD pipeline fully automated

---

**Document Version:** 2.0  
**Next Review:** 2026-05-12  
**Maintainer:** Development Team  
**Status:** Active Development