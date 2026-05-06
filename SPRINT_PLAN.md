# Sprint Plan - Post Trace Order Fix Validation

**Sprint Period**: 2026-05-05 to 2026-05-19 (2 weeks)  
**Last Updated**: 2026-05-05. **Baseline Clarification (PARITY-001A)**: Resolved discrepancy between claimed 41.4% baseline and actual measured rates. The 41.4% claim was determined to be inaccurate (likely a projection or target); the verified baseline progression is 12.4% → 33.8%.  
**Current Match Rate**: 33.8% (404/1,197 MATLAB pairs)  
**Sprint Goal**: Reach 50% match rate (598+ matched pairs)

---

## 🎯 Sprint Objectives

1. **Understand the gap**: Categorized 793 missing MATLAB pairs and 84 extra Python pairs. Identified top 3 root causes (Frontier Ordering, Hub Vertex Exploration, Filtering Gaps).
2. **Fix top divergences**: Implement and validate top 3 root cause fixes
3. **Achieve milestone**: Reach 50% match rate through systematic parity improvements
4. **Document progress**: Maintain clear audit trail of findings and fixes

---

## 📋 Week 1 Tasks (2026-05-05 to 2026-05-12)

### Critical Priority

#### ✅ COMPLETED: PARITY-001 - Trace Order Fix Validation
- **Status**: Complete (2026-05-05)
- **Result**: 33.8% match rate, 2.7x improvement in matches
- **Next**: Document findings ✅ and promote report

#### ✅ COMPLETED: PARITY-001A - Investigate Baseline Discrepancy
- **Status**: Complete (2026-05-05)
- **Result**: 41.4% claim resolved as inaccurate; verified baseline 12.4% → 33.8%
- **Tasks**:
  - [x] Search for experiments with 41.4% match rate claim
  - [x] Review experiment history in index.jsonl
  - [x] Update TODO.md with accurate baseline
  - [x] Document in EXACT_PROOF_FINDINGS.md
- **Deliverable**: Clarified baseline history

### High Priority

#### ✅ COMPLETED: INVEST-001 - Categorize 793 Missing MATLAB Pairs
- **Status**: Complete (2026-05-05)
- **Result**: Categorized root causes with impact assessment (Frontier, Hubs, Filtering)
- **Tasks**:
  - [x] Extract sample of 50-100 missing pairs
  - [x] Trace execution path for each sample
  - [x] Identify divergence points (seed selection, frontier, etc.)
  - [x] Group by root cause pattern
  - [x] Estimate fix effort and impact per category
  - [x] Create prioritized fix roadmap
- **Deliverable**: Categorized root causes with impact assessment

#### ✅ COMPLETED: INVEST-006 - Analyze 84 Extra Python Pairs
- **Status**: Complete (2026-05-05)
- **Result**: Root cause analysis with fix proposals (Filtering/Cleanup alignment)
- **Tasks**:
  - [x] Extract sample of 20-30 extra pairs
  - [x] Compare with MATLAB logic for these cases
  - [x] Identify why Python generates but MATLAB doesn't
  - [x] Categorize root causes (over-generation, filtering, etc.)
  - [x] Propose fixes
- **Deliverable**: Root cause analysis with fix proposals

---

## 📋 Week 2 Tasks (2026-05-12 to 2026-05-19)

### Critical Priority

#### 🔴 PARITY-002 - Achieve 50% Match Rate
- **Priority**: Critical
- **Effort**: 2-3 days
- **Owner**: TBD
- **Dependencies**: INVEST-001, INVEST-006
- **Tasks**:
  - [ ] Implement top 3 fixes from INVEST-001
  - [ ] Run parity experiment for each fix
  - [ ] Validate match rate improvement
  - [ ] Add regression tests
  - [ ] Document fixes in EXACT_PROOF_FINDINGS.md
- **Target**: 598+ matched pairs (50% of 1,197)

### High Priority

#### ✅ COMPLETED: INVEST-002 - Frontier Ordering Semantics
- **Status**: Complete (2026-05-05)
- **Result**: Identified and fixed `r/R` distance normalization and energy map integrity.
- **Tasks**:
  - [x] Document MATLAB frontier insertion algorithm
  - [x] Compare with Python implementation
  - [x] Implement `r/R` normalization fix
  - [x] Fix energy map integrity (stopped penalty propagation)
  - [x] Verify with 54 passing tests

#### ✅ COMPLETED: INVEST-003 - Join Cleanup Logic Verification
- **Status**: Complete (2026-05-05)
- **Result**: Verified tracing, zero-pointer termination, and join available_locations reset logic.
- **Tasks**:
  - [x] Document MATLAB join cleanup algorithm
  - [x] Compare with Python implementation
  - [x] Verify timing and trigger conditions
  - [x] Verified cycle detection safety
  - [x] Verified reset join locations behavior

---

## 📊 Success Metrics

### Week 1 Targets
- [x] PARITY-001A completed (baseline clarified) ✅
- [x] INVEST-001 completed (793 pairs categorized) ✅
- [x] INVEST-006 completed (84 extra pairs analyzed) ✅
- [x] Top 3 fix priorities identified ✅
- [x] Fix implementation started (and verified in unit tests) ✅

### Week 2 Targets
- [ ] Top 3 fixes implemented and validated
- [ ] Match rate ≥ 40% (minimum acceptable)
- [ ] Match rate ≥ 50% (sprint goal)
- [ ] Regression tests added for all fixes
- [ ] Documentation updated

### Sprint Success Criteria
- ✅ Match rate improvement: 33.8% → 50%+ (16.2+ percentage points)
- ✅ Matched pairs: 404 → 598+ (194+ new matches)
- ✅ Root causes documented and prioritized
- ✅ Regression test coverage for fixes
- ✅ Clear roadmap for 80% milestone

---

## 🚧 Risks & Mitigations

### Risk 1: Root causes more complex than expected
- **Impact**: High
- **Probability**: Medium
- **Mitigation**: Start investigation early, allocate buffer time, focus on high-impact fixes first

### Risk 2: Fixes introduce regressions
- **Impact**: High
- **Probability**: Low
- **Mitigation**: Comprehensive regression testing, parity experiments after each fix

### Risk 3: 50% target unreachable in 2 weeks
- **Impact**: Medium
- **Probability**: Medium
- **Mitigation**: Set minimum acceptable target (40%), prioritize by impact, defer low-impact fixes

---

## 📝 Daily Standup Template

### What did I complete yesterday?
- Task completions
- Blockers resolved

### What will I work on today?
- Planned tasks
- Expected deliverables

### Any blockers or concerns?
- Technical blockers
- Resource needs
- Timeline concerns

---

## 🔗 Key Resources

- **TODO.md**: Comprehensive task tracking
- **EXACT_PROOF_FINDINGS.md**: Parity findings and validation results
- **Parity Experiment Tool**: `dev/scripts/cli/parity_experiment.py`
- **Oracle**: `D:\slavv_comparisons\experiments\live-parity\oracles\180709_E_batch_190910-103039`
- **Latest Run**: `D:\slavv_comparisons\experiments\live-parity\runs\trace_order_fix`

---

## 📅 Review Schedule

- **Daily**: Quick progress check (5 min)
- **Mid-sprint** (2026-05-12): Week 1 retrospective, adjust Week 2 plan
- **End-sprint** (2026-05-19): Sprint retrospective, plan next sprint

---

**Sprint Owner**: Development Team  
**Status**: Active  
**Next Review**: 2026-05-12