# Sprint Plan - Post Trace Order Fix Validation

**Sprint Period**: 2026-05-05 to 2026-05-19 (2 weeks)  
**Last Updated**: 2026-05-05  
**Current Match Rate**: 33.8% (404/1,197 MATLAB pairs)  
**Sprint Goal**: Reach 50% match rate (598+ matched pairs)

---

## 🎯 Sprint Objectives

1. **Understand the gap**: Categorize 793 missing MATLAB pairs and 84 extra Python pairs
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

#### 🔴 PARITY-001A - Investigate Baseline Discrepancy
- **Priority**: Critical
- **Effort**: 2-3 hours
- **Owner**: TBD
- **Tasks**:
  - [ ] Search for experiments with 41.4% match rate claim
  - [ ] Review experiment history in index.jsonl
  - [ ] Update TODO.md with accurate baseline
  - [ ] Document in EXACT_PROOF_FINDINGS.md
- **Deliverable**: Clarified baseline history

### High Priority

#### 🟢 INVEST-001 - Categorize 793 Missing MATLAB Pairs
- **Priority**: High
- **Effort**: 1-2 days
- **Owner**: TBD
- **Tasks**:
  - [ ] Extract sample of 50-100 missing pairs
  - [ ] Trace execution path for each sample
  - [ ] Identify divergence points (seed selection, frontier, etc.)
  - [ ] Group by root cause pattern
  - [ ] Estimate fix effort and impact per category
  - [ ] Create prioritized fix roadmap
- **Deliverable**: Categorized root causes with impact assessment

#### 🟢 INVEST-006 - Analyze 84 Extra Python Pairs
- **Priority**: High
- **Effort**: 4-6 hours
- **Owner**: TBD
- **Tasks**:
  - [ ] Extract sample of 20-30 extra pairs
  - [ ] Compare with MATLAB logic for these cases
  - [ ] Identify why Python generates but MATLAB doesn't
  - [ ] Categorize root causes (over-generation, filtering, etc.)
  - [ ] Propose fixes
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

#### 🟡 INVEST-002 - Frontier Ordering Semantics
- **Priority**: High
- **Effort**: 6-8 hours
- **Owner**: TBD
- **Tasks**:
  - [ ] Document MATLAB frontier insertion algorithm
  - [ ] Compare with Python implementation
  - [ ] Identify semantic differences
  - [ ] Create implementation plan
  - [ ] Add test cases for edge cases
- **Deliverable**: Frontier alignment plan

#### 🟡 INVEST-003 - Join Cleanup Logic Verification
- **Priority**: High
- **Effort**: 4-6 hours
- **Owner**: TBD
- **Tasks**:
  - [ ] Document MATLAB join cleanup algorithm
  - [ ] Compare with Python implementation
  - [ ] Verify timing and trigger conditions
  - [ ] Add test cases
  - [ ] Align implementation if needed
- **Deliverable**: Join cleanup verification report

---

## 📊 Success Metrics

### Week 1 Targets
- [ ] PARITY-001A completed (baseline clarified)
- [ ] INVEST-001 completed (793 pairs categorized)
- [ ] INVEST-006 completed (84 extra pairs analyzed)
- [ ] Top 3 fix priorities identified
- [ ] Fix implementation started

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