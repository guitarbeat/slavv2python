# Documentation Audit & Improvement Plan

**Date**: 2026-06-09  
**Purpose**: Improve AI agent navigability and consolidate documentation structure

---

## Executive Summary

The slavv2python documentation is **well-structured** overall, with good separation of concerns and clear entry points. However, there are opportunities to improve AI navigability through better cross-linking, consolidation of redundant content, and clearer decision trees.

**Key Strengths:**
- ✅ Clear ownership boundaries (TODO.md for tasks, EXACT_PROOF_FINDINGS.md for status)
- ✅ AGENTS.md provides comprehensive domain glossary
- ✅ Good separation between maintained docs (reference/) and archival (investigations/)
- ✅ Recent monitoring documentation is well-integrated

**Key Opportunities:**
- 🔄 Redundant parity workflow content across multiple files
- 🔄 Decision tree in AGENTS.md could be more prominent
- 🔄 docs/README.md has evolved into a complex navigation file that could be simplified
- 🔄 Glossary exists in two places (AGENTS.md and reference/core/GLOSSARY.md) - needs sync strategy

---

## Audit Findings

### 1. Redundancy Issues

#### Parity Workflow Duplication
**Location**: README.md, AGENTS.md, docs/README.md all describe parity closure fast path

**Current state:**
- README.md has "Parity Closure Fast Path" (5 steps)
- docs/README.md has "Parity Closure Fast Path" (6 steps)
- AGENTS.md has "Work Decision Tree → I'm working on MATLAB parity" (5 steps)

**Recommendation**: **Consolidate to single source**
- Keep detailed version in docs/README.md (it's the docs index)
- README.md should point to docs/README.md
- AGENTS.md should point to docs/README.md with brief summary

---

#### Glossary Duplication
**Location**: AGENTS.md and reference/core/GLOSSARY.md

**Current state:**
- AGENTS.md has full glossary (16 terms)
- reference/core/GLOSSARY.md also exists (not read in this audit)
- No clear sync strategy documented

**Recommendation**: **Establish canonical source**
- AGENTS.md should be canonical (auto-loaded into AI context)
- reference/core/GLOSSARY.md should be supplementary with additional technical details
- Add note in both files about the relationship

---

### 2. Navigation Issues

#### Decision Tree Buried
**Location**: AGENTS.md has excellent decision tree but it's after TOC

**Current state:**
- Work Decision Tree is comprehensive and helpful
- Appears after table of contents
- Could be missed by AI agents scanning the file

**Recommendation**: **Make more prominent**
- Add visual marker (🧭) to stand out
- Consider moving summary to very top
- Add to README.md as primary navigation aid

---

#### docs/README.md Complexity
**Location**: docs/README.md has grown complex with multiple navigation schemes

**Current state:**
- "Quick Start By Use Case" (4 sections)
- "Core Entry Points" (9 items)
- "Documentation Map" (large hierarchy diagram)
- "Navigation Rules" (table)
- "Content Ownership" (table)
- "Anti-Patterns" (list)
- "Parity Closure Fast Path" (6 steps)
- Traditional sections

**Recommendation**: **Simplify for AI consumption**
- Keep "Quick Start By Use Case" at top (most useful)
- Consolidate navigation aids into one section
- Move documentation map to separate DOCUMENTATION_MAP.md
- Keep file focused on "what to read when"

---

### 3. Missing Links

#### Cross-References Need Strengthening
**Areas needing more links:**

1. **AGENTS.md → docs/README.md**: Should prominently link to docs index
2. **README.md → PARITY_JOB_MONITORING.md**: New feature not mentioned in main README
3. **Parity docs**: Should cross-link more (Pre-Gate ↔ Certification ↔ Monitoring)

---

### 4. Terminology Consistency

#### Good Consistency Overall
- Terms like "Oracle", "Parity Run", "Canonical Volume" used consistently
- Domain-specific vocabulary well-defined

#### Minor Issues:
- "Parity Closure Fast Path" vs "Parity Fast Path" vs "Parity Work" (pick one)
- "slavv jobs" sometimes formatted as code, sometimes plain text

---

## Improvement Plan

### Phase 1: Quick Wins (30 min)
**Goal**: Improve immediate navigability without major refactoring

1. ✅ **Add prominent navigation aid to AGENTS.md**
   - Already has decision tree, just needs better visual prominence
   
2. **Add monitoring to README.md**
   - Add `slavv jobs` commands to Common Commands section
   - Add PARITY_JOB_MONITORING.md to Documentation section

3. **Consolidate parity fast paths**
   - Make docs/README.md canonical
   - Update README.md to point there
   - Update AGENTS.md to point there

4. **Add glossary sync note**
   - Add note to AGENTS.md: "This glossary is the canonical source. reference/core/GLOSSARY.md contains supplementary technical details."
   - Add note to GLOSSARY.md pointing back

---

### Phase 2: Consolidation (1-2 hours)
**Goal**: Eliminate redundancy, create single sources of truth

1. **Consolidate parity workflow documentation**
   - Create PARITY_WORKFLOW_GUIDE.md that combines:
     * Cold-start protocol
     * Common commands
     * Workflow sequence (Pre-Gate → Certification → Monitoring)
   - Update all three current locations to point to it

2. **Simplify docs/README.md**
   - Extract documentation map to separate file
   - Focus on "use case → docs to read" mapping
   - Keep anti-patterns as they're valuable

3. **Strengthen cross-links**
   - Add "Related Documents" sections to each major doc
   - Use consistent linking format
   - Add breadcrumbs to workflow docs

---

### Phase 3: Structure Enhancement (2-3 hours)
**Goal**: Make documentation self-navigating for AI agents

1. **Create visual documentation map**
   - Mermaid diagram showing doc relationships
   - Include in DOCUMENTATION_MAP.md
   - Link from README.md and docs/README.md

2. **Add "Start Here" badges**
   - Add ⭐ to critical entry points across all docs
   - Make it easy for AI to spot key documents

3. **Create quick reference cards**
   - One-page cheat sheets for:
     * Parity workflow
     * Common commands
     * Documentation navigation
   - Place in docs/reference/quick-start/

---

## Recommended Immediate Actions

### Action 1: Add Monitoring to README.md
**File**: README.md  
**Change**: Add to "Common Commands" section and "Documentation" section

### Action 2: Consolidate Parity Fast Paths
**Files**: README.md, docs/README.md, AGENTS.md  
**Change**: Make docs/README.md canonical, others point there

### Action 3: Add Glossary Sync Notes
**Files**: AGENTS.md, reference/core/GLOSSARY.md  
**Change**: Document which is canonical and relationship

### Action 4: Strengthen Monitoring Integration
**Files**: All parity-related docs  
**Change**: Ensure PARITY_JOB_MONITORING.md is linked from all relevant places

---

## Metrics

### Current State
- **Total documentation files**: ~50+ (estimate)
- **Entry points**: 4 (README.md, AGENTS.md, docs/README.md, reference/README.md)
- **Parity-related docs**: 6 core + multiple supporting
- **Cross-references**: Good but could be better
- **Redundant content**: ~15% (parity workflows, glossary)

### Target State
- **Entry points**: Same 4, but clearly differentiated
- **Redundant content**: <5%
- **Cross-references**: Every major doc has "Related Documents" section
- **Navigation time**: <30 seconds for AI to find right doc for any task

---

## Validation Criteria

Documentation improvements are successful when:

1. ✅ AI agent can find parity workflow commands in <30 seconds
2. ✅ No duplicate "single source of truth" content
3. ✅ Every workflow doc has clear "Next Steps" or "Related Docs"
4. ✅ Glossary terms are consistent across all docs
5. ✅ New features (like monitoring) are integrated into all relevant navigation

---

## Notes

- **Do not** create new documentation files unless absolutely necessary
- **Do** consolidate existing content
- **Do** improve cross-linking aggressively
- **Keep** the substance of all docs; only improve structure
- **Follow** AGENTS.md constraint: max 1000 lines per file

