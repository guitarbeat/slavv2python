# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) that document important technical decisions made in the SLAVV project.

## What is an ADR?

An ADR captures a significant architectural decision, including the context, decision, and consequences. ADRs are numbered sequentially and never deleted—if a decision is reversed, a new ADR supersedes the old one.

## Format

Each ADR follows this structure:

- **Status:** Proposed | Accepted | Deprecated | Superseded
- **Date:** When the decision was made
- **Context:** The circumstances and constraints
- **Decision:** What we chose to do
- **Consequences:** The results, both positive and negative

---

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](0001-use-python-over-matlab.md) | Use Python over MATLAB for Data Processing | Accepted | 2026-01-27 |

---

## Creating New ADRs

1. Copy the template: `cp 0001-use-python-over-matlab.md NNNN-short-title.md`
2. Update the number and title
3. Fill in Context, Decision, and Consequences
4. Add to the index table above
5. Submit for review
