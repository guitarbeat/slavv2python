# Architecture Decision Records

This document indexes Architecture Decision Records (ADRs) for the SLAVV project.

## What is an ADR?

An ADR captures a significant architectural decision, including context, decision, and consequences. ADRs are numbered sequentially and never deleted; if a decision is reversed, a new ADR supersedes the old one.

## Format

Each ADR should include:

- `Status`: Proposed | Accepted | Deprecated | Superseded
- `Date`: when the decision was made
- `Context`: constraints and background
- `Decision`: what was chosen
- `Consequences`: tradeoffs and impacts

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [0001](ADR_0001-use-python-over-matlab.md) | Use Python over MATLAB for Data Processing | Accepted | 2026-01-27 |

## Creating a new ADR

1. Copy template: `cp ADR_0001-use-python-over-matlab.md ADR_NNNN-short-title.md`
2. Update number and title.
3. Fill in context, decision, and consequences.
4. Add the ADR to the index table above.
