# Modularization Plan

This file now tracks the post-compatibility codebase.

## Completed

- test support and ownership cleanup
- typed pipeline result models
- workflow extraction from `source/slavv/core/pipeline.py`
- runtime helper split under `source/slavv/runtime/_run_state/`
- app state and service extraction
- CLI service extraction
- removal of parity, MATLAB import, comparison, and legacy checkpoint compatibility surfaces

## Current Architecture

- `source/slavv/core/`: extraction algorithms and stage implementations
- `source/slavv/workflows/`: pipeline orchestration helpers
- `source/slavv/runtime/`: structured run metadata and status handling
- `source/slavv/apps/`: CLI, Streamlit pages, and app services
- `source/slavv/io/`: supported file and network formats
- `source/slavv/models/`: typed result payloads

## Remaining Cleanup

- keep thinning `source/slavv/core/pipeline.py` until it is a small facade over `workflows/`
- keep pushing Streamlit pages toward presentation-only modules
- remove stale naming inside core modules where parity-era labels no longer describe the production behavior
- continue deleting dead docs, tests, and helpers as the scientific core is simplified

## Working Rules

- prefer deletion over compatibility wrappers
- keep only the structured `run_dir` resumable flow
- treat removed parity/MATLAB surfaces as intentionally unsupported
- validate every cleanup with focused `pytest`, `ruff check`, and `compileall` as appropriate
