# Glossary

[Up: Documentation Index](../../README.md)

Maintain this reference for domain-specific and project-specific terms used throughout the `slavv2python` repository.

## Core Concepts

| Term | Definition |
| --- | --- |
| **Origin** | A starting vertex or seed point from which the extraction pipeline begins searching for edge candidates. |
| **Candidate** | A potential edge trace identified during the frontier-searching or watershed phase. Candidates are evaluated for ownership and cleanup before becoming final edges. |
| **Edge** | A finalized trace connecting two vertices. Edges represent the local skeleton of the vascular network. |
| **Strand** | A connected sequence of one or more edges that represents a distinct vascular branch or segment between junction points. |
| **Neighborhood** | The local spatial region around an **Origin**. Parity analysis often focuses on "shared neighborhoods" where multiple origins compete for candidates. |
| **Frontier** | The active set of pixels at the leading edge of a trace expansion or watershed search. |
| **Parity** | The goal of achieving exact numerical and topological agreement between the Python implementation and the original MATLAB codebase. |

## Workflow & Infrastructure

| Term | Definition |
| --- | --- |
| **Staged Run** | A structured run directory that follows the canonical `01_Input/`, `02_Output/`, `03_Analysis/`, `99_Metadata/` layout. See [COMPARISON_LAYOUT.md](COMPARISON_LAYOUT.md). |
| **Checkpoint** | A persisted intermediate state of a run (e.g., `vertices.pkl`, `edges.pkl`). Checkpoints allow the pipeline to resume from a specific stage without recomputing earlier steps. |
| **Energy** | A pre-processed image volume (e.g., vesselness, objectness, or Hessian map) that serves as the numerical input for vertex and edge extraction. |
| **Imported-MATLAB Workflow** | A parity-validation mode where Python runs using energy or vertices imported directly from a MATLAB execution, isolating the downstream extraction logic. |

## Data Formats

| Term | Definition |
| --- | --- |
| **network.json** | The canonical portable JSON export for Python vascular networks, containing vertices, edges, and summary metrics. |
| **VMV / CASX** | Legacy binary and MATLAB-native file formats used for interoperability with original SLAVV tools. |
| **Zarr** | An optional chunked, compressed, N-dimensional array format used for storing large energy volumes during resumable runs. |

## Up: [Project Docs](../../README.md)