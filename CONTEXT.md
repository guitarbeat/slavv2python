# SLAVV Domain Glossary

## Lowest Linear Index Priority
The secondary tie-breaking rule for [Vertex](#vertex) and [Edge Discovery](#edge-discovery). When two voxels have identical energy values, the one with the lower Fortran-order linear index is prioritized.

## Pipeline
The authoritative sequence of computational stages (Energy → Vertices → Edges → Network) required to transform a 3D vascular volume into a vectorized graph representation.

## Vertex
A localized point of interest in the vascular volume, characterized by a 3D position, an estimated radius, and a local energy value.

## Seed Vertex
A [Vertex](#vertex) identified directly from the energy field as a local minimum. These serve as the initial discovery points for the [Pipeline](#pipeline).

## Bridge Vertex
A structural [Vertex](#vertex) inserted during edge selection to resolve overlaps or connectivity gaps. These are topologically necessary but were not originally identified as energy minima.

## Vertex Set
The authoritative collection of [Vertices](#vertex) for a given stage of a [Run](#run). A Vertex Set can contain both Seed and Bridge vertices.

## Edge Discovery
The process of identifying potential connectivity between [Vertices](#vertex) by analyzing the energy field.

## Tracing Discovery
An [Edge Discovery](#edge-discovery) strategy that identifies centerlines via frontier propagation from individual Seed Vertices.

## Watershed Discovery
An [Edge Discovery](#edge-discovery) strategy that partitions the volume into regional influence zones (catchment basins) to identify adjacent [Vertices](#vertex).

## Run State
The complete collection of data persisted during a [Run](#run).

## Stage Result
The authoritative output of a [Pipeline](#pipeline) stage, serving as the interface for subsequent stages.

## Checkpoint
Internal state persisted during a stage's execution to allow a [Run](#run) to recover from interruption or to skip recalculation.

## Artifact
Supplemental data produced by a stage for diagnostics, auditing, or visualization that is not strictly required for [Pipeline](#pipeline) progression.
