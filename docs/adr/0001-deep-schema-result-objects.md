# ADR 0001: Deep Schema Result Objects

## Status
Accepted

## Context
Prior to this decision, the SLAVV pipeline passed results between stages using raw Python dictionaries (`dict[str, Any]`). While flexible, this approach created several architectural problems:
1. **Shallow Interfaces**: Each computational stage had to manually handle data type coercion (e.g., NumPy dtypes), validation of array shapes, and persistence boilerplate (e.g., calling `joblib.dump`).
2. **Implicit Contracts**: The "contract" between stages (what keys must exist in the dictionary) was documented only in comments or discovered at runtime, making the system brittle.
3. **Leaked Persistence**: Knowledge of how to save and load specific data types (like large 4D energy volumes) was scattered across the `resumable.py` logic of individual stages.

## Decision
We have transitioned to "Deep" Schema classes (`EnergyResult`, `VertexSet`, `EdgeSet`, `NetworkResult`, `PipelineResult`) that provide high leverage through a small interface.

1. **Authoritative Builders**: Classes feature a `@classmethod create()` that handles all dtype coercion and structural validation (e.g., matching vertex positions to energies).
2. **Encapsulated Persistence**: Classes manage their own serialization via `.save()` and `.load()` methods, allowing the schema to select the most efficient format (joblib, npy, zarr) for the data type.
3. **Internalized Logic**: Logic that was previously "auxiliary" to the stages (like normalizing radius axes) has been moved into the schema classes.

## Consequences
- **Improved Locality**: Validation and persistence logic now lives with the data it governs, not in the calling orchestration code.
- **Enhanced Type Safety**: Stage signatures now explicitly expect schema types, making the data flow self-documenting and AI-navigable.
- **Maintenance Cost**: Adding new attributes to a stage result now requires updating the schema class, which is a desirable constraint to prevent "dictionary bloat."
- **Backward Compatibility**: Stage schemas retain `.to_dict()` for legacy consumers. `SlavvPipeline.run()` returns `PipelineResult`, which implements `Mapping[str, Any]` so existing `results["vertices"]` access in the Streamlit app and tests continues to work.
- **Pipeline envelope**: `RunState.to_pipeline_result()` is the authoritative conversion at run completion; prefer typed fields (`.vertices`, `.edges`) in new code.
- **UI envelope**: `AppRunState` (`slavv_python.schema.app_run`) wraps `PipelineResult` plus run metadata for Streamlit/session storage. Internal helpers use typed access; `.to_dict()` / `normalize_state_results()` are reserved for JSON export and share-report boundaries.
