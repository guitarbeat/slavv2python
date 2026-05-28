"""Compatibility facade for parity execution helpers.

Implementation is split across:
- ``params_audit`` — exact-route parameter audit and persistence
- ``surfaces`` — dataset/oracle/run authority surfaces and manifests
- ``bootstrap`` — init-exact-run derivation and resume classification
"""

from __future__ import annotations

from .bootstrap import (
    _copy_exact_bootstrap_refs,
    _finalize_init_exact_run,
    _reorient_exact_input_volume,
    _resolve_existing_init_exact_run,
    derive_exact_params_from_oracle,
    maybe_sync_exact_vertex_checkpoint,
)
from .params_audit import (
    _normalize_param_value,
    build_exact_params_audit,
    load_params_file,
    normalize_param_value,
    persist_param_storage,
)
from .surfaces import (
    _oracle_energy_size_of_image,
    copy_source_surface,
    ensure_dest_run_layout,
    load_dataset_surface,
    load_oracle_surface,
    oracle_energy_size_of_image,
    resolve_input_file,
    validate_exact_proof_source_surface,
    validate_source_run_surface,
    write_run_manifest,
)

__all__ = [
    "_copy_exact_bootstrap_refs",
    "_finalize_init_exact_run",
    "_normalize_param_value",
    "_oracle_energy_size_of_image",
    "_reorient_exact_input_volume",
    "_resolve_existing_init_exact_run",
    "build_exact_params_audit",
    "copy_source_surface",
    "derive_exact_params_from_oracle",
    "ensure_dest_run_layout",
    "load_dataset_surface",
    "load_oracle_surface",
    "load_params_file",
    "maybe_sync_exact_vertex_checkpoint",
    "normalize_param_value",
    "oracle_energy_size_of_image",
    "persist_param_storage",
    "resolve_input_file",
    "validate_exact_proof_source_surface",
    "validate_source_run_surface",
    "write_run_manifest",
]
