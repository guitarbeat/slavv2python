"""Load and normalize preserved MATLAB Oracle vectors for exact-route proof."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.io import loadmat

from slavv_python.analytics.parity.array_normalization import (
    _normalize_connection_array,
    _normalize_float_array,
    _normalize_float_matrix,
    _normalize_float_matrix_list,
    _normalize_float_vector,
    _normalize_float_vector_list,
    _normalize_int_array,
    _normalize_int_vector,
    _normalize_spatial_matrix,
    _normalize_spatial_matrix_list,
    _normalize_spatial_scale_matrix_list,
    _normalize_spatial_vector_list,
    _optional_field,
    _require_key,
)
from slavv_python.analytics.parity.exact_proof_contract import EXACT_STAGE_ORDER

if TYPE_CHECKING:
    from pathlib import Path


def find_single_matlab_batch_dir(run_root: Path) -> Path:
    """Return the single preserved MATLAB batch directory under a staged run root."""
    matlab_results_dir = run_root / "01_Input" / "matlab_results"
    if not matlab_results_dir.is_dir():
        raise ValueError(f"missing MATLAB results directory: {matlab_results_dir}")

    batch_dirs = sorted(
        path
        for path in matlab_results_dir.iterdir()
        if path.is_dir() and path.name.startswith("batch_")
    )
    if not batch_dirs:
        raise ValueError(f"no MATLAB batch directory found under {matlab_results_dir}")
    if len(batch_dirs) > 1:
        joined = ", ".join(str(path) for path in batch_dirs)
        raise ValueError(
            f"expected one MATLAB batch directory under {matlab_results_dir}, found: {joined}"
        )
    return batch_dirs[0]


def _is_matlab_energy_hdf5(path: Path) -> bool:
    """Return True when ``path`` is a MATLAB energy HDF5 bundle (dataset ``d``)."""
    import h5py

    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return "d" in handle
    except OSError:
        return False


def _matlab_energy_hdf5_companion(mat_path: Path) -> Path | None:
    """Return the extensionless HDF5 companion for a MATLAB energy metadata ``.mat``."""
    candidate = mat_path.with_suffix("")
    if _is_matlab_energy_hdf5(candidate):
        return candidate
    return None


def _resolve_matlab_energy_settings_path(batch_dir: Path, energy_asset_stem: str) -> Path:
    """Locate the workflow settings mat that matches an energy artifact stem."""
    prefix = "energy_"
    if not energy_asset_stem.startswith(prefix):
        raise ValueError(f"unexpected MATLAB energy artifact stem: {energy_asset_stem}")
    timestamp = energy_asset_stem[len(prefix) :].split("_", 1)[0]
    settings_path = batch_dir / "settings" / f"energy_{timestamp}.mat"
    if not settings_path.is_file():
        raise ValueError(f"missing MATLAB energy settings file: {settings_path}")
    return settings_path


def _find_matlab_energy_path(data_dir: Path) -> Path:
    """Resolve the authoritative MATLAB energy artifact under ``data/``."""
    mat_candidates = sorted(path for path in data_dir.glob("energy*.mat") if path.is_file())
    if mat_candidates:
        if len(mat_candidates) > 1:
            joined = ", ".join(str(path) for path in mat_candidates)
            raise ValueError(
                f"expected one raw MATLAB energy artifact under {data_dir}, found: {joined}"
            )
        h5_path = _matlab_energy_hdf5_companion(mat_candidates[0])
        if h5_path is not None:
            return h5_path
        return mat_candidates[0]

    h5_candidates = sorted(
        path
        for path in data_dir.glob("energy_*")
        if path.is_file() and _is_matlab_energy_hdf5(path)
    )
    if not h5_candidates:
        raise ValueError(f"missing raw MATLAB energy artifact under {data_dir}")
    if len(h5_candidates) > 1:
        joined = ", ".join(str(path) for path in h5_candidates)
        raise ValueError(
            f"expected one raw MATLAB energy artifact under {data_dir}, found: {joined}"
        )
    return h5_candidates[0]


def find_matlab_vector_paths(
    batch_dir: Path,
    stages: tuple[str, ...] = EXACT_STAGE_ORDER,
) -> dict[str, Path]:
    """Locate the raw MATLAB exact-proof files for the requested stages."""
    vectors_dir = batch_dir / "vectors"
    data_dir = batch_dir / "data"
    stage_locations: dict[str, tuple[Path, tuple[str, ...]]] = {
        "vertices": (vectors_dir, ("curated_vertices*.mat", "vertices*.mat")),
        "edges": (vectors_dir, ("edges*.mat", "curated_edges*.mat")),
        "network": (vectors_dir, ("network*.mat",)),
    }
    stage_paths: dict[str, Path] = {}
    for stage in stages:
        if stage == "energy":
            if not data_dir.is_dir():
                raise ValueError(f"missing MATLAB energy directory: {data_dir}")
            stage_paths[stage] = _find_matlab_energy_path(data_dir)
            continue

        root_dir, patterns = stage_locations[stage]
        if not root_dir.is_dir():
            raise ValueError(f"missing MATLAB {stage} directory: {root_dir}")
        candidates: list[Path] = []
        seen: set[Path] = set()
        for pattern in patterns:
            for path in sorted(root_dir.glob(pattern)):
                if not path.is_file() or path in seen:
                    continue
                candidates.append(path)
                seen.add(path)
            if candidates:
                break
        if not candidates:
            raise ValueError(f"missing raw MATLAB {stage} artifact under {root_dir}")
        if len(candidates) > 1:
            joined = ", ".join(str(path) for path in candidates)
            raise ValueError(
                f"expected one raw MATLAB {stage} artifact under {root_dir}, found: {joined}"
            )
        stage_paths[stage] = candidates[0]
    return stage_paths


def load_normalized_matlab_vectors(
    batch_dir: Path,
    stages: tuple[str, ...] = EXACT_STAGE_ORDER,
) -> dict[str, dict[str, Any]]:
    """Load and normalize the requested raw MATLAB vector files."""
    vector_paths = find_matlab_vector_paths(batch_dir, stages)
    normalized: dict[str, dict[str, Any]] = {}
    for stage in stages:
        normalized[stage] = load_normalized_matlab_stage(vector_paths[stage], stage)
    return normalized


def load_normalized_matlab_stage(path: Path, stage: str) -> dict[str, Any]:
    """Load and normalize a single raw MATLAB vector file."""
    if stage == "energy" and _is_matlab_energy_hdf5(path):
        return _load_normalized_matlab_energy_from_hdf5(path)
    matlab_payload = cast(
        "dict[str, Any]",
        loadmat(path, squeeze_me=stage != "energy", struct_as_record=False),
    )
    if stage == "vertices":
        normalized = _normalize_matlab_vertices_payload(matlab_payload)
        if path.name.startswith("curated_vertices"):
            # MATLAB curation overwrites vertex_energies with a normalized rank
            # ramp. Recover the true physical energies from the raw vertices*.mat
            # (matched by exact integer voxel position) so the gate certifies the
            # energy Python actually computes, not a display artifact.
            normalized["energies"] = _true_vertex_energies_from_raw(path, matlab_payload)
        return normalized
    if stage == "energy":
        return _normalize_matlab_energy_payload(matlab_payload)
    if stage == "edges":
        return _normalize_matlab_edges_payload(matlab_payload)
    if stage == "network":
        return _normalize_matlab_network_payload(matlab_payload)
    raise ValueError(f"unsupported exact-proof stage: {stage}")


def load_normalized_matlab_edge_input_vertices(batch_dir: Path) -> dict[str, Any] | None:
    """Load the vertex surface embedded in the preserved MATLAB edges artifact when present."""
    edge_path = find_matlab_vector_paths(batch_dir, ("edges",)).get("edges")
    if edge_path is None:
        return None
    matlab_payload = cast(
        "dict[str, Any]",
        loadmat(edge_path, squeeze_me=True, struct_as_record=False),
    )
    required_fields = (
        "vertex_space_subscripts",
        "vertex_scale_subscripts",
        "vertex_energies",
    )
    if not all(field_name in matlab_payload for field_name in required_fields):
        return None
    return _normalize_matlab_vertex_fields_payload(matlab_payload)


def _normalize_matlab_vertices_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _normalize_matlab_vertex_fields_payload(payload)


def _find_raw_vertices_sibling(curated_path: Path) -> Path | None:
    """Locate the raw (uncurated) vertices*.mat beside a curated_vertices*.mat."""
    for candidate in sorted(curated_path.parent.glob("vertices*.mat")):
        if candidate.is_file() and not candidate.name.startswith("curated"):
            return candidate
    return None


def _true_vertex_energies_from_raw(
    curated_path: Path, curated_payload: dict[str, Any]
) -> np.ndarray:
    """Return true vertex energies for the curated set, sourced from raw vertices.mat.

    The curated artifact's ``vertex_energies`` are a curation rank-ramp; the raw
    ``vertices*.mat`` keeps the physical energies. Match each curated vertex to its
    raw counterpart by exact integer voxel subscript (no rounding) and return the
    raw energy in curated row order. Falls back to the curated energies if the raw
    sibling is absent or any curated vertex is missing from it.
    """
    raw_path = _find_raw_vertices_sibling(curated_path)
    curated_positions = np.asarray(_require_key(curated_payload, "vertex_space_subscripts"))
    if raw_path is None:
        return _normalize_float_vector(_require_key(curated_payload, "vertex_energies"))

    raw_payload = cast(
        "dict[str, Any]",
        loadmat(raw_path, squeeze_me=True, struct_as_record=False),
    )
    raw_positions = np.asarray(_require_key(raw_payload, "vertex_space_subscripts"))
    raw_energies = np.asarray(
        _require_key(raw_payload, "vertex_energies"), dtype=np.float64
    ).ravel()
    energy_by_position = {
        tuple(int(value) for value in raw_positions[index]): float(raw_energies[index])
        for index in range(raw_positions.shape[0])
    }
    true_energies = np.empty(curated_positions.shape[0], dtype=np.float64)
    for row in range(curated_positions.shape[0]):
        key = tuple(int(value) for value in curated_positions[row])
        if key not in energy_by_position:
            # A curated vertex absent from raw would be unexpected; keep the
            # curated value rather than fabricate, so the gate surfaces it.
            return _normalize_float_vector(_require_key(curated_payload, "vertex_energies"))
        true_energies[row] = energy_by_position[key]
    return _normalize_float_vector(true_energies)


def _normalize_matlab_vertex_fields_payload(
    payload: dict[str, Any],
    *,
    spatial_field: str = "vertex_space_subscripts",
    scale_field: str = "vertex_scale_subscripts",
    energy_field: str = "vertex_energies",
) -> dict[str, Any]:
    return {
        "positions": _normalize_matlab_spatial_matrix(_require_key(payload, spatial_field)),
        "scales": _normalize_matlab_int_vector(_require_key(payload, scale_field)),
        "energies": _normalize_float_vector(_require_key(payload, energy_field)),
    }


def _load_normalized_matlab_energy_from_hdf5(path: Path) -> dict[str, Any]:
    """Load MATLAB energy written as an HDF5 bundle (dataset ``d``) plus settings mat."""
    import h5py

    batch_dir = path.parent.parent
    settings_path = _resolve_matlab_energy_settings_path(batch_dir, path.name)
    settings_payload = cast(
        "dict[str, Any]",
        loadmat(settings_path, squeeze_me=True, struct_as_record=False),
    )
    with h5py.File(path, "r") as handle:
        planes = np.asarray(handle["d"], dtype=np.float64)
    if planes.ndim != 4 or planes.shape[0] < 2:
        raise ValueError(f"unexpected MATLAB energy HDF5 shape for {path}: {planes.shape}")

    scale_indices = planes[0]
    energy = planes[1]
    lumen_radius_microns = settings_payload.get("lumen_radius_in_microns_range")
    if lumen_radius_microns is None:
        lumen_radius_microns = settings_payload.get("lumen_radius_microns")

    # MATLAB writes 3D volumes in Fortran (column-major) order; h5py reads them
    # back in C order, reversing the axis indices relative to MATLAB's convention.
    # The Python checkpoint stores energy in the reoriented frame produced by
    # _reorient_exact_input_volume (permutation [2,0,1] on the original z,y,x
    # input), which swaps the last two spatial axes compared to the raw HDF5
    # layout.  Transposing with (0, 2, 1) here aligns the MATLAB oracle with the
    # Python checkpoint so that voxel [i, j, k] refers to the same physical voxel
    # in both arrays.
    energy = np.ascontiguousarray(energy.transpose(0, 2, 1))
    scale_indices = np.ascontiguousarray(scale_indices.transpose(0, 2, 1))

    return {
        "energy": _normalize_float_array(energy),
        # get_energy_V202 stores 1-based global scale subscripts in plane 0
        # (energy_chunk_scale_min + sum of prior-octave scale counts). Invalid
        # voxels are written as 0; keep them at 0 after the 1-based shift.
        "scale_indices": _normalize_int_array(scale_indices, one_based=True),
        "energy_4d": _normalize_float_array(None),
        "lumen_radius_microns": _normalize_float_vector(lumen_radius_microns),
    }


def _normalize_matlab_energy_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "energy": _normalize_float_array(_require_key(payload, "energy")),
        "scale_indices": _normalize_int_array(
            _require_key(payload, "scale_indices"),
            one_based=True,
        ),
        "energy_4d": _normalize_float_array(_require_key(payload, "energy_4d")),
        "lumen_radius_microns": _normalize_float_vector(
            _require_key(payload, "lumen_radius_microns"),
        ),
    }


def _normalize_matlab_edges_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "connections": _normalize_matlab_connections(_require_key(payload, "edges2vertices")),
        "traces": _normalize_matlab_spatial_matrix_list(
            _require_key(payload, "edge_space_subscripts"),
        ),
        "scale_traces": _normalize_matlab_float_vector_list(
            _require_key(payload, "edge_scale_subscripts"),
        ),
        "energy_traces": _normalize_float_vector_list(_require_key(payload, "edge_energies")),
        "energies": _normalize_float_vector(
            payload.get("mean_edge_energies", payload.get("energies")),
        ),
        "bridge_vertex_positions": _normalize_optional_matlab_spatial_matrix(
            payload,
            ("bridge_vertex_space_subscripts", "bridge_vertex_positions"),
        ),
        "bridge_vertex_scales": _normalize_optional_matlab_int_vector(
            payload,
            ("bridge_vertex_scale_subscripts", "bridge_vertex_scales"),
        ),
        "bridge_vertex_energies": _normalize_float_vector(
            _optional_field(payload, "bridge_vertex_energies"),
        ),
        "bridge_edges": _normalize_matlab_bridge_payload(payload),
    }


def _normalize_matlab_network_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "strands": _normalize_matlab_strands(_require_key(payload, "strands2vertices")),
        "bifurcations": _normalize_matlab_int_vector(
            _require_key(payload, "bifurcation_vertices"),
        ),
        "strand_subscripts": _normalize_matlab_spatial_scale_matrix_list(
            _require_key(payload, "strand_subscripts"),
        ),
        "strand_energy_traces": _normalize_float_vector_list(
            _require_key(payload, "strand_energies"),
        ),
        "mean_strand_energies": _normalize_float_vector(
            _require_key(payload, "mean_strand_energies"),
        ),
        "vessel_directions": _normalize_spatial_vector_list(
            _require_key(payload, "vessel_directions"),
        ),
    }


def _normalize_matlab_bridge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("bridge_edges")
    nested_mapping = nested.__dict__ if hasattr(nested, "__dict__") else {}
    has_prefixed_bridge_fields = any(key.startswith("bridge_") for key in payload)
    if not nested_mapping and not has_prefixed_bridge_fields:
        return {
            "connections": np.empty((0, 2), dtype=np.int64),
            "traces": [],
            "scale_traces": [],
            "energy_traces": [],
            "energies": np.empty((0,), dtype=np.float64),
        }

    source_payload = nested_mapping if nested_mapping else payload
    return {
        "connections": _normalize_optional_matlab_connections(
            source_payload,
            ("bridge_edges2vertices", "bridge_connections", "edges2vertices", "connections"),
        ),
        "traces": _normalize_optional_matlab_spatial_matrix_list(
            source_payload,
            ("bridge_edge_space_subscripts", "traces", "edge_space_subscripts"),
        ),
        "scale_traces": _normalize_optional_matlab_float_vector_list(
            source_payload,
            ("bridge_edge_scale_subscripts", "scale_traces", "edge_scale_subscripts"),
        ),
        "energy_traces": _normalize_float_vector_list(
            _optional_field(
                source_payload,
                "bridge_edge_energies",
                "energy_traces",
                "edge_energies",
            )
        ),
        "energies": _normalize_float_vector(
            _optional_field(
                source_payload,
                "bridge_mean_edge_energies",
                "energies",
                "mean_edge_energies",
            )
        ),
    }


def _normalize_matlab_strands(value: Any) -> list[np.ndarray]:
    connections = _normalize_matlab_connections(value)
    return [row.astype(np.int64, copy=False) for row in connections]


def _normalize_matlab_connections(value: Any) -> np.ndarray:
    return _normalize_connection_array(value, one_based=True)


def _normalize_optional_matlab_connections(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> np.ndarray:
    value = _optional_field(payload, *field_names)
    return _normalize_matlab_connections(value)


def _normalize_matlab_int_vector(value: Any) -> np.ndarray:
    return _normalize_int_vector(value, one_based=True)


def _normalize_optional_matlab_int_vector(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> np.ndarray:
    return _normalize_int_vector(_optional_field(payload, *field_names), one_based=True)


def _normalize_matlab_float_matrix(value: Any, *, columns: int) -> np.ndarray:
    return _normalize_float_matrix(value, columns=columns, one_based=True)


def _normalize_matlab_spatial_matrix(value: Any) -> np.ndarray:
    return _normalize_spatial_matrix(value, one_based=True)


def _normalize_optional_matlab_float_matrix(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
    *,
    columns: int,
) -> np.ndarray:
    return _normalize_float_matrix(
        _optional_field(payload, *field_names), columns=columns, one_based=True
    )


def _normalize_optional_matlab_spatial_matrix(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> np.ndarray:
    return _normalize_spatial_matrix(_optional_field(payload, *field_names), one_based=True)


def _normalize_matlab_float_matrix_list(value: Any, *, columns: int) -> list[np.ndarray]:
    return _normalize_float_matrix_list(value, columns=columns, one_based=True)


def _normalize_matlab_spatial_matrix_list(value: Any) -> list[np.ndarray]:
    return _normalize_spatial_matrix_list(value, one_based=True)


def _normalize_optional_matlab_float_matrix_list(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
    *,
    columns: int,
) -> list[np.ndarray]:
    return _normalize_float_matrix_list(
        _optional_field(payload, *field_names), columns=columns, one_based=True
    )


def _normalize_optional_matlab_spatial_matrix_list(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> list[np.ndarray]:
    return _normalize_spatial_matrix_list(_optional_field(payload, *field_names), one_based=True)


def _normalize_matlab_float_vector_list(value: Any) -> list[np.ndarray]:
    return _normalize_float_vector_list(value, one_based=True)


def _normalize_matlab_spatial_scale_matrix_list(value: Any) -> list[np.ndarray]:
    return _normalize_spatial_scale_matrix_list(value, one_based=True)


def _normalize_optional_matlab_float_vector_list(
    payload: dict[str, Any],
    field_names: tuple[str, ...],
) -> list[np.ndarray]:
    return _normalize_float_vector_list(_optional_field(payload, *field_names), one_based=True)


is_matlab_energy_hdf5 = _is_matlab_energy_hdf5

__all__ = [
    "find_matlab_vector_paths",
    "find_single_matlab_batch_dir",
    "is_matlab_energy_hdf5",
    "load_normalized_matlab_edge_input_vertices",
    "load_normalized_matlab_stage",
    "load_normalized_matlab_vectors",
]
