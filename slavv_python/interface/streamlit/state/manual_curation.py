"""Pure state, validation, and materialization for MATLAB-style browser curation."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.schema.results import EdgeSet, VertexSet

if TYPE_CHECKING:
    from collections.abc import Sequence


CURATION_SCHEMA_VERSION = 1
MAX_ADDED_OBJECTS = 10_000
MAX_HISTORY_ENTRIES = 20_000
MAX_TRACE_POINTS = 100_000


class CurationSessionError(ValueError):
    """Raised when a browser curation session is unsafe or incompatible."""


@dataclass
class CurationSessionV1:
    """Versioned, replayable browser curation state."""

    baseline_signature: str
    dataset_name: str
    stage: str = "vertices"
    view: dict[str, Any] = field(default_factory=dict)
    vertex_truth: list[bool] = field(default_factory=list)
    vertex_deleted: list[bool] = field(default_factory=list)
    edge_truth: list[bool] = field(default_factory=list)
    edge_deleted: list[bool] = field(default_factory=list)
    added_vertices: list[dict[str, Any]] = field(default_factory=list)
    added_edges: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    cursor: int = 0
    schema_version: int = CURATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def _hash_array(hasher: Any, value: Any, *, dtype: Any | None = None) -> None:
    array = np.asarray(value, dtype=dtype)
    hasher.update(str(array.shape).encode("ascii"))
    hasher.update(str(array.dtype).encode("ascii"))
    hasher.update(np.ascontiguousarray(array).tobytes())


def build_curation_baseline_signature(
    vertices: Mapping[str, Any],
    edges: Mapping[str, Any],
    image_shape: Sequence[int],
) -> str:
    """Fingerprint the exact object collection a curation session addresses."""
    hasher = hashlib.sha256()
    _hash_array(hasher, image_shape, dtype=np.int64)
    _hash_array(hasher, vertices.get("positions", []), dtype=np.float32)
    _hash_array(hasher, vertices.get("scales", []), dtype=np.int16)
    _hash_array(hasher, edges.get("connections", []), dtype=np.int32)
    for trace in edges.get("traces", []):
        _hash_array(hasher, trace, dtype=np.float32)
    return hasher.hexdigest()


def new_curation_session(
    vertices: Mapping[str, Any],
    edges: Mapping[str, Any],
    *,
    image_shape: Sequence[int],
    dataset_name: str,
) -> CurationSessionV1:
    """Create an untouched curation session for a typed pipeline result."""
    vertex_count = len(np.asarray(vertices.get("positions", [])).reshape(-1, 3))
    edge_count = len(np.asarray(edges.get("connections", [])).reshape(-1, 2))
    return CurationSessionV1(
        baseline_signature=build_curation_baseline_signature(vertices, edges, image_shape),
        dataset_name=str(dataset_name),
        vertex_truth=[True] * vertex_count,
        vertex_deleted=[False] * vertex_count,
        edge_truth=[True] * edge_count,
        edge_deleted=[False] * edge_count,
        view={
            "axis": 2,
            "depth": max(int(image_shape[2]) // 2, 0),
            "thickness": max(int(image_shape[2]) // 8, 1),
            "invert": True,
            "binary": False,
        },
    )


def _require_bool_list(value: Any, *, name: str, expected: int) -> list[bool]:
    if not isinstance(value, list) or len(value) != expected:
        raise CurationSessionError(f"{name} must contain exactly {expected} values")
    if any(type(item) is not bool for item in value):
        raise CurationSessionError(f"{name} must contain only booleans")
    return list(value)


def _finite_triplet(value: Any, *, name: str, shape: Sequence[int]) -> list[float]:
    try:
        result = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError) as exc:
        raise CurationSessionError(f"{name} must be a three-coordinate position") from exc
    if not np.isfinite(result).all():
        raise CurationSessionError(f"{name} contains a non-finite coordinate")
    bounds = np.asarray(shape, dtype=float)
    if np.any(result < 0) or np.any(result >= bounds):
        raise CurationSessionError(f"{name} is outside the image volume")
    return result.tolist()


def _validate_added_vertex(value: Any, *, index: int, shape: Sequence[int]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CurationSessionError(f"added_vertices[{index}] must be an object")
    position = _finite_triplet(
        value.get("position"), name=f"added_vertices[{index}].position", shape=shape
    )
    try:
        energy = float(value.get("energy", 0.0))
        scale = int(value.get("scale", 0))
        radius_pixels = np.asarray(value.get("radii_pixels", [1.0]), dtype=float)
        radius_microns = float(value.get("radius_microns", 1.0))
    except (TypeError, ValueError) as exc:
        raise CurationSessionError(f"added_vertices[{index}] has invalid attributes") from exc
    if not np.isfinite(energy) or not np.isfinite(radius_microns) or radius_microns < 0:
        raise CurationSessionError(f"added_vertices[{index}] has non-finite attributes")
    if radius_pixels.size not in {1, 3} or not np.isfinite(radius_pixels).all():
        raise CurationSessionError(
            f"added_vertices[{index}].radii_pixels must have 1 or 3 values"
        )
    return {
        "position": position,
        "energy": energy,
        "scale": scale,
        "radii_pixels": radius_pixels.reshape(-1).tolist(),
        "radius_microns": radius_microns,
    }


def _validate_added_edge(
    value: Any,
    *,
    index: int,
    vertex_count: int,
    shape: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CurationSessionError(f"added_edges[{index}] must be an object")
    try:
        endpoints = np.asarray(value.get("connections"), dtype=int).reshape(2)
        trace = np.asarray(value.get("trace"), dtype=float).reshape(-1, 3)
        energy = float(value.get("energy", -1.0e30))
    except (TypeError, ValueError) as exc:
        raise CurationSessionError(f"added_edges[{index}] has invalid geometry") from exc
    if np.any(endpoints < 0) or np.any(endpoints >= vertex_count) or endpoints[0] == endpoints[1]:
        raise CurationSessionError(f"added_edges[{index}] has invalid endpoints")
    if len(trace) < 2 or len(trace) > MAX_TRACE_POINTS:
        raise CurationSessionError(f"added_edges[{index}] has an invalid trace length")
    if not np.isfinite(trace).all():
        raise CurationSessionError(f"added_edges[{index}] contains non-finite trace coordinates")
    bounds = np.asarray(shape, dtype=float)
    if np.any(trace < 0) or np.any(trace >= bounds):
        raise CurationSessionError(f"added_edges[{index}] leaves the image volume")
    if not np.isfinite(energy):
        raise CurationSessionError(f"added_edges[{index}] has an invalid energy")
    return {
        "connections": endpoints.tolist(),
        "trace": trace.tolist(),
        "energy": energy,
    }


def validate_curation_session(
    value: CurationSessionV1 | Mapping[str, Any] | str | bytes,
    *,
    expected_signature: str,
    baseline_vertex_count: int,
    baseline_edge_count: int,
    image_shape: Sequence[int],
) -> CurationSessionV1:
    """Validate an imported or component-returned curation session."""
    if isinstance(value, CurationSessionV1):
        raw = value.to_dict()
    elif isinstance(value, bytes):
        if len(value) > 20_000_000:
            raise CurationSessionError("Curation session file is too large")
        try:
            raw = json.loads(value.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CurationSessionError("Curation session is not valid UTF-8 JSON") from exc
    elif isinstance(value, str):
        if len(value) > 20_000_000:
            raise CurationSessionError("Curation session file is too large")
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CurationSessionError("Curation session is not valid JSON") from exc
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise CurationSessionError("Curation session must be a JSON object")

    if not isinstance(raw, dict):
        raise CurationSessionError("Curation session must be a JSON object")
    if raw.get("schema_version") != CURATION_SCHEMA_VERSION:
        raise CurationSessionError(
            f"Unsupported curation schema version {raw.get('schema_version')!r}; "
            f"expected {CURATION_SCHEMA_VERSION}"
        )
    if raw.get("baseline_signature") != expected_signature:
        raise CurationSessionError("This curation belongs to a different pipeline result")
    stage = str(raw.get("stage", "vertices"))
    if stage not in {"vertices", "edges"}:
        raise CurationSessionError("Curation stage must be 'vertices' or 'edges'")

    added_vertex_values = raw.get("added_vertices", [])
    added_edge_values = raw.get("added_edges", [])
    if not isinstance(added_vertex_values, list) or len(added_vertex_values) > MAX_ADDED_OBJECTS:
        raise CurationSessionError("Too many added vertices")
    if not isinstance(added_edge_values, list) or len(added_edge_values) > MAX_ADDED_OBJECTS:
        raise CurationSessionError("Too many added edges")
    added_vertices = [
        _validate_added_vertex(item, index=index, shape=image_shape)
        for index, item in enumerate(added_vertex_values)
    ]
    vertex_count = baseline_vertex_count + len(added_vertices)
    added_edges = [
        _validate_added_edge(
            item,
            index=index,
            vertex_count=vertex_count,
            shape=image_shape,
        )
        for index, item in enumerate(added_edge_values)
    ]
    edge_count = baseline_edge_count + len(added_edges)

    history = raw.get("history", [])
    if not isinstance(history, list) or len(history) > MAX_HISTORY_ENTRIES:
        raise CurationSessionError("Curation history is invalid or too large")
    if any(not isinstance(entry, dict) for entry in history):
        raise CurationSessionError("Curation history entries must be objects")
    try:
        cursor = int(raw.get("cursor", len(history)))
    except (TypeError, ValueError) as exc:
        raise CurationSessionError("Curation history cursor is invalid") from exc
    if cursor < 0 or cursor > len(history):
        raise CurationSessionError("Curation history cursor is outside the history")

    view = raw.get("view", {})
    if not isinstance(view, dict):
        raise CurationSessionError("Curation view must be an object")
    if int(view.get("axis", 2)) not in {0, 1, 2}:
        raise CurationSessionError("Projection axis must be 0, 1, or 2")

    return CurationSessionV1(
        baseline_signature=expected_signature,
        dataset_name=str(raw.get("dataset_name", "Current run"))[:500],
        stage=stage,
        view=copy.deepcopy(view),
        vertex_truth=_require_bool_list(
            raw.get("vertex_truth"), name="vertex_truth", expected=vertex_count
        ),
        vertex_deleted=_require_bool_list(
            raw.get("vertex_deleted"), name="vertex_deleted", expected=vertex_count
        ),
        edge_truth=_require_bool_list(
            raw.get("edge_truth"), name="edge_truth", expected=edge_count
        ),
        edge_deleted=_require_bool_list(
            raw.get("edge_deleted"), name="edge_deleted", expected=edge_count
        ),
        added_vertices=added_vertices,
        added_edges=added_edges,
        history=copy.deepcopy(history),
        cursor=cursor,
    )


def _filter_item_payload(
    payload: Mapping[str, Any],
    keep: np.ndarray,
    size: int,
    *,
    unaligned_keys: Sequence[str] = (),
) -> dict[str, Any]:
    """Copy a stage payload while filtering fields aligned to its item axis."""
    filtered: dict[str, Any] = {}
    for key, value in payload.items():
        if key in unaligned_keys:
            filtered[key] = copy.deepcopy(value)
        elif isinstance(value, np.ndarray) and value.ndim and len(value) == size:
            filtered[key] = value[keep].copy()
        elif isinstance(value, list) and len(value) == size:
            filtered[key] = [copy.deepcopy(value[index]) for index in np.flatnonzero(keep)]
        else:
            filtered[key] = copy.deepcopy(value)
    return filtered


def _append_vertices(payload: dict[str, Any], added: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not added:
        return payload
    result = copy.deepcopy(payload)
    positions = np.asarray([item["position"] for item in added], dtype=np.float32)
    energies = np.asarray([item["energy"] for item in added], dtype=np.float32)
    scales = np.asarray([item["scale"] for item in added], dtype=np.int16)
    radii_microns = np.asarray(
        [item["radius_microns"] for item in added], dtype=np.float32
    )
    existing_radii = np.asarray(result.get("radii_pixels", []), dtype=np.float32)
    radii_width = existing_radii.shape[1] if existing_radii.ndim == 2 else 1
    radii_rows = []
    for item in added:
        row = np.asarray(item["radii_pixels"], dtype=np.float32).reshape(-1)
        if row.size == 1:
            row = np.repeat(row, radii_width)
        if row.size != radii_width:
            row = np.resize(row, radii_width)
        radii_rows.append(row)
    result["positions"] = np.concatenate(
        [np.asarray(result["positions"], dtype=np.float32).reshape(-1, 3), positions], axis=0
    )
    result["energies"] = np.concatenate([np.asarray(result["energies"]), energies])
    result["scales"] = np.concatenate([np.asarray(result["scales"]), scales])
    result["radii_microns"] = np.concatenate(
        [np.asarray(result.get("radii_microns", []), dtype=np.float32), radii_microns]
    )
    radii_added = np.asarray(radii_rows, dtype=np.float32)
    if existing_radii.ndim == 1:
        radii_added = radii_added[:, 0]
    result["radii_pixels"] = np.concatenate([existing_radii, radii_added], axis=0)
    return result


def _append_edges(payload: dict[str, Any], added: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not added:
        return payload
    result = copy.deepcopy(payload)
    result["traces"] = list(result.get("traces", [])) + [
        np.asarray(item["trace"], dtype=np.float32) for item in added
    ]
    result["connections"] = np.concatenate(
        [
            np.asarray(result.get("connections", []), dtype=np.int32).reshape(-1, 2),
            np.asarray([item["connections"] for item in added], dtype=np.int32).reshape(-1, 2),
        ],
        axis=0,
    )
    result["energies"] = np.concatenate(
        [
            np.asarray(result.get("energies", []), dtype=np.float32).reshape(-1),
            np.asarray([item["energy"] for item in added], dtype=np.float32),
        ]
    )
    return result


def materialize_curation_session(
    vertices: Mapping[str, Any],
    edges: Mapping[str, Any],
    session: CurationSessionV1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the current truth/deleted state and additions to typed stage payloads."""
    vertex_payload = _append_vertices(
        VertexSet.from_dict(dict(vertices)).to_dict(), session.added_vertices
    )
    edge_payload = _append_edges(EdgeSet.from_dict(dict(edges)).to_dict(), session.added_edges)
    vertex_count = len(np.asarray(vertex_payload["positions"]).reshape(-1, 3))
    vertex_keep = np.asarray(session.vertex_truth, dtype=bool) & ~np.asarray(
        session.vertex_deleted, dtype=bool
    )
    if len(vertex_keep) != vertex_count:
        raise CurationSessionError("Vertex state no longer matches the materialized Vertex Set")

    connections = np.asarray(edge_payload.get("connections", []), dtype=np.int64).reshape(-1, 2)
    edge_count = len(connections)
    edge_keep = np.asarray(session.edge_truth, dtype=bool) & ~np.asarray(
        session.edge_deleted, dtype=bool
    )
    if len(edge_keep) != edge_count:
        raise CurationSessionError("Edge state no longer matches the materialized Edge Set")
    bridge_positions = np.asarray(
        edge_payload.get("bridge_vertex_positions", np.empty((0, 3))), dtype=np.float64
    ).reshape(-1, 3)
    bridge_count = len(bridge_positions)
    endpoint_count = vertex_count + bridge_count
    valid_endpoints = np.all((connections >= 0) & (connections < endpoint_count), axis=1)
    edge_keep &= valid_endpoints
    if connections.size:
        endpoint_keep = np.concatenate([vertex_keep, np.ones(bridge_count, dtype=bool)])
        safe_connections = np.clip(connections, 0, max(endpoint_count - 1, 0))
        edge_keep &= np.all(endpoint_keep[safe_connections], axis=1)

    vertex_map = np.full(endpoint_count, -1, dtype=np.int64)
    vertex_map[np.flatnonzero(vertex_keep)] = np.arange(
        int(vertex_keep.sum()), dtype=np.int64
    )
    if bridge_count:
        vertex_map[vertex_count:] = np.arange(
            int(vertex_keep.sum()), int(vertex_keep.sum()) + bridge_count, dtype=np.int64
        )
    curated_vertices = _filter_item_payload(vertex_payload, vertex_keep, vertex_count)
    curated_edges = _filter_item_payload(
        edge_payload,
        edge_keep,
        edge_count,
        unaligned_keys=(
            "bridge_vertex_positions",
            "bridge_vertex_scales",
            "bridge_vertex_energies",
            "lumen_radius_microns",
            "lumen_radius_pixels",
            "lumen_radius_pixels_axes",
        ),
    )
    curated_edges["connections"] = vertex_map[connections[edge_keep]].astype(np.int32)
    curated_vertices.pop("status", None)
    curated_edges.pop("status", None)
    return curated_vertices, curated_edges


def curate_manual_selection(
    vertices: Mapping[str, Any],
    edges: Mapping[str, Any],
    *,
    rejected_vertex_ids: Sequence[int] = (),
    rejected_edge_ids: Sequence[int] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compatibility helper for simple table-based rejection workflows."""
    vertex_count = len(np.asarray(vertices.get("positions", [])).reshape(-1, 3))
    edge_count = len(np.asarray(edges.get("connections", [])).reshape(-1, 2))
    vertex_truth = np.ones(vertex_count, dtype=bool)
    edge_truth = np.ones(edge_count, dtype=bool)
    for index in rejected_vertex_ids:
        if 0 <= int(index) < vertex_count:
            vertex_truth[int(index)] = False
    for index in rejected_edge_ids:
        if 0 <= int(index) < edge_count:
            edge_truth[int(index)] = False
    session = CurationSessionV1(
        baseline_signature="compatibility",
        dataset_name="Current run",
        vertex_truth=vertex_truth.tolist(),
        vertex_deleted=[False] * vertex_count,
        edge_truth=edge_truth.tolist(),
        edge_deleted=[False] * edge_count,
    )
    return materialize_curation_session(vertices, edges, session)


def serialize_curation_session(session: CurationSessionV1) -> bytes:
    """Serialize a validated session using stable, readable JSON."""
    return json.dumps(
        session.to_dict(), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8")


__all__ = [
    "CURATION_SCHEMA_VERSION",
    "CurationSessionError",
    "CurationSessionV1",
    "build_curation_baseline_signature",
    "curate_manual_selection",
    "materialize_curation_session",
    "new_curation_session",
    "serialize_curation_session",
    "validate_curation_session",
]
