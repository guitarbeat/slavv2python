"""Python comparison result loading helpers."""

from __future__ import annotations

import glob
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from ...utils.safe_unpickle import safe_load

if TYPE_CHECKING:
    from pathlib import Path

_PYTHON_RESULT_SOURCE_CHOICES = {
    "auto",
    "checkpoints-only",
    "export-json-only",
    "network-json-only",
}


def _resolve_python_energy_source(energy_data: dict[str, Any] | None) -> str:
    """Infer the origin of Python energy data for diagnostics."""
    if energy_data is None:
        return "native_python"
    comparison_mode = energy_data.get("comparison_mode")
    if isinstance(comparison_mode, dict):
        result_source = comparison_mode.get("result_source")
        if result_source is not None:
            return str(result_source)
    energy_origin = energy_data.get("energy_origin")
    if energy_origin is not None:
        return str(energy_origin)
    source = energy_data.get("source")
    if source is not None:
        return str(source)
    energy_source = energy_data.get("energy_source")
    if energy_source is not None:
        return str(energy_source)
    return "native_python"


def _load_python_results_from_checkpoints(python_root: Path) -> dict[str, Any] | None:
    """Prefer stage checkpoints over exported JSON when reconstructing comparison inputs."""
    checkpoint_dir = python_root / "checkpoints"
    energy_path = checkpoint_dir / "checkpoint_energy.pkl"
    vertices_path = checkpoint_dir / "checkpoint_vertices.pkl"
    edges_path = checkpoint_dir / "checkpoint_edges.pkl"
    network_path = checkpoint_dir / "checkpoint_network.pkl"
    if not (vertices_path.exists() and edges_path.exists() and network_path.exists()):
        return None

    try:
        energy_data = safe_load(energy_path) if energy_path.exists() else None
        vertices = safe_load(vertices_path)
        edges = safe_load(edges_path)
        network = safe_load(network_path)
    except Exception:
        return None

    results = {
        "energy_data": energy_data,
        "vertices": vertices,
        "edges": edges,
        "network": network,
    }
    candidate_edges = _load_python_candidate_edges(python_root)
    if candidate_edges is not None:
        results["candidate_edges"] = candidate_edges
    candidate_audit = _load_python_candidate_audit(python_root)
    if candidate_audit is not None:
        results["candidate_audit"] = candidate_audit
    candidate_lifecycle = _load_python_candidate_lifecycle(python_root)
    if candidate_lifecycle is not None:
        results["candidate_lifecycle"] = candidate_lifecycle
    return results


def _load_python_candidate_edges(python_root: Path) -> dict[str, Any] | None:
    """Load the persisted pre-cleanup candidate edge manifest when available."""
    candidate_path = python_root / "stages" / "edges" / "candidates.pkl"
    if not candidate_path.exists():
        return None
    try:
        return safe_load(candidate_path)
    except Exception:
        return None


def _load_python_candidate_audit(python_root: Path) -> dict[str, Any] | None:
    """Load the persisted candidate provenance audit when available."""
    candidate_audit_path = python_root / "stages" / "edges" / "candidate_audit.json"
    if not candidate_audit_path.exists():
        return None
    try:
        with open(candidate_audit_path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_python_candidate_lifecycle(python_root: Path) -> dict[str, Any] | None:
    """Load the persisted frontier lifecycle artifact when available."""
    candidate_lifecycle_path = python_root / "stages" / "edges" / "candidate_lifecycle.json"
    if not candidate_lifecycle_path.exists():
        return None
    try:
        with open(candidate_lifecycle_path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_python_results_from_source(
    python_root: Path,
    python_result_source: str,
) -> dict[str, Any]:
    """Load Python comparison data from the requested source."""
    normalized_source = python_result_source.strip().lower()
    if normalized_source not in _PYTHON_RESULT_SOURCE_CHOICES:
        raise ValueError(
            f"Unsupported python result source '{python_result_source}'. "
            f"Expected one of: {sorted(_PYTHON_RESULT_SOURCE_CHOICES)}"
        )

    source_order: list[str]
    if normalized_source == "auto":
        source_order = ["checkpoints-only", "export-json-only", "network-json-only"]
    else:
        source_order = [normalized_source]

    result_payload = {"success": True, "output_dir": str(python_root), "elapsed_time": 0.0}
    source_errors: list[str] = []

    for source_name in source_order:
        if source_name == "checkpoints-only":
            checkpoint_results = _load_python_results_from_checkpoints(python_root)
            if checkpoint_results is None:
                source_errors.append("checkpoints unavailable")
                continue

            print(f"Loading Python results from checkpoints in: {python_root / 'checkpoints'}")
            result_payload["results"] = checkpoint_results
            result_payload["vertices_count"] = len(
                checkpoint_results.get("vertices", {}).get("positions", [])
            )
            result_payload["edges_count"] = len(
                checkpoint_results.get("edges", {}).get("traces", [])
            )
            result_payload["network_strands_count"] = len(
                checkpoint_results.get("network", {}).get("strands", [])
            )
            result_payload["comparison_mode"] = {
                "result_source": "checkpoints",
                "energy_source": _resolve_python_energy_source(
                    checkpoint_results.get("energy_data")
                ),
            }
            return result_payload

        if source_name == "export-json-only":
            json_files = [
                path
                for path in glob.glob(str(python_root / "python_comparison_*.json"))
                if not path.endswith("_parameters.json")
            ]
            if not json_files:
                source_errors.append("export_json unavailable")
                continue

            latest_json = sorted(json_files)[-1]
            print(f"Loading Python results from: {latest_json}")
            try:
                with open(latest_json) as f:
                    loaded_data = json.load(f)

                if "vertices" in loaded_data and "positions" in loaded_data["vertices"]:
                    loaded_data["vertices"]["positions"] = np.array(
                        loaded_data["vertices"]["positions"]
                    )
                    if "radii" in loaded_data["vertices"]:
                        loaded_data["vertices"]["radii"] = np.array(
                            loaded_data["vertices"]["radii"]
                        )

                if "edges" in loaded_data and "traces" in loaded_data["edges"]:
                    loaded_data["edges"]["traces"] = [
                        np.array(t) for t in loaded_data["edges"]["traces"]
                    ]

                result_payload["results"] = loaded_data
                result_payload["vertices_count"] = len(
                    loaded_data.get("vertices", {}).get("positions", [])
                )
                result_payload["edges_count"] = len(loaded_data.get("edges", {}).get("traces", []))
                result_payload["network_strands_count"] = len(
                    loaded_data.get("network", {}).get("strands", [])
                )
                result_payload["comparison_mode"] = {
                    "result_source": "export_json",
                    "energy_source": "native_python",
                }
                return result_payload
            except Exception as e:
                result_payload["success"] = False
                source_errors.append(f"export_json error: {e}")
        elif source_name == "network-json-only":
            network_json_paths = glob.glob(str(python_root / "network.json"))
            if not network_json_paths:
                source_errors.append("network_json unavailable")
                continue

            network_json = network_json_paths[0]
            print(f"Loading Python results from fallback: {network_json}")
            try:
                with open(network_json) as f:
                    net_data = json.load(f)

                loaded_data = {
                    "vertices": net_data.get("vertices", {}),
                    "edges": net_data.get("edges", {}),
                    "network": net_data.get("network", {}),
                }

                if "positions" in loaded_data["vertices"]:
                    loaded_data["vertices"]["positions"] = np.array(
                        loaded_data["vertices"]["positions"]
                    )

                result_payload["results"] = loaded_data
                result_payload["vertices_count"] = len(
                    loaded_data.get("vertices", {}).get("positions", [])
                )
                if "connections" in loaded_data.get("edges", {}):
                    result_payload["edges_count"] = len(loaded_data["edges"]["connections"])
                else:
                    result_payload["edges_count"] = len(
                        loaded_data.get("edges", {}).get("traces", [])
                    )
                result_payload["network_strands_count"] = len(
                    loaded_data.get("network", {}).get("strands", [])
                )
                result_payload["comparison_mode"] = {
                    "result_source": "network_json",
                    "energy_source": "native_python",
                }
                return result_payload
            except Exception as e:
                result_payload["success"] = False
                source_errors.append(f"network_json error: {e}")
    result_payload["success"] = False
    result_payload["error"] = (
        "; ".join(source_errors) if source_errors else "No result files found."
    )
    return result_payload
