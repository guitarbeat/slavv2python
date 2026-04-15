"""Maintained proof artifacts for successful stage-isolated network-gate runs."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .run_layout import resolve_run_layout

logger = logging.getLogger(__name__)

_PROOF_INDEX_NAME = "proof_artifact_index.json"


@dataclass
class StageIsolatedNetworkProof:
    """Durable proof artifact for a successful stage-isolated network gate."""

    run_root: str
    generated_at: str
    matlab_batch_folder: str
    matlab_batch_timestamp: str | None
    matlab_edges_fingerprint: str
    matlab_vertices_fingerprint: str | None
    matlab_energy_fingerprint: str | None
    python_network_fingerprint: str | None
    python_vertices_fingerprint: str | None
    python_edges_fingerprint: str | None
    execution_timestamp: str
    elapsed_seconds: float
    comparison_exact_network_forced: bool
    vertices_exact_parity: bool
    edges_exact_parity: bool
    strands_exact_parity: bool
    overall_parity_achieved: bool
    peak_memory_mb: float | None = None
    cpu_time_seconds: float | None = None
    artifact_json_path: str = ""
    artifact_markdown_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary view."""
        return asdict(self)


@dataclass
class ProofArtifactIndex:
    """Chronological index of persisted stage-isolated network proofs."""

    run_root: str
    proof_artifacts: list[dict[str, Any]] = field(default_factory=list)
    latest_proof: dict[str, Any] | None = None
    total_proofs: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary view."""
        return asdict(self)


def _compute_file_fingerprint(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return f"sha256:{hasher.hexdigest()}"


def _proof_artifacts_dir(run_root: Path) -> Path:
    layout = resolve_run_layout(run_root)
    analysis_dir = layout["run_root"] / "03_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir / "proof_artifacts"


def _proof_timestamp_label(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return parsed.strftime("%Y%m%d_%H%M%S")


def _proof_paths(run_root: Path, timestamp: str) -> tuple[Path, Path]:
    proof_dir = _proof_artifacts_dir(run_root)
    proof_dir.mkdir(parents=True, exist_ok=True)
    label = _proof_timestamp_label(timestamp)
    return (
        proof_dir / f"network_gate_proof_{label}.json",
        proof_dir / f"network_gate_proof_{label}.md",
    )


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _find_latest_matlab_batch(matlab_dir: Path) -> Path | None:
    if not matlab_dir.exists():
        return None
    candidates = [
        child
        for child in matlab_dir.iterdir()
        if child.is_dir() and child.name.startswith("batch_")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _build_proof_markdown(proof: StageIsolatedNetworkProof) -> str:
    status = "exact parity achieved" if proof.overall_parity_achieved else "parity not achieved"
    lines = [
        "# Stage-Isolated Network Proof",
        "",
        f"- Run root: `{proof.run_root}`",
        f"- Generated at: `{proof.generated_at}`",
        f"- Execution timestamp: `{proof.execution_timestamp}`",
        f"- MATLAB batch folder: `{proof.matlab_batch_folder}`",
        f"- Status: `{status}`",
        f"- Elapsed seconds: `{proof.elapsed_seconds:.3f}`",
        "",
        "## Exact Parity Status",
        "",
        f"- Vertices: `{proof.vertices_exact_parity}`",
        f"- Edges: `{proof.edges_exact_parity}`",
        f"- Strands: `{proof.strands_exact_parity}`",
        "",
        "## Fingerprints",
        "",
        f"- MATLAB edges: `{proof.matlab_edges_fingerprint}`",
        f"- MATLAB vertices: `{proof.matlab_vertices_fingerprint or 'n/a'}`",
        f"- MATLAB energy: `{proof.matlab_energy_fingerprint or 'n/a'}`",
        f"- Python network: `{proof.python_network_fingerprint or 'n/a'}`",
        f"- Python vertices: `{proof.python_vertices_fingerprint or 'n/a'}`",
        f"- Python edges: `{proof.python_edges_fingerprint or 'n/a'}`",
    ]
    if proof.peak_memory_mb is not None or proof.cpu_time_seconds is not None:
        lines.extend(["", "## Resource Usage", ""])
        lines.append(f"- Peak memory MB: `{proof.peak_memory_mb}`")
        lines.append(f"- CPU time seconds: `{proof.cpu_time_seconds}`")
    return "\n".join(lines) + "\n"


def _build_index_entry(proof: StageIsolatedNetworkProof) -> dict[str, Any]:
    return {
        "generated_at": proof.generated_at,
        "execution_timestamp": proof.execution_timestamp,
        "overall_parity_achieved": proof.overall_parity_achieved,
        "vertices_exact_parity": proof.vertices_exact_parity,
        "edges_exact_parity": proof.edges_exact_parity,
        "strands_exact_parity": proof.strands_exact_parity,
        "artifact_json_path": proof.artifact_json_path,
        "artifact_markdown_path": proof.artifact_markdown_path,
    }


def _proof_from_payload(payload: dict[str, Any]) -> StageIsolatedNetworkProof:
    return StageIsolatedNetworkProof(
        run_root=str(payload.get("run_root", "")),
        generated_at=str(payload.get("generated_at", "")),
        matlab_batch_folder=str(payload.get("matlab_batch_folder", "")),
        matlab_batch_timestamp=payload.get("matlab_batch_timestamp"),
        matlab_edges_fingerprint=str(payload.get("matlab_edges_fingerprint", "")),
        matlab_vertices_fingerprint=payload.get("matlab_vertices_fingerprint"),
        matlab_energy_fingerprint=payload.get("matlab_energy_fingerprint"),
        python_network_fingerprint=payload.get("python_network_fingerprint"),
        python_vertices_fingerprint=payload.get("python_vertices_fingerprint"),
        python_edges_fingerprint=payload.get("python_edges_fingerprint"),
        execution_timestamp=str(payload.get("execution_timestamp", "")),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        comparison_exact_network_forced=bool(payload.get("comparison_exact_network_forced", False)),
        vertices_exact_parity=bool(payload.get("vertices_exact_parity", False)),
        edges_exact_parity=bool(payload.get("edges_exact_parity", False)),
        strands_exact_parity=bool(payload.get("strands_exact_parity", False)),
        overall_parity_achieved=bool(payload.get("overall_parity_achieved", False)),
        peak_memory_mb=payload.get("peak_memory_mb"),
        cpu_time_seconds=payload.get("cpu_time_seconds"),
        artifact_json_path=str(payload.get("artifact_json_path", "")),
        artifact_markdown_path=str(payload.get("artifact_markdown_path", "")),
    )


def generate_proof_artifact(
    execution_metadata_path: Path,
    *,
    run_root: Path,
) -> StageIsolatedNetworkProof:
    """Generate and persist a proof artifact from network-gate execution metadata."""
    payload = _read_json(execution_metadata_path)
    if not payload.get("parity_achieved"):
        raise ValueError("Network-gate execution did not achieve exact parity; no proof artifact.")
    if not payload.get("comparison_exact_network_forced"):
        raise ValueError(
            "Proof artifact requires comparison_exact_network=True execution metadata."
        )

    layout = resolve_run_layout(run_root)
    matlab_dir = layout["matlab_dir"]
    python_dir = layout["python_dir"]
    batch_folder = (
        Path(str(payload.get("matlab_batch_folder", "")))
        if payload.get("matlab_batch_folder")
        else _find_latest_matlab_batch(matlab_dir)
    )
    if batch_folder is None:
        raise FileNotFoundError("Could not resolve the staged MATLAB batch folder for the proof.")

    proof = StageIsolatedNetworkProof(
        run_root=str(layout["run_root"]),
        generated_at=datetime.now().isoformat(),
        matlab_batch_folder=str(batch_folder),
        matlab_batch_timestamp=batch_folder.name.replace("batch_", "")
        if batch_folder.name.startswith("batch_")
        else None,
        matlab_edges_fingerprint=str(
            payload.get("validation", {}).get("matlab_edges_fingerprint") or ""
        ),
        matlab_vertices_fingerprint=payload.get("validation", {}).get(
            "matlab_vertices_fingerprint"
        ),
        matlab_energy_fingerprint=_compute_file_fingerprint(matlab_dir / "energy.mat"),
        python_network_fingerprint=payload.get("python_network_fingerprint")
        or _compute_file_fingerprint(python_dir / "network.json")
        or _compute_file_fingerprint(python_dir / "checkpoints" / "checkpoint_network.pkl"),
        python_vertices_fingerprint=_compute_file_fingerprint(
            python_dir / "checkpoints" / "checkpoint_vertices.pkl"
        ),
        python_edges_fingerprint=_compute_file_fingerprint(
            python_dir / "checkpoints" / "checkpoint_edges.pkl"
        ),
        execution_timestamp=str(payload.get("completed_at") or payload.get("started_at") or ""),
        elapsed_seconds=float(payload.get("elapsed_seconds", 0.0)),
        comparison_exact_network_forced=bool(payload.get("comparison_exact_network_forced", False)),
        vertices_exact_parity=bool(payload.get("vertices_match", False)),
        edges_exact_parity=bool(payload.get("edges_match", False)),
        strands_exact_parity=bool(payload.get("strands_match", False)),
        overall_parity_achieved=bool(payload.get("parity_achieved", False)),
        peak_memory_mb=payload.get("peak_memory_mb"),
        cpu_time_seconds=payload.get("cpu_time_seconds"),
    )

    json_path, markdown_path = _proof_paths(layout["run_root"], proof.execution_timestamp)
    proof.artifact_json_path = str(json_path)
    proof.artifact_markdown_path = str(markdown_path)

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(proof.to_dict(), handle, indent=2, sort_keys=True)
    markdown_path.write_text(_build_proof_markdown(proof), encoding="utf-8")
    return proof


def maintain_proof_artifact_index(
    run_root: Path,
    new_proof: StageIsolatedNetworkProof | None = None,
) -> ProofArtifactIndex:
    """Persist or rebuild the canonical proof artifact index for a run root."""
    del new_proof

    layout = resolve_run_layout(run_root)
    analysis_dir = layout["run_root"] / "03_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    proof_dir = _proof_artifacts_dir(layout["run_root"])
    proof_dir.mkdir(parents=True, exist_ok=True)
    proofs: list[StageIsolatedNetworkProof] = []

    for path in sorted(proof_dir.glob("network_gate_proof_*.json")):
        try:
            proof = _proof_from_payload(_read_json(path))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable proof artifact %s: %s", path, exc)
            continue
        proofs.append(proof)

    proofs.sort(key=lambda proof: (proof.execution_timestamp, proof.generated_at))
    proof_entries = [_build_index_entry(proof) for proof in proofs]
    latest = proof_entries[-1] if proof_entries else None
    index = ProofArtifactIndex(
        run_root=str(layout["run_root"]),
        proof_artifacts=proof_entries,
        latest_proof=latest,
        total_proofs=len(proof_entries),
    )

    index_path = analysis_dir / _PROOF_INDEX_NAME
    with open(index_path, "w", encoding="utf-8") as handle:
        json.dump(index.to_dict(), handle, indent=2, sort_keys=True)
    return index


def display_latest_proof_summary(run_root: Path) -> str:
    """Render the latest proof artifact summary for CLI display."""
    layout = resolve_run_layout(run_root)
    analysis_dir = layout["run_root"] / "03_Analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    index_path = analysis_dir / _PROOF_INDEX_NAME
    if index_path.exists():
        try:
            index_payload = _read_json(index_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Proof index unreadable at %s: %s; rebuilding.", index_path, exc)
            index = maintain_proof_artifact_index(layout["run_root"])
        else:
            index = ProofArtifactIndex(
                run_root=str(index_payload.get("run_root", "")),
                proof_artifacts=list(index_payload.get("proof_artifacts", [])),
                latest_proof=index_payload.get("latest_proof"),
                total_proofs=int(index_payload.get("total_proofs", 0)),
            )
    else:
        index = maintain_proof_artifact_index(layout["run_root"])

    if index.total_proofs == 0 or index.latest_proof is None:
        return "No network-gate proof artifacts are available for this run root yet."

    latest = index.latest_proof
    lines = [
        "Latest network-gate proof:",
        f"  execution: {latest.get('execution_timestamp', 'unknown')}",
        "  parity: "
        f"vertices={latest.get('vertices_exact_parity')} "
        f"edges={latest.get('edges_exact_parity')} "
        f"strands={latest.get('strands_exact_parity')}",
        f"  json: {latest.get('artifact_json_path', '')}",
        "",
        "Known proof artifacts:",
    ]
    for entry in index.proof_artifacts:
        lines.append(
            "  - "
            f"{entry.get('execution_timestamp', 'unknown')} | "
            f"overall={entry.get('overall_parity_achieved')} | "
            f"{entry.get('artifact_json_path', '')}"
        )
    return "\n".join(lines)
