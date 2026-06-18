"""Oracle artifact readiness helpers for exact-route parity runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np

from slavv_python.analytics.parity.constants import NORMALIZED_DIR
from slavv_python.analytics.parity.exact_proof_contract import EXACT_STAGE_ORDER
from slavv_python.analytics.parity.matlab_vector_loader import (
    find_single_matlab_batch_dir,
    load_normalized_matlab_vectors,
)
from slavv_python.analytics.parity.utils import persist_normalized_payloads
from slavv_python.engine.state import fingerprint_file, load_json_dict

from .constants import ORACLE_MANIFEST_PATH
from .utils import now_iso, write_json_with_hash

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class OracleArtifactStatus:
    """Readiness status for one normalized Oracle stage Artifact."""

    stage: str
    path: Path
    exists: bool
    readable: bool
    repaired: bool
    sha256: str | None
    summary: dict[str, Any]
    error: str | None = None

    @property
    def ready(self) -> bool:
        """Return True when the normalized Artifact exists and can be loaded."""
        return self.exists and self.readable

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready status payload."""
        return {
            "stage": self.stage,
            "path": str(self.path),
            "exists": self.exists,
            "readable": self.readable,
            "ready": self.ready,
            "repaired": self.repaired,
            "sha256": self.sha256,
            "summary": self.summary,
            "error": self.error,
        }


def ensure_oracle_artifacts(
    oracle_root: Path,
    *,
    stages: tuple[str, ...] = EXACT_STAGE_ORDER,
    matlab_batch_dir: Path | None = None,
    repair: bool = True,
) -> dict[str, OracleArtifactStatus]:
    """Verify and optionally materialize normalized Oracle stage Artifacts."""
    resolved_root = oracle_root.expanduser().resolve()
    requested_stages = _normalize_stages(stages)
    missing = [
        stage for stage in requested_stages if not _artifact_path(resolved_root, stage).is_file()
    ]
    repaired_stages: set[str] = set()

    if repair and missing:
        raw_batch_dir = (
            matlab_batch_dir.expanduser().resolve()
            if matlab_batch_dir is not None
            else find_single_matlab_batch_dir(resolved_root)
        )
        payloads = load_normalized_matlab_vectors(raw_batch_dir, tuple(missing))
        persist_normalized_payloads(resolved_root, group_name="oracle", payloads=payloads)
        repaired_stages.update(missing)

    statuses = {
        stage: inspect_oracle_artifact(
            resolved_root,
            stage,
            repaired=stage in repaired_stages,
        )
        for stage in requested_stages
    }
    if repair:
        _sync_manifest_normalized_artifacts(resolved_root, statuses)
    return statuses


def inspect_oracle_artifact(
    oracle_root: Path,
    stage: str,
    *,
    repaired: bool = False,
) -> OracleArtifactStatus:
    """Load one normalized Oracle Artifact and return its readiness status."""
    resolved_root = oracle_root.expanduser().resolve()
    normalized_stage = _normalize_stages((stage,))[0]
    path = _artifact_path(resolved_root, normalized_stage)
    if not path.is_file():
        return OracleArtifactStatus(
            stage=normalized_stage,
            path=path,
            exists=False,
            readable=False,
            repaired=False,
            sha256=None,
            summary={},
            error="missing normalized Oracle Artifact",
        )

    try:
        payload = joblib.load(path)
    except Exception as exc:  # pragma: no cover - defensive report path
        return OracleArtifactStatus(
            stage=normalized_stage,
            path=path,
            exists=True,
            readable=False,
            repaired=repaired,
            sha256=_sidecar_text(path),
            summary={},
            error=f"{type(exc).__name__}: {exc}",
        )

    return OracleArtifactStatus(
        stage=normalized_stage,
        path=path,
        exists=True,
        readable=True,
        repaired=repaired,
        sha256=_sidecar_text(path) or fingerprint_file(path),
        summary=_payload_summary(payload),
    )


def _artifact_path(oracle_root: Path, stage: str) -> Path:
    return oracle_root / NORMALIZED_DIR / "oracle" / f"{stage}.pkl"


def _sync_manifest_normalized_artifacts(
    oracle_root: Path,
    statuses: dict[str, OracleArtifactStatus],
) -> None:
    manifest_path = oracle_root / ORACLE_MANIFEST_PATH
    manifest = load_json_dict(manifest_path)
    if manifest is None:
        return

    normalized_artifacts = manifest.setdefault("normalized_artifacts", {})
    if not isinstance(normalized_artifacts, dict):
        normalized_artifacts = {}
        manifest["normalized_artifacts"] = normalized_artifacts

    changed = False
    for stage, status in statuses.items():
        if not status.ready:
            continue
        path_text = str(status.path)
        if normalized_artifacts.get(stage) != path_text:
            normalized_artifacts[stage] = path_text
            changed = True

    if not changed:
        return

    timestamps = manifest.setdefault("timestamps", {})
    if isinstance(timestamps, dict):
        timestamps["updated_at"] = now_iso()
    else:
        manifest["timestamps"] = {"updated_at": now_iso()}
    write_json_with_hash(manifest_path, manifest)


def _normalize_stages(stages: tuple[str, ...]) -> tuple[str, ...]:
    if stages == ("all",):
        return EXACT_STAGE_ORDER
    if "all" in stages:
        raise ValueError("'all' cannot be combined with explicit Oracle Artifact stages")
    unknown = [stage for stage in stages if stage not in EXACT_STAGE_ORDER]
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unsupported Oracle Artifact stage(s): {joined}")
    return stages


def _sidecar_text(path: Path) -> str | None:
    sidecar = path.with_name(f"{path.name}.sha256")
    if not sidecar.is_file():
        return None
    return sidecar.read_text(encoding="utf-8").strip() or None


def _payload_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): _value_summary(item) for key, item in value.items()}
    return {"payload": _value_summary(value)}


def _value_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, np.ndarray):
        return {"kind": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {"kind": "dict", "keys": sorted(str(key) for key in value)}
    if isinstance(value, (list, tuple)):
        return {"kind": type(value).__name__, "length": len(value)}
    return {"kind": type(value).__name__}


__all__ = [
    "OracleArtifactStatus",
    "ensure_oracle_artifacts",
    "inspect_oracle_artifact",
]
