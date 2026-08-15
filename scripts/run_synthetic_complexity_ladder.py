"""Synthetic complexity ladder until first MATLAB↔Python divergence.

Fixed four-rung fake-volume ladder reusing the tiny Y-junction dual-run pattern.
Results are informative only — NOT Certification / NOT Phase 1.

Usage (repo root):
  .\\.venv\\Scripts\\python.exe scripts\\run_synthetic_complexity_ladder.py
  .\\.venv\\Scripts\\python.exe scripts\\run_synthetic_complexity_ladder.py --rung y_junction_32
  .\\.venv\\Scripts\\python.exe scripts\\run_synthetic_complexity_ladder.py --skip-matlab --reuse-python
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy.io as sio
import tifffile

from slavv_python.analytics.parity.probes.synthetic_dual_run_compare import (
    first_break_surface,
    strict_compare_summary,
)
from slavv_python.analytics.parity.probes.synthetic_ladder_report import (
    DEFAULT_SOFT_SIZE_MAX_DIM,
    DEFAULT_SOFT_TIME_SEC,
    NON_CERTIFICATION_NOTE,
    assemble_ladder_report,
    soft_cap_blocks_next_rung,
)
from slavv_python.engine import SlavvPipeline
from slavv_python.storage import load_tiff_volume
from slavv_python.utils.synthetic import (
    LADDER_RUNG_IDS,
    LADDER_RUNG_MAX_DIM,
    generate_ladder_rung_volume,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = REPO_ROOT / "workspace" / "experiments" / "synthetic_complexity_ladder"
MATLAB_DRIVER = Path(__file__).resolve().parent / "vectorize_ladder_rung.m"
REPORT_PATH = EXP_DIR / "ladder_report.json"

SEED = 20260814
BG_UINT16 = 40
VESSEL_UINT16 = 2200

SHARED_PARAMS: dict[str, Any] = {
    "pipeline_profile": "matlab_compat",
    "comparison_exact_network": True,
    "comparison_exact_network_use_conflict_painting": False,
    "microns_per_voxel": [1.0, 1.0, 1.0],
    "radius_of_smallest_vessel_in_microns": 1.5,
    "radius_of_largest_vessel_in_microns": 5.0,
    "approximating_PSF": False,
    "scales_per_octave": 1.0,
    "gaussian_to_ideal_ratio": 1.0,
    "spherical_to_annular_ratio": 1.0,
    "max_voxels_per_node_energy": 100000,
    "space_strel_apothem": 1,
    "energy_upper_bound": 0.0,
    "max_voxels_per_node": 6000,
    "length_dilation_ratio": 1.0,
    "max_edge_length_per_origin_radius": 30.0,
    "space_strel_apothem_edges": 1,
    "number_of_edges_per_vertex": 4,
    "energy_method": "hessian",
    "energy_projection_mode": "matlab",
    "direction_method": "hessian",
    "discrete_tracing": False,
    "edge_method": "tracing",
    "n_jobs": 1,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rung_dirs(rung_id: str) -> dict[str, Path]:
    root = EXP_DIR / rung_id
    return {
        "root": root,
        "input": root / "input",
        "matlab_batches": root / "matlab_batches",
        "python_run": root / "python_run",
        "tiff": root / "input" / f"{rung_id}.tif",
        "matlab_stdout": root / "matlab_stdout.txt",
        "matlab_stderr": root / "matlab_stderr.txt",
    }


def resolve_matlab_exe() -> Path | None:
    env = os.environ.get("MATLAB_EXE")
    if env and Path(env).is_file():
        return Path(env)
    which = shutil.which("matlab.exe") or shutil.which("matlab")
    if which and Path(which).is_file():
        return Path(which)
    default = Path(r"C:\Program Files\MATLAB\R2019a\bin\matlab.exe")
    if default.is_file():
        return default
    return None


def write_rung_tiff(rung_id: str) -> dict[str, Any]:
    dirs = _rung_dirs(rung_id)
    dirs["input"].mkdir(parents=True, exist_ok=True)
    rung_salt = int(hashlib.sha256(rung_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    rng = np.random.default_rng(SEED + rung_salt)
    volume = generate_ladder_rung_volume(rung_id)
    noise = rng.normal(0.0, 0.02, size=volume.shape).astype(np.float32)
    volume = np.clip(volume + noise, 0.0, 1.0)
    uint16 = (BG_UINT16 + volume * (VESSEL_UINT16 - BG_UINT16)).astype(np.uint16)
    tifffile.imwrite(str(dirs["tiff"]), uint16)
    return {
        "path": str(dirs["tiff"]),
        "rung_id": rung_id,
        "shape_zyx": list(volume.shape),
        "max_dim": LADDER_RUNG_MAX_DIM[rung_id],
        "dtype": "uint16",
        "seed": SEED,
        "n_vessel_voxels": int((volume > 0.5).sum()),
    }


def run_matlab_for_rung(rung_id: str, *, skip: bool = False) -> dict[str, Any]:
    dirs = _rung_dirs(rung_id)
    dirs["matlab_batches"].mkdir(parents=True, exist_ok=True)
    if skip:
        batches = sorted(dirs["matlab_batches"].glob("batch_*"), key=lambda p: p.stat().st_mtime)
        latest = batches[-1] if batches else None
        return {
            "available": True,
            "ok": latest is not None,
            "batch_dir": str(latest) if latest else None,
            "timing": {"wall_sec": None, "reused": True},
            "reused": True,
        }

    matlab = resolve_matlab_exe()
    if matlab is None:
        return {"available": False, "ok": False, "error": "MATLAB executable not found"}

    if not MATLAB_DRIVER.is_file():
        return {
            "available": True,
            "ok": False,
            "error": f"MATLAB driver missing: {MATLAB_DRIVER}",
        }

    tif = str(dirs["tiff"]).replace("\\", "/")
    out = str(dirs["matlab_batches"]).replace("\\", "/")
    scripts_dir = str(MATLAB_DRIVER.parent).replace("\\", "/")
    # Function file: addpath then call (run() is for scripts only).
    batch_cmd = f"addpath('{scripts_dir}'); vectorize_ladder_rung('{tif}','{out}')"
    cmd = [str(matlab), "-batch", batch_cmd]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wall = time.perf_counter() - t0
    dirs["matlab_stdout"].write_text(proc.stdout or "", encoding="utf-8")
    dirs["matlab_stderr"].write_text(proc.stderr or "", encoding="utf-8")
    batches = sorted(dirs["matlab_batches"].glob("batch_*"), key=lambda p: p.stat().st_mtime)
    latest = batches[-1] if batches else None
    return {
        "available": True,
        "ok": proc.returncode == 0 and latest is not None,
        "batch_dir": str(latest) if latest else None,
        "timing": {"wall_sec": wall, "returncode": proc.returncode},
        "stdout_tail": (proc.stdout or "")[-2000:],
        "stderr_tail": (proc.stderr or "")[-2000:],
    }


def _as_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        return dict(obj.to_dict())
    raise TypeError(f"Cannot convert {type(obj)!r} to mapping")


def _load_python_artifacts(run_dir: Path) -> dict[str, Any]:
    ckpt = run_dir / "02_Output" / "python_results" / "checkpoints"
    vertices = joblib.load(ckpt / "checkpoint_vertices.pkl")
    edges = joblib.load(ckpt / "checkpoint_edges.pkl")
    network = joblib.load(ckpt / "checkpoint_network.pkl")
    positions = np.asarray(vertices["positions"], dtype=np.float64)
    connections = np.asarray(edges["connections"], dtype=np.int64)
    strands = network.get("strands") if isinstance(network, dict) else None
    return {
        "ok": True,
        "positions": positions,
        "connections": connections,
        "n_vertices": int(positions.reshape(-1, 3).shape[0]),
        "n_edges": int(connections.reshape(-1, 2).shape[0]),
        "n_strands": len(strands) if strands is not None else None,
    }


def run_python_for_rung(rung_id: str, *, reuse: bool = False) -> dict[str, Any]:
    dirs = _rung_dirs(rung_id)
    run_dir = dirs["python_run"] / "exact_run"
    if reuse and (run_dir / "02_Output" / "python_results" / "checkpoints").exists():
        arts = _load_python_artifacts(run_dir)
        return {
            **arts,
            "reused": True,
            "run_dir": str(run_dir),
            "timing": {"wall_sec": None, "note": "reused prior run"},
        }

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    image = load_tiff_volume(str(dirs["tiff"]))
    t0 = time.perf_counter()
    results = SlavvPipeline().run(
        image,
        dict(SHARED_PARAMS),
        run_dir=str(run_dir),
        stop_after="network",
    )
    wall = time.perf_counter() - t0
    try:
        vertices = _as_mapping(results["vertices"])
        edges = _as_mapping(results["edges"])
        network = _as_mapping(results["network"])
        positions = np.asarray(vertices["positions"], dtype=np.float64)
        connections = np.asarray(edges["connections"], dtype=np.int64)
        strands = network.get("strands")
        arts = {
            "ok": True,
            "positions": positions,
            "connections": connections,
            "n_vertices": int(positions.reshape(-1, 3).shape[0]),
            "n_edges": int(connections.reshape(-1, 2).shape[0]),
            "n_strands": len(strands) if strands is not None else None,
        }
    except Exception:
        arts = _load_python_artifacts(run_dir)
    return {
        **arts,
        "reused": False,
        "run_dir": str(run_dir),
        "timing": {"wall_sec": wall},
        "image_shape_yxz": list(image.shape),
    }


def _latest_mat(batch_dir: Path, prefix: str) -> Path | None:
    hits = sorted((batch_dir / "vectors").glob(f"{prefix}_*.mat"))
    return hits[-1] if hits else None


def load_matlab_artifacts(batch_dir: Path) -> dict[str, Any]:
    edges_mat = _latest_mat(batch_dir, "edges")
    curated_edges = _latest_mat(batch_dir, "curated_edges")
    network_mat = _latest_mat(batch_dir, "network")
    vertices_mat = _latest_mat(batch_dir, "vertices")
    curated_verts = _latest_mat(batch_dir, "curated_vertices")
    vert_path = curated_verts or vertices_mat
    edge_path = curated_edges or edges_mat
    if vert_path is None or edge_path is None:
        return {"ok": False, "error": "missing vectors mats"}

    v = sio.loadmat(str(vert_path), squeeze_me=True, struct_as_record=False)
    e = sio.loadmat(str(edge_path), squeeze_me=True, struct_as_record=False)
    positions = np.asarray(v["vertex_space_subscripts"], dtype=np.float64)
    connections = np.asarray(e["edges2vertices"], dtype=np.int64)
    out: dict[str, Any] = {
        "ok": True,
        "positions": positions,
        "connections": connections,
        "n_vertices": int(positions.reshape(-1, 3).shape[0]),
        "n_edges": int(connections.reshape(-1, 2).shape[0]),
        "n_strands": None,
    }
    if network_mat is not None:
        n = sio.loadmat(str(network_mat), squeeze_me=True, struct_as_record=False)
        strands = n.get("strands2vertices")
        if strands is None:
            out["n_strands"] = None
        elif isinstance(strands, np.ndarray) and strands.dtype == object:
            out["n_strands"] = int(strands.size)
        elif isinstance(strands, (list, tuple)):
            out["n_strands"] = len(strands)
        else:
            out["n_strands"] = 1
    return out


def _safe_side(d: dict[str, Any]) -> dict[str, Any]:
    skip = {"positions", "connections", "candidate_connections"}
    return {k: v for k, v in d.items() if k not in skip}


def run_one_rung(
    rung_id: str,
    *,
    skip_matlab: bool,
    reuse_python: bool,
) -> dict[str, Any]:
    tiff_meta = write_rung_tiff(rung_id)
    matlab_run = run_matlab_for_rung(rung_id, skip=skip_matlab)
    try:
        python_run = run_python_for_rung(rung_id, reuse=reuse_python)
    except Exception as exc:
        python_run = {"ok": False, "error": repr(exc), "timing": {"wall_sec": None}}

    matlab_art: dict[str, Any] = {"ok": False}
    if matlab_run.get("ok") and matlab_run.get("batch_dir"):
        matlab_art = load_matlab_artifacts(Path(matlab_run["batch_dir"]))

    if not matlab_run.get("available"):
        status = "inconclusive"
        surface = None
        compare: dict[str, Any] = {
            "comparable": False,
            "reason": matlab_run.get("error", "MATLAB unavailable"),
        }
    elif not matlab_art.get("ok") or not python_run.get("ok"):
        status = "failed" if python_run.get("ok") is False else "inconclusive"
        if not matlab_art.get("ok") and not python_run.get("ok"):
            status = "failed"
        surface = None
        compare = {"comparable": False, "reason": "one side failed or non-comparable"}
    else:
        compare = strict_compare_summary(matlab_art, python_run)
        try:
            surface = first_break_surface(matlab_art, python_run)
        except Exception as exc:
            status = "inconclusive"
            surface = None
            compare = {"comparable": False, "reason": str(exc)}
        else:
            status = "match" if surface is None else "first_break"

    matlab_wall = (matlab_run.get("timing") or {}).get("wall_sec")
    python_wall = (python_run.get("timing") or {}).get("wall_sec")
    return {
        "rung_id": rung_id,
        "executed": True,
        "status": status,
        "first_break_surface": surface,
        "tiff": tiff_meta,
        "matlab_wall_sec": matlab_wall,
        "python_wall_sec": python_wall,
        "matlab_run": _safe_side(matlab_run),
        "matlab_artifacts": _safe_side(matlab_art),
        "python_run": _safe_side(python_run),
        "compare": compare,
    }


def run_ladder(
    *,
    rung_ids: tuple[str, ...] | list[str],
    skip_matlab: bool,
    reuse_python: bool,
    soft_time_sec: float,
    soft_size_max_dim: int,
) -> dict[str, Any]:
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    executed: list[dict[str, Any]] = []
    ids = list(rung_ids)

    for idx, rung_id in enumerate(ids):
        if idx > 0:
            prior = executed[-1]
            block = soft_cap_blocks_next_rung(
                next_rung_id=rung_id,
                prior_matlab_wall_sec=prior.get("matlab_wall_sec"),
                prior_python_wall_sec=prior.get("python_wall_sec"),
                soft_time_sec=soft_time_sec,
                soft_size_max_dim=soft_size_max_dim,
            )
            if block is not None:
                prior["soft_cap_blocked"] = block
                return assemble_ladder_report(
                    rung_results=executed,
                    outcome="soft_cap_full_match",
                    soft_cap_reason=block,
                    created_utc=_utc_now(),
                    soft_time_sec=soft_time_sec,
                    soft_size_max_dim=soft_size_max_dim,
                )

        print(f"[{_utc_now()}] Ladder rung: {rung_id}")
        result = run_one_rung(rung_id, skip_matlab=skip_matlab, reuse_python=reuse_python)
        executed.append(result)
        print(
            f"  status={result['status']} surface={result.get('first_break_surface')} "
            f"matlab_wall={result.get('matlab_wall_sec')} "
            f"python_wall={result.get('python_wall_sec')}"
        )

        if result["status"] == "first_break":
            return assemble_ladder_report(
                rung_results=executed,
                outcome="first_break",
                first_break_rung=rung_id,
                first_break_surface=result.get("first_break_surface"),
                created_utc=_utc_now(),
                soft_time_sec=soft_time_sec,
                soft_size_max_dim=soft_size_max_dim,
            )
        if result["status"] in {"inconclusive", "failed"}:
            return assemble_ladder_report(
                rung_results=executed,
                outcome=result["status"],
                created_utc=_utc_now(),
                soft_time_sec=soft_time_sec,
                soft_size_max_dim=soft_size_max_dim,
            )

    if executed:
        executed[-1]["soft_cap_blocked"] = "end_of_ladder"
    return assemble_ladder_report(
        rung_results=executed,
        outcome="soft_cap_full_match",
        soft_cap_reason="end_of_ladder",
        created_utc=_utc_now(),
        soft_time_sec=soft_time_sec,
        soft_size_max_dim=soft_size_max_dim,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rung",
        choices=list(LADDER_RUNG_IDS),
        help="Run only this named rung (smoke / single-step).",
    )
    parser.add_argument(
        "--skip-matlab",
        action="store_true",
        help="Reuse latest per-rung matlab_batches/batch_* instead of launching MATLAB.",
    )
    parser.add_argument(
        "--reuse-python",
        action="store_true",
        help="Reuse existing per-rung python_run/exact_run checkpoints.",
    )
    parser.add_argument(
        "--soft-time-sec",
        type=float,
        default=DEFAULT_SOFT_TIME_SEC,
        help="Soft wall-clock budget per side before refusing the next rung (default 180).",
    )
    parser.add_argument(
        "--soft-size-max-dim",
        type=int,
        default=DEFAULT_SOFT_SIZE_MAX_DIM,
        help="Refuse starting a rung whose max dim exceeds this (default 64).",
    )
    args = parser.parse_args(argv)

    rung_ids: tuple[str, ...] = (args.rung,) if args.rung else LADDER_RUNG_IDS
    print(f"[{_utc_now()}] Synthetic complexity ladder")
    print(f"  note: {NON_CERTIFICATION_NOTE}")
    print(f"  rungs: {list(rung_ids)}")
    print(f"  MATLAB: {resolve_matlab_exe()}")

    # Mirror driver into the experiment dir for operator browsing (workspace is gitignored).
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    if MATLAB_DRIVER.is_file():
        shutil.copy2(MATLAB_DRIVER, EXP_DIR / MATLAB_DRIVER.name)

    report = run_ladder(
        rung_ids=rung_ids,
        skip_matlab=bool(args.skip_matlab),
        reuse_python=bool(args.reuse_python),
        soft_time_sec=float(args.soft_time_sec),
        soft_size_max_dim=int(args.soft_size_max_dim),
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nOUTCOME: {report['outcome']}")
    if report.get("first_break_rung"):
        print(
            f"  first_break_rung={report['first_break_rung']} "
            f"surface={report.get('first_break_surface')}"
        )
    if report.get("soft_cap_reason"):
        print(f"  soft_cap_reason={report['soft_cap_reason']}")
    print(f"Report: {REPORT_PATH}")
    return 0 if report["outcome"] in {"first_break", "soft_cap_full_match"} else 1


if __name__ == "__main__":
    sys.exit(main())
