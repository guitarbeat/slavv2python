"""Developer helpers for imported-MATLAB parity experiments."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copy2, copytree
from typing import Any, cast

import numpy as np
import psutil

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from source import SLAVVProcessor
from source.core._edges.bridge_vertices import add_vertices_to_edges_matlab_style
from source.core._edges.postprocess import finalize_edges_matlab_style
from source.core.edge_candidates import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from source.core.edge_selection import choose_edges_for_workflow
from source.core.vertices import paint_vertex_center_image
from source.io import load_tiff_volume
from source.io.matlab_exact_proof import (
    EXACT_STAGE_ORDER,
    compare_exact_artifacts,
    find_matlab_vector_paths,
    find_single_matlab_batch_dir,
    load_normalized_matlab_vectors,
    load_normalized_python_checkpoints,
    normalize_python_stage_payload,
    render_exact_proof_report,
    sync_exact_vertex_checkpoint_from_matlab,
)
from source.io.matlab_fail_fast import (
    build_candidate_coverage_report,
    build_candidate_snapshot_payload,
    compare_lut_fixture_payload,
    load_builtin_lut_fixture,
    render_candidate_coverage_report,
    render_lut_proof_report,
)
from source.runtime.run_state import (
    atomic_joblib_dump,
    atomic_write_json,
    atomic_write_text,
    load_json_dict,
)
from source.utils.safe_unpickle import safe_load

CHECKPOINTS_DIR = Path("02_Output") / "python_results" / "checkpoints"
COMPARISON_REPORT_PATH = Path("03_Analysis") / "comparison_report.json"
EDGE_CANDIDATE_CHECKPOINT_PATH = CHECKPOINTS_DIR / "checkpoint_edge_candidates.pkl"
EDGE_REPLAY_PROOF_JSON_PATH = Path("03_Analysis") / "edge_replay_proof.json"
EDGE_REPLAY_PROOF_TEXT_PATH = Path("03_Analysis") / "edge_replay_proof.txt"
EXACT_PROOF_JSON_PATH = Path("03_Analysis") / "exact_proof.json"
EXACT_PROOF_TEXT_PATH = Path("03_Analysis") / "exact_proof.txt"
LUT_PROOF_JSON_PATH = Path("03_Analysis") / "lut_proof.json"
LUT_PROOF_TEXT_PATH = Path("03_Analysis") / "lut_proof.txt"
PREFLIGHT_EXACT_JSON_PATH = Path("03_Analysis") / "preflight_exact.json"
PREFLIGHT_EXACT_TEXT_PATH = Path("03_Analysis") / "preflight_exact.txt"
RUN_SNAPSHOT_PATH = Path("99_Metadata") / "run_snapshot.json"
SUMMARY_JSON_PATH = Path("03_Analysis") / "experiment_summary.json"
SUMMARY_TEXT_PATH = Path("03_Analysis") / "experiment_summary.txt"
VALIDATED_PARAMS_PATH = Path("99_Metadata") / "validated_params.json"
CANDIDATE_COVERAGE_JSON_PATH = Path("03_Analysis") / "candidate_coverage.json"
CANDIDATE_COVERAGE_TEXT_PATH = Path("03_Analysis") / "candidate_coverage.txt"
HEARTBEAT_INTERVAL_ITERATIONS = 512
DEFAULT_MEMORY_SAFETY_FRACTION = 0.8
EXACT_ROUTE_ARRAY_BYTES_PER_VOXEL: tuple[tuple[str, int], ...] = (
    ("energy", 4),
    ("scale_indices", 2),
    ("vertex_center_image", 4),
    ("energy_map_temp", 4),
    ("energy_map", 4),
    ("branch_order_map", 1),
    ("d_over_r_map", 4),
    ("pointer_map", 4),
    ("vertex_index_map", 4),
    ("size_map", 2),
)


@dataclass(frozen=True)
class RunCounts:
    vertices: int
    edges: int
    strands: int


@dataclass(frozen=True)
class SourceRunSurface:
    run_root: Path
    checkpoints_dir: Path
    comparison_report_path: Path
    validated_params_path: Path
    run_snapshot_path: Path | None


@dataclass(frozen=True)
class ExactProofSourceSurface:
    run_root: Path
    checkpoints_dir: Path
    validated_params_path: Path
    matlab_batch_dir: Path
    matlab_vector_paths: dict[str, Path]


@dataclass(frozen=True)
class ExactPreflightSurface:
    source_surface: ExactProofSourceSurface
    dest_run_root: Path
    image_shape: tuple[int, int, int]


def build_parser() -> argparse.ArgumentParser:
    """Build the developer parity experiment parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Developer helpers for rerunning and proving imported-MATLAB exact-route parity "
            "against an existing staged comparison run."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    rerun = subparsers.add_parser(
        "rerun-python",
        help="Copy a reusable Python checkpoint surface into a fresh run root and rerun from edges or network.",
    )
    rerun.add_argument(
        "--source-run-root",
        required=True,
        help="Existing staged comparison run root that still retains python checkpoints.",
    )
    rerun.add_argument(
        "--dest-run-root",
        required=True,
        help="Fresh destination run root for the current-code experiment.",
    )
    rerun.add_argument(
        "--input",
        help=(
            "Override the input volume path. If omitted, the script resolves the path from "
            "99_Metadata/run_snapshot.json provenance."
        ),
    )
    rerun.add_argument(
        "--rerun-from",
        choices=("edges", "network"),
        default="edges",
        help="Pipeline stage to recompute after copying reusable checkpoints.",
    )
    rerun.add_argument(
        "--params-file",
        help=(
            "Optional JSON parameters file. Defaults to 99_Metadata/validated_params.json "
            "from the source run root."
        ),
    )
    rerun.set_defaults(handler=_handle_rerun_python)

    summarize = subparsers.add_parser(
        "summarize",
        help="Print the saved experiment summary for a destination run root.",
    )
    summarize.add_argument(
        "--run-root",
        required=True,
        help="Run root containing 03_Analysis/experiment_summary.{txt,json}.",
    )
    summarize.set_defaults(handler=_handle_summarize)

    prove = subparsers.add_parser(
        "prove-exact",
        help="Compare rerun Python checkpoints against preserved raw MATLAB vectors on the exact imported-MATLAB route.",
    )
    prove.add_argument(
        "--source-run-root",
        required=True,
        help="Existing staged comparison run root containing raw MATLAB vectors and exact-route provenance.",
    )
    prove.add_argument(
        "--dest-run-root",
        required=True,
        help="Destination run root containing the current-code rerun Python checkpoints to prove.",
    )
    prove.add_argument(
        "--stage",
        choices=(*EXACT_STAGE_ORDER, "all"),
        default="all",
        help="Proof surface to compare. Defaults to all exact-route stages.",
    )
    prove.add_argument(
        "--report-path",
        help=(
            "Optional proof report override. Defaults to 03_Analysis/exact_proof.{json,txt} "
            "under the destination run root."
        ),
    )
    prove.set_defaults(handler=_handle_prove_exact)

    preflight = subparsers.add_parser(
        "preflight-exact",
        help="Validate exact-route source/destination surfaces, memory budget, and process collisions.",
    )
    preflight.add_argument("--source-run-root", required=True)
    preflight.add_argument("--dest-run-root", required=True)
    preflight.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=DEFAULT_MEMORY_SAFETY_FRACTION,
        help="Refuse the run if projected exact-route memory exceeds this fraction of available RAM.",
    )
    preflight.add_argument(
        "--force",
        action="store_true",
        help="Ignore destination-root process collisions during preflight.",
    )
    preflight.set_defaults(handler=_handle_preflight_exact)

    prove_luts = subparsers.add_parser(
        "prove-luts",
        help="Compare the shared Python watershed LUT builder against the checked-in fixture surface.",
    )
    prove_luts.add_argument("--source-run-root", required=True)
    prove_luts.add_argument("--dest-run-root", required=True)
    prove_luts.set_defaults(handler=_handle_prove_luts)

    capture = subparsers.add_parser(
        "capture-candidates",
        help="Run only exact candidate generation and write a slim candidate checkpoint plus coverage report.",
    )
    capture.add_argument("--source-run-root", required=True)
    capture.add_argument("--dest-run-root", required=True)
    capture.add_argument(
        "--debug-maps",
        action="store_true",
        help="Include full-volume debug maps in checkpoint_edge_candidates.pkl.",
    )
    capture.set_defaults(handler=_handle_capture_candidates)

    replay = subparsers.add_parser(
        "replay-edges",
        help="Replay exact edge choosing from a saved candidate snapshot without regenerating candidates.",
    )
    replay.add_argument("--source-run-root", required=True)
    replay.add_argument("--dest-run-root", required=True)
    replay.set_defaults(handler=_handle_replay_edges)

    fail_fast = subparsers.add_parser(
        "fail-fast",
        help="Run the fail-fast exact parity funnel and stop at the first failing gate.",
    )
    fail_fast.add_argument("--source-run-root", required=True)
    fail_fast.add_argument("--dest-run-root", required=True)
    fail_fast.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=DEFAULT_MEMORY_SAFETY_FRACTION,
    )
    fail_fast.add_argument("--debug-maps", action="store_true")
    fail_fast.add_argument("--force", action="store_true")
    fail_fast.set_defaults(handler=_handle_fail_fast)
    return parser


def validate_source_run_surface(source_run_root: Path) -> SourceRunSurface:
    """Validate the reusable staged source surface for a Python rerun."""
    run_root = source_run_root.resolve()
    checkpoints_dir = run_root / CHECKPOINTS_DIR
    comparison_report_path = run_root / COMPARISON_REPORT_PATH
    validated_params_path = run_root / VALIDATED_PARAMS_PATH
    run_snapshot_path = run_root / RUN_SNAPSHOT_PATH

    missing: list[Path] = []
    if not checkpoints_dir.is_dir():
        missing.append(checkpoints_dir)
    if not comparison_report_path.is_file():
        missing.append(comparison_report_path)
    if not validated_params_path.is_file():
        missing.append(validated_params_path)
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"source run root is missing required artifacts: {joined}")

    return SourceRunSurface(
        run_root=run_root,
        checkpoints_dir=checkpoints_dir,
        comparison_report_path=comparison_report_path,
        validated_params_path=validated_params_path,
        run_snapshot_path=run_snapshot_path if run_snapshot_path.is_file() else None,
    )


def validate_exact_proof_source_surface(source_run_root: Path) -> ExactProofSourceSurface:
    """Validate the staged source surface for full-artifact exact proof."""
    run_root = source_run_root.resolve()
    checkpoints_dir = run_root / CHECKPOINTS_DIR
    validated_params_path = run_root / VALIDATED_PARAMS_PATH
    energy_checkpoint_path = checkpoints_dir / "checkpoint_energy.pkl"

    missing: list[Path] = []
    if not checkpoints_dir.is_dir():
        missing.append(checkpoints_dir)
    if not validated_params_path.is_file():
        missing.append(validated_params_path)
    if not energy_checkpoint_path.is_file():
        missing.append(energy_checkpoint_path)
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ValueError(f"source run root is missing required exact-proof artifacts: {joined}")

    validated_params = load_json_dict(validated_params_path)
    if validated_params is None:
        raise ValueError(f"expected JSON object in params file: {validated_params_path}")
    if not bool(validated_params.get("comparison_exact_network")):
        raise ValueError(
            f"source run root does not enable comparison_exact_network in {validated_params_path}"
        )

    energy_payload = _expect_mapping(
        safe_load(energy_checkpoint_path),
        str(energy_checkpoint_path),
    )
    energy_origin = energy_payload.get("energy_origin", energy_payload.get("energy_source"))
    if energy_origin != "matlab_batch_hdf5":
        raise ValueError(
            f"source energy provenance must be matlab_batch_hdf5, found: {energy_origin!r}"
        )

    matlab_batch_dir = find_single_matlab_batch_dir(run_root)
    matlab_vector_paths = find_matlab_vector_paths(matlab_batch_dir)
    return ExactProofSourceSurface(
        run_root=run_root,
        checkpoints_dir=checkpoints_dir,
        validated_params_path=validated_params_path,
        matlab_batch_dir=matlab_batch_dir,
        matlab_vector_paths=matlab_vector_paths,
    )


def validate_exact_preflight_surface(
    source_run_root: Path,
    dest_run_root: Path,
) -> ExactPreflightSurface:
    """Validate the source exact surface and load the image shape needed for memory preflight."""
    source_surface = validate_exact_proof_source_surface(source_run_root)
    dest_root = dest_run_root.expanduser().resolve()
    if dest_root.exists() and not dest_root.is_dir():
        raise ValueError(f"destination run root must be a directory path: {dest_root}")

    energy_payload = _load_exact_energy_payload(source_surface)
    energy = np.asarray(energy_payload.get("energy"))
    if energy.ndim != 3:
        raise ValueError(
            f"expected 3D energy volume in {source_surface.checkpoints_dir / 'checkpoint_energy.pkl'}"
        )
    return ExactPreflightSurface(
        source_surface=source_surface,
        dest_run_root=dest_root,
        image_shape=cast("tuple[int, int, int]", tuple(int(value) for value in energy.shape)),
    )


def build_exact_preflight_report(
    source_run_root: Path,
    dest_run_root: Path,
    *,
    memory_safety_fraction: float,
    force: bool,
) -> dict[str, Any]:
    """Build the fail-fast preflight report for an exact imported-MATLAB run."""
    surface = validate_exact_preflight_surface(source_run_root, dest_run_root)
    memory_estimate = estimate_exact_route_memory(surface.image_shape)
    available_memory_bytes = int(psutil.virtual_memory().available)
    safety_fraction = float(max(min(memory_safety_fraction, 1.0), 0.05))
    allowed_memory_bytes = int(available_memory_bytes * safety_fraction)
    collisions = find_parity_process_collisions(surface.dest_run_root)
    passed = memory_estimate["estimated_required_bytes"] <= allowed_memory_bytes and (
        force or not collisions
    )
    return {
        "passed": passed,
        "source_run_root": str(surface.source_surface.run_root),
        "dest_run_root": str(surface.dest_run_root),
        "report_scope": "exact imported-MATLAB preflight only",
        "image_shape": list(surface.image_shape),
        "memory_estimate": memory_estimate,
        "available_memory_bytes": available_memory_bytes,
        "allowed_memory_bytes": allowed_memory_bytes,
        "memory_safety_fraction": safety_fraction,
        "collision_count": len(collisions),
        "collisions": collisions,
        "force": bool(force),
    }


def render_exact_preflight_report(report_payload: dict[str, Any]) -> str:
    """Render a compact exact-route preflight report."""
    memory_estimate = _mapping_item(report_payload, "memory_estimate")
    lines = [
        "Exact preflight report",
        f"Status: {'PASS' if report_payload.get('passed') else 'FAIL'}",
        f"Source run root: {report_payload.get('source_run_root')}",
        f"Destination run root: {report_payload.get('dest_run_root')}",
        f"Image shape: {report_payload.get('image_shape')}",
        (
            "Memory: "
            f"estimated={memory_estimate.get('estimated_required_bytes')} "
            f"allowed={report_payload.get('allowed_memory_bytes')} "
            f"available={report_payload.get('available_memory_bytes')}"
        ),
        f"Collision count: {report_payload.get('collision_count', 0)}",
    ]
    collisions = report_payload.get("collisions", [])
    if collisions:
        lines.append(f"Collisions: {collisions}")
    return "\n".join(lines)


def estimate_exact_route_memory(image_shape: tuple[int, int, int]) -> dict[str, Any]:
    """Estimate the peak exact-route memory footprint from the planned full-volume arrays."""
    voxel_count = int(np.prod(np.asarray(image_shape, dtype=np.int64)))
    planned_arrays: list[dict[str, int | str]] = []
    subtotal_bytes = 0
    for name, bytes_per_voxel in EXACT_ROUTE_ARRAY_BYTES_PER_VOXEL:
        estimated_bytes = int(voxel_count * bytes_per_voxel)
        planned_arrays.append(
            {
                "name": name,
                "bytes_per_voxel": bytes_per_voxel,
                "estimated_bytes": estimated_bytes,
            }
        )
        subtotal_bytes += estimated_bytes
    overhead_bytes = round(subtotal_bytes * 0.25)
    return {
        "voxel_count": voxel_count,
        "planned_arrays": planned_arrays,
        "subtotal_bytes": subtotal_bytes,
        "overhead_bytes": overhead_bytes,
        "estimated_required_bytes": subtotal_bytes + overhead_bytes,
    }


def find_parity_process_collisions(dest_run_root: Path) -> list[dict[str, Any]]:
    """Return live parity processes already targeting the same destination run root."""
    current_pid = int(psutil.Process().pid)
    collisions: list[dict[str, Any]] = []
    normalized_dest = str(dest_run_root.resolve()).lower()
    owner_commands = {"rerun-python", "capture-candidates", "replay-edges", "fail-fast"}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        info = process.info
        pid = int(info.get("pid", -1))
        if pid == current_pid:
            continue
        cmdline = info.get("cmdline") or []
        if not isinstance(cmdline, list):
            continue
        joined = " ".join(str(part) for part in cmdline).lower()
        if "parity_experiment.py" not in joined:
            continue
        if normalized_dest not in joined:
            continue
        if not any(f" {command} " in f" {joined} " for command in owner_commands):
            continue
        collisions.append(
            {
                "pid": pid,
                "name": str(info.get("name", "")),
                "cmdline": [str(part) for part in cmdline],
            }
        )
    return collisions


def resolve_input_file(
    source_surface: SourceRunSurface,
    input_arg: str | None,
    *,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Resolve the input file either from the CLI or the source run snapshot provenance."""
    if input_arg:
        candidate = Path(input_arg).expanduser()
    else:
        if source_surface.run_snapshot_path is None:
            raise ValueError(
                "source run root does not contain 99_Metadata/run_snapshot.json and no --input was provided"
            )
        snapshot_payload = load_json_dict(source_surface.run_snapshot_path)
        provenance = (
            snapshot_payload.get("provenance", {}) if isinstance(snapshot_payload, dict) else {}
        )
        raw_input = provenance.get("input_file") if isinstance(provenance, dict) else None
        if not isinstance(raw_input, str) or not raw_input.strip():
            raise ValueError(
                "source run snapshot does not record provenance.input_file; pass --input explicitly"
            )
        candidate = Path(raw_input)
        if not candidate.is_absolute():
            candidate = repo_root / candidate

    resolved = candidate.resolve()
    if not resolved.is_file():
        raise ValueError(f"input file not found: {resolved}")
    return resolved


def load_params_file(
    source_surface: SourceRunSurface,
    params_file: str | None,
) -> dict[str, Any]:
    """Load the JSON parameter payload for the rerun."""
    params_path = (
        Path(params_file).expanduser().resolve()
        if params_file is not None
        else source_surface.validated_params_path
    )
    payload = load_json_dict(params_path)
    if payload is None:
        raise ValueError(f"expected JSON object in params file: {params_path}")
    return cast("dict[str, Any]", payload)


def load_exact_params_file(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    """Load the exact-route validated params payload."""
    payload = load_json_dict(source_surface.validated_params_path)
    if payload is None:
        raise ValueError(
            f"expected JSON object in params file: {source_surface.validated_params_path}"
        )
    return cast("dict[str, Any]", payload)


def ensure_dest_run_layout(dest_run_root: Path) -> None:
    """Ensure the minimal staged directories used by the developer parity helpers."""
    (dest_run_root / CHECKPOINTS_DIR).mkdir(parents=True, exist_ok=True)
    (dest_run_root / "03_Analysis").mkdir(parents=True, exist_ok=True)
    (dest_run_root / "99_Metadata").mkdir(parents=True, exist_ok=True)


def _load_exact_energy_payload(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    checkpoint_path = source_surface.checkpoints_dir / "checkpoint_energy.pkl"
    return _expect_mapping(safe_load(checkpoint_path), str(checkpoint_path))


def _load_exact_vertices_payload(source_surface: ExactProofSourceSurface) -> dict[str, Any]:
    checkpoint_path = source_surface.checkpoints_dir / "checkpoint_vertices.pkl"
    payload = dict(_expect_mapping(safe_load(checkpoint_path), str(checkpoint_path)))
    normalized_vertices = load_normalized_matlab_vectors(
        source_surface.matlab_batch_dir,
        ("vertices",),
    )["vertices"]
    payload["positions"] = np.asarray(normalized_vertices["positions"], dtype=np.float32)
    payload["scales"] = np.asarray(normalized_vertices["scales"], dtype=np.int16)
    payload["energies"] = np.asarray(normalized_vertices["energies"], dtype=np.float32)
    payload["count"] = len(payload["positions"])
    return payload


def _lumen_radius_pixels_axes(
    energy_payload: dict[str, Any],
    params: dict[str, Any],
) -> np.ndarray:
    if "lumen_radius_pixels_axes" in energy_payload:
        return cast(
            "np.ndarray",
            np.asarray(energy_payload["lumen_radius_pixels_axes"], dtype=np.float32),
        )
    lumen_radius_microns = np.asarray(
        energy_payload["lumen_radius_microns"], dtype=np.float32
    ).reshape(-1, 1)
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]),
        dtype=np.float32,
    ).reshape(1, 3)
    return cast(
        "np.ndarray",
        lumen_radius_microns / np.maximum(microns_per_voxel, 1e-6),
    )


def extract_matlab_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract preserved MATLAB count truth from a comparison report."""
    matlab = _mapping_item(report_payload, "matlab")
    vertices = _coerce_int(
        matlab.get("vertices_count", _mapping_item(report_payload, "vertices").get("matlab_count")),
        label="matlab vertices count",
    )
    edges = _coerce_int(
        matlab.get("edges_count", _mapping_item(report_payload, "edges").get("matlab_count")),
        label="matlab edges count",
    )
    strands = _coerce_int(
        matlab.get(
            "strand_count", _mapping_item(report_payload, "network").get("matlab_strand_count")
        ),
        label="matlab strand count",
    )
    return RunCounts(vertices=vertices, edges=edges, strands=strands)


def extract_source_python_counts(report_payload: dict[str, Any]) -> RunCounts:
    """Extract the preserved source-run Python counts from a comparison report."""
    python_counts = _mapping_item(report_payload, "python")
    vertices = _coerce_int(
        python_counts.get(
            "vertices_count",
            _mapping_item(report_payload, "vertices").get("python_count"),
        ),
        label="source python vertices count",
    )
    edges = _coerce_int(
        python_counts.get(
            "edges_count", _mapping_item(report_payload, "edges").get("python_count")
        ),
        label="source python edges count",
    )
    strands = _coerce_int(
        python_counts.get(
            "network_strands_count",
            _mapping_item(report_payload, "network").get("python_strand_count"),
        ),
        label="source python strand count",
    )
    return RunCounts(vertices=vertices, edges=edges, strands=strands)


def read_python_counts_from_run(run_root: Path) -> RunCounts:
    """Read Python stage counts from the structured checkpoint surface."""
    checkpoints_dir = run_root / CHECKPOINTS_DIR
    vertices_payload = _expect_mapping(
        safe_load(checkpoints_dir / "checkpoint_vertices.pkl"),
        "checkpoint_vertices.pkl",
    )
    edges_payload = _expect_mapping(
        safe_load(checkpoints_dir / "checkpoint_edges.pkl"),
        "checkpoint_edges.pkl",
    )
    network_payload = _expect_mapping(
        safe_load(checkpoints_dir / "checkpoint_network.pkl"),
        "checkpoint_network.pkl",
    )
    return RunCounts(
        vertices=_payload_count(
            vertices_payload, preferred_keys=("positions", "count"), label="vertices"
        ),
        edges=_payload_count(
            edges_payload, preferred_keys=("connections", "traces", "count"), label="edges"
        ),
        strands=_payload_count(
            network_payload, preferred_keys=("strands", "count"), label="network"
        ),
    )


def build_experiment_summary(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    input_file: Path,
    rerun_from: str,
    matlab_counts: RunCounts,
    source_python_counts: RunCounts,
    new_python_counts: RunCounts,
) -> dict[str, Any]:
    """Build a JSON-serializable experiment summary."""
    return {
        "source_run_root": str(source_run_root),
        "dest_run_root": str(dest_run_root),
        "input_file": str(input_file),
        "rerun_from": rerun_from,
        "matlab_counts": asdict(matlab_counts),
        "source_python_counts": asdict(source_python_counts),
        "new_python_counts": asdict(new_python_counts),
        "diff_vs_matlab": _diff_counts(new_python_counts, matlab_counts),
        "diff_vs_source_python": _diff_counts(new_python_counts, source_python_counts),
    }


def render_experiment_summary(summary_payload: dict[str, Any]) -> str:
    """Render a human-readable experiment summary."""
    matlab_counts = _mapping_item(summary_payload, "matlab_counts")
    source_python_counts = _mapping_item(summary_payload, "source_python_counts")
    new_python_counts = _mapping_item(summary_payload, "new_python_counts")
    diff_vs_matlab = _mapping_item(summary_payload, "diff_vs_matlab")
    diff_vs_source_python = _mapping_item(summary_payload, "diff_vs_source_python")
    return "\n".join(
        [
            "Parity experiment summary",
            f"Source run root: {summary_payload['source_run_root']}",
            f"Destination run root: {summary_payload['dest_run_root']}",
            f"Input file: {summary_payload['input_file']}",
            f"Rerun from: {summary_payload['rerun_from']}",
            "",
            "Counts",
            f"MATLAB: vertices={matlab_counts['vertices']} edges={matlab_counts['edges']} strands={matlab_counts['strands']}",
            (
                "Source Python: "
                f"vertices={source_python_counts['vertices']} "
                f"edges={source_python_counts['edges']} "
                f"strands={source_python_counts['strands']}"
            ),
            (
                "New Python: "
                f"vertices={new_python_counts['vertices']} "
                f"edges={new_python_counts['edges']} "
                f"strands={new_python_counts['strands']}"
            ),
            "",
            "Delta vs MATLAB",
            _format_delta_line(diff_vs_matlab),
            "Delta vs source Python",
            _format_delta_line(diff_vs_source_python),
        ]
    )


def copy_source_surface(source_surface: SourceRunSurface, dest_run_root: Path) -> None:
    """Copy the reusable source checkpoints and reference metadata into a fresh destination root."""
    destination = dest_run_root.resolve()
    if destination.exists():
        raise ValueError(f"destination run root already exists: {destination}")

    checkpoints_dir = destination / CHECKPOINTS_DIR
    checkpoints_dir.mkdir(parents=True, exist_ok=False)
    for artifact in source_surface.checkpoints_dir.iterdir():
        target = checkpoints_dir / artifact.name
        if artifact.is_dir():
            copytree(artifact, target)
        else:
            copy2(artifact, target)

    metadata_dir = destination / "99_Metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (destination / "01_Input").mkdir(parents=True, exist_ok=True)
    (destination / "03_Analysis").mkdir(parents=True, exist_ok=True)
    copy2(source_surface.validated_params_path, metadata_dir / "source_validated_params.json")
    copy2(source_surface.comparison_report_path, metadata_dir / "source_comparison_report.json")
    if source_surface.run_snapshot_path is not None:
        copy2(source_surface.run_snapshot_path, metadata_dir / "source_run_snapshot.json")


def maybe_sync_exact_vertex_checkpoint(
    source_run_root: Path,
    dest_run_root: Path,
) -> bool:
    """Refresh the destination vertex checkpoint from canonical MATLAB vectors on the exact route."""
    try:
        exact_surface = validate_exact_proof_source_surface(source_run_root)
    except ValueError:
        return False

    checkpoint_path = dest_run_root / CHECKPOINTS_DIR / "checkpoint_vertices.pkl"
    if not checkpoint_path.is_file():
        raise ValueError(f"destination run root is missing vertex checkpoint: {checkpoint_path}")
    sync_exact_vertex_checkpoint_from_matlab(checkpoint_path, exact_surface.matlab_batch_dir)
    return True


def persist_experiment_summary(dest_run_root: Path, summary_payload: dict[str, Any]) -> None:
    """Persist the JSON and text summaries under 03_Analysis."""
    summary_text = render_experiment_summary(summary_payload)
    atomic_write_json(dest_run_root / SUMMARY_JSON_PATH, summary_payload)
    atomic_write_text(dest_run_root / SUMMARY_TEXT_PATH, summary_text)


def persist_exact_proof_report(
    json_path: Path,
    text_path: Path,
    report_payload: dict[str, Any],
) -> None:
    """Persist the JSON and text exact-proof reports."""
    atomic_write_json(json_path, report_payload)
    atomic_write_text(text_path, render_exact_proof_report(report_payload))


def persist_text_and_json_report(
    json_path: Path,
    text_path: Path,
    report_payload: dict[str, Any],
    *,
    renderer: Any,
) -> None:
    """Persist a JSON report and its paired text rendering."""
    atomic_write_json(json_path, report_payload)
    atomic_write_text(text_path, str(renderer(report_payload)))


def _build_candidate_exact_proof_report(
    matlab_edges_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build an exact-proof-compatible report from the candidate boundary only."""
    coverage_report = build_candidate_coverage_report(matlab_edges_payload, candidate_payload)
    first_failure: dict[str, Any] | None = None
    if not bool(coverage_report.get("passed")):
        first_failure = {
            "stage": "edges",
            "field_path": "edges.connections",
            "mismatch_type": "value mismatch",
            "matlab_preview": f"pair_count={int(coverage_report.get('matlab_pair_count', 0))}",
            "python_preview": (
                f"pair_count={int(coverage_report.get('python_pair_count', 0))} "
                f"matched={int(coverage_report.get('matched_pair_count', 0))} "
                f"missing={int(coverage_report.get('missing_pair_count', 0))} "
                f"extra={int(coverage_report.get('extra_pair_count', 0))}"
            ),
        }

    stage_summary: dict[str, Any] = {
        "passed": bool(coverage_report.get("passed")),
        "field_count": 1,
        "proof_surface": "candidate_connections_only",
    }
    if first_failure is not None:
        stage_summary["first_failure"] = first_failure

    return {
        "passed": bool(coverage_report.get("passed")),
        "stages": ["edges"],
        "stage_summaries": {"edges": stage_summary},
        "first_failing_stage": first_failure["stage"] if first_failure is not None else None,
        "first_failing_field_path": first_failure["field_path"] if first_failure is not None else None,
        "first_failure": first_failure,
        "candidate_surface": coverage_report,
        "report_scope": "candidate boundary fallback (edges.connections only)",
    }


def _run_preflight_exact(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    memory_safety_fraction: float,
    force: bool,
) -> tuple[dict[str, Any], Path, Path]:
    report_payload = build_exact_preflight_report(
        source_run_root,
        dest_run_root,
        memory_safety_fraction=memory_safety_fraction,
        force=force,
    )
    dest_root = dest_run_root.expanduser().resolve()
    ensure_dest_run_layout(dest_root)
    json_path = dest_root / PREFLIGHT_EXACT_JSON_PATH
    text_path = dest_root / PREFLIGHT_EXACT_TEXT_PATH
    persist_text_and_json_report(
        json_path,
        text_path,
        report_payload,
        renderer=render_exact_preflight_report,
    )
    return report_payload, json_path, text_path


def _run_prove_luts(
    *,
    source_run_root: Path,
    dest_run_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    source_surface = validate_exact_proof_source_surface(source_run_root)
    params = load_exact_params_file(source_surface)
    energy_payload = _load_exact_energy_payload(source_surface)
    size_of_image = cast(
        "tuple[int, int, int]",
        tuple(int(value) for value in np.asarray(energy_payload["energy"]).shape),
    )
    report_payload = compare_lut_fixture_payload(
        load_builtin_lut_fixture(),
        size_of_image=size_of_image,
        microns_per_voxel=np.asarray(
            params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
        ),
        lumen_radius_microns=np.asarray(energy_payload["lumen_radius_microns"], dtype=np.float32),
    )
    report_payload.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_run_root.expanduser().resolve()),
        }
    )
    dest_root = dest_run_root.expanduser().resolve()
    ensure_dest_run_layout(dest_root)
    json_path = dest_root / LUT_PROOF_JSON_PATH
    text_path = dest_root / LUT_PROOF_TEXT_PATH
    persist_text_and_json_report(
        json_path,
        text_path,
        report_payload,
        renderer=render_lut_proof_report,
    )
    return report_payload, json_path, text_path


def _run_capture_candidates(
    *,
    source_run_root: Path,
    dest_run_root: Path,
    include_debug_maps: bool,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    source_surface = validate_exact_proof_source_surface(source_run_root)
    params = load_exact_params_file(source_surface)
    energy_payload = _load_exact_energy_payload(source_surface)
    vertices_payload = _load_exact_vertices_payload(source_surface)

    dest_root = dest_run_root.expanduser().resolve()
    ensure_dest_run_layout(dest_root)

    energy = np.asarray(energy_payload["energy"], dtype=np.float32)
    scale_indices = energy_payload.get("scale_indices")
    vertex_positions = np.asarray(vertices_payload["positions"], dtype=np.float32)
    vertex_scales = np.asarray(vertices_payload["scales"], dtype=np.int16)
    lumen_radius_microns = np.asarray(energy_payload["lumen_radius_microns"], dtype=np.float32)
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    )
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    candidates = _generate_edge_candidates_matlab_frontier(
        energy,
        None if scale_indices is None else np.asarray(scale_indices, dtype=np.int16),
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        vertex_center_image,
        params,
    )
    candidates = _finalize_matlab_parity_candidates(
        candidates,
        energy,
        None if scale_indices is None else np.asarray(scale_indices, dtype=np.int16),
        vertex_positions,
        float(energy_payload.get("energy_sign", params.get("energy_sign", -1.0))),
        params,
        microns_per_voxel,
    )

    snapshot_payload = build_candidate_snapshot_payload(
        candidates,
        include_debug_maps=include_debug_maps,
    )
    atomic_joblib_dump(snapshot_payload, dest_root / EDGE_CANDIDATE_CHECKPOINT_PATH)
    matlab_edges = load_normalized_matlab_vectors(source_surface.matlab_batch_dir, ("edges",))[
        "edges"
    ]
    coverage_report = build_candidate_coverage_report(matlab_edges, snapshot_payload)
    coverage_report.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_root),
            "debug_maps_included": bool(include_debug_maps),
            "candidate_checkpoint_path": str(dest_root / EDGE_CANDIDATE_CHECKPOINT_PATH),
        }
    )
    json_path = dest_root / CANDIDATE_COVERAGE_JSON_PATH
    text_path = dest_root / CANDIDATE_COVERAGE_TEXT_PATH
    persist_text_and_json_report(
        json_path,
        text_path,
        coverage_report,
        renderer=render_candidate_coverage_report,
    )
    return coverage_report, snapshot_payload, json_path, text_path


def _run_replay_edges(
    *,
    source_run_root: Path,
    dest_run_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    source_surface = validate_exact_proof_source_surface(source_run_root)
    params = load_exact_params_file(source_surface)
    energy_payload = _load_exact_energy_payload(source_surface)
    vertices_payload = _load_exact_vertices_payload(source_surface)

    dest_root = dest_run_root.expanduser().resolve()
    candidate_checkpoint_path = dest_root / EDGE_CANDIDATE_CHECKPOINT_PATH
    if not candidate_checkpoint_path.is_file():
        raise ValueError(f"missing candidate checkpoint for replay: {candidate_checkpoint_path}")
    candidates = _expect_mapping(
        safe_load(candidate_checkpoint_path), str(candidate_checkpoint_path)
    )

    energy = np.asarray(energy_payload["energy"], dtype=np.float32)
    scale_indices = energy_payload.get("scale_indices")
    lumen_radius_microns = np.asarray(energy_payload["lumen_radius_microns"], dtype=np.float32)
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=np.float32
    )
    lumen_radius_pixels_axes = _lumen_radius_pixels_axes(energy_payload, params)
    vertex_positions = np.asarray(vertices_payload["positions"], dtype=np.float32)
    vertex_scales = np.asarray(vertices_payload["scales"], dtype=np.int16)

    chosen = choose_edges_for_workflow(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        lumen_radius_pixels_axes,
        tuple(int(value) for value in energy.shape),
        params,
    )
    chosen = add_vertices_to_edges_matlab_style(
        chosen,
        vertices_payload,
        energy=energy,
        scale_indices=None if scale_indices is None else np.asarray(scale_indices, dtype=np.int16),
        microns_per_voxel=microns_per_voxel,
        lumen_radius_microns=lumen_radius_microns,
        lumen_radius_pixels_axes=lumen_radius_pixels_axes,
        size_of_image=tuple(int(value) for value in energy.shape),
        params=params,
    )
    chosen = finalize_edges_matlab_style(
        chosen,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=tuple(int(value) for value in energy.shape),
    )
    chosen["lumen_radius_microns"] = lumen_radius_microns.astype(np.float32, copy=True)
    atomic_joblib_dump(chosen, dest_root / CHECKPOINTS_DIR / "checkpoint_edges.pkl")

    matlab_edges = load_normalized_matlab_vectors(source_surface.matlab_batch_dir, ("edges",))
    python_edges = {
        "edges": normalize_python_stage_payload("edges", chosen),
    }
    report_payload = compare_exact_artifacts(matlab_edges, python_edges, ("edges",))
    report_payload.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_root),
            "report_scope": "edge replay proof only",
        }
    )
    json_path = dest_root / EDGE_REPLAY_PROOF_JSON_PATH
    text_path = dest_root / EDGE_REPLAY_PROOF_TEXT_PATH
    persist_exact_proof_report(json_path, text_path, report_payload)
    return report_payload, json_path, text_path


def _handle_rerun_python(args: argparse.Namespace) -> None:
    source_surface = validate_source_run_surface(Path(args.source_run_root))
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    input_file = resolve_input_file(source_surface, args.input)
    params = load_params_file(source_surface, args.params_file)
    copy_source_surface(source_surface, dest_run_root)
    exact_vertex_sync = maybe_sync_exact_vertex_checkpoint(source_surface.run_root, dest_run_root)

    atomic_write_json(
        dest_run_root / "99_Metadata" / "experiment_provenance.json",
        {
            "source_run_root": str(source_surface.run_root),
            "source_comparison_report": str(source_surface.comparison_report_path),
            "source_validated_params": str(source_surface.validated_params_path),
            "source_run_snapshot": (
                str(source_surface.run_snapshot_path)
                if source_surface.run_snapshot_path is not None
                else None
            ),
            "input_file": str(input_file),
            "rerun_from": args.rerun_from,
            "exact_vertex_checkpoint_sync": exact_vertex_sync,
        },
    )

    image = load_tiff_volume(input_file)
    processor = SLAVVProcessor()
    processor.process_image(
        image,
        params,
        run_dir=str(dest_run_root),
        force_rerun_from=args.rerun_from,
    )

    report_payload = load_json_dict(source_surface.comparison_report_path)
    if report_payload is None:
        raise ValueError(
            f"expected JSON object in comparison report: {source_surface.comparison_report_path}"
        )
    summary_payload = build_experiment_summary(
        source_run_root=source_surface.run_root,
        dest_run_root=dest_run_root,
        input_file=input_file,
        rerun_from=args.rerun_from,
        matlab_counts=extract_matlab_counts(report_payload),
        source_python_counts=extract_source_python_counts(report_payload),
        new_python_counts=read_python_counts_from_run(dest_run_root),
    )
    persist_experiment_summary(dest_run_root, summary_payload)
    print(render_experiment_summary(summary_payload))


def _handle_summarize(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).expanduser().resolve()
    summary_text_path = run_root / SUMMARY_TEXT_PATH
    if summary_text_path.is_file():
        print(summary_text_path.read_text(encoding="utf-8"))
        return

    summary_payload = load_json_dict(run_root / SUMMARY_JSON_PATH)
    if summary_payload is None:
        raise ValueError(
            f"no experiment summary found under {run_root / SUMMARY_TEXT_PATH} or {run_root / SUMMARY_JSON_PATH}"
        )
    print(render_experiment_summary(summary_payload))


def _handle_prove_exact(args: argparse.Namespace) -> None:
    source_surface = validate_exact_proof_source_surface(Path(args.source_run_root))
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    checkpoints_dir = dest_run_root / CHECKPOINTS_DIR
    if not checkpoints_dir.is_dir():
        raise ValueError(f"destination run root is missing python checkpoints: {checkpoints_dir}")

    selected_stages = _selected_exact_stages(args.stage)
    matlab_artifacts = load_normalized_matlab_vectors(
        source_surface.matlab_batch_dir,
        selected_stages,
    )
    edge_checkpoint_path = checkpoints_dir / "checkpoint_edges.pkl"
    candidate_checkpoint_path = dest_run_root / EDGE_CANDIDATE_CHECKPOINT_PATH

    if (
        selected_stages == ("edges",)
        and not edge_checkpoint_path.is_file()
        and candidate_checkpoint_path.is_file()
    ):
        candidate_payload = _expect_mapping(
            safe_load(candidate_checkpoint_path),
            str(candidate_checkpoint_path),
        )
        report_payload = _build_candidate_exact_proof_report(
            matlab_artifacts["edges"],
            candidate_payload,
        )
        report_payload.update(
            {
                "candidate_checkpoint_path": str(candidate_checkpoint_path),
                "edge_checkpoint_path": str(edge_checkpoint_path),
            }
        )
    else:
        python_artifacts = load_normalized_python_checkpoints(checkpoints_dir, selected_stages)
        report_payload = compare_exact_artifacts(matlab_artifacts, python_artifacts, selected_stages)
    report_payload.update(
        {
            "source_run_root": str(source_surface.run_root),
            "dest_run_root": str(dest_run_root),
            "matlab_batch_dir": str(source_surface.matlab_batch_dir),
            "report_scope": str(
                report_payload.get("report_scope", "imported-matlab exact route only")
            ),
            "exact_route_gate": "comparison_exact_network + matlab_batch_hdf5",
        }
    )

    report_json_path, report_text_path = _resolve_exact_report_paths(
        dest_run_root,
        args.report_path,
    )
    persist_exact_proof_report(report_json_path, report_text_path, report_payload)

    rendered = render_exact_proof_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_preflight_exact(args: argparse.Namespace) -> None:
    report_payload, _json_path, _text_path = _run_preflight_exact(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
    )
    rendered = render_exact_preflight_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_prove_luts(args: argparse.Namespace) -> None:
    report_payload, _json_path, _text_path = _run_prove_luts(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
    )
    rendered = render_lut_proof_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_capture_candidates(args: argparse.Namespace) -> None:
    report_payload, _snapshot_payload, _json_path, _text_path = _run_capture_candidates(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
        include_debug_maps=bool(args.debug_maps),
    )
    rendered = render_candidate_coverage_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_replay_edges(args: argparse.Namespace) -> None:
    report_payload, _json_path, _text_path = _run_replay_edges(
        source_run_root=Path(args.source_run_root),
        dest_run_root=Path(args.dest_run_root),
    )
    rendered = render_exact_proof_report(report_payload)
    print(rendered)
    if not bool(report_payload.get("passed")):
        raise SystemExit(1)


def _handle_fail_fast(args: argparse.Namespace) -> None:
    source_run_root = Path(args.source_run_root)
    dest_run_root = Path(args.dest_run_root)
    preflight_report, _json_path, _text_path = _run_preflight_exact(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        memory_safety_fraction=float(args.memory_safety_fraction),
        force=bool(args.force),
    )
    print(render_exact_preflight_report(preflight_report))
    if not bool(preflight_report.get("passed")):
        raise SystemExit(1)

    lut_report, _json_path, _text_path = _run_prove_luts(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
    )
    print(render_lut_proof_report(lut_report))
    if not bool(lut_report.get("passed")):
        raise SystemExit(1)

    candidate_report, _snapshot_payload, _json_path, _text_path = _run_capture_candidates(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
        include_debug_maps=bool(args.debug_maps),
    )
    print(render_candidate_coverage_report(candidate_report))
    if not bool(candidate_report.get("passed")):
        raise SystemExit(1)

    replay_report, _json_path, _text_path = _run_replay_edges(
        source_run_root=source_run_root,
        dest_run_root=dest_run_root,
    )
    print(render_exact_proof_report(replay_report))
    if not bool(replay_report.get("passed")):
        raise SystemExit(1)

    exact_args = argparse.Namespace(
        source_run_root=str(source_run_root),
        dest_run_root=str(dest_run_root),
        stage="all",
        report_path=None,
    )
    _handle_prove_exact(exact_args)


def _expect_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping payload for {label}")
    return cast("dict[str, Any]", value)


def _mapping_item(payload: dict[str, Any], key: str) -> dict[str, Any]:
    return _expect_mapping(payload.get(key), key)


def _coerce_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"expected integer value for {label}")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise ValueError(f"expected integer value for {label}")


def _payload_count(payload: dict[str, Any], *, preferred_keys: tuple[str, ...], label: str) -> int:
    for key in preferred_keys:
        if key not in payload:
            continue
        value = payload[key]
        if key == "count":
            return _coerce_int(value, label=f"{label}.{key}")
        try:
            return len(value)
        except TypeError as exc:
            raise ValueError(f"expected sized payload for {label}.{key}") from exc
    expected = ", ".join(preferred_keys)
    raise ValueError(f"could not determine count for {label}; expected one of: {expected}")


def _diff_counts(current: RunCounts, baseline: RunCounts) -> dict[str, int]:
    return {
        "vertices": current.vertices - baseline.vertices,
        "edges": current.edges - baseline.edges,
        "strands": current.strands - baseline.strands,
    }


def _format_delta(value: int) -> str:
    return f"{value:+d}"


def _format_delta_line(diff_payload: dict[str, Any]) -> str:
    return (
        f"vertices={_format_delta(_coerce_int(diff_payload.get('vertices'), label='delta vertices'))} "
        f"edges={_format_delta(_coerce_int(diff_payload.get('edges'), label='delta edges'))} "
        f"strands={_format_delta(_coerce_int(diff_payload.get('strands'), label='delta strands'))}"
    )


def _selected_exact_stages(stage_arg: str) -> tuple[str, ...]:
    if stage_arg == "all":
        return cast("tuple[str, ...]", EXACT_STAGE_ORDER)
    return (stage_arg,)


def _resolve_exact_report_paths(
    dest_run_root: Path,
    report_path_arg: str | None,
) -> tuple[Path, Path]:
    if report_path_arg is None:
        return dest_run_root / EXACT_PROOF_JSON_PATH, dest_run_root / EXACT_PROOF_TEXT_PATH

    base_path = Path(report_path_arg).expanduser().resolve()
    if base_path.suffix.lower() == ".json":
        return base_path, base_path.with_suffix(".txt")
    if base_path.suffix.lower() == ".txt":
        return base_path.with_suffix(".json"), base_path
    return base_path.with_suffix(".json"), base_path.with_suffix(".txt")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    if argv is not None and len(argv) == 0:
        parser.print_help()
        raise SystemExit(0)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()


