"""Developer helper for counts-only SLAVV parity experiments."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from shutil import copy2, copytree
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIR = REPO_ROOT / "source"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from slavv import SLAVVProcessor
from slavv.io import load_tiff_volume
from slavv.runtime.run_state import atomic_write_json, atomic_write_text, load_json_dict
from slavv.utils.safe_unpickle import safe_load

CHECKPOINTS_DIR = Path("02_Output") / "python_results" / "checkpoints"
COMPARISON_REPORT_PATH = Path("03_Analysis") / "comparison_report.json"
RUN_SNAPSHOT_PATH = Path("99_Metadata") / "run_snapshot.json"
SUMMARY_JSON_PATH = Path("03_Analysis") / "experiment_summary.json"
SUMMARY_TEXT_PATH = Path("03_Analysis") / "experiment_summary.txt"
VALIDATED_PARAMS_PATH = Path("99_Metadata") / "validated_params.json"


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


def build_parser() -> argparse.ArgumentParser:
    """Build the developer parity experiment parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Counts-only developer helper for rerunning the live Python edges/network "
            "stages against an existing staged comparison run."
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
            snapshot_payload.get("provenance", {})
            if isinstance(snapshot_payload, dict)
            else {}
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
        matlab.get("strand_count", _mapping_item(report_payload, "network").get("matlab_strand_count")),
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
        python_counts.get("edges_count", _mapping_item(report_payload, "edges").get("python_count")),
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
        vertices=_payload_count(vertices_payload, preferred_keys=("positions", "count"), label="vertices"),
        edges=_payload_count(edges_payload, preferred_keys=("connections", "traces", "count"), label="edges"),
        strands=_payload_count(network_payload, preferred_keys=("strands", "count"), label="network"),
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


def persist_experiment_summary(dest_run_root: Path, summary_payload: dict[str, Any]) -> None:
    """Persist the JSON and text summaries under 03_Analysis."""
    summary_text = render_experiment_summary(summary_payload)
    atomic_write_json(dest_run_root / SUMMARY_JSON_PATH, summary_payload)
    atomic_write_text(dest_run_root / SUMMARY_TEXT_PATH, summary_text)


def _handle_rerun_python(args: argparse.Namespace) -> None:
    source_surface = validate_source_run_surface(Path(args.source_run_root))
    dest_run_root = Path(args.dest_run_root).expanduser().resolve()
    input_file = resolve_input_file(source_surface, args.input)
    params = load_params_file(source_surface, args.params_file)
    copy_source_surface(source_surface, dest_run_root)

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
