"""Command registry for the developer parity experiment CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

from slavv_python.analytics.parity.cli import (
    handle_capture_candidates,
    handle_dedupe,
    handle_diagnose_gaps,
    handle_ensure_oracle_artifacts,
    handle_fail_fast,
    handle_init_exact_run,
    handle_launch_exact_run,
    handle_normalize_recordings,
    handle_preflight_exact,
    handle_promote_dataset,
    handle_promote_oracle,
    handle_promote_report,
    handle_prove_exact,
    handle_prove_exact_sequence,
    handle_prove_luts,
    handle_replay_edges,
    handle_rerun_python,
    handle_resume_exact_run,
    handle_status_exact_run,
    handle_summarize,
    handle_trace_vertex,
)
from slavv_python.analytics.parity.constants import (
    DEFAULT_MEMORY_SAFETY_FRACTION,
    EXACT_STAGE_ORDER,
)


@dataclass(frozen=True)
class ArgumentSpec:
    """One argparse argument declaration."""

    flags: tuple[str, ...]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandSpec:
    """One parity CLI command declaration."""

    name: str
    handler: Callable[[argparse.Namespace], None]
    arguments: tuple[ArgumentSpec, ...]
    help: str | None = None


def build_parity_parser() -> argparse.ArgumentParser:
    """Build the developer parity experiment parser from command specs."""
    parser = argparse.ArgumentParser(
        description=(
            "Developer helpers for rerunning and proving native-first exact-route parity "
            "against an existing staged comparison run."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True
    for spec in PARITY_COMMAND_SPECS:
        subparser = subparsers.add_parser(spec.name, help=spec.help)
        for argument in spec.arguments:
            subparser.add_argument(*argument.flags, **argument.kwargs)
        subparser.set_defaults(handler=spec.handler)
    return parser


def arg(*flags: str, **kwargs: Any) -> ArgumentSpec:
    """Create an argparse argument declaration."""
    return ArgumentSpec(flags=flags, kwargs=kwargs)


_RUN_ROOT_PAIR = (
    arg("--source-run-root", required=True),
    arg("--dest-run-root", required=True),
)


PARITY_COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "rerun-python",
        handle_rerun_python,
        (
            *_RUN_ROOT_PAIR,
            arg("--input"),
            arg("--rerun-from", choices=("edges", "network"), default="edges"),
            arg("--params-file"),
        ),
    ),
    CommandSpec(
        "trace-vertex",
        handle_trace_vertex,
        (
            arg("--source-run-root", required=True),
            arg("--vertex-idx", type=int, required=True),
            arg("--output-trace", required=True),
        ),
    ),
    CommandSpec("summarize", handle_summarize, (arg("--run-root", required=True),)),
    CommandSpec(
        "normalize-recordings",
        handle_normalize_recordings,
        (arg("--run-root", required=True),),
    ),
    CommandSpec(
        "diagnose-gaps",
        handle_diagnose_gaps,
        (arg("--run-root", required=True), arg("--limit", type=int, default=10)),
    ),
    CommandSpec(
        "prove-exact",
        handle_prove_exact,
        (
            *_RUN_ROOT_PAIR,
            arg("--oracle-root"),
            arg("--stage", choices=(*EXACT_STAGE_ORDER, "all"), default="all"),
            arg("--report-path"),
        ),
    ),
    CommandSpec(
        "prove-exact-sequence",
        handle_prove_exact_sequence,
        (*_RUN_ROOT_PAIR, arg("--oracle-root")),
        help="Run prove-exact for energy, vertices, edges, network in order.",
    ),
    CommandSpec(
        "preflight-exact",
        handle_preflight_exact,
        (
            *_RUN_ROOT_PAIR,
            arg("--oracle-root"),
            arg("--dataset-root"),
            arg("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION),
            arg("--force", action="store_true"),
        ),
    ),
    CommandSpec(
        "resume-exact-run",
        handle_resume_exact_run,
        (
            arg("--dest-run-root", required=True),
            arg("--dataset-root"),
            arg("--oracle-root"),
            arg("--stop-after", default="network"),
            arg("--force-rerun-from", choices=EXACT_STAGE_ORDER),
            arg("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION),
            arg("--force", action="store_true"),
            arg("--skip-preflight", action="store_true"),
            arg("--n-jobs", type=int),
            arg("--monitor", action="store_true", help="Monitor job and send notifications"),
            arg("--force-kill", action="store_true", help="Kill active writer if exists"),
        ),
        help="Resume an interrupted init-exact-run after preflight checks.",
    ),
    CommandSpec(
        "launch-exact-run",
        handle_launch_exact_run,
        (
            arg("--dest-run-root", required=True),
            arg("--dataset-root"),
            arg("--oracle-root"),
            arg("--stop-after", default="network"),
            arg("--force-rerun-from", choices=EXACT_STAGE_ORDER),
            arg("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION),
            arg("--force", action="store_true"),
            arg("--skip-preflight", action="store_true"),
            arg("--n-jobs", type=int),
            arg("--monitor", action="store_true", help="Monitor job and send notifications"),
            arg("--force-kill", action="store_true", help="Kill active writer if exists"),
        ),
        help="Launch resume-exact-run in a detached OS-owned process.",
    ),
    CommandSpec(
        "status-exact-run",
        handle_status_exact_run,
        (arg("--run-dir", required=True),),
        help="Print detached parity job and run snapshot status.",
    ),
    CommandSpec("prove-luts", handle_prove_luts, (*_RUN_ROOT_PAIR, arg("--oracle-root"))),
    CommandSpec(
        "capture-candidates",
        handle_capture_candidates,
        (
            *_RUN_ROOT_PAIR,
            arg("--oracle-root"),
            arg("--include-debug-maps", dest="debug_maps", action="store_true"),
        ),
    ),
    CommandSpec("replay-edges", handle_replay_edges, (*_RUN_ROOT_PAIR, arg("--oracle-root"))),
    CommandSpec(
        "fail-fast",
        handle_fail_fast,
        (
            *_RUN_ROOT_PAIR,
            arg("--oracle-root"),
            arg("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION),
            arg("--force", action="store_true"),
            arg("--debug-maps", action="store_true"),
        ),
    ),
    CommandSpec(
        "ensure-oracle-artifacts",
        handle_ensure_oracle_artifacts,
        (
            arg("--oracle-root", required=True),
            arg("--matlab-batch-dir"),
            arg("--stage", action="append", choices=(*EXACT_STAGE_ORDER, "all"), default=None),
            arg("--no-repair", action="store_true"),
        ),
        help="Verify and optionally repair normalized Oracle Artifacts.",
    ),
    CommandSpec(
        "promote-oracle",
        handle_promote_oracle,
        (
            arg("--matlab-batch-dir", required=True),
            arg("--oracle-root", required=True),
            arg("--dataset-file"),
            arg("--dataset-hash"),
            arg("--oracle-id"),
            arg("--matlab-source-version"),
        ),
    ),
    CommandSpec(
        "promote-report",
        handle_promote_report,
        (arg("--run-root", required=True), arg("--report-root")),
    ),
    CommandSpec(
        "promote-dataset",
        handle_promote_dataset,
        (arg("--dataset-file", required=True), arg("--experiment-root", required=True)),
    ),
    CommandSpec(
        "init-exact-run",
        handle_init_exact_run,
        (
            arg("--dataset-root", required=True),
            arg("--oracle-root", required=True),
            arg("--dest-run-root", required=True),
            arg("--stop-after", default="vertices"),
            arg("--energy-storage-format", default="npy"),
            arg("--resume", action="store_true"),
            arg("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION),
            arg("--force", action="store_true"),
            arg("--skip-preflight", action="store_true"),
        ),
    ),
    CommandSpec(
        "dedupe",
        handle_dedupe,
        (arg("--dry-run", action="store_true", help="Preview deletions without modifying files."),),
    ),
)


__all__ = ["PARITY_COMMAND_SPECS", "CommandSpec", "build_parity_parser"]
