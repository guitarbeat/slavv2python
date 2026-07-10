# ADR 0008: Exact Proof Coordinator

## Status
Accepted

## Context
Exact-route parity orchestration was split across a monolithic `execution.py`, `proofs.py` (candidate capture calling frontier internals directly), and duplicate `RunCounts` helpers with incompatible report key shapes. The old `execution` barrel was deleted; call `params_audit`, `surfaces`, and `bootstrap` by name. Writer lifecycle lives in `writer_session`.

## Decision
1. **`ExactProofCoordinator`** (`analytics/parity/proof/coordinator.py`) — `prove()`, `capture_candidates()`, `prepare_dest_run()`, typed loaders for exact energy/vertices.
2. **`counts.py`** — canonical `extract_matlab_counts`, `extract_source_python_counts`, `read_python_counts_from_run` (snapshot + typed checkpoint loaders).
3. **Candidate capture** — `EdgeManager.discover_candidates()` via the discovery strategy seam, not direct `_generate_edge_candidates_matlab_frontier` calls.
4. **`reports.py`** — re-exports count helpers from `counts.py` for CLI/report consumers.

## Consequences
- One locality for prove/capture orchestration and count normalization.
- Parity reruns use the same Edge Discovery interface as production.
- Preflight (memory gate, params audit, provenance) lives under `analytics/parity/runs/preflight.py`; see `resume-exact-run`.

## Addendum (2026-07-04): Collapse parity CLI facades; sharpen coordinator boundary

Further architecture review collapsed shallow indirection without changing parity behavior:

1. **Deleted `analytics/parity/cli.py`.** The module was a 28-export re-export facade between `commands.py` and `cli_handlers/*`. `commands.py` now imports handlers directly from the owning `cli_handlers` submodules (`cli_proofs`, `cli_runs`, `cli_edges`, `cli_diagnostics`).

2. **Deleted `analytics/parity/proof/proofs.py`.** Its only remaining function, `run_exact_preflight`, moved to `runs/preflight.py` alongside `build_exact_preflight_report` and `run_exact_preflight_for_surfaces`. Preflight is run-lifecycle (gates execution); artifact comparison stays in `proof/`.

3. **Unified auxiliary proof workflows on the coordinator class.** `run_lut_proof` and `run_edge_replay` are `@staticmethod`s on `ExactProofCoordinator`, matching `estimate_exact_route_memory`. Handlers call one type for prove, capture, LUT proof, and edge replay.

**Rejected alternatives:** keeping `proofs.py` as a named barrel (misleading name after preflight moved); folding preflight into `ExactProofCoordinator.preflight()` (preflight runs before a proof surface exists); keeping LUT/replay as module-level functions in `coordinator.py` (split entry style).

**Consequences:** Parity CLI navigation is `commands.py` → `cli_handlers/*` → `ExactProofCoordinator` / `runs/preflight.py`. ADR 0008's single-coordinator intent is reinforced; preflight ownership is explicit. Update any docs or tests that referenced `parity.cli` or `proof.proofs`.

## Addendum (2026-07-04): Writer session for in-process resume

Architecture review candidate #5 (scope A: in-process resume only) extracted the hand-coded lease/manifest/registry transaction from `handle_resume_exact_run`:

1. **`runs/writer_session.py`** — `resume_writer_session` context manager owns claim → run → finalize across `writer_lease.json`, run-local `parity_job.json`, and global `JobRegistry`. Exported helpers: `reconcile_stale_writer_lease`, `reconcile_registry_writer_conflict`, `register_monitor_job`.

2. **`handle_resume_exact_run`** — reduced to `with resume_writer_session(...): resume_exact_run(...)`. Detached launch (`handle_launch_exact_run`) unchanged for this iteration; launch may adopt shared reconcile/register helpers later.

**Consequences:** The one-writer invariant has a single testable home for in-process writers. Resume and launch can share reconcile/register helpers incrementally without duplicating finalize logic.

## Addendum (2026-07-10): Detached launch + delete execution facade

Architecture review candidate #3 finished the Writer Session surface:

1. **`launch_writer_session`** — reconcile lease/registry → prepare (preflight/probe) → detach spawn → optional monitor register. `handle_launch_exact_run` is a thin CLI adapter.

2. **`assert_no_conflicting_registry_writer`** — delegates to `reconcile_registry_writer_conflict` (single conflict policy).

3. **Deleted `runs/execution.py`** — pure re-export barrel of `params_audit` / `surfaces` / `bootstrap`. Callers import real owners (e.g. `bootstrap.derive_exact_params_from_oracle`).

**Unchanged:** Preflight remains under `runs/preflight` (not ExactProofCoordinator). Detached child still enters via `resume-exact-run` → `resume_writer_session`.

## Addendum (2026-07-10): Exact proof report locality

Architecture review candidate #4 sharpened proof vs run-lifecycle report ownership:

| Surface | Owner |
|---------|--------|
| Exact proof JSON/text | `proof/proof_report.py` (`render_exact_proof_report`, `persist_exact_proof_report`) |
| Count extraction | `proof/counts.py` (no more re-export aliases from `reports`) |
| Experiment tables / delta summaries | `proof/reports.py` |
| Preflight text | `runs/preflight.py` (`render_exact_preflight_report` + existing `persist_exact_preflight_report`) |

`ExactProofCoordinator.prove` persists via `persist_exact_proof_report`. CLI/preflight no longer pull preflight rendering from the proof package.
