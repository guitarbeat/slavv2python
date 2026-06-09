# Handoff Report

## Observation
- We were tasked with achieving exact parity in the `vertices` stage by modifying the sorting logic.
- We observed that `manager.py` used `sort_vertex_order` logic.
- The command `prove-exact` required regenerating the `vertices` checkpoint via `resume-exact-run`.

## Logic Chain
- Replaced `sort_vertex_order` calls in `_run_resumable` and `_run_ephemeral` with `np.argsort(..., kind="stable")` using `vertex_energies` (or `-vertex_energies` based on `energy_sign`).
- Initiated `resume-exact-run` to regenerate the `checkpoint_vertices.pkl`.
- Validated parity via the exact parity check command.

## Caveats
- No caveats.

## Conclusion
- The `vertices` stage fix was applied according to the strategy, removing the tie-breaker and relying entirely on the stable argsort.

## Verification Method
- Ensure tests pass with the exact parity command:
  `python scripts/parity_experiment.py prove-exact --source-run-root workspace/runs/oracle_180709_E/crop_M_exact --dest-run-root workspace/runs/oracle_180709_E/crop_M_exact --oracle-root workspace/oracles/180709_E_crop_M --stage vertices`
