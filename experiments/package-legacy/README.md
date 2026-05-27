# Legacy benchmark scripts

This directory holds the original benchmark scripts that produced the
early data underlying §6 of the paper. They were previously housed
inside `TruncatedDistributions.jl/test/experiments/` and have been
relocated here so the package repository can focus on the package
itself.

The current paper-table scripts are one level up:
`../moment_table.jl` (§4 table) and `../gradient_table.jl` (§4
gradient table). The scripts here are kept for historical reference
and for the ablation/diagnostic runs they perform.

## Scripts

* `quick_bcd_n6.jl`         — hybrid BCD experiment for n = 6, …, 10
                              (full table in §6.3 of the paper).
* `quick_bcd_n2to5.jl`      — same experiment for n = 2, …, 5.
* `quick_bcd_sweep.jl`,
  `quick_bcd_highn.jl`,
  `quick_bcd_smoke.jl`      — additional BCD configurations / smoke tests.
* `benchmark_kr_mvn_high_n.jl`,
  `benchmark_kr_mvn_improvements.jl` — joint LBFGS + warm-start timing
                              data (§6.1, §6.2).
* `experiment_true_loss_gradient.jl` — finite-difference cross-check of
                              the explicit true-loss gradient.
* `profile_kr.jl`           — profiling of the Kan–Robotti recursion.
* `quick_alloc_check.jl`,
  `quick_update_dist.jl`,
  `quick_n6_probe.jl`       — diagnostics used during development.

## Running

These scripts originally expected to be run from the package root with
`--project=.`. To re-run them now, clone
`TruncatedDistributions.jl`, copy a script next to its `Project.toml`,
and invoke as before.
