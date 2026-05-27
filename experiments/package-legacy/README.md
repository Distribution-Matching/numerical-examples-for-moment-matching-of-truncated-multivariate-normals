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

§6.1 LBFGS-FD/AD/explicit/NEWUOA comparison:
* `benchmark_fd_ad_explicit.jl` — LBFGS with finite-difference vs
                              ForwardDiff vs the closed-form gradient
                              (n = 2).
* `compare_methods.jl`      — four-method comparison including PRIMA
                              NEWUOA on the bundled examples.
* `benchmark_solvers.jl`    — broader sweep over solver variants on
                              n = 2, 3, 4.

§6.2 warm-start ablation:
* `benchmark_kr_mvn_improvements.jl` — joint LBFGS with/without the
                              coordinate warm-start (§6.2 baseline).
* `benchmark_kr_mvn_high_n.jl` — joint LBFGS at n = 5, 6, 7.

§6.3 hybrid BCD timings:
* `quick_bcd_n6.jl`         — hybrid BCD for n = 6, …, 10.
* `quick_bcd_n2to5.jl`      — same experiment for n = 2, …, 5.
* `quick_bcd_sweep.jl`,
  `quick_bcd_highn.jl`,
  `quick_bcd_smoke.jl`      — additional BCD configurations / smoke tests.

Other:
* `experiment_true_loss_gradient.jl` — finite-difference cross-check of
                              the explicit gradient.
* `experiment_lbfgs_our_gradient.jl` — historical LBFGS-with-matched
                              (loss, gradient) experiment.
* `profile_kr.jl`           — profiling of the Kan–Robotti recursion.
* `quick_alloc_check.jl`,
  `quick_update_dist.jl`,
  `quick_n6_probe.jl`       — diagnostics used during development.

## Running

These scripts originally expected to be run from the package root with
`--project=.`. To re-run them now, clone
`TruncatedDistributions.jl`, copy a script next to its `Project.toml`,
and invoke as before.
