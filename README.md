# Numerical examples for moment matching of truncated multivariate normals

Reproducibility scripts for the numerical examples in the paper
*Moment Matching of Box-Truncated Multivariate Normal and Student-t
Distributions* (Carrizo Molina & Nazarathy).

## Companion artefacts

- Paper source: <https://github.com/Distribution-Matching/paper-truncated-mv-normal>
- Julia package: <https://github.com/Distribution-Matching/TruncatedDistributions.jl>

## Layout

```
experiments/moment_table.jl      # paper Table 1: Kan–Robotti vs direct cubature
experiments/gradient_table.jl    # paper Table 2: closed-form gradient vs AD vs FD,
                                 #   n = 1, …, 7
experiments/bcd_trace.jl         # paper Figure 2 (data): a worked hybrid-BCD
                                 #   run at n = 6 (subset path, accept/reject, loss)
experiments/bcd_trace_plot.py    # paper Figure 2 (plot): the loss curve with
                                 #   sub-blocks S labelled along it (needs matplotlib)
experiments/fit_table.jl         # paper Tables 3–5: gradient-source comparison,
                                 #   warm-start ablation, hybrid BCD, n = 2, …, 10
experiments/tmvtnorm_comparison.jl  # paper Table "tmvtnorm": fit_mvnormal vs the R
                                 #   package tmvtnorm (mle.tmvnorm, gmm.tmvnorm) on
                                 #   truncated-normal data, n = 2, …, 5 (calls R)
experiments/tmvtnorm_fit.R       # R half of the above (sampling + tmvtnorm fits)
experiments/tmvtnorm_loglik.R    # log-likelihood / residual check with tmvtnorm's
                                 #   own dtmvnorm and mtmvnorm
experiments/tmvtnorm_adapter.jl  # tmvtnorm as a moment matcher on the fit-table
experiments/tmvtnorm_adapter.R   #   targets, via an in-box pseudo-data set (calls R)
experiments/t_fit_table.jl       # paper Student-t fit table: FD / AD / NEWUOA /
                                 #   explicit / explicit+WS, n = 2, 3, 4
experiments/t_generic_loss.jl    # type-generic cubature loss of the t (FD/AD baselines)
experiments/t_moment_table.jl    # paper moment-cost table, t rows
experiments/t_gradient_table.jl  # paper gradient-cost table, t columns (closed form
                                 #   vs ForwardDiff vs FD), n = 1, …, 5
experiments/intro_figure.py      # paper Figure 1: the introduction's worked n = 3
                                 #   example (dots + wall marginals; needs matplotlib)
experiments/results/             # captured benchmark outputs, dated
```

Earlier exploratory and diagnostic scripts are preserved in this
repository's git history (the `experiments/package-legacy/` directory,
prior to its removal).

## Running

Scripts are plain Julia, runnable from the top of this repo. All paper
timings are single-threaded:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/moment_table.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/gradient_table.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/fit_table.jl      # all three tables
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/fit_table.jl A    # one table (A, B, or C)
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/tmvtnorm_comparison.jl
Rscript experiments/tmvtnorm_loglik.R experiments/results/tmvtnorm
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_fit_table.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_gradient_table.jl
```

The `tmvtnorm` comparison needs R with the `tmvtnorm` package on the
`PATH`; set `REUSE_R=1` to re-score saved R fits without refitting.
The Student-t scripts need TruncatedDistributions.jl 0.5 or later.

Each script prints a human-readable summary plus ready-to-paste LaTeX
table rows. Dependencies are declared in `experiments/Project.toml`.
