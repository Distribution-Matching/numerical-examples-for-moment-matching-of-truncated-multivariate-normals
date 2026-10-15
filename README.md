# Numerical examples for moment matching of truncated multivariate normals

Reproducibility scripts for the numerical examples in the paper
*Moment Matching of Box-Truncated Multivariate Normal and Student-t
Distributions* (Carrizo Molina & Nazarathy).

## Companion artefacts

- Paper source: <https://github.com/Distribution-Matching/paper-truncated-mv-normal>
- Julia package: <https://github.com/Distribution-Matching/TruncatedDistributions.jl>

## Layout

```
experiments/moment_table.jl      # paper moment-cost table, normal rows
experiments/gradient_table.jl    # paper gradient-cost table, normal columns:
                                 #   closed form vs AD vs FD, n = 1, …, 10
experiments/bcd_trace.jl         # earlier draft's Figure 2 (data): a worked hybrid-BCD
                                 #   run at n = 6 (subset path, accept/reject, loss)
experiments/bcd_trace_plot.py    # earlier draft's Figure 2 (plot): the loss curve with
                                 #   sub-blocks S labelled along it (needs matplotlib)
experiments/fit_table.jl         # FD / AD columns of the fit tables; also the earlier draft's
                                 #   warm-start and hybrid-BCD tables, n = 2, …, 10
experiments/fit_table_closed.jl  # paper normal fit table: NEWUOA / explicit /
                                 #   explicit+WS, rows 2.1–8.2 (FD/AD from fit_table.jl)
experiments/tmvtnorm_comparison.jl  # paper "maximum likelihood from data": fit_mvnormal vs
                                 #   package tmvtnorm (mle.tmvnorm, gmm.tmvnorm) on
                                 #   truncated-normal data, n = 2, …, 5 (calls R)
experiments/tmvtnorm_fit.R       # R half of the above (sampling + tmvtnorm fits)
experiments/tmvtnorm_loglik.R    # log-likelihood / residual check with tmvtnorm's
                                 #   own dtmvnorm and mtmvnorm
experiments/tmvtnorm_adapter.jl  # tmvtnorm as a moment matcher on the fit-table
experiments/tmvtnorm_adapter.R   #   targets, via an in-box pseudo-data set (calls R)
experiments/t_fit_table.jl       # paper Student-t fit table: FD / AD / NEWUOA /
                                 #   explicit / explicit+WS, n = 2, …, 5
                                 #   (METHODS, ROWS env vars select a subset)
experiments/t_limit.jl           # paper normal-limit table: t_ν fits for
                                 #   ν = 10, …, 200 vs the normal fit
experiments/t_generic_loss.jl    # type-generic cubature loss of the t (FD/AD baselines)
experiments/t_moment_table.jl    # paper moment-cost table, t rows
experiments/t_gradient_table.jl  # paper gradient-cost table, t columns (closed form
                                 #   vs ForwardDiff vs FD), n = 1, …, 6
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
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/fit_table_closed.jl  # paper normal fit table: NEWUOA / explicit /
                                 #   explicit+WS, rows 2.1–8.2 (FD/AD from fit_table.jl)
experiments/tmvtnorm_comparison.jl
Rscript experiments/tmvtnorm_loglik.R experiments/results/tmvtnorm
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_fit_table.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_gradient_table.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/fit_table_closed.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_moment_table.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_limit.jl
OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/tmvtnorm_adapter.jl
```

The `tmvtnorm` comparison needs R with the `tmvtnorm` package on the
`PATH`; set `REUSE_R=1` to re-score saved R fits without refitting.
The Student-t scripts need TruncatedDistributions.jl 0.5 or later.

Each script prints a human-readable summary plus ready-to-paste LaTeX
table rows. Dependencies are declared in `experiments/Project.toml`.
