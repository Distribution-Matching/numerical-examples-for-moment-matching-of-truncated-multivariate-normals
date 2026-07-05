# Numerical examples for moment matching of truncated multivariate normals

Reproducibility scripts for the numerical examples in the paper
*Moment Matching of Box Truncated Multivariate Normal Distributions*
(Carrizo Molina & Nazarathy).

## Companion artefacts

- Paper source: <https://github.com/Distribution-Matching/paper-truncated-mv-normal>
- Julia package: <https://github.com/Distribution-Matching/TruncatedDistributions.jl>

## Layout

```
experiments/moment_table.jl      # paper Table 1: Kan–Robotti vs direct cubature
experiments/gradient_table.jl    # paper Table 2: closed-form gradient vs AD vs FD,
                                 #   n = 1, …, 7
experiments/fit_table.jl         # paper Tables 3–5: gradient-source comparison,
                                 #   warm-start ablation, hybrid BCD, n = 2, …, 10
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
```

Each script prints a human-readable summary plus ready-to-paste LaTeX
table rows. Dependencies are declared in `experiments/Project.toml`.
