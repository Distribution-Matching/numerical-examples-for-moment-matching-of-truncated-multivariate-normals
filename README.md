# Numerical examples for moment matching of truncated multivariate normals

Reproducibility scripts for the numerical examples in the paper
*Moment Matching of Box Truncated Multivariate Normal Distributions*
(Carrizo Molina & Nazarathy).

## Companion artefacts

- Paper source: <https://github.com/Distribution-Matching/paper-truncated-mv-normal>
- Julia package: <https://github.com/Distribution-Matching/TruncatedDistributions.jl>

## Layout

```
experiments/moment_table.jl      # §4 K-R vs hcubature comparison
experiments/gradient_table.jl    # §4 closed-form vs FD vs AD
experiments/results/             # captured benchmark outputs
experiments/package-legacy/      # earlier benchmark scripts that
                                 #   used to live in
                                 #   TruncatedDistributions.jl/test/
                                 #   experiments
```

## Running

Scripts are plain Julia, runnable from the top of this repo:

```sh
julia --project=experiments experiments/<name>.jl
```

Dependencies are declared in `experiments/Project.toml`.
