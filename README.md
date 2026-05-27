# Numerical examples for moment matching of truncated multivariate normals

Reproducibility scripts for the numerical examples in the paper
*Moment Matching of Box Truncated Multivariate Normal Distributions*
(Carrizo Molina & Nazarathy).

## Companion artefacts

- Paper source: <https://github.com/Distribution-Matching/paper-truncated-mv-normal>
- Julia package: <https://github.com/yoninazarathy/TruncatedDistributions.jl>

## Layout

```
experiments/   # one Julia script per numerical example in the paper
output/        # generated tables and figures (gitignored)
```

## Running

Scripts are plain Julia, runnable from the top of this repo:

```sh
julia --project=experiments experiments/<name>.jl
```

Each script declares its own dependencies via `experiments/Project.toml`
(once initialised).

## Status

This repo is intentionally minimal and is expected to grow alongside
the paper. The first planned experiment is the §4 comparison of the
35-moment table at $n = 3$ computed via the Kan--Robotti recursion
plus Genz--Bretz QMC against direct adaptive cubature.
