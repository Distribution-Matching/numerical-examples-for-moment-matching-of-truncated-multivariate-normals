#=
example_moments.jl

Companion to Appendix A of the paper. For each example, prints the box
probability m^(0) and the truncated mean and covariance under the normal
and, up to n = 5, under t_4 and t_10 with the example's (μ, Σ) as location
and scale. Cubature base case to n = 4; at n ≥ 5 the average of eight
seeded quasi-Monte-Carlo runs, with `spread` the largest range over the
seeds of any entry. Rounding these moments to one decimal gives the
targets of §6.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/example_moments.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using LinearAlgebra
using Printf
using Random
using Statistics

BLAS.set_num_threads(1)
set_moment_mc!(; above = 100)   # recursion, not rejection sampling, at n = 7, 8

const EXAMPLES = vcat(
    [("1.1", [0.0], fill(1.0, 1, 1), [-1.0], [1.0]),
     ("1.2", [0.5], fill(2.25, 1, 1), [-1.0], [2.0])],
    [("$n.$i", n, i) for (n, is) in ((2, 1:4), (3, 1:3), (4, 1:1), (5, 1:2), (6, 1:2), (7, 1:2), (8, 1:2))
     for i in is])

unpack(e::Tuple{String,Int,Int}) = begin
    ex = get_example(n = e[2], index = e[3])
    (e[1], collect(Float64, ex.μ), Matrix{Float64}(ex.Σ), collect(Float64, ex.a), collect(Float64, ex.b))
end
unpack(e) = e

function moments(ν, μ, Σ, a, b)
    d = ν === nothing ? TruncatedMvNormal(μ, Σ, a, b) : TruncatedMvT(ν, μ, Σ, a, b; max_moment_levels = 2)
    return tp(d), collect(mean(d)), Matrix(cov(d))
end

vecstr(v) = join((@sprintf("%.5f", x) for x in v), " ")

for e in EXAMPLES
    id, μ, Σ, a, b = unpack(e)
    n = length(μ)
    for ν in (nothing, 4.0, 10.0)
        ν !== nothing && n > 5 && continue
        law = ν === nothing ? "normal" : "t$(Int(ν))"
        if n <= 4
            set_kr_base_backend!(:hcubature)
            m0, m, S = moments(ν, μ, Σ, a, b)
            spread = 0.0
        else
            set_kr_base_backend!(:mvnormalcdf)
            runs = map(1:8) do s
                set_kr_base_rng!(MersenneTwister(20260706 + s))
                moments(ν, μ, Σ, a, b)
            end
            m0 = mean(r[1] for r in runs)
            m = mean(r[2] for r in runs)
            S = mean(r[3] for r in runs)
            rng(f) = maximum(f(r) for r in runs) - minimum(f(r) for r in runs)
            spread = max(rng(r -> r[1]),
                         maximum(rng(r -> r[2][i]) for i in 1:n),
                         maximum(rng(r -> r[3][i, j]) for i in 1:n, j in 1:n))
        end
        @printf("%s|%s|%.5f|%s|%s|%.1e\n", id, law, m0, vecstr(m), vecstr(vec(S)), spread)
        flush(stdout)
    end
end
