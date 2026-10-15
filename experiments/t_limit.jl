#=
t_limit.jl

Consistency check of the Student-t pipeline against the normal one as
ν → ∞. On the normal targets of panel rows 2.1, 2.3, 3.1 and 4.1 — the
exact truncated normal moments of each example, unrounded, so that an exact
solution exists (the rounded target of row 2.3 is not attainable: the loss
plateaus near 1e-4 while the parameters diverge) — fit the normal with fit_mvnormal and the t_ν
with fit_mvt for ν ∈ {10, 50, 100, 200}, both to a tight loss target, and
report the distance between the fitted t location/scale and the fitted
normal mean/covariance. Since t_ν(μ, Σ) → N(μ, Σ), the distance should
vanish, at rate O(1/ν).

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_limit.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using LinearAlgebra
using Printf
using Random

BLAS.set_num_threads(1)

const ROWS = [("2.1", 2, 1), ("2.3", 2, 3), ("3.1", 3, 1), ("4.1", 4, 1)]
const NUS = (10.0, 50.0, 100.0, 200.0)
const SEED = 20261015

function targets(n, idx)
    ex = get_example(n = n, index = idx)
    set_kr_base_backend!(:mvnormalcdf); set_kr_base_rng!(MersenneTwister(SEED))
    a = collect(Float64, ex.a); b = collect(Float64, ex.b)
    d = RecursiveMomentsBoxTruncatedMvNormal(collect(Float64, ex.μ),
            TruncatedDistributions.PDMat(Matrix{Float64}(ex.Σ)), a, b; max_moment_levels = 2)
    m0 = raw_moment_from_indices(d, Int[])
    m1 = [raw_moment_from_indices(d, [i]) for i in 1:n] ./ m0
    m2 = [raw_moment_from_indices(d, [i, j]) for i in 1:n, j in 1:n] ./ m0
    return m1, m2 .- m1 * m1', a, b
end

function main()
    set_kr_base_backend!(:mvnormalcdf)
    kw = (; gamma = 1.0, ftarget = 1e-10, iterations = 300, time_limit = 300.0, seed = SEED)
    println(@sprintf("%-5s %6s %10s %12s %12s %8s", "ex.", "ν", "loss", "max|Δμ|", "max|ΔΣ|", "time"))
    latex = String[]
    for (id, n, idx) in ROWS
        μ̂, Σ̂, a, b = targets(n, idx)
        # naive starts throughout (the coordinate warm start fails on row 2.3)
        tN = @elapsed μN, ΣN, iN = fit_mvnormal(μ̂, Σ̂, a, b; method = :lbfgs,
                                                μ_init = μ̂, Σ_init = Σ̂, kw...)
        println(@sprintf("%-5s %6s %10.1e %12s %12s %8.2f", id, "normal", iN.loss, "", "", tN))
        cells = String[]
        for ν in NUS
            μt, Σt, it = fit_mvt(ν, μ̂, Σ̂, a, b; μ_init = μ̂, Σ_init = ((ν - 2) / ν) .* Σ̂, kw...)
            dμ = maximum(abs, μt - μN); dΣ = maximum(abs, Matrix(Σt) - Matrix(ΣN))
            println(@sprintf("%-5s %6.0f %10.1e %12.1e %12.1e %8.2f", id, ν, it.loss, dμ, dΣ,
                             it.time))
            flush(stdout)
            push!(cells, @sprintf("%.1e / %.1e", dμ, dΣ))
        end
        push!(latex, id * " & " * join(cells, " & ") * " \\\\")
    end
    println("\nRows (max|Δμ| / max|ΔΣ| per ν):")
    foreach(println, latex)
end

main()
