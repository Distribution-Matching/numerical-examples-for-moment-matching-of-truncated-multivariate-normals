#=
tmvtnorm_comparison.jl

Companion to the paper's comparison with the R package tmvtnorm
(Wilhelm and Manjunath). For the box-truncated normal the moment-matching
equations with targets (x̄, S) — the sample mean and the (1/N) sample
covariance — are exactly the maximum-likelihood equations, so
tmvtnorm::mle.tmvnorm, tmvtnorm::gmm.tmvnorm ("ManjunathWilhelm" moment
conditions) and fit_mvnormal(x̄, S, a, b) all target the same point. We
compare them on that common point rather than on sampling error:

  1. write the panel (rows 2.1–5.2 of the fit tables) to panel.csv;
  2. call tmvtnorm_fit.R, which draws N points per row (seeded rejection
     sampling), fits with mle.tmvnorm and gmm.tmvnorm from the sample
     moments, and records wall-clock and estimates;
  3. read the same data, form (x̄, S), and run fit_mvnormal;
  4. score every estimate by its *solver error*: the residual of the
     moment-matching equations, max|μ_A(θ̂) − x̄| and max|Σ_A(θ̂) − S|,
     with (μ_A, Σ_A) evaluated by one common, accurate evaluator
     (Kan–Robotti with adaptive-cubature base case at rtol 1e-8).

Run from the repository root (needs R with tmvtnorm on the PATH):
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/tmvtnorm_comparison.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using DelimitedFiles
using Distributions
using LinearAlgebra
using Printf
using Random
using Statistics

BLAS.set_num_threads(1)

const OUTDIR = joinpath(@__DIR__, "results", "tmvtnorm")
const NSAMP = 2000
const SEED = 20261015
const PANEL = [("2.1", 2, 1), ("2.2", 2, 2), ("2.3", 2, 3), ("2.4", 2, 4),
               ("3.1", 3, 1), ("3.2", 3, 2), ("4.1", 4, 1),
               ("5.1", 5, 1), ("5.2", 5, 2)]

joinnum(v) = join((isinf(x) ? (x > 0 ? "Inf" : "-Inf") : repr(x) for x in v), ";")

function write_panel()
    mkpath(OUTDIR)
    open(joinpath(OUTDIR, "panel.csv"), "w") do io
        println(io, "id,n,mu,Sigma,a,b")
        for (id, n, idx) in PANEL
            ex = get_example(n = n, index = idx)
            Σ = Matrix{Float64}(ex.Σ)
            println(io, join([id, n, joinnum(ex.μ), joinnum(vec(permutedims(Σ))),
                              joinnum(ex.a), joinnum(ex.b)], ","))
        end
    end
end

# tmvtnorm coefficient vector → (μ, Σ), parsed by name (mu_i, sigma_i.j).
function parse_r_coef(names::AbstractString, coefs::AbstractString, n::Int)
    nm = split(names, ";")
    v = parse.(Float64, strip.(split(coefs, ";")))
    μ = fill(NaN, n); Σ = fill(NaN, n, n)
    for (k, s) in enumerate(nm)
        if startswith(s, "mu_")
            μ[parse(Int, s[4:end])] = v[k]
        elseif startswith(s, "sigma_")
            i, j = parse.(Int, split(s[7:end], "."))
            Σ[i, j] = Σ[j, i] = v[k]
        end
    end
    return μ, Σ
end

# Accurate truncated mean and covariance: Kan–Robotti on an
# adaptive-cubature base case at tight tolerance.
function accurate_moments(μ, Σ, a, b)
    prev = set_kr_base_backend!(:hcubature)
    try
        Σs = Symmetric(0.5 .* (Σ .+ Σ'))
        d = RecursiveMomentsBoxTruncatedMvNormal(μ, TruncatedDistributions.PDMat(Matrix(Σs)), a, b;
                                                 max_moment_levels = 2)
        n = length(μ)
        m0 = raw_moment_from_indices(d, Int[])
        m1 = [raw_moment_from_indices(d, [i]) for i in 1:n]
        m2 = [raw_moment_from_indices(d, [i, j]) for i in 1:n, j in 1:n]
        μA = m1 ./ m0
        return μA, m2 ./ m0 .- μA * μA'
    finally
        set_kr_base_backend!(prev)
    end
end

function solver_error(μ, Σ, a, b, x̄, S)
    (any(isnan, μ) || any(isnan, Σ) || !isposdef(Symmetric(Σ))) && return (NaN, NaN)
    μA, ΣA = accurate_moments(μ, Σ, a, b)
    return maximum(abs, μA .- x̄), maximum(abs, ΣA .- S)
end

function main()
    write_panel()
    # Set REUSE_R=1 to score an existing r_fits.csv without refitting in R.
    if get(ENV, "REUSE_R", "0") != "1"
        println("Running tmvtnorm (R) on the panel, N = $NSAMP ...")
        run(`Rscript $(joinpath(@__DIR__, "tmvtnorm_fit.R")) $OUTDIR $NSAMP`)
    end
    rdata, rhdr = readdlm(joinpath(OUTDIR, "r_fits.csv"), ',', String; header = true)
    unq(s) = strip(s, '"')
    col = Dict(unq(h) => k for (k, h) in enumerate(vec(rhdr)))
    rfits = rdata

    set_kr_base_backend!(:mvnormalcdf)
    # JIT warm-up so the timings below are post-compilation.
    let ex = get_example(n = 2, index = 1)
        fit_mvnormal(ex.μ̂, Matrix(ex.Σ̂), ex.a, ex.b; gamma = 1.0, seed = SEED)
    end

    ours = open(joinpath(OUTDIR, "ours_fits.csv"), "w")
    println(ours, "id,mu,Sigma")
    rows = String[]
    hdr = @sprintf("%-5s %-14s %9s %11s %11s", "ex.", "method", "time (s)",
                   "max|Δμ_A|", "max|ΔΣ_A|")
    println(hdr); push!(rows, hdr)
    for (id, n, idx) in PANEL
        ex = get_example(n = n, index = idx)
        a = collect(Float64, ex.a); b = collect(Float64, ex.b)
        X = readdlm(joinpath(OUTDIR, "X_$(id).csv"), ',', Float64)
        x̄ = vec(mean(X; dims = 1))
        S = cov(X; corrected = false)

        results = Tuple{String,Float64,Float64,Float64}[]
        for r in eachrow(rfits)
            unq(r[col["id"]]) == id || continue
            method = unq(r[col["method"]]) == "mle" ? "mle.tmvnorm" : "gmm.tmvnorm"
            t = parse(Float64, unq(r[col["time"]]))
            if startswith(unq(r[col["status"]]), "error")
                push!(results, (method, t, NaN, NaN)); continue
            end
            μr, Σr = parse_r_coef(unq(r[col["names"]]), unq(r[col["coef"]]), n)
            push!(results, (method, t, solver_error(μr, Σr, a, b, x̄, S)...))
        end
        t = @elapsed μf, Σf, info = fit_mvnormal(x̄, S, a, b; method = :lbfgs,
                                                 gamma = 1.0, ftarget = 1e-9,
                                                 iterations = 200, time_limit = 60.0,
                                                 seed = SEED)
        push!(results, ("fit_mvnormal", t, solver_error(μf, Matrix(Σf), a, b, x̄, S)...))
        println(ours, join([id, joinnum(μf), joinnum(vec(permutedims(Matrix(Σf))))], ","))
        for (m, t, eμ, eΣ) in results
            line = @sprintf("%-5s %-14s %9.3f %11.1e %11.1e", id, m, t, eμ, eΣ)
            println(line); push!(rows, line); flush(stdout)
        end
    end
    close(ours)
    open(joinpath(OUTDIR, "tmvtnorm_table.txt"), "w") do io
        foreach(l -> println(io, l), rows)
    end
end

main()
