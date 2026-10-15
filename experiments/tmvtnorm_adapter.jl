#=
tmvtnorm_adapter.jl

The R package tmvtnorm as a moment matcher on the targets of the fit
tables (rows 2.1–5.2, the same rounded (μ̂, Σ̂) as fit_table.jl). For the
truncated normal the maximum-likelihood estimate depends on the data only
through the sample mean and (1/N) covariance, so tmvtnorm_adapter.R feeds
mle.tmvnorm and gmm.tmvnorm a small pseudo-data set inside the box whose
sample moments equal the target exactly, started from (μ̂, Σ̂) as in the fit
tables. Each estimate is then scored by the moment-matching loss of the fit
tables (γ = 1) with an accurate common evaluator (Kan–Robotti on an
adaptive-cubature base case).

Run from the repository root (needs R with tmvtnorm on the PATH):
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/tmvtnorm_adapter.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using DelimitedFiles
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)

const OUTDIR = joinpath(@__DIR__, "results", "tmvtnorm_adapter")
const PANEL = [("2.1", 2, 1), ("2.2", 2, 2), ("2.3", 2, 3), ("2.4", 2, 4),
               ("3.1", 3, 1), ("3.2", 3, 2), ("4.1", 4, 1),
               ("5.1", 5, 1), ("5.2", 5, 2)]

joinnum(v) = join((isinf(x) ? (x > 0 ? "Inf" : "-Inf") : repr(x) for x in v), ";")

# Targets exactly as in fit_table.jl: Kan–Robotti moments rounded to 1 decimal.
function targets(n, idx)
    ex = get_example(n = n, index = idx)
    set_kr_base_backend!(:mvnormalcdf)
    d = dist_from_example(ex)
    μ̂ = round.(collect(mean(d)); digits = 1)
    Σ̂ = round.(Matrix(cov(d)); digits = 1)
    return μ̂, Σ̂, collect(Float64, ex.a), collect(Float64, ex.b)
end

function moments_accurate(μ, Σ, a, b)
    prev = set_kr_base_backend!(:hcubature)
    try
        Σs = 0.5 .* (Σ .+ Σ')
        d = RecursiveMomentsBoxTruncatedMvNormal(μ, TruncatedDistributions.PDMat(Σs), a, b;
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

function loss_gamma1(μ, Σ, a, b, μ̂, Σ̂)
    (any(isnan, μ) || any(isnan, Σ) || !isposdef(Symmetric(Σ))) && return NaN
    μA, ΣA = moments_accurate(μ, Σ, a, b)
    n = length(μ)
    return 0.5 * (sum(abs2, μA .- μ̂) + 2 / (n + 1) * sum(abs2, ΣA .- Σ̂))
end

function parse_r_coef(names, coefs, n)
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

function main()
    mkpath(OUTDIR)
    tg = Dict{String,Any}()
    open(joinpath(OUTDIR, "targets.csv"), "w") do io
        println(io, "id,n,mu_hat,Sigma_hat,a,b")
        for (id, n, idx) in PANEL
            μ̂, Σ̂, a, b = targets(n, idx)
            tg[id] = (μ̂, Σ̂, a, b)
            println(io, join([id, n, joinnum(μ̂), joinnum(vec(permutedims(Σ̂))),
                              joinnum(a), joinnum(b)], ","))
        end
    end
    run(`Rscript $(joinpath(@__DIR__, "tmvtnorm_adapter.R")) $OUTDIR`)
    data, hdr = readdlm(joinpath(OUTDIR, "r_fits.csv"), ',', String; header = true)
    unq(s) = strip(s, '"')
    col = Dict(unq(h) => k for (k, h) in enumerate(vec(hdr)))
    println(@sprintf("%-5s %-12s %3s %-17s %9s %11s %5s", "ex.", "method", "N", "design",
                     "time (s)", "L (γ=1)", "conv"))
    latex = Dict{String,Vector{String}}()
    for r in eachrow(data)
        id = string(unq(r[col["id"]])); n = length(tg[id][1])
        method = unq(r[col["method"]]) == "mle" ? "mle.tmvnorm" : "gmm.tmvnorm"
        status = unq(r[col["status"]])
        t = status == "ok" ? parse(Float64, unq(r[col["time"]])) : NaN
        L = NaN
        if status == "ok"
            μ, Σ = parse_r_coef(unq(r[col["names"]]), unq(r[col["coef"]]), n)
            L = loss_gamma1(μ, Σ, tg[id][3], tg[id][4], tg[id][1], tg[id][2])
        end
        println(@sprintf("%-5s %-12s %3s %-17s %9.2f %11.1e %5s", id, method,
                         unq(r[col["N"]]), unq(r[col["design"]]), t, L,
                         unq(r[col["convergence"]])))
        push!(get!(latex, id, String[]), isnan(t) ? "--" : @sprintf("%.2f / %.1e", t, L))
    end
    println("\nCells (mle, gmm) per row:")
    for (id, _, _) in PANEL
        println(id, " & ", join(latex[id], " & "))
    end
end

main()
