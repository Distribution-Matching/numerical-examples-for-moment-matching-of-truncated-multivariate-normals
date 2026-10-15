#=
t_moment_table.jl

Student-t counterpart of moment_table.jl. For each configuration of
moment_table.jl, the truncated t_7 (the order-4 table needed by the
gradient when fitting a t_5) with the example's (μ, Σ, A) (Σ as the
scale matrix), compute every primitive moment {m^(κ) : |κ| ≤ 4} (or a
single κ at n = 4) via
    (a) the t recursion (TruncatedMvT) on the package's base case:
        Genz–Bretz QMC from AdditionalDistributions.jl at n ≥ 3
        (4·10^4 samples per probability), adaptive cubature at n = 2;
    (b) direct adaptive cubature on each moment integral (hcubature_inf,
        rtol = 1e-6).
The reference is adaptive cubature at rtol = 1e-9 after mapping infinite
coordinates to [0, 1) by x = a + t/(1-t) (resp. b - t/(1-t)) — clipping
at μ ± 10σ, as done for the normal, would bias heavy-tailed t moments.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 julia --project=experiments experiments/t_moment_table.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Distributions
using HCubature
using BenchmarkTools
using LinearAlgebra
using Printf
using Random

BLAS.set_num_threads(1)

# The order-4 table the moment-matching gradient of a t_ν law consumes is that
# of t_{ν+2}; we benchmark t_7, the law needed when fitting a t_5.
const NU = 7.0
const SEED = 20261015

struct Cfg
    name::String
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    a::Vector{Float64}
    b::Vector{Float64}
    single_κ::Union{Nothing,Vector{Int}}
end
bundled(name, n, idx; single_κ = nothing) = (ex = get_example(n = n, index = idx);
    Cfg(name, collect(Float64, ex.μ), Matrix{Float64}(ex.Σ), collect(Float64, ex.a),
        collect(Float64, ex.b), single_κ))

const CONFIGS = [
    bundled("2.1", 2, 1), bundled("2.4", 2, 4),
    bundled("3.1", 3, 1), bundled("3.3", 3, 3),
    bundled("4.1", 4, 1; single_κ = [1, 1, 1, 1]),
]

function multiindices(n, L)
    out = Vector{Vector{Int}}()
    rec(p, r) = length(p) == n - 1 ? push!(out, vcat(p, r)) :
                foreach(k -> rec(vcat(p, k), r - k), 0:r)
    for t in 0:L
        n == 1 ? push!(out, [t]) : rec(Int[], t)
    end
    out
end

function moments_recursion(c::Cfg, κs)
    set_kr_base_backend!(:mvnormalcdf)
    set_kr_base_rng!(MersenneTwister(SEED))
    d = TruncatedMvT(NU, c.μ, c.Σ, c.a, c.b; max_moment_levels = 4)
    return Dict(κ => raw_moment(d, κ) for κ in κs)
end

function moments_direct(c::Cfg, κs; rtol = 1e-6)
    D = MvTDist(NU, c.μ, c.Σ)
    return Dict(κ => hcubature_inf(x -> prod(x[i]^κ[i] for i in eachindex(κ)) * pdf(D, x),
                                   c.a, c.b; rtol = rtol, maxevals = 10^7)[1] for κ in κs)
end

function moments_reference(c::Cfg, κs)
    D = MvTDist(NU, c.μ, c.Σ); n = length(c.μ)
    function mom(κ)
        g(t) = begin
            x = similar(t); J = 1.0
            for i in 1:n
                lo, hi = c.a[i], c.b[i]
                if isfinite(lo) && isfinite(hi)
                    x[i] = lo + (hi - lo) * t[i]; J *= hi - lo
                elseif isfinite(lo)
                    x[i] = lo + t[i] / (1 - t[i]); J /= (1 - t[i])^2
                elseif isfinite(hi)
                    x[i] = hi - t[i] / (1 - t[i]); J /= (1 - t[i])^2
                else
                    s = 2t[i] - 1; x[i] = s / (1 - s^2); J *= 2 * (1 + s^2) / (1 - s^2)^2
                end
            end
            isfinite(J) || return 0.0     # cells at the end of the mapped range
            v = prod(x[i]^κ[i] for i in 1:n) * pdf(D, x) * J
            isfinite(v) ? v : 0.0
        end
        hcubature(g, zeros(n), ones(n); rtol = 1e-9, maxevals = 5 * 10^7)[1]
    end
    return Dict(κ => mom(κ) for κ in κs)
end

maxrel(A, R, κs) = maximum(abs(A[κ] - R[κ]) / max(abs(R[κ]), 1e-15) for κ in κs)

fmt_time(t) = t < 1e-3 ? @sprintf("\$%.0f\$\\,\\textmu s", t * 1e6) :
              t < 1.0  ? @sprintf("\$%.1f\$\\,ms", t * 1e3) : @sprintf("\$%.2f\$\\,s", t)
fmt_err(e) = replace(@sprintf("\$%.1e\$", e), r"e-0?(\d+)" => s"\\mathrm{e}\\text{-}\1")

function main()
    latex = String[]
    @printf("%-4s %2s %4s %12s %10s %12s %10s %8s\n", "Cfg", "n", "M", "recursion", "rel.err",
            "cubature", "rel.err", "speedup")
    for c in CONFIGS
        n = length(c.μ)
        κs = c.single_κ === nothing ? multiindices(n, 4) : [c.single_κ]
        ref = moments_reference(c, κs)
        mr = moments_recursion(c, κs); er = maxrel(mr, ref, κs)
        tr = @belapsed moments_recursion($c, $κs) samples = 5 seconds = 30
        md = moments_direct(c, κs); ed = maxrel(md, ref, κs)
        td = @belapsed moments_direct($c, $κs) samples = 2 seconds = 60
        @printf("%-4s %2d %4d %12.3e %10.1e %12.3e %10.1e %8.0f\n", c.name, n, length(κs),
                tr, er, td, ed, td / tr)
        flush(stdout)
        push!(latex, @sprintf("%s & %d & %3d & %s & %s && %s & %s & \$%.0f\\times\$ \\\\",
                              c.name, n, length(κs), fmt_time(tr), fmt_err(er),
                              fmt_time(td), fmt_err(ed), td / tr))
    end
    println("\nLaTeX rows:")
    foreach(println, latex)
end

main()
