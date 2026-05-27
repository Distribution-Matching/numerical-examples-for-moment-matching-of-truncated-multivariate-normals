#=
moment_table.jl

Populates Table~\ref{tab:kr-bench} in section "On efficient computation of
truncated moments" of the paper "Moment Matching of Box Truncated
Multivariate Normal Distributions". For each (μ, Σ, A) configuration,
compute every primitive moment {m^{(κ)} : 0 ≤ |κ| ≤ 4} at the underlying
parameters via
    (a) the Kan–Robotti recursion with the Genz–Bretz QMC base case
        (TruncatedDistributions.jl with set_kr_base_backend!(:mvnormalcdf)),
        m = 10_000 QMC samples per box probability (package default);
    (b) direct adaptive cubature on each moment integral
        (HCubature.jl at rtol = 1e-6).
A tight-tolerance hcubature run gives the reference moments for the
max-relative-error column. Semi-infinite bounds are clipped to μ ± 10·σ
for the hcubature methods; the Kan–Robotti recursion handles ±Inf
natively via the Genz–Bretz separation-of-variables step.

The configurations tie back to the bundled
`TruncatedDistributions.get_example(n = n, index = k)` panel used
elsewhere in the paper, plus the Genz–Bretz textbook example
(Genz & Bretz 2009, §1.3.1) at n = 3.

Run from the repository root:
    julia --project=experiments experiments/moment_table.jl
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Distributions
using HCubature
using BenchmarkTools
using LinearAlgebra
using Printf

# ----------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------

struct Config
    name::String           # short label, e.g. "2A"
    n::Int                 # dimension
    descr::String          # human-readable description
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    a::Vector{Float64}     # may contain -Inf
    b::Vector{Float64}     # may contain +Inf
end

function bundled_config(label, n, idx, descr)
    ex = TruncatedDistributions.get_example(n = n, index = idx)
    return Config(label, n, descr,
                  Vector{Float64}(ex.μ), Matrix{Float64}(ex.Σ),
                  Vector{Float64}(ex.a), Vector{Float64}(ex.b))
end

const CONFIGS = [
    bundled_config("2A", 2, 1, "bundled n=2 ex 1: off-centre, mild correlation"),
    bundled_config("2B", 2, 4, "bundled n=2 ex 4: centred, mildly correlated, tight box"),
    bundled_config("3A", 3, 1, "bundled n=3 ex 1: off-centre, finite asymmetric box"),
    bundled_config("3G", 3, 3, "Genz–Bretz Phi3ex (G&B 2009 §1.3.1): centred, semi-infinite"),
    bundled_config("4A", 4, 1, "bundled n=4 ex 1: off-centre, finite asymmetric box"),
]

# ----------------------------------------------------------------------
# Multi-index enumeration: |κ| ≤ L over n dimensions
# ----------------------------------------------------------------------

function multiindices(n::Int, L::Int)
    out = Vector{Vector{Int}}()
    function rec(prefix, remaining)
        if length(prefix) == n - 1
            push!(out, vcat(prefix, remaining))
            return
        end
        for k in 0:remaining
            rec(vcat(prefix, k), remaining - k)
        end
    end
    for total in 0:L
        rec(Int[], total)
    end
    return out
end

# ----------------------------------------------------------------------
# Method (a): KR + Genz–Bretz QMC via TruncatedDistributions.jl
# ----------------------------------------------------------------------

function moments_kr_qmc(cfg::Config, κs)
    set_kr_base_backend!(:mvnormalcdf)
    d = TruncatedMvNormal(cfg.μ, cfg.Σ, cfg.a, cfg.b; max_moment_levels = 4)
    out = Dict{Vector{Int}, Float64}()
    for κ in κs
        out[κ] = raw_moment(d, κ)
    end
    return out
end

# ----------------------------------------------------------------------
# Method (b): direct adaptive cubature on each moment, with ±Inf clipped
# to μ ± 10·σ.
# ----------------------------------------------------------------------

function clip_inf_bounds(cfg::Config)
    a = copy(cfg.a)
    b = copy(cfg.b)
    for i in 1:cfg.n
        σi = sqrt(cfg.Σ[i, i])
        if isinf(a[i]); a[i] = cfg.μ[i] - 10 * σi; end
        if isinf(b[i]); b[i] = cfg.μ[i] + 10 * σi; end
    end
    return a, b
end

function moment_hcubature(cfg::Config, κ; rtol, atol, maxevals)
    nrm = MvNormal(cfg.μ, cfg.Σ)
    a_clip, b_clip = clip_inf_bounds(cfg)
    integrand = x -> begin
        p = 1.0
        @inbounds for i in eachindex(κ)
            if κ[i] != 0
                p *= x[i]^κ[i]
            end
        end
        return p * pdf(nrm, x)
    end
    val, _ = hcubature(integrand, a_clip, b_clip;
                       rtol = rtol, atol = atol, maxevals = maxevals)
    return val
end

function moments_hcubature(cfg::Config, κs; rtol, atol, maxevals = 50_000_000)
    out = Dict{Vector{Int}, Float64}()
    for κ in κs
        out[κ] = moment_hcubature(cfg, κ; rtol = rtol, atol = atol,
                                  maxevals = maxevals)
    end
    return out
end

# Reference: tight-tolerance hcubature.
moments_reference(cfg::Config, κs) =
    moments_hcubature(cfg, κs; rtol = 1e-12, atol = 1e-14,
                      maxevals = 200_000_000)

# ----------------------------------------------------------------------
# Accuracy metric: max relative error across the moments (absolute when
# the reference value is below 1e-15).
# ----------------------------------------------------------------------

function max_relerr(approx::Dict, ref::Dict, κs)
    err = 0.0
    for κ in κs
        r = ref[κ]
        e = if abs(r) > 1e-15
            abs(approx[κ] - r) / abs(r)
        else
            abs(approx[κ] - r)
        end
        err = max(err, e)
    end
    return err
end

# ----------------------------------------------------------------------
# Output formatters for the LaTeX table rows
# ----------------------------------------------------------------------

fmt_time(t) = t < 1e-3 ? @sprintf("%.0f\\,\\textmu s", t * 1e6) :
              t < 1.0  ? @sprintf("%.1f\\,ms",        t * 1e3) :
                         @sprintf("%.2f\\,s",         t)

fmt_err(e) = e < 1e-12 ? @sprintf("%.1g", e) :
                         @sprintf("%.1e", e)

fmt_speedup(s) = @sprintf("%.0f\\times", s)

# ----------------------------------------------------------------------
# Per-configuration: reference + (KR+QMC, hcubature) timings + errors
# ----------------------------------------------------------------------

struct ConfigResult
    name::String
    n::Int
    M::Int          # number of moments (= binom(n+4, 4))
    kr_time::Float64
    kr_err::Float64
    hc_time::Float64
    hc_err::Float64
end

function run_config(cfg::Config; kr_samples = 20, kr_seconds = 10,
                                 hc_samples = 3, hc_seconds = 120)
    κs = multiindices(cfg.n, 4)
    M = length(κs)

    @info "config $(cfg.name) (n = $(cfg.n), M = $M): computing tight-tolerance reference..."
    t_ref = @elapsed (ref = moments_reference(cfg, κs))
    @info "config $(cfg.name): reference ready in $(round(t_ref, digits = 2)) s, m^{(0)} = $(round(ref[zeros(Int, cfg.n)], sigdigits = 6))"

    @info "config $(cfg.name): timing KR + Genz–Bretz QMC..."
    moments_kr = moments_kr_qmc(cfg, κs)  # warm-up & accuracy
    kr_err = max_relerr(moments_kr, ref, κs)
    kr_time = @belapsed moments_kr_qmc($cfg, $κs) samples=kr_samples seconds=kr_seconds

    @info "config $(cfg.name): timing direct hcubature..."
    moments_hc = moments_hcubature(cfg, κs; rtol = 1e-6, atol = 1e-10)
    hc_err = max_relerr(moments_hc, ref, κs)
    hc_time = @belapsed moments_hcubature($cfg, $κs;
                                          rtol = 1e-6, atol = 1e-10) samples=hc_samples seconds=hc_seconds

    return ConfigResult(cfg.name, cfg.n, M, kr_time, kr_err, hc_time, hc_err)
end

# ----------------------------------------------------------------------
# Main: run + print table.
# ----------------------------------------------------------------------

function main()
    println("\n", "="^80)
    println("Moment-table benchmark for the section 4 / tab:kr-bench")
    println("="^80)

    results = ConfigResult[]
    for cfg in CONFIGS
        @info "----- $(cfg.name): $(cfg.descr) -----"
        push!(results, run_config(cfg))
    end

    println()
    @printf("%-6s | %3s | %4s | %-23s | %-23s | %-8s\n",
            "Cfg", "n", "M", "KR + Genz–Bretz QMC", "Direct hcubature", "Speedup")
    @printf("%-6s | %3s | %4s | %-23s | %-23s | %-8s\n",
            "", "", "", "time (s) / max rel err", "time (s) / max rel err", "")
    println("-"^85)
    for r in results
        speedup = r.hc_time / r.kr_time
        @printf("%-6s | %3d | %4d | %.3e s / %.2e   | %.3e s / %.2e   | %6.1fx\n",
                r.name, r.n, r.M, r.kr_time, r.kr_err, r.hc_time, r.hc_err, speedup)
    end

    println()
    println("LaTeX table rows (paste into tab:kr-bench):")
    for r in results
        speedup = r.hc_time / r.kr_time
        @printf("%s & %d & %d & %s & %s & %s & %s & %s \\\\\n",
                r.name, r.n, r.M,
                fmt_time(r.kr_time), fmt_err(r.kr_err),
                fmt_time(r.hc_time), fmt_err(r.hc_err),
                fmt_speedup(speedup))
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
