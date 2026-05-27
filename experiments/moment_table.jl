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
    name::String                    # short label, e.g. "2A"
    n::Int                          # dimension
    descr::String                   # human-readable description
    μ::Vector{Float64}
    Σ::Matrix{Float64}
    a::Vector{Float64}              # may contain -Inf
    b::Vector{Float64}              # may contain +Inf
    # If non-empty, only these κ are evaluated (and compared). When
    # empty, the full order-≤4 multi-index set is used.
    single_κ::Union{Nothing, Vector{Int}}
    # If true, skip the direct-hcubature comparison (e.g. when the
    # cubature on the full moment table is infeasible).
    skip_hcubature::Bool
end

function bundled_config(label, n, idx, descr;
                        single_κ = nothing, skip_hcubature = false)
    ex = TruncatedDistributions.get_example(n = n, index = idx)
    return Config(label, n, descr,
                  Vector{Float64}(ex.μ), Matrix{Float64}(ex.Σ),
                  Vector{Float64}(ex.a), Vector{Float64}(ex.b),
                  single_κ, skip_hcubature)
end

const CONFIGS = [
    # --- KR vs hcubature on the full moment table (n = 2, 3) ---
    bundled_config("2A", 2, 1, "bundled n=2 ex 1: off-centre, mild correlation"),
    bundled_config("2B", 2, 4, "bundled n=2 ex 4: centred, mildly correlated, tight box"),
    bundled_config("3A", 3, 1, "bundled n=3 ex 1: off-centre, finite asymmetric box"),
    bundled_config("3G", 3, 3, "Genz–Bretz Phi3ex (G&B 2009 §1.3.1): centred, semi-infinite"),
    # --- Single complicated 4th-order moment at n = 4 ---
    # Direct hcubature on all 70 moments is too slow at the reference
    # tolerance; the joint cross-moment κ = (1,1,1,1) is enough to
    # show that even one 4D quartic-Gaussian integral is slower than
    # the KR pipeline computing the whole 70-moment table.
    bundled_config("4A", 4, 1,
                   "bundled n=4 ex 1: single κ = (1,1,1,1)";
                   single_κ = [1, 1, 1, 1]),
    # --- KR-only scalability sweep (n = 5, 6, 7) ---
    # hcubature on the full moment table is infeasible at these n,
    # so we report only the KR pipeline.
    bundled_config("5",  5, 1, "bundled n=5 ex 1: light truncation, tridiag Σ";
                   skip_hcubature = true),
    bundled_config("6",  6, 1, "bundled n=6 ex 1: light truncation, tridiag Σ";
                   skip_hcubature = true),
    bundled_config("7",  7, 1, "bundled n=7 ex 1: light truncation, tridiag Σ";
                   skip_hcubature = true),
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

# Reference: tight-tolerance hcubature. rtol = 1e-8 is more than
# four orders of magnitude tighter than either method being measured
# (Genz–Bretz QMC at m = 10_000 reaches ~1e-2 on the worst moments;
# direct adaptive cubature at rtol = 1e-6 reaches ~1e-6 by request).
# Tightening further to 1e-12 makes the n = 4 reference prohibitive.
moments_reference(cfg::Config, κs) =
    moments_hcubature(cfg, κs; rtol = 1e-8, atol = 1e-12,
                      maxevals = 50_000_000)

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
    M::Int                      # number of moments evaluated
    kr_time::Float64
    kr_err::Float64
    hc_time::Union{Float64, Nothing}   # `nothing` if skipped
    hc_err::Union{Float64, Nothing}
end

function run_config(cfg::Config; kr_samples = 5, kr_seconds = 30,
                                 hc_samples = 2, hc_seconds = 60)
    κs = cfg.single_κ === nothing ? multiindices(cfg.n, 4) : [cfg.single_κ]
    M = length(κs)

    if cfg.skip_hcubature
        @info "config $(cfg.name) (n = $(cfg.n), M = $M): KR only (skip_hcubature)"
        # No reference: kr_err is reported as NaN for the scalability sweep.
        moments_kr = moments_kr_qmc(cfg, κs)
        kr_time = @belapsed moments_kr_qmc($cfg, $κs) samples=kr_samples seconds=kr_seconds
        r = ConfigResult(cfg.name, cfg.n, M, kr_time, NaN, nothing, nothing)
        @info "config $(cfg.name): DONE  KR $(fmt_time(r.kr_time))  (no hcubature reference)"
        return r
    end

    @info "config $(cfg.name) (n = $(cfg.n), M = $M): computing tight-tolerance reference..."
    t_ref = @elapsed (ref = moments_reference(cfg, κs))
    @info "config $(cfg.name): reference ready in $(round(t_ref, digits = 2)) s"

    @info "config $(cfg.name): timing KR + Genz–Bretz QMC..."
    moments_kr = moments_kr_qmc(cfg, κs)
    kr_err = max_relerr(moments_kr, ref, κs)
    kr_time = @belapsed moments_kr_qmc($cfg, $κs) samples=kr_samples seconds=kr_seconds

    @info "config $(cfg.name): timing direct hcubature..."
    moments_hc = moments_hcubature(cfg, κs; rtol = 1e-6, atol = 1e-10)
    hc_err = max_relerr(moments_hc, ref, κs)
    hc_time = @belapsed moments_hcubature($cfg, $κs;
                                          rtol = 1e-6, atol = 1e-10) samples=hc_samples seconds=hc_seconds

    r = ConfigResult(cfg.name, cfg.n, M, kr_time, kr_err, hc_time, hc_err)
    speedup = r.hc_time / r.kr_time
    @info "config $(cfg.name): DONE  KR $(fmt_time(r.kr_time)) ($(fmt_err(r.kr_err)))  HC $(fmt_time(r.hc_time)) ($(fmt_err(r.hc_err)))  speedup $(fmt_speedup(speedup))"
    return r
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
        kr_s = @sprintf("%.3e s / %s", r.kr_time, isnan(r.kr_err) ? "  --  " : @sprintf("%.2e", r.kr_err))
        if r.hc_time === nothing
            @printf("%-6s | %3d | %4d | %-23s | %-23s | %-8s\n",
                    r.name, r.n, r.M, kr_s, "  (skipped)", "  --")
        else
            speedup = r.hc_time / r.kr_time
            hc_s = @sprintf("%.3e s / %.2e", r.hc_time, r.hc_err)
            @printf("%-6s | %3d | %4d | %s   | %s | %6.1fx\n",
                    r.name, r.n, r.M, kr_s, hc_s, speedup)
        end
    end

    println()
    println("LaTeX table rows (paste into tab:kr-bench):")
    for r in results
        kr_err_str = isnan(r.kr_err) ? "--" : fmt_err(r.kr_err)
        if r.hc_time === nothing
            @printf("%s & %d & %d & %s & %s & -- & -- & -- \\\\\n",
                    r.name, r.n, r.M, fmt_time(r.kr_time), kr_err_str)
        else
            speedup = r.hc_time / r.kr_time
            @printf("%s & %d & %d & %s & %s & %s & %s & %s \\\\\n",
                    r.name, r.n, r.M,
                    fmt_time(r.kr_time), kr_err_str,
                    fmt_time(r.hc_time), fmt_err(r.hc_err),
                    fmt_speedup(speedup))
        end
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
