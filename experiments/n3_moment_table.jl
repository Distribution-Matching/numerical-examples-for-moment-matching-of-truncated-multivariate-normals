#=
n3_moment_table.jl

Populates Table~\ref{tab:kr-bench} of the paper section "On efficient
computation of truncated moments". For each of three (μ, Σ, A) configurations
in n=3, compute the 35 primitive moments {m^{(κ)} : 0 ≤ |κ| ≤ 4} via
    (a) the Kan–Robotti recursion with the Genz–Bretz QMC base case
        (TruncatedDistributions.jl with set_kr_base_backend!(:mvnormalcdf)), and
    (b) direct adaptive cubature on each moment integral (HCubature.jl).
A tight-tolerance hcubature run gives the reference moments for the
max-relative-error column.

Run from this directory:
    julia --project=. n3_moment_table.jl

Wall-clock timings use BenchmarkTools.@belapsed with the methods' typical
defaults; for the reference we just run hcubature once at rtol=1e-12.
=#

using Pkg
Pkg.activate(@__DIR__)

using TruncatedDistributions
using Distributions
using HCubature
using BenchmarkTools
using LinearAlgebra
using Printf
using Statistics

# ----------------------------------------------------------------------
# Configurations (all n = 3).
# Config A matches eq:kr-example-params in the paper.
# ----------------------------------------------------------------------

const CONFIGS = [
    (name = "A",
     μ = [0.5, 0.0, -0.3],
     Σ = [1.0 0.5 0.3;
          0.5 1.0 0.2;
          0.3 0.2 1.0],
     a = [-1.0, -1.0, -1.0],
     b = [ 1.0,  1.0,  1.0]),
    (name = "B",
     μ = [0.0, 0.0, 0.0],
     Σ = Matrix{Float64}(I, 3, 3),
     a = [-2.0, -2.0, -2.0],
     b = [ 2.0,  2.0,  2.0]),
    (name = "C",
     μ = [-0.5,  0.2,  0.4],
     Σ = [1.0 0.7 0.5;
          0.7 1.0 0.4;
          0.5 0.4 1.0],
     a = [-2.0, -1.0, -1.0],
     b = [ 1.0,  1.5,  1.0]),
]

# ----------------------------------------------------------------------
# The 35 multi-indices κ = (κ_1, κ_2, κ_3) with |κ| ≤ 4.
# ----------------------------------------------------------------------

function multiindices(n::Int, L::Int)
    out = Vector{Vector{Int}}()
    function rec(prefix, remaining)
        if length(prefix) == n - 1
            push!(out, vcat(prefix, remaining))  # last entry consumes the remainder
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

const KAPPA_LIST = multiindices(3, 4)
@assert length(KAPPA_LIST) == 35

# ----------------------------------------------------------------------
# Method (a): Kan–Robotti + Genz–Bretz QMC via TruncatedDistributions.jl
# ----------------------------------------------------------------------

function moments_kr_qmc(μ, Σ, a, b)
    set_kr_base_backend!(:mvnormalcdf)
    d = TruncatedMvNormal(μ, Σ, a, b; max_moment_levels = 4)
    out = Dict{Vector{Int}, Float64}()
    for κ in KAPPA_LIST
        out[κ] = raw_moment(d, κ)
    end
    return out
end

# ----------------------------------------------------------------------
# Method (b): direct adaptive cubature on each of the 35 moments.
# ----------------------------------------------------------------------

function moment_hcubature(μ, Σ, a, b, κ; rtol, atol, maxevals = 10_000_000)
    nrm = MvNormal(μ, Σ)
    integrand = x -> begin
        p = 1.0
        for i in eachindex(κ)
            if κ[i] != 0
                p *= x[i]^κ[i]
            end
        end
        return p * pdf(nrm, x)
    end
    val, _ = hcubature(integrand, a, b; rtol = rtol, atol = atol,
                       maxevals = maxevals)
    return val
end

function moments_hcubature(μ, Σ, a, b; rtol, atol)
    out = Dict{Vector{Int}, Float64}()
    for κ in KAPPA_LIST
        out[κ] = moment_hcubature(μ, Σ, a, b, κ; rtol = rtol, atol = atol)
    end
    return out
end

# ----------------------------------------------------------------------
# Reference: tight-tolerance hcubature.
# ----------------------------------------------------------------------

moments_reference(μ, Σ, a, b) =
    moments_hcubature(μ, Σ, a, b; rtol = 1e-12, atol = 1e-14)

# ----------------------------------------------------------------------
# Accuracy metric: max relative error across the 35 moments (absolute for
# moments whose reference value is below 1e-15).
# ----------------------------------------------------------------------

function max_relerr(approx::Dict, ref::Dict)
    err = 0.0
    for κ in KAPPA_LIST
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
# Per-configuration: compute reference, then time + accuracy of (a), (b).
# ----------------------------------------------------------------------

struct ConfigResult
    name::String
    kr_time::Float64
    kr_err::Float64
    hc_time::Float64
    hc_err::Float64
end

function run_config(cfg)
    @info "config $(cfg.name): computing tight-tolerance reference (rtol = 1e-12)..."
    t_ref = @elapsed (ref = moments_reference(cfg.μ, cfg.Σ, cfg.a, cfg.b))
    @info "config $(cfg.name): reference ready in $(round(t_ref, digits = 2)) s, m^{(0)} = $(round(ref[[0,0,0]], sigdigits = 6))"

    # --- KR + Genz-Bretz QMC ---
    @info "config $(cfg.name): timing KR + Genz–Bretz QMC..."
    moments_kr_warmup = moments_kr_qmc(cfg.μ, cfg.Σ, cfg.a, cfg.b)  # warmup + accuracy
    kr_err = max_relerr(moments_kr_warmup, ref)
    kr_time = @belapsed moments_kr_qmc($(cfg.μ), $(cfg.Σ), $(cfg.a), $(cfg.b)) samples=20 seconds=10

    # --- direct hcubature ---
    @info "config $(cfg.name): timing direct hcubature (rtol = 1e-6)..."
    moments_hc_warmup = moments_hcubature(cfg.μ, cfg.Σ, cfg.a, cfg.b;
                                          rtol = 1e-6, atol = 1e-10)
    hc_err = max_relerr(moments_hc_warmup, ref)
    hc_time = @belapsed moments_hcubature($(cfg.μ), $(cfg.Σ), $(cfg.a), $(cfg.b);
                                          rtol = 1e-6, atol = 1e-10) samples=5 seconds=60

    return ConfigResult(cfg.name, kr_time, kr_err, hc_time, hc_err)
end

# ----------------------------------------------------------------------
# Main: run + print table.
# ----------------------------------------------------------------------

function main()
    println("\n", "="^80)
    println("n = 3 moment-table benchmark: 35 primitive moments per configuration")
    println("="^80)

    results = ConfigResult[]
    for cfg in CONFIGS
        push!(results, run_config(cfg))
    end

    println()
    @printf("%-8s | %-23s | %-23s | %-8s\n",
            "Config", "KR + Genz–Bretz QMC", "Direct hcubature", "Speedup")
    @printf("%-8s | %-23s | %-23s | %-8s\n",
            "", "time (s) / max rel err", "time (s) / max rel err", "")
    println("-"^80)
    for r in results
        speedup = r.hc_time / r.kr_time
        @printf("%-8s | %.3e s / %.2e   | %.3e s / %.2e   | %6.1fx\n",
                r.name, r.kr_time, r.kr_err, r.hc_time, r.hc_err, speedup)
    end

    println()
    println("LaTeX table rows (paste into tab:kr-bench, dropping the TODO row):")
    for r in results
        speedup = r.hc_time / r.kr_time
        @printf("%s & \\SI{%.2e}{\\second} & %.1e & \\SI{%.2e}{\\second} & %.1e & %.1f\$\\times\$ \\\\\n",
                r.name, r.kr_time, r.kr_err, r.hc_time, r.hc_err, speedup)
    end

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
